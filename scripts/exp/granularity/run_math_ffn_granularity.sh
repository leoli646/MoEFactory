#!/usr/bin/env bash
# Run one model's Math FFN-granularity learning-rate sweep on four GPUs.
# Matrix: 4 FFN granularities x 3 learning rates = 12 settings.

set -uo pipefail

MODE="${1:-run}"
case "$MODE" in
  run) RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-0}" ;;
  dry-run) RUNNER_DRY_RUN=1 ;;
  *) echo "Usage: $0 [run|dry-run]" >&2; exit 2 ;;
esac

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"
MODEL="${MODEL:-}"

# Position ablation: ATTACH_ATTN=0 drops the attention LoRA branch (FFN+Router only).
# Outputs are isolated under *_noattn roots so the with-attention runs are never touched.
ATTACH_ATTN="${ATTACH_ATTN:-0}"
case "$ATTACH_ATTN" in
  0) ATTACH_ATTN_BOOL=false; NOATTN_SUFFIX="_noattn" ;;
  1) ATTACH_ATTN_BOOL=true; NOATTN_SUFFIX="" ;;
  *) echo "ATTACH_ATTN must be 0 or 1; got '$ATTACH_ATTN'." >&2; exit 2 ;;
esac

ARMS="${ARMS:-expert_weight_r1 expert_weight_r2 expert_bypass_r2 layer_bypass_r256}"
LR_TRIALS="${LR_TRIALS:-lr1e-4 lr1e-3 lr1e-5}"
GPUS="${GPUS:-0,1,2,3}"
PHASE="${PHASE:-full}"
MATH_EPOCHS="${MATH_EPOCHS:-1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30400}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
OUT_ROOT="${OUT_ROOT:-$REPO/saves/ffn_granularity_math$NOATTN_SUFFIX}"
RES_ROOT="${RES_ROOT:-$REPO/results/ffn_granularity_math$NOATTN_SUFFIX}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
MATH500_EXPECTED_N="${MATH500_EXPECTED_N:-500}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
SEED_SUBDIR="${SEED_SUBDIR:-}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"
BUDGET_CHECK="${BUDGET_CHECK:-$REPO/scripts/exp/granularity/check_intent_budget.py}"

model_meta() {
  case "$MODEL" in
    qwen3)
      MODEL_TAG=qwen3
      MODEL_PATH="${MODEL_PATH:-${QWEN3_BASE:-$CKPT_ROOT/Qwen3-30B-A3B-Base}}"
      BASE_CONFIG="$REPO/examples/ffn_granularity/qwen3_math.yaml"
      MODEL_N_EXPERTS=128
      MODEL_TOP_K=8
      ;;
    gemma4)
      MODEL_TAG=gemma4
      MODEL_PATH="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT_ROOT/gemma-4-26B-A4B}}"
      BASE_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gemma4_math.yaml"
      MODEL_N_EXPERTS=128
      MODEL_TOP_K=8
      ;;
    gptoss)
      MODEL_TAG=gptoss
      MODEL_PATH="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT_ROOT/gpt-oss-20b}}"
      BASE_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gptoss_math.yaml"
      MODEL_N_EXPERTS=32
      MODEL_TOP_K=4
      ;;
    *) echo "MODEL must be qwen3, gemma4, or gptoss; got '${MODEL:-}'." >&2; return 1 ;;
  esac
  TRAIN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
}

validate_arm() {
  case "$1" in
    layer_bypass_r*)
      local rank="${1#layer_bypass_r}"
      [[ "$rank" =~ ^[1-9][0-9]*$ ]] || {
        echo "Layer-bypass arms must use layer_bypass_r<positive-rank>; got '$1'." >&2
        return 1
      }
      ;;
    expert_bypass_r2|expert_weight_r1|expert_weight_r2) ;;
    *) echo "Unknown FFN-granularity arm '$1'." >&2; return 1 ;;
  esac
}

lr_value() {
  case "$1" in
    lr1e-3) echo "1.0e-3" ;;
    lr1e-4) echo "1.0e-4" ;;
    lr1e-5) echo "1.0e-5" ;;
    *) echo "Unknown LR trial '$1' (use lr1e-3/lr1e-4/lr1e-5)." >&2; return 1 ;;
  esac
}

set_arm_flags() {
  case "$1" in
    layer_bypass_r*)
      local rank="${1#layer_bypass_r}"
      ARM_FLAGS=(
        moe_lora_ffn_granularity=bypass
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        "moe_lora_rank=$rank"
        "moe_lora_alpha=$((rank * 2))"
      )
      ;;
    expert_bypass_r2)
      ARM_FLAGS=(
        moe_lora_ffn_granularity=bypass
        moe_lora_routing_mode=follow_moe
        "moe_lora_n_experts=$MODEL_N_EXPERTS"
        "moe_lora_top_k=$MODEL_TOP_K"
        moe_lora_rank=2
        moe_lora_alpha=4
      )
      ;;
    expert_weight_r1)
      ARM_FLAGS=(
        moe_lora_ffn_granularity=expert_weight
        moe_lora_routing_mode=follow_moe
        "moe_lora_n_experts=$MODEL_N_EXPERTS"
        "moe_lora_top_k=$MODEL_TOP_K"
        moe_lora_rank=1
        moe_lora_alpha=2
      )
      ;;
    expert_weight_r2)
      ARM_FLAGS=(
        moe_lora_ffn_granularity=expert_weight
        moe_lora_routing_mode=follow_moe
        "moe_lora_n_experts=$MODEL_N_EXPERTS"
        "moe_lora_top_k=$MODEL_TOP_K"
        moe_lora_rank=2
        moe_lora_alpha=4
      )
      ;;
  esac
}

print_command() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
}

prepare_log() {
  local logfile="$1"
  if [[ "$RUNNER_DRY_RUN" == "1" ]]; then
    printf '/dev/null'
    return
  fi
  mkdir -p "$(dirname "$logfile")"
  touch "$logfile"
  printf '%s' "$logfile"
}

run_logged() {
  local logfile="$1"
  shift
  print_command "$@" | tee -a "$logfile"
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  "$@" 2>&1 | tee -a "$logfile"
  return "${PIPESTATUS[0]}"
}

adapter_complete() {
  local output_dir="$1"
  [[ -f "$output_dir/moe_lora_state.safetensors" ]] || return 1
  [[ -f "$output_dir/trainer_state.json" ]] || return 1
  [[ -f "$output_dir/train_results.json" ]]
}

math500_complete() {
  local result_file="$1" expected="$2"
  [[ -s "$result_file" ]] || return 1
  "$PYTHON_BIN" - "$result_file" "$expected" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
indices = []
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                indices.append(int(json.loads(line)["idx"]))
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

if len(indices) != expected or sorted(indices) != list(range(expected)):
    raise SystemExit(1)
PY
}


run_trial() {
  local arm="$1" trial="$2" learning_rate="$3" port="$4"
  local tag="${arm}_${trial}_ep${MATH_EPOCHS}"
  local model_root="$MODEL_TAG${SEED_SUBDIR:+/$SEED_SUBDIR}"
  local output_dir="$OUT_ROOT/$model_root/$tag"
  local result_dir="$RES_ROOT/$model_root/$tag"
  local result_file="$result_dir/math500.jsonl"
  local expected="${SMOKE_EVAL_LIMIT:-$MATH500_EXPECTED_N}"
  local logfile
  logfile="$(prepare_log "$RES_ROOT/$model_root/logs/$tag.log")"
  set_arm_flags "$arm"

  echo "[$MODEL_TAG/$arm/$trial] lr=$learning_rate output=$output_dir" | tee -a "$logfile"

  if [[ "$PHASE" == "train" || "$PHASE" == "full" ]]; then
    if [[ "$SKIP_EXISTING" == "1" ]] && adapter_complete "$output_dir"; then
      echo "[$MODEL_TAG/$arm/$trial] complete adapter found; training skipped." | tee -a "$logfile"
    else
      local max_samples="${SMOKE_MAX_SAMPLES:-80000}"
      local -a train_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=$GPUS"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        FORCE_TORCHRUN=1
        "MASTER_PORT=$port"
        llamafactory-cli train "$BASE_CONFIG"
        "model_name_or_path=$MODEL_PATH"
        "deepspeed=$DS_CONFIG"
        "dataset_dir=$DATA_ROOT"
        dataset=metamathqa_r1
        cutoff_len=2048
        "max_samples=$max_samples"
        "num_train_epochs=$MATH_EPOCHS"
        "learning_rate=$learning_rate"
        "seed=$SEED"
        "data_seed=$DATA_SEED"
        moe_lora_attach_ffn=true
        "moe_lora_attach_attn=$ATTACH_ATTN_BOOL"
        moe_lora_attach_router=true
        moe_lora_attn_targets=auto
        moe_lora_attn_rank=32
        moe_lora_attn_alpha=64
        moe_lora_router_rank=8
        moe_lora_router_alpha=16
        moe_lora_b_init=normal
        moe_lora_wp_b_init=normal
        moe_lora_wl_init=default
        moe_lora_pool_share=per_layer
        moe_lora_w_share=per_layer
        moe_lora_target_layers=all
        moe_lora_balance_loss_coef=0.0
      )
      train_cmd+=("${TRAIN_BATCH[@]}")
      train_cmd+=("${ARM_FLAGS[@]}")
      [[ -n "$SMOKE_MAX_STEPS" ]] && train_cmd+=("max_steps=$SMOKE_MAX_STEPS")
      train_cmd+=(
        report_to=none
        overwrite_output_dir=true
        "run_name=${MODEL_TAG}_math_${tag}"
        "output_dir=$output_dir"
      )
      run_logged "$logfile" "${train_cmd[@]}" || return 1
    fi
  fi

  if [[ "$PHASE" == "eval" || "$PHASE" == "full" ]]; then
    if [[ "$RUNNER_DRY_RUN" != "1" ]] && ! adapter_complete "$output_dir"; then
      echo "[$MODEL_TAG/$arm/$trial] incomplete adapter; MATH500 evaluation cannot start." | tee -a "$logfile" >&2
      return 1
    fi
    if [[ "$SKIP_EXISTING" == "1" ]] && math500_complete "$result_file" "$expected"; then
      echo "[$MODEL_TAG/$arm/$trial] complete MATH500 result found; evaluation skipped." | tee -a "$logfile"
    else
      [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir"
      local -a eval_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=$GPUS"
        HF_HUB_OFFLINE=1
        HF_DATASETS_OFFLINE=1
        TRANSFORMERS_OFFLINE=1
        torchrun
        --nproc_per_node=4
        "--master_port=$((port + 1))"
        "$REPO/eval_scripts/eval_math500.py"
        --base_model "$MODEL_PATH"
        --adapter_path "$output_dir"
        --batch_size "$EVAL_BATCH_SIZE"
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS"
        --save_path "$result_file"
      )
      [[ -n "$SMOKE_EVAL_LIMIT" ]] && eval_cmd+=(--limit "$SMOKE_EVAL_LIMIT")
      run_logged "$logfile" "${eval_cmd[@]}" || return 1
    fi
  fi
}

case "$PHASE" in
  full|train|eval) ;;
  *) echo "PHASE must be full, train, or eval; got '$PHASE'." >&2; exit 2 ;;
esac
case "$RUNNER_DRY_RUN:$SKIP_EXISTING" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUNNER_DRY_RUN and SKIP_EXISTING must each be 0 or 1." >&2; exit 2 ;;
esac
if ! [[ "$MATH_EPOCHS" =~ ^[1-9][0-9]*$ && "$BASE_MASTER_PORT" =~ ^[0-9]+$ \
  && "$MATH500_EXPECTED_N" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ \
  && "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "MATH_EPOCHS/MATH500_EXPECTED_N/SEED/DATA_SEED must be non-negative integers; BASE_MASTER_PORT must be numeric." >&2
  exit 2
fi
if [[ -n "$SEED_SUBDIR" && ! "$SEED_SUBDIR" =~ ^seed[0-9]+$ ]]; then
  echo "SEED_SUBDIR must be empty or have the form seed<integer>; got '$SEED_SUBDIR'." >&2
  exit 2
fi
model_meta || exit 2
LAYER_BYPASS_RANKS=()
ARM_COUNT=0
for arm in $ARMS; do
  validate_arm "$arm" || exit 2
  ARM_COUNT=$((ARM_COUNT + 1))
  if [[ "$arm" =~ ^layer_bypass_r([1-9][0-9]*)$ ]]; then
    LAYER_BYPASS_RANKS+=("${BASH_REMATCH[1]}")
  fi
done
LR_COUNT=0
for trial in $LR_TRIALS; do
  lr_value "$trial" >/dev/null || exit 2
  LR_COUNT=$((LR_COUNT + 1))
done
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" != "4" ]]; then
  echo "This protocol requires four GPUs; GPUS='$GPUS' contains ${#GPU_IDS[@]} devices." >&2
  exit 2
fi

cd "$REPO" || exit 2
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Missing Python interpreter: $PYTHON_BIN" >&2; exit 2; }
# Preview needs only Python and the local model config; execution needs the active environment's CLIs.
if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
  if [[ "$PHASE" != "eval" ]]; then
    command -v llamafactory-cli >/dev/null 2>&1 || { echo "Missing llamafactory-cli in PATH; activate the installed environment." >&2; exit 2; }
  fi
  if [[ "$PHASE" != "train" ]]; then
    command -v torchrun >/dev/null 2>&1 || { echo "Missing torchrun in PATH; activate the installed environment." >&2; exit 2; }
  fi
fi
if [[ -z "$PYTHON_BIN" || ! -f "$MODEL_PATH/config.json" || ! -f "$BASE_CONFIG" || ! -f "$BUDGET_CHECK" ]]; then
  echo "Missing Python, MODEL_PATH/config.json, BASE_CONFIG, or BUDGET_CHECK." >&2
  exit 2
fi
budget_args=(--model "$MODEL_TAG" --config "$MODEL_PATH/config.json" --attach-attn "$ATTACH_ATTN")
if (( ${#LAYER_BYPASS_RANKS[@]} > 0 )); then
  budget_args+=(--layer-bypass-ranks "$(IFS=,; printf '%s' "${LAYER_BYPASS_RANKS[*]}")")
fi
"$PYTHON_BIN" "$BUDGET_CHECK" "${budget_args[@]}" || exit 2

echo "=================================================================="
echo " $MODEL_TAG Math FFN-granularity learning-rate sweep"
echo " arms=$ARMS"
echo " learning_rates=$LR_TRIALS"
echo " matrix=$ARM_COUNT arms x $LR_COUNT learning rates = $((ARM_COUNT * LR_COUNT)) settings"
if [[ "$ATTACH_ATTN" == "1" ]]; then
  echo " common=attention r32/a64 + Router r8/a16"
else
  echo " common=NO attention (position ablation) + Router r8/a16"
fi
echo " train=MetaMathQA-R1 epochs=$MATH_EPOCHS batch=6x6x4=144"
echo " eval=MATH500 | phase=$PHASE GPUs=$GPUS"
echo " skip_existing=$SKIP_EXISTING dry_run=$RUNNER_DRY_RUN"
echo " adapters=$OUT_ROOT"
echo " results=$RES_ROOT"
echo "=================================================================="

declare -a summary=()
fail_count=0
port="$BASE_MASTER_PORT"
for trial in $LR_TRIALS; do
  learning_rate="$(lr_value "$trial")"
  for arm in $ARMS; do
    if run_trial "$arm" "$trial" "$learning_rate" "$port"; then
      summary+=("[OK]   $MODEL_TAG/$arm/$trial")
    else
      summary+=("[FAIL] $MODEL_TAG/$arm/$trial")
      fail_count=$((fail_count + 1))
    fi
    port=$((port + 2))
  done
done


echo
echo "========== Math FFN-granularity summary =========="
printf '  %s\n' "${summary[@]}"
echo "Adapters: $OUT_ROOT/$MODEL_TAG/<arm>_<lr-trial>_ep${MATH_EPOCHS}/"
echo "Results:  $RES_ROOT/$MODEL_TAG/<arm>_<lr-trial>_ep${MATH_EPOCHS}/math500.jsonl"

(( fail_count == 0 ))
