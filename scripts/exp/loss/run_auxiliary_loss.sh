#!/usr/bin/env bash
# Balance-loss coefficient ablation for weight-granularity LoRA (expert_weight_r1),
# in the FFN+Router-only (no-attention) setting.
#
# Matrix per (MODEL, TASK): COEF_TRIALS x LR_TRIALS. Train/eval commands mirror the
# FFN-granularity runners exactly; the only additions are
#   moe_lora_balance_loss_coef=<c>  moe_lora_balance_loss_schedule=constant
# (constant schedule = the aux-sweep convention; per-layer-mean LBL, ideal value = top_k).
# aux0 (coef=0) is byte-identical to the noattn granularity chain's expert_weight_r1 arm;
# with REUSE_COEF0=1 (default) a complete result there is copied instead of retrained.

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"

MODEL="${MODEL:?Set MODEL=qwen3|gemma4|gptoss}"
TASK="${TASK:?Set TASK=intent|math}"
ARM="${ARM:-expert_weight_r1}"
COEF_TRIALS="${COEF_TRIALS:-aux0 aux1e-4 aux1e-3 aux1e-2 aux1e-1}"
LR_TRIALS="${LR_TRIALS:-lr1e-4 lr1e-3}"
ATTACH_ATTN="${ATTACH_ATTN:-0}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
SEED_SUBDIR="${SEED_SUBDIR:-seed$SEED}"
PHASE="${PHASE:-full}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30760}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-0}"
REUSE_COEF0="${REUSE_COEF0:-1}"
UNIFIED_LAYOUT="${UNIFIED_LAYOUT:-0}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
OUT_ROOT="${OUT_ROOT:-$REPO/saves/aux_coef_noattn}"
RES_ROOT="${RES_ROOT:-$REPO/results/aux_coef_noattn}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
MATH_EPOCHS="${MATH_EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-80000}"
INTENT_EVAL_BATCH_SIZE="${INTENT_EVAL_BATCH_SIZE:-8}"
MATH_EVAL_BATCH_SIZE="${MATH_EVAL_BATCH_SIZE:-32}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
MATH500_EXPECTED_N="${MATH500_EXPECTED_N:-500}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"

case "$ATTACH_ATTN" in
  0) ATTACH_ATTN_BOOL=false; GRAN_SUFFIX="_noattn" ;;
  1) ATTACH_ATTN_BOOL=true; GRAN_SUFFIX="" ;;
  *) echo "ATTACH_ATTN must be 0 or 1; got '$ATTACH_ATTN'." >&2; exit 2 ;;
esac
case "$PHASE" in
  full|train|eval) ;;
  *) echo "PHASE must be full, train, or eval; got '$PHASE'." >&2; exit 2 ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ && "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED and DATA_SEED must be non-negative integers; got SEED=$SEED DATA_SEED=$DATA_SEED." >&2
  exit 2
fi
if ! [[ "$SEED_SUBDIR" =~ ^seed[0-9]+$ ]]; then
  echo "SEED_SUBDIR must have the form seed<integer>; got '$SEED_SUBDIR'." >&2
  exit 2
fi

model_meta() {
  case "$MODEL" in
    qwen3)
      MODEL_PATH="${MODEL_PATH:-${QWEN3_BASE:-$CKPT_ROOT/Qwen3-30B-A3B-Base}}"
      BASE_CONFIG="$REPO/examples/ffn_granularity/qwen3_math.yaml"
      MODEL_N_EXPERTS=128; MODEL_TOP_K=8
      ;;
    gemma4)
      MODEL_PATH="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT_ROOT/gemma-4-26B-A4B}}"
      BASE_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gemma4_math.yaml"
      MODEL_N_EXPERTS=128; MODEL_TOP_K=8
      ;;
    gptoss)
      MODEL_PATH="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT_ROOT/gpt-oss-20b}}"
      MODEL_N_EXPERTS=32; MODEL_TOP_K=4
      BASE_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gptoss_math.yaml"
      ;;
    *) echo "MODEL must be qwen3, gemma4, or gptoss; got '$MODEL'." >&2; return 1 ;;
  esac
}

task_meta() {
  case "$TASK" in
    intent)
      EPOCHS="$INTENT_EPOCHS"
      TASK_FLAGS=(dataset=esft_intent template=esft_raw cutoff_len=4096 packing=false)
      if [[ "$MODEL" == "gemma4" ]]; then
        TRAIN_BATCH=(per_device_train_batch_size=4 gradient_accumulation_steps=9)
      else
        TRAIN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
      fi
      EXTRA_TRAIN_FLAGS=(); EXTRA_EVAL_FLAGS=()
      if [[ "$MODEL" == "gptoss" ]]; then
        EXTRA_TRAIN_FLAGS=(flash_attn=disabled); EXTRA_EVAL_FLAGS=(--flash-attn disabled)
      fi
      GRAN_ROOT_TAG="ffn_granularity_intent$GRAN_SUFFIX"
      ;;
    math)
      EPOCHS="$MATH_EPOCHS"
      TASK_FLAGS=(dataset=metamathqa_r1 cutoff_len=2048)
      TRAIN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
      EXTRA_TRAIN_FLAGS=(); EXTRA_EVAL_FLAGS=()
      GRAN_ROOT_TAG="ffn_granularity_math$GRAN_SUFFIX"
      ;;
    *) echo "TASK must be intent or math; got '$TASK'." >&2; return 1 ;;
  esac
}

lr_value() {
  case "$1" in
    lr1e-3) echo "1.0e-3" ;;
    lr1e-4) echo "1.0e-4" ;;
    lr1e-5) echo "1.0e-5" ;;
    *) echo "Unknown LR trial '$1'." >&2; return 1 ;;
  esac
}

coef_value() {
  case "$1" in
    aux0) echo "0.0" ;;
    aux1e-4) echo "1.0e-4" ;;
    aux1e-3) echo "1.0e-3" ;;
    aux1e-2) echo "1.0e-2" ;;
    aux1e-1) echo "1.0e-1" ;;
    aux*) echo "${1#aux}" ;;
    *) echo "Unknown coef trial '$1' (use auxN, e.g. aux1e-3)." >&2; return 1 ;;
  esac
}

set_arm_flags() {
  case "$1" in
    expert_weight_r1) ARM_RANK=1 ;;
    expert_weight_r2) ARM_RANK=2 ;;
    *) echo "ARM must be expert_weight_r1 or expert_weight_r2; got '$1'." >&2; return 1 ;;
  esac
  ARM_FLAGS=(
    moe_lora_ffn_granularity=expert_weight
    moe_lora_routing_mode=follow_moe
    "moe_lora_n_experts=$MODEL_N_EXPERTS"
    "moe_lora_top_k=$MODEL_TOP_K"
    "moe_lora_rank=$ARM_RANK"
    "moe_lora_alpha=$((2 * ARM_RANK))"
  )
}

print_command() { printf '+'; printf ' %q' "$@"; printf '\n'; }

prepare_log() {
  local logfile="$1"
  if [[ "$RUNNER_DRY_RUN" == "1" ]]; then printf '/dev/null'; return; fi
  mkdir -p "$(dirname "$logfile")"; touch "$logfile"; printf '%s' "$logfile"
}

run_logged() {
  local logfile="$1"; shift
  print_command "$@" | tee -a "$logfile"
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  "$@" 2>&1 | tee -a "$logfile"
  return "${PIPESTATUS[0]}"
}

adapter_complete() { [[ -f "$1/moe_lora_state.safetensors" ]]; }

intent_complete() {
  local result_dir="$1" expected="${SMOKE_EVAL_LIMIT:-500}" count
  [[ -f "$result_dir/intent_predictions.jsonl" && -f "$result_dir/intent_metrics.json" ]] || return 1
  count="$(wc -l < "$result_dir/intent_predictions.jsonl" 2>/dev/null || echo 0)"
  (( count >= expected ))
}

math500_complete() {
  local result_file="$1" expected="${SMOKE_EVAL_LIMIT:-$MATH500_EXPECTED_N}"
  [[ -s "$result_file" ]] || return 1
  "$PYTHON_BIN" - "$result_file" "$expected" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
try:
    idx = [int(json.loads(l)["idx"]) for l in open(path, encoding="utf-8") if l.strip()]
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if len(idx) == expected and sorted(idx) == list(range(expected)) else 1)
PY
}

result_complete() {
  local result_dir="$1"
  if [[ "$TASK" == "intent" ]]; then intent_complete "$result_dir"; else math500_complete "$result_dir/math500.jsonl"; fi
}

try_reuse_coef0() {
  # aux0 == the noattn granularity chain's expert_weight arm. Prefer the new
  # unified layout, then preserved local history, then an optional LEGACY_REPO.
  local result_dir="$1" logfile="$2" lr_trial="$3"
  [[ "$REUSE_COEF0" == "1" ]] || return 1
  local src_tag="${ARM}_${lr_trial}_ep${EPOCHS}"
  local src_res="" src_out="" legacy_repo candidate index
  legacy_repo="${LEGACY_REPO:-$REPO}"
  local -a result_candidates=(
    "$REPO/results/four_experiments/granularity/$ARM/$TASK/$MODEL/$SEED_SUBDIR/$src_tag"
    "$REPO/results/$GRAN_ROOT_TAG/$MODEL/$SEED_SUBDIR/$src_tag"
  )
  local -a output_candidates=(
    "$REPO/saves/four_experiments/granularity/$ARM/$TASK/$MODEL/$SEED_SUBDIR/$src_tag"
    "$REPO/saves/$GRAN_ROOT_TAG/$MODEL/$SEED_SUBDIR/$src_tag"
  )
  # Unlabelled historical runs belong to the original seed-42 protocol only.
  if [[ "$SEED" == "42" && "$DATA_SEED" == "42" ]]; then
    result_candidates+=(
      "$REPO/results/intent_repro/granularity/$TASK/$MODEL/$src_tag"
      "$REPO/results/$GRAN_ROOT_TAG/$MODEL/$src_tag"
      "$legacy_repo/results/$GRAN_ROOT_TAG/$MODEL/$src_tag"
    )
    output_candidates+=(
      "$REPO/saves/intent_repro/granularity/$TASK/$MODEL/$src_tag"
      "$REPO/saves/$GRAN_ROOT_TAG/$MODEL/$src_tag"
      "$legacy_repo/saves/$GRAN_ROOT_TAG/$MODEL/$src_tag"
    )
  fi
  for index in "${!result_candidates[@]}"; do
    candidate="${result_candidates[$index]}"
    if result_complete "$candidate" && adapter_complete "${output_candidates[$index]}"; then
      src_res="$candidate"
      src_out="${output_candidates[$index]}"
      break
    fi
  done
  [[ -n "$src_res" ]] || return 1
  [[ "$RUNNER_DRY_RUN" == "1" ]] && { echo "[reuse] would copy $src_res" | tee -a "$logfile"; return 0; }
  mkdir -p "$result_dir"
  cp -a "$src_res"/. "$result_dir/"
  printf '{"reused_from": "%s"}\n' "$src_res" > "$result_dir/aux0_provenance.json"
  if [[ -n "$src_out" && ! -e "$output_dir" ]]; then
    mkdir -p "$(dirname "$output_dir")"
    ln -s "$src_out" "$output_dir"
    echo "[reuse] aux0 adapter linked from $src_out" | tee -a "$logfile"
  fi
  echo "[reuse] aux0 result copied from $src_res" | tee -a "$logfile"
  result_complete "$result_dir"
}

run_trial() {
  local coef_trial="$1" lr_trial="$2" port="$3"
  local coef lr tag output_dir result_dir logfile
  coef="$(coef_value "$coef_trial")" || return 1
  lr="$(lr_value "$lr_trial")" || return 1
  tag="${ARM}_${lr_trial}_${coef_trial}_ep${EPOCHS}"
  if [[ "$UNIFIED_LAYOUT" == "1" ]]; then
    output_dir="$OUT_ROOT/$coef_trial/$TASK/$MODEL/$SEED_SUBDIR/$tag"
    result_dir="$RES_ROOT/$coef_trial/$TASK/$MODEL/$SEED_SUBDIR/$tag"
    logfile="$(prepare_log "$RES_ROOT/$coef_trial/$TASK/$MODEL/$SEED_SUBDIR/logs/$tag.log")"
  else
    output_dir="$OUT_ROOT/$MODEL/$TASK/$tag"
    result_dir="$RES_ROOT/$MODEL/$TASK/$tag"
    logfile="$(prepare_log "$RES_ROOT/$MODEL/$TASK/logs/$tag.log")"
  fi

  echo "[$MODEL/$TASK/$coef_trial/$lr_trial] coef=$coef lr=$lr output=$output_dir" | tee -a "$logfile"

  if [[ "$coef_trial" == "aux0" && "$SKIP_EXISTING" == "1" ]] && ! result_complete "$result_dir"; then
    if try_reuse_coef0 "$result_dir" "$logfile" "$lr_trial"; then
      echo "[$MODEL/$TASK/$coef_trial/$lr_trial] reused noattn granularity baseline; done." | tee -a "$logfile"
      return 0
    fi
  fi

  if [[ "$PHASE" == "train" || "$PHASE" == "full" ]]; then
    if [[ "$RUNNER_DRY_RUN" != "1" && "$SKIP_EXISTING" == "1" ]] && adapter_complete "$output_dir"; then
      echo "[$MODEL/$TASK/$coef_trial/$lr_trial] existing adapter found; training skipped." | tee -a "$logfile"
    else
      local max_samples="${SMOKE_MAX_SAMPLES:-$MAX_SAMPLES}"
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
        "${TASK_FLAGS[@]}"
        "max_samples=$max_samples"
        "num_train_epochs=$EPOCHS"
        "learning_rate=$lr"
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
        moe_lora_pool_share=per_layer
        moe_lora_w_share=per_layer
        moe_lora_target_layers=all
        "moe_lora_balance_loss_coef=$coef"
        moe_lora_balance_loss_schedule=constant
      )
      train_cmd+=("${TRAIN_BATCH[@]}" "${ARM_FLAGS[@]}" "${EXTRA_TRAIN_FLAGS[@]}")
      [[ -n "$SMOKE_MAX_STEPS" ]] && train_cmd+=("max_steps=$SMOKE_MAX_STEPS")
      train_cmd+=(
        report_to=none
        overwrite_output_dir=true
        "run_name=${MODEL}_${TASK}_${tag}"
        "output_dir=$output_dir"
      )
      run_logged "$logfile" "${train_cmd[@]}" || return 1
    fi
  fi

  if [[ "$PHASE" == "eval" || "$PHASE" == "full" ]]; then
    if [[ "$RUNNER_DRY_RUN" != "1" ]] && ! adapter_complete "$output_dir"; then
      echo "[$MODEL/$TASK/$coef_trial/$lr_trial] adapter missing; evaluation skipped." | tee -a "$logfile" >&2
      return 1
    fi
    if [[ "$RUNNER_DRY_RUN" != "1" && "$SKIP_EXISTING" == "1" ]] && result_complete "$result_dir"; then
      echo "[$MODEL/$TASK/$coef_trial/$lr_trial] complete result found; evaluation skipped." | tee -a "$logfile"
      return 0
    fi
    [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir"
    if [[ "$TASK" == "intent" ]]; then
      local -a eval_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=$GPUS"
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        "PYTHONPATH=$REPO/src:$REPO/eval_scripts/esft:$REPO${PYTHONPATH:+:$PYTHONPATH}"
        torchrun --nproc_per_node=4 "--master_port=$((port + 1))"
        "$REPO/eval_scripts/esft/eval_esft.py"
        --task intent --mode all --template esft_raw --prompt-protocol auto
        --runtime-merge --runtime-merge-device-map auto
        --hf-generate-batch-size "$INTENT_EVAL_BATCH_SIZE"
        --tensor-parallel-size 1
        --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
        --model-name-or-path "$MODEL_PATH"
        --adapter-name-or-path "$output_dir"
        --output-dir "$result_dir"
      )
      eval_cmd+=("${EXTRA_EVAL_FLAGS[@]}")
      [[ -n "$SMOKE_EVAL_LIMIT" ]] && eval_cmd+=(--debug --debug-samples "$SMOKE_EVAL_LIMIT")
      run_logged "$logfile" "${eval_cmd[@]}" || return 1
    else
      local -a eval_cmd=(
        env
        "CUDA_VISIBLE_DEVICES=$GPUS"
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        torchrun --nproc_per_node=4 "--master_port=$((port + 1))"
        "$REPO/eval_scripts/eval_math500.py"
        --base_model "$MODEL_PATH"
        --adapter_path "$output_dir"
        --batch_size "$MATH_EVAL_BATCH_SIZE"
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS"
        --save_path "$result_dir/math500.jsonl"
      )
      [[ -n "$SMOKE_EVAL_LIMIT" ]] && eval_cmd+=(--limit "$SMOKE_EVAL_LIMIT")
      run_logged "$logfile" "${eval_cmd[@]}" || return 1
    fi
  fi
}


model_meta || exit 2
task_meta || exit 2
set_arm_flags "$ARM" || exit 2
for t in $COEF_TRIALS; do coef_value "$t" >/dev/null || exit 2; done
for t in $LR_TRIALS; do lr_value "$t" >/dev/null || exit 2; done
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ "${#GPU_IDS[@]}" == "4" ]] || { echo "Need exactly 4 GPUs; GPUS='$GPUS'." >&2; exit 2; }

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
[[ -n "$PYTHON_BIN" && -f "$MODEL_PATH/config.json" && -f "$BASE_CONFIG" ]] || {
  echo "Missing Python, MODEL_PATH/config.json, or BASE_CONFIG." >&2; exit 2; }

echo "=================================================================="
echo " Balance-loss coefficient ablation (weight-granularity LoRA)"
echo " model=$MODEL task=$TASK arm=$ARM attach_attn=$ATTACH_ATTN (FFN+Router only)"
echo " coefs=$COEF_TRIALS x lrs=$LR_TRIALS | schedule=constant, per-layer-mean LBL"
echo " aux0 reuse from results/$GRAN_ROOT_TAG (REUSE_COEF0=$REUSE_COEF0)"
echo " seed=$SEED data_seed=$DATA_SEED"
echo " outputs=$OUT_ROOT/<coef>/$TASK/$MODEL/$SEED_SUBDIR results=$RES_ROOT/<coef>/$TASK/$MODEL/$SEED_SUBDIR"
echo " paper reference (main.tex:206-225): aux0=68.7%, aux1e-4=68.5%, aux1e-3=65.9%, aux1e-2=64.4%, aux1e-1=58.6%."
echo " phase=$PHASE GPUs=$GPUS skip_existing=$SKIP_EXISTING dry_run=$RUNNER_DRY_RUN"
echo "=================================================================="

declare -a SUMMARY=()
failures=0
port="$BASE_MASTER_PORT"
for lr_trial in $LR_TRIALS; do
  for coef_trial in $COEF_TRIALS; do
    if run_trial "$coef_trial" "$lr_trial" "$port"; then
      SUMMARY+=("[OK]   $MODEL/$TASK/$coef_trial/$lr_trial")
    else
      SUMMARY+=("[FAIL] $MODEL/$TASK/$coef_trial/$lr_trial")
      failures=$((failures + 1))
    fi
    port=$((port + 2))
  done
done

echo
echo "========== balance-coef ablation summary =========="
printf '  %s\n' "${SUMMARY[@]}"
(( failures == 0 ))
