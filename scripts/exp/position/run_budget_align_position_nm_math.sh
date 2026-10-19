#!/usr/bin/env bash
# Gemma4/GPT-OSS budget-align-position experiments on Math.
#
# This is the Math counterpart of the lr=1e-4 Intent entries collected in:
#   results/budget_align_position/intent/plots/dense_router_positions_lr1e-4.csv
#
# For each model it trains/evaluates four budget-aligned base arms and/or their
# router-r8/alpha16 variants. Training uses MetaMathQA-R1 for one epoch; the
# only evaluation is MATH500. All runs are sequential on the same GPU set.

set -uo pipefail

MODE="${1:-run}"
case "$MODE" in
  run) RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-${DRY_RUN:-0}}" ;;
  dry-run) RUNNER_DRY_RUN=1 ;;
  *) echo "Usage: $0 [run|dry-run]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/common_nm.sh"

MODELS="${MODELS:-gemma4 gptoss}"
ARMS="${ARMS:-attn_r64 dense_r256 attn32_dense128 attn64_dense256}"
VARIANTS="${VARIANTS:-base router}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
SEED_SUBDIR="${SEED_SUBDIR:-}"
MATH_EPOCHS="${MATH_EPOCHS:-1}"
ROUTER_RANK="${ROUTER_RANK:-8}"
ROUTER_ALPHA="${ROUTER_ALPHA:-16}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30320}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MATH500_EXPECTED_N="${MATH500_EXPECTED_N:-500}"

if [[ -n "$SMOKE_MAX_STEPS$SMOKE_MAX_SAMPLES$SMOKE_EVAL_LIMIT" ]]; then
  OUT_ROOT="${OUT_ROOT:-$REPO/saves/budget_align_position_smoke/math}"
  RES_ROOT="${RES_ROOT:-$REPO/results/budget_align_position_smoke/math}"
else
  OUT_ROOT="${OUT_ROOT:-$REPO/saves/budget_align_position/math}"
  RES_ROOT="${RES_ROOT:-$REPO/results/budget_align_position/math}"
fi

case "$RUNNER_DRY_RUN:$SKIP_EXISTING" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUNNER_DRY_RUN and SKIP_EXISTING must each be 0 or 1." >&2; exit 2 ;;
esac
case "$LEARNING_RATE" in
  1e-3|1.0e-3|0.001) LR_TAG=lr1e-3 ;;
  1e-4|1.0e-4|0.0001) LR_TAG=lr1e-4 ;;
  1e-5|1.0e-5|0.00001) LR_TAG=lr1e-5 ;;
  *) echo "LEARNING_RATE must be 1e-3, 1e-4, or 1e-5; got '$LEARNING_RATE'." >&2; exit 2 ;;
esac
if ! [[ "$MATH_EPOCHS" =~ ^[1-9][0-9]*$ && "$ROUTER_RANK" =~ ^[1-9][0-9]*$ \
  && "$ROUTER_ALPHA" =~ ^[1-9][0-9]*$ && "$BASE_MASTER_PORT" =~ ^[0-9]+$ ]]; then
  echo "Epochs/ranks/alpha must be positive integers and BASE_MASTER_PORT must be numeric." >&2
  exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ && "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED and DATA_SEED must be non-negative integers; got SEED=$SEED DATA_SEED=$DATA_SEED." >&2
  exit 2
fi
if [[ -n "$SEED_SUBDIR" && ! "$SEED_SUBDIR" =~ ^seed[0-9]+$ ]]; then
  echo "SEED_SUBDIR must be empty or have the form seed<integer>; got '$SEED_SUBDIR'." >&2
  exit 2
fi

validate_model() {
  case "$1" in
    gemma4|gptoss) ;;
    *) echo "Unknown model '$1' (use gemma4 or gptoss)." >&2; return 1 ;;
  esac
}

validate_arm() {
  case "$1" in
    attn_r64|dense_r256|attn32_dense128|attn64_dense256) ;;
    *) echo "Unknown position-budget arm '$1'." >&2; return 1 ;;
  esac
}

validate_variant() {
  case "$1" in
    base|router) ;;
    *) echo "Unknown variant '$1' (use base or router)." >&2; return 1 ;;
  esac
}

for model in $MODELS; do validate_model "$model" || exit 2; done
for arm in $ARMS; do validate_arm "$arm" || exit 2; done
for variant in $VARIANTS; do validate_variant "$variant" || exit 2; done

_print_command() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
}

_prepare_log() {
  local logfile="$1"
  if [[ "$RUNNER_DRY_RUN" == "1" ]]; then
    printf '/dev/null'
    return
  fi
  mkdir -p "$(dirname "$logfile")"
  touch "$logfile"
  printf '%s' "$logfile"
}

_run_array() {
  local logfile="$1"
  shift
  _print_command "$@" | tee -a "$logfile"
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  "$@" 2>&1 | tee -a "$logfile"
  return "${PIPESTATUS[0]}"
}

_adapter_complete() {
  local out="$1"
  [[ -f "$out/moe_lora_state.safetensors" ]] || return 1
  [[ -f "$out/trainer_state.json" ]] || return 1
  [[ -f "$out/train_results.json" ]]
}

_math500_complete() {
  local path="$1" expected="$2"
  [[ -s "$path" ]] || return 1
  "${PYTHON_BIN:-python}" - "$path" "$expected" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
indices = []
try:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                indices.append(int(row["idx"]))
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

if len(indices) != expected or sorted(indices) != list(range(expected)):
    raise SystemExit(1)
PY
}

variant_meta() {
  case "$1" in
    base)
      ATTACH_ROUTER=false
      RUN_TAG="$LR_TAG"
      ;;
    router)
      ATTACH_ROUTER=true
      RUN_TAG="router_r${ROUTER_RANK}_${LR_TAG}"
      ;;
  esac
}

arm_args() {
  local arm="$1"
  ARM_ARGS=(
    "moe_lora_attach_router=$ATTACH_ROUTER"
    moe_lora_attn_targets=auto
  )
  if [[ "$ATTACH_ROUTER" == "true" ]]; then
    ARM_ARGS+=(
      "moe_lora_router_rank=$ROUTER_RANK"
      "moe_lora_router_alpha=$ROUTER_ALPHA"
    )
  fi

  case "$arm" in
    attn_r64)
      ARM_ARGS+=(
        moe_lora_attach_ffn=false
        moe_lora_attach_attn=true
        moe_lora_attn_rank=64
        moe_lora_attn_alpha=128
      )
      ;;
    dense_r256)
      ARM_ARGS+=(
        moe_lora_attach_ffn=true
        moe_lora_attach_attn=false
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        moe_lora_rank=256
        moe_lora_alpha=512
      )
      ;;
    attn32_dense128)
      ARM_ARGS+=(
        moe_lora_attach_ffn=true
        moe_lora_attach_attn=true
        moe_lora_attn_rank=32
        moe_lora_attn_alpha=64
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        moe_lora_rank=128
        moe_lora_alpha=256
      )
      ;;
    attn64_dense256)
      ARM_ARGS+=(
        moe_lora_attach_ffn=true
        moe_lora_attach_attn=true
        moe_lora_attn_rank=64
        moe_lora_attn_alpha=128
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        moe_lora_rank=256
        moe_lora_alpha=512
      )
      ;;
  esac
}

run_train() {
  local model="$1" arm="$2" out="$3" port="$4" logfile="$5"
  if [[ "$SKIP_EXISTING" == "1" ]] && _adapter_complete "$out"; then
    echo "[$model/$arm/$RUN_TAG] complete adapter exists, training SKIP: $out" | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$out"

  local -a task_args cmd
  read -r -a task_args <<< "$TTRAIN_OVERRIDE"
  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    FORCE_TORCHRUN=1
    "MASTER_PORT=$port"
    llamafactory-cli train
    "$MBASELINE_CFG"
    "deepspeed=$DS_CONFIG"
    "model_name_or_path=${MODEL_BASE:-$MBASE}"
    "dataset_dir=$DATA_ROOT"
    "learning_rate=$LEARNING_RATE"
    "seed=$SEED"
    "data_seed=$DATA_SEED"
  )
  cmd+=("${task_args[@]}")
  cmd+=("${ARM_ARGS[@]}")
  cmd+=(
    report_to=none
    overwrite_output_dir=true
    "run_name=${model}_math_${arm}_${RUN_TAG}_ep${MATH_EPOCHS}"
    "output_dir=$out"
  )

  echo "========== [$model/$arm/$RUN_TAG] MetaMathQA-R1 training ==========" | tee -a "$logfile"
  _run_array "$logfile" "${cmd[@]}"
}

run_eval() {
  local model="$1" arm="$2" adapter="$3" resdir="$4" port="$5" logfile="$6"
  local expected="${SMOKE_EVAL_LIMIT:-$MATH500_EXPECTED_N}"
  local result="$resdir/adapter_math500.jsonl"

  if [[ "$RUNNER_DRY_RUN" != "1" ]] && ! _adapter_complete "$adapter"; then
    echo "[$model/$arm/$RUN_TAG] missing adapter: $adapter/moe_lora_state.safetensors" | tee -a "$logfile" >&2
    return 1
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && _math500_complete "$result" "$expected"; then
    echo "[$model/$arm/$RUN_TAG] complete MATH500 result exists, eval SKIP: $result" | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$resdir"

  local -a launch cmd _gpu_ids
  IFS=',' read -r -a _gpu_ids <<< "$GPUS"
  if (( ${#_gpu_ids[@]} > 1 )); then
    launch=(torchrun "--nproc_per_node=${#_gpu_ids[@]}" "--master_port=$port")
  else
    launch=("${PYTHON_BIN:-python}")
  fi
  cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    "${launch[@]}"
    "$REPO/eval_scripts/eval_math500.py"
    --base_model "$MBASE"
    --adapter_path "$adapter"
    --batch_size "$TEVAL_BSZ"
    --max_new_tokens "$EVAL_MAXNEW"
    --save_path "$result"
  )
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && cmd+=(--limit "$SMOKE_EVAL_LIMIT")

  echo "========== [$model/$arm/$RUN_TAG] MATH500 evaluation ==========" | tee -a "$logfile"
  _run_array "$logfile" "${cmd[@]}"
}

echo "=================================================================="
echo " Gemma4/GPT-OSS budget-align-position Math"
echo " models=$MODELS"
echo " arms=$ARMS"
echo " variants=$VARIANTS"
echo " train=MetaMathQA-R1 epochs=$MATH_EPOCHS lr=$LEARNING_RATE"
echo " seed=$SEED data_seed=$DATA_SEED"
echo " eval=MATH500 only | GPUs=$GPUS | skip_existing=$SKIP_EXISTING"
echo " router=r${ROUTER_RANK}/alpha${ROUTER_ALPHA}"
echo " adapters=$OUT_ROOT"
echo " results=$RES_ROOT"
echo "=================================================================="

port="$BASE_MASTER_PORT"
fail_count=0
declare -a summary=()

for model in $MODELS; do
  model_meta "$model" || { summary+=("[FAIL] $model metadata"); fail_count=$((fail_count + 1)); continue; }
  task_meta math || { summary+=("[FAIL] $model Math protocol"); fail_count=$((fail_count + 1)); continue; }
  if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
    [[ -d "$MBASE" ]] || { echo "Missing model path: $MBASE" >&2; summary+=("[FAIL] $model missing model"); fail_count=$((fail_count + 1)); continue; }
    [[ -f "$MBASELINE_CFG" ]] || { echo "Missing config: $MBASELINE_CFG" >&2; summary+=("[FAIL] $model missing config"); fail_count=$((fail_count + 1)); continue; }
  fi

  for variant in $VARIANTS; do
    variant_meta "$variant"
    for arm in $ARMS; do
      arm_args "$arm"
      model_root="$model${SEED_SUBDIR:+/$SEED_SUBDIR}"
      out="$OUT_ROOT/$model_root/${arm}_${RUN_TAG}_ep${MATH_EPOCHS}"
      resdir="$RES_ROOT/$model_root/${arm}_${RUN_TAG}_ep${MATH_EPOCHS}"
      logfile="$(_prepare_log "$RES_ROOT/$model_root/logs/${arm}_${RUN_TAG}_ep${MATH_EPOCHS}.log")"

      if run_train "$model" "$arm" "$out" "$port" "$logfile"; then
        if run_eval "$model" "$arm" "$out" "$resdir" "$((port + 1))" "$logfile"; then
          summary+=("[OK]   $model/$arm/$RUN_TAG train+MATH500")
        else
          summary+=("[FAIL] $model/$arm/$RUN_TAG MATH500")
          fail_count=$((fail_count + 1))
        fi
      else
        summary+=("[FAIL] $model/$arm/$RUN_TAG training")
        fail_count=$((fail_count + 1))
      fi
      port=$((port + 2))
    done
  done
done

echo
echo "========== new-model budget-align-position Math summary =========="
printf '  %s\n' "${summary[@]:-no runs}"
echo "Adapters: $OUT_ROOT/<gemma4|gptoss>/<arm>_<lr-or-router-tag>_ep${MATH_EPOCHS}/"
echo "Results:  $RES_ROOT/<gemma4|gptoss>/<arm>_<lr-or-router-tag>_ep${MATH_EPOCHS}/adapter_math500.jsonl"

(( fail_count == 0 ))
