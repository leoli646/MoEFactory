#!/usr/bin/env bash
# Reproduce the lr=1e-4 budget-align-position table on Math.
#
# Target table:
#   results/budget_align_position/intent/plots/dense_router_positions_lr1e-4.csv
#
# Per model:
#   attn_r64, dense_r256, attn32_dense128, attn64_dense256
#   plus the same four arms with router LoRA r8/alpha16.
#
# Qwen3 base-arm Math results already exist under saves/results/budget_align_lr
# and are reused by default. This runner fills the remaining Qwen3 router arms
# and all Gemma4/GPT-OSS base+router arms, sequentially on one four-GPU node.

set -uo pipefail

MODE="${1:-run}"
case "$MODE" in
  run) RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-${DRY_RUN:-0}}" ;;
  dry-run) RUNNER_DRY_RUN=1 ;;
  *) echo "Usage: $0 [run|dry-run]" >&2; exit 2 ;;
esac

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
QWEN_RUNNER="$REPO/scripts/exp/position/run_budget_align.sh"
QWEN_LR_RUNNER="$REPO/scripts/exp/position/run_budget_align_lr_sweep.sh"
NM_RUNNER="$REPO/scripts/exp/position/run_budget_align_position_nm_math.sh"

ARMS="${ARMS:-attn_r64 dense_r256 attn32_dense128 attn64_dense256}"
GPUS="${GPUS:-0,1,2,3}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
ROUTER_RANK="${ROUTER_RANK:-8}"
ROUTER_ALPHA="${ROUTER_ALPHA:-16}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30220}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"

# Existing adapters/results are intentionally not copied into MoEFactory, so
# reproduce the four Qwen3 base arms by default. Set this to 0 only when
# QWEN_BASE_OUT_ROOT and QWEN_BASE_RES_ROOT point to complete reusable artifacts.
RUN_QWEN3_BASE="${RUN_QWEN3_BASE:-1}"
RUN_QWEN3_ROUTER="${RUN_QWEN3_ROUTER:-1}"
RUN_NEW_MODELS="${RUN_NEW_MODELS:-1}"

QWEN_BASE_OUT_ROOT="${QWEN_BASE_OUT_ROOT:-$REPO/saves/budget_align_lr/math}"
QWEN_BASE_RES_ROOT="${QWEN_BASE_RES_ROOT:-$REPO/results/budget_align_lr/math}"
OUT_ROOT="${OUT_ROOT:-$REPO/saves/budget_align_position/math}"
RES_ROOT="${RES_ROOT:-$REPO/results/budget_align_position/math}"

case "$RUNNER_DRY_RUN:$SKIP_EXISTING:$RUN_QWEN3_BASE:$RUN_QWEN3_ROUTER:$RUN_NEW_MODELS" in
  [01]:[01]:[01]:[01]:[01]) ;;
  *) echo "Dry-run, skip, and run-selection flags must each be 0 or 1." >&2; exit 2 ;;
esac
case "$LEARNING_RATE" in
  1e-4|1.0e-4|0.0001) ;;
  *) echo "This experiment is fixed to lr=1e-4, got LEARNING_RATE='$LEARNING_RATE'." >&2; exit 2 ;;
esac
if ! [[ "$ROUTER_RANK" =~ ^[1-9][0-9]*$ && "$ROUTER_ALPHA" =~ ^[1-9][0-9]*$ \
  && "$BASE_MASTER_PORT" =~ ^[0-9]+$ ]]; then
  echo "Router rank/alpha must be positive integers and BASE_MASTER_PORT must be numeric." >&2
  exit 2
fi
for runner in "$QWEN_RUNNER" "$QWEN_LR_RUNNER" "$NM_RUNNER"; do
  [[ -f "$runner" ]] || { echo "Missing runner: $runner" >&2; exit 2; }
done

_print_command() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
}

_run_array() {
  _print_command "$@"
  "$@"
}

qwen_base_complete() {
  local arm="$1"
  local adapter_dir="$QWEN_BASE_OUT_ROOT/qwen3_${arm}_lr1e-4_ep1"
  local result="$QWEN_BASE_RES_ROOT/qwen3/${arm}_lr1e-4_ep1/adapter_math500.jsonl"
  [[ -f "$adapter_dir/moe_lora_state.safetensors" ]] || return 1
  [[ -f "$adapter_dir/trainer_state.json" ]] || return 1
  [[ -f "$adapter_dir/train_results.json" && -s "$result" ]] || return 1
  python - "$result" <<'PY'
import json
import sys

indices = []
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                indices.append(int(json.loads(line)["idx"]))
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)

if len(indices) != 500 or sorted(indices) != list(range(500)):
    raise SystemExit(1)
PY
}

echo "=================================================================="
echo " budget-align-position Math (MetaMathQA-R1 -> MATH500)"
echo " arms=$ARMS"
echo " learning_rate=$LEARNING_RATE | router=r${ROUTER_RANK}/alpha${ROUTER_ALPHA}"
echo " GPUs=$GPUS | skip_existing=$SKIP_EXISTING"
echo " run_qwen3_base=$RUN_QWEN3_BASE (default 1: reproduce missing artifacts)"
echo " run_qwen3_router=$RUN_QWEN3_ROUTER | run_new_models=$RUN_NEW_MODELS"
echo " new adapters=$OUT_ROOT"
echo " new results=$RES_ROOT"
echo "=================================================================="

fail_count=0
declare -a summary=()

if [[ "$RUN_QWEN3_BASE" == "1" ]]; then
  cmd=(
    env
    DOMAIN=math
    "ARMS=$ARMS"
    LR_TRIALS=lr1e-4
    EVALS=math500
    "GPUS=$GPUS"
    "BASE_MASTER_PORT=$BASE_MASTER_PORT"
    "SKIP_EXISTING=$SKIP_EXISTING"
    "RUNNER_DRY_RUN=$RUNNER_DRY_RUN"
    "DS_CONFIG=$DS_CONFIG"
    "OUT_ROOT=$QWEN_BASE_OUT_ROOT"
    "RES_ROOT=$QWEN_BASE_RES_ROOT"
    bash "$QWEN_LR_RUNNER"
  )
  if _run_array "${cmd[@]}"; then
    summary+=("[OK]   qwen3 base arms lr1e-4")
  else
    summary+=("[FAIL] qwen3 base arms lr1e-4")
    fail_count=$((fail_count + 1))
  fi
else
  for arm in $ARMS; do
    if qwen_base_complete "$arm"; then
      echo "[REUSE] qwen3/$arm/lr1e-4: complete adapter + 500-row MATH500"
      summary+=("[REUSE] qwen3/$arm/lr1e-4")
    else
      echo "[MISSING] qwen3/$arm/lr1e-4 expected reuse artifact is incomplete; set RUN_QWEN3_BASE=1." >&2
      summary+=("[FAIL] qwen3/$arm/lr1e-4 reuse artifact missing")
      fail_count=$((fail_count + 1))
    fi
  done
fi

if [[ "$RUN_QWEN3_ROUTER" == "1" ]]; then
  cmd=(
    env
    DOMAIN=math
    PHASE=full
    EVALS=math500
    "ARMS=$ARMS"
    "GPUS=$GPUS"
    "LEARNING_RATE=$LEARNING_RATE"
    "RUN_TAG=router_r${ROUTER_RANK}_lr1e-4"
    ATTACH_ROUTER=true
    "ROUTER_RANK=$ROUTER_RANK"
    "ROUTER_ALPHA=$ROUTER_ALPHA"
    "MASTER_PORT=$((BASE_MASTER_PORT + 20))"
    "SKIP_EXISTING=$SKIP_EXISTING"
    "RUNNER_DRY_RUN=$RUNNER_DRY_RUN"
    "DS_CONFIG=$DS_CONFIG"
    "OUT_ROOT=$OUT_ROOT"
    "RES_ROOT=$RES_ROOT"
    bash "$QWEN_RUNNER"
  )
  if _run_array "${cmd[@]}"; then
    summary+=("[OK]   qwen3 four router-r${ROUTER_RANK} arms train+MATH500")
  else
    summary+=("[FAIL] qwen3 router-r${ROUTER_RANK} arms")
    fail_count=$((fail_count + 1))
  fi
fi

if [[ "$RUN_NEW_MODELS" == "1" ]]; then
  cmd=(
    env
    MODELS="gemma4 gptoss"
    "ARMS=$ARMS"
    VARIANTS="base router"
    "GPUS=$GPUS"
    "LEARNING_RATE=$LEARNING_RATE"
    MATH_EPOCHS=1
    "ROUTER_RANK=$ROUTER_RANK"
    "ROUTER_ALPHA=$ROUTER_ALPHA"
    "BASE_MASTER_PORT=$((BASE_MASTER_PORT + 60))"
    "SKIP_EXISTING=$SKIP_EXISTING"
    "RUNNER_DRY_RUN=$RUNNER_DRY_RUN"
    "DS_CONFIG=$DS_CONFIG"
    "OUT_ROOT=$OUT_ROOT"
    "RES_ROOT=$RES_ROOT"
    bash "$NM_RUNNER" run
  )
  if _run_array "${cmd[@]}"; then
    summary+=("[OK]   gemma4+gptoss base/router arms train+MATH500")
  else
    summary+=("[FAIL] gemma4+gptoss base/router arms")
    fail_count=$((fail_count + 1))
  fi
fi

echo
echo "========== budget-align-position Math summary =========="
printf '  %s\n' "${summary[@]:-no runs}"
echo "Reused Qwen3 base: $QWEN_BASE_OUT_ROOT + $QWEN_BASE_RES_ROOT"
echo "New adapters:      $OUT_ROOT"
echo "New MATH500:        $RES_ROOT/<model>/<arm>_<tag>_ep1/adapter_math500.jsonl"

(( fail_count == 0 ))
