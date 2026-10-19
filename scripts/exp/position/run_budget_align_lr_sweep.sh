#!/usr/bin/env bash
# Qwen3 budget-align learning-rate sweep for Intent or math.
#
# Each selected setting is trained and evaluated at every LR trial. The
# underlying budget-align runner remains the source of truth for arm structure,
# batch size, and training protocol. DOMAIN=intent runs the 500-example Intent
# evaluation; DOMAIN=math trains on MetaMathQA-R1 and evaluates MATH500 only.

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
BASE_RUNNER="$REPO/scripts/exp/position/run_budget_align.sh"

ALL_ARMS="attn_r64 dense_r256 attn64_dense256 attn32_dense128"
MATH_ARMS="attn_r64 dense_r256 attn64_dense256 attn32_dense128"
LR_TRIALS="${LR_TRIALS:-lr1e-3 lr1e-4 lr1e-5}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
DOMAIN="${DOMAIN:-intent}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29920}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-${DRY_RUN:-0}}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
MODEL_SUBDIR="${MODEL_SUBDIR:-}"
RUN_TAG_SUFFIX="${RUN_TAG_SUFFIX:-}"
ATTACH_ROUTER="${ATTACH_ROUTER:-false}"
ROUTER_RANK="${ROUTER_RANK:-8}"
ROUTER_ALPHA="${ROUTER_ALPHA:-16}"

case "$DOMAIN" in
  intent)
    ARMS="${ARMS:-$ALL_ARMS}"
    EVALS="${EVALS:-intent}"
    RUN_EPOCHS="$INTENT_EPOCHS"
    ;;
  math)
    ARMS="${ARMS:-$MATH_ARMS}"
    EVALS="${EVALS:-math500}"
    RUN_EPOCHS=1
    ;;
  *)
    echo "Unknown DOMAIN '$DOMAIN' (use intent or math)." >&2
    exit 2
    ;;
esac

if ! [[ "$SEED" =~ ^[0-9]+$ && "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED and DATA_SEED must be non-negative integers; got SEED=$SEED DATA_SEED=$DATA_SEED." >&2
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-$REPO/saves/budget_align_lr/$DOMAIN}"
RES_ROOT="${RES_ROOT:-$REPO/results/budget_align_lr/$DOMAIN}"

SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"

lr_value() {
  case "$1" in
    lr1e-3) echo "1.0e-3" ;;
    lr1e-4) echo "1.0e-4" ;;
    lr1e-5) echo "1.0e-5" ;;
    *)
      echo "Unknown LR trial '$1' (use lr1e-3/lr1e-4/lr1e-5)." >&2
      return 1
      ;;
  esac
}

validate_arm() {
  case "$1" in
    attn_r64|dense_r256|attn64_dense256|attn32_dense128) ;;
    *)
      echo "Unknown setting '$1'." >&2
      return 1
      ;;
  esac
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_trial() {
  local arm="$1" trial="$2" lr="$3" port="$4"
  local -a cmd=(
    env
    "DOMAIN=$DOMAIN"
    PHASE=full
    "EVALS=$EVALS"
    "ARMS=$arm"
    "GPUS=$GPUS"
    "SEED=$SEED"
    "DATA_SEED=$DATA_SEED"
    "INTENT_EPOCHS=$INTENT_EPOCHS"
    "LEARNING_RATE=$lr"
    "RUN_TAG=${trial}${RUN_TAG_SUFFIX:+_$RUN_TAG_SUFFIX}"
    "MASTER_PORT=$port"
    "SKIP_EXISTING=$SKIP_EXISTING"
    "DS_CONFIG=$DS_CONFIG"
    "OUT_ROOT=$OUT_ROOT"
    "RES_ROOT=$RES_ROOT"
    "MODEL_SUBDIR=$MODEL_SUBDIR"
    "RUN_TAG_SUFFIX=$RUN_TAG_SUFFIX"
    "ATTACH_ROUTER=$ATTACH_ROUTER"
    "ROUTER_RANK=$ROUTER_RANK"
    "ROUTER_ALPHA=$ROUTER_ALPHA"
  )
  [[ -n "$SMOKE_MAX_STEPS" ]] && cmd+=("SMOKE_MAX_STEPS=$SMOKE_MAX_STEPS")
  [[ -n "$SMOKE_MAX_SAMPLES" ]] && cmd+=("SMOKE_MAX_SAMPLES=$SMOKE_MAX_SAMPLES")
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && cmd+=("SMOKE_EVAL_LIMIT=$SMOKE_EVAL_LIMIT")
  cmd+=(bash "$BASE_RUNNER")

  print_command "${cmd[@]}"
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  "${cmd[@]}"
}

if [[ ! -f "$BASE_RUNNER" ]]; then
  echo "Missing base runner: $BASE_RUNNER" >&2
  exit 2
fi

declare -a summary=()
fail_count=0
port="$BASE_MASTER_PORT"

echo "=================================================================="
echo " Qwen3 budget-align $DOMAIN LR sweep"
echo " settings: $ARMS"
echo " LR trials: $LR_TRIALS"
echo " GPUs: $GPUS | epochs=$RUN_EPOCHS | evals=$EVALS | skip_existing=$SKIP_EXISTING"
echo " outputs: $OUT_ROOT + $RES_ROOT"
echo "=================================================================="

for arm in $ARMS; do
  if ! validate_arm "$arm"; then
    summary+=("[FAIL] $arm (unknown setting)")
    fail_count=$((fail_count + 1))
    continue
  fi
  for trial in $LR_TRIALS; do
    if ! lr="$(lr_value "$trial")"; then
      summary+=("[FAIL] $arm/$trial (unknown LR trial)")
      fail_count=$((fail_count + 1))
      continue
    fi

    echo
    echo "========== [$arm/$trial] learning_rate=$lr =========="
    if run_trial "$arm" "$trial" "$lr" "$port"; then
      summary+=("[OK]   $arm/$trial train+$EVALS (lr=$lr)")
    else
      summary+=("[FAIL] $arm/$trial train+$EVALS (lr=$lr)")
      fail_count=$((fail_count + 1))
    fi
    port=$((port + 2))
  done
done

echo
echo "========== Qwen3 budget-align $DOMAIN LR sweep summary =========="
printf '  %s\n' "${summary[@]:-（无）}"
echo "Adapters: $OUT_ROOT/qwen3_<setting>_lr<value>_ep${RUN_EPOCHS}/"
echo "Results:  $RES_ROOT/qwen3/<setting>_lr<value>_ep${RUN_EPOCHS}/"

(( fail_count == 0 ))
