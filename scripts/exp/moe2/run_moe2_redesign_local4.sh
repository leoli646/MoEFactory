#!/usr/bin/env bash
# Native P0 single-factor -> P1 combined arm -> P2 measured-results workflow.
# Preview is the default. Activate the environment and prepare data/models first.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 2
export MOEFACTORY_DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
MODEL="${MODEL:-qwen3}"
GPUS="${GPUS:-0,1,2,3}"
PHASES="${PHASES:-P0 P1 P2}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE="${SMOKE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TRAIN_GAS="${TRAIN_GAS:-6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RES_UP="${RES_UP:-$REPO/results/upstream_moe2_redesign/$MODEL}"
OUT_ROOT_P0="${OUT_ROOT_P0:-$REPO/saves/moe2_redesign_p0}"
RES_ROOT_P0="${RES_ROOT_P0:-$REPO/results/moe2_redesign_p0}"
OUT_ROOT_P1="${OUT_ROOT_P1:-$REPO/saves/moe2_redesign_sall}"
RES_ROOT_P1="${RES_ROOT_P1:-$REPO/results/moe2_redesign_sall}"
# Isolate short smoke adapters/results from complete experiments.
if (( SMOKE )); then
  OUT_ROOT_P0="${SMOKE_OUT_ROOT_P0:-${OUT_ROOT_P0}_smoke}"
  RES_ROOT_P0="${SMOKE_RES_ROOT_P0:-${RES_ROOT_P0}_smoke}"
  OUT_ROOT_P1="${SMOKE_OUT_ROOT_P1:-${OUT_ROOT_P1}_smoke}"
  RES_ROOT_P1="${SMOKE_RES_ROOT_P1:-${RES_ROOT_P1}_smoke}"
  RES_UP="${SMOKE_RES_UP:-${RES_UP}_smoke}"
fi
P0_LR="${P0_LR:-lr1e-3}"
P0_ARMS="${P0_ARMS:-wp-a11|n128_r32_wp_a11
wp-bz|n128_r32_wp_bz
wp-ntk|n128_r32_wp_ntk
wp-ntk-a128|n128_r32_wp_ntk_a128
wp-up|n128_r32_wp_up}"
S_ALL="${S_ALL:-n128_r32_wp_up_ntk_a11_bz}"
P1_LRS="${P1_LRS:-lr1e-3 lr1e-4}"
IFS=',' read -r -a gpu_ids <<< "$GPUS"
(( ${#gpu_ids[@]} == 4 )) || { echo 'GPUS requires four IDs' >&2; exit 2; }

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [[ "$DRY_RUN" == 1 ]]; then
    # Nested native previews print the actual train/eval commands, including overrides.
    case "${*: -1}" in
      scripts/exp/moe2/run_moe2_official_pilot.sh|scripts/reproduce/run_upstream_eval.sh) DRY_RUN=1 "$@" ;;
    esac
  else
    "$@"
  fi
}
train_downstream() {
  local setting="$1" lr="$2" output="$3" result="$4"
  local smoke=()
  (( SMOKE )) && smoke=(SMOKE_MAX_STEPS=3 SMOKE_MAX_SAMPLES=256 SMOKE_EVAL_LIMIT=16)
  run env "MODEL=$MODEL" TASK=intent "SETTINGS=$setting" "LR_TRIALS=$lr" \
    "RUNNER_DRY_RUN=$DRY_RUN" "GPUS=$GPUS" "TRAIN_GAS=$TRAIN_GAS" PHASE=full \
    "SKIP_EXISTING=$SKIP_EXISTING" "OUT_ROOT=$output" "RES_ROOT=$result" \
    "INTENT_EVAL_BATCH_SIZE=${BSZ:-8}" "${smoke[@]}" bash scripts/exp/moe2/run_moe2_official_pilot.sh
}
eval_upstream() {
  local label="$1" adapter="$2"
  if [[ "$DRY_RUN" != 1 && ! -f "$adapter/moe_lora_config.json" ]]; then
    echo "missing adapter: $adapter" >&2; return 1
  fi
  local mmlu="${MMLU_LIMIT:-0}" ifeval="${IFEVAL_LIMIT:-0}"
  (( SMOKE )) && { mmlu=64; ifeval=32; }
  run env "MODEL=$MODEL" "ADAPTER=$adapter" "OUT_DIR=$RES_UP/$label" "GPUS=$GPUS" \
    'TASKS=mmlu_pro ifeval' "ENGINE=${ENGINE:-hf}" "DRY_RUN=$DRY_RUN" \
    "SKIP_EXISTING=$SKIP_EXISTING" "MAX_LEN=${MAX_LEN:-4096}" \
    "IFEVAL_MAXNEW=${IFEVAL_MAXNEW:-1280}" "HF_BSZ=${BSZ:-8}" \
    "MMLU_LIMIT=$mmlu" "IFEVAL_LIMIT=$ifeval" bash scripts/reproduce/run_upstream_eval.sh
}
phase_p0() {
  local label setting failed=0
  while IFS='|' read -r label setting; do
    [[ -z "$label" ]] && continue
    [[ -n "$setting" ]] || { echo "invalid P0 arm: $label" >&2; return 2; }
    train_downstream "$setting" "$P0_LR" "$OUT_ROOT_P0" "$RES_ROOT_P0" || { failed=1; continue; }
    eval_upstream "$label" "$OUT_ROOT_P0/$MODEL/$P0_LR/intent/moe2official_$setting" || failed=1
  done <<< "$P0_ARMS"
  return "$failed"
}
phase_p1() {
  local lr failed=0
  for lr in $P1_LRS; do
    train_downstream "$S_ALL" "$lr" "$OUT_ROOT_P1" "$RES_ROOT_P1" || { failed=1; continue; }
    eval_upstream "sall-$lr" "$OUT_ROOT_P1/$MODEL/$lr/intent/moe2official_$S_ALL" || failed=1
  done
  return "$failed"
}
phase_p2() {
  local base=()
  [[ -n "${BASE_UPSTREAM_DIR:-}" ]] && base=(--base-upstream-dir "$BASE_UPSTREAM_DIR")
  run "$PYTHON_BIN" scripts/exp/moe2/collect_moe2_redesign.py --repo "$REPO" \
    --upstream-root "$RES_UP" --downstream-root "$RES_ROOT_P0" --downstream-root "$RES_ROOT_P1" \
    --out "$RES_UP/summary_$MODEL.csv" --model "$MODEL" "${base[@]}"
}
failures=0
for phase in $PHASES; do
  case "$phase" in
    P0) phase_p0 || failures=$((failures+1)) ;;
    P1) phase_p1 || failures=$((failures+1)) ;;
    P2) phase_p2 || failures=$((failures+1)) ;;
    *) echo "unknown phase: $phase" >&2; exit 2 ;;
  esac
done
echo "failures=$failures dry_run=$DRY_RUN"
exit "$((failures > 0))"
