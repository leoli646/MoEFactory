#!/usr/bin/env bash
# Intent-only budget-align experiments for Gemma 4 and GPT-OSS.
#
# The six arm names match the Qwen3 budget-align experiment.  The expert FFN
# rank is model-specific so that n_experts * rank remains comparable:
#   gemma4:  2 / 1, because it has 128 experts
#   gptoss:  8 / 4, because it has 32 experts
#
# For each model, one no-adapter Intent zero-shot baseline runs first, followed
# by the six train + adapter-Intent-eval arms.  This runner intentionally does
# not call any general-evaluation script.
#
# The default Intent learning rate is 1e-4; set LEARNING_RATE explicitly to
# override it without changing the output layout.

set -uo pipefail

MODE="${1:-run}"
case "$MODE" in
  run|inside) ;;
  *) echo "Usage: $0 [run|inside]" >&2; exit 2 ;;
esac

SCRIPT_PATH="$(readlink -f "$0")"
HERE="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-${DATA_ROOT:-$REPO/data}}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

MODELS="${MODELS:-gemma4 gptoss}"
ARMS="${ARMS:-attn_r64 dense_r256 attn64_dense256 attn32_dense128}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
SEED_SUBDIR="${SEED_SUBDIR:-}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29960}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-${DRY_RUN:-0}}"
RUN_TAG="${RUN_TAG:-}"
ATTACH_ROUTER="${ATTACH_ROUTER:-false}"
ROUTER_RANK="${ROUTER_RANK:-8}"
ROUTER_ALPHA="${ROUTER_ALPHA:-$((ROUTER_RANK * 2))}"
RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"

OUT_ROOT="${OUT_ROOT:-$REPO/saves/budget_align_nm/intent}"
RES_ROOT="${RES_ROOT:-$REPO/results/budget_align_nm/intent}"

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if [[ "$NPROC_PER_NODE" -lt 1 ]]; then
  echo "GPUS must contain at least one device: '$GPUS'" >&2
  exit 2
fi
case "$ATTACH_ROUTER" in
  true|false) ;;
  *) echo "ATTACH_ROUTER must be true or false, got '$ATTACH_ROUTER'." >&2; exit 2 ;;
esac
case "$RUNNER_DRY_RUN:$RUN_ZERO_SHOT" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUNNER_DRY_RUN and RUN_ZERO_SHOT must each be 0 or 1." >&2; exit 2 ;;
esac
if ! [[ "$ROUTER_RANK" =~ ^[1-9][0-9]*$ && "$ROUTER_ALPHA" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROUTER_RANK and ROUTER_ALPHA must be positive integers." >&2
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

_print_command() {
  printf '+'
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
  if [[ "$RUNNER_DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee -a "$logfile"
  return "${PIPESTATUS[0]}"
}

_adapter_complete() {
  local out="$1"
  [[ -f "$out/moe_lora_state.safetensors" ]] || return 1
  [[ -f "$out/trainer_state.json" ]] || return 1
  [[ -f "$out/train_results.json" ]] || return 1
}

_intent_result_complete() {
  local resdir="$1"
  local predictions="$resdir/intent_predictions.jsonl"
  local metrics="$resdir/intent_metrics.json"
  [[ -f "$predictions" && -f "$metrics" ]] || return 1
  local count
  count="$(wc -l < "$predictions" 2>/dev/null || echo 0)"
  (( count >= 500 ))
}

model_meta() {
  local model="$1"
  case "$model" in
    gemma4)
      MODEL_TAG=gemma4
      MODEL_BASE="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT_ROOT/gemma-4-26B-A4B}}"
      MODEL_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gemma4_math.yaml"
      MODEL_N_EXPERTS=128
      MODEL_TOP_K=8
      MODEL_FFN_RANK=2
      MODEL_FFN_BATCH=(per_device_train_batch_size=4 gradient_accumulation_steps=9)
      ;;
    gptoss)
      MODEL_TAG=gptoss
      MODEL_BASE="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT_ROOT/gpt-oss-20b}}"
      MODEL_CONFIG="$REPO/examples/train_moe_lora_newmodels/baseline1_gptoss_math.yaml"
      MODEL_N_EXPERTS=32
      MODEL_TOP_K=4
      MODEL_FFN_RANK=8
      MODEL_FFN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
      ;;
    *)
      echo "Unknown model '$model' (use gemma4 or gptoss)." >&2
      return 1
      ;;
  esac
  return 0
}

# Fill ARM_ARGS for one arm.  Expert ranks are selected from MODEL_FFN_RANK;
# dense ranks stay fixed because the six arm definitions use the same d->d
# budget convention as the original Qwen3 experiment.
arm_args() {
  local arm="$1"
  local half_ffn_rank=$(( MODEL_FFN_RANK / 2 ))
  local half_ffn_alpha=$(( half_ffn_rank * 2 ))
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
    attn64_dense256)
      ARM_ARGS+=(
        moe_lora_attach_ffn=true
        moe_lora_attach_attn=true
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        moe_lora_rank=256
        moe_lora_alpha=512
        moe_lora_attn_rank=64
        moe_lora_attn_alpha=128
      )
      ;;
    attn32_dense128)
      ARM_ARGS+=(
        moe_lora_attach_ffn=true
        moe_lora_attach_attn=true
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        moe_lora_rank=128
        moe_lora_alpha=256
        moe_lora_attn_rank=32
        moe_lora_attn_alpha=64
      )
      ;;
    *)
      echo "Unknown arm '$arm'." >&2
      return 1
      ;;
  esac
  return 0
}

run_train() {
  local model="$1" arm="$2" out="$3" port="$4" raw_log="$5"
  local logfile
  logfile="$(_prepare_log "$raw_log")"

  if [[ "$SKIP_EXISTING" == "1" ]] && _adapter_complete "$out"; then
    echo "[$model/$arm] complete adapter exists, training SKIP: $out" | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$out"

  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    FORCE_TORCHRUN=1
    "MASTER_PORT=$port"
    llamafactory-cli train
    "$MODEL_CONFIG"
    "deepspeed=$DS_CONFIG"
    "model_name_or_path=$MODEL_BASE"
    "dataset_dir=$DATA_ROOT"
    dataset=esft_intent
    template=esft_raw
    cutoff_len=4096
    packing=false
    max_samples=80000
    "num_train_epochs=$INTENT_EPOCHS"
    "seed=$SEED"
    "data_seed=$DATA_SEED"
  )
  cmd+=("${MODEL_FFN_BATCH[@]}")
  cmd+=("learning_rate=$LEARNING_RATE")
  cmd+=("${ARM_ARGS[@]}")
  [[ "$model" == "gptoss" ]] && cmd+=(flash_attn=disabled)
  cmd+=(report_to=none overwrite_output_dir=true \
        "run_name=${model}_intent_${arm}${RUN_TAG:+_$RUN_TAG}_ep${INTENT_EPOCHS}" "output_dir=$out")

  echo "========== [$model/$arm] Intent training ==========" | tee -a "$logfile"
  _run_array "$logfile" "${cmd[@]}"
}

run_eval() {
  local model="$1" arm="$2" adapter="$3" resdir="$4" port="$5" raw_log="$6"
  local logfile
  logfile="$(_prepare_log "$raw_log")"

  if [[ "$RUNNER_DRY_RUN" != "1" && ! -f "$adapter/moe_lora_state.safetensors" ]]; then
    echo "[$model/$arm] missing adapter: $adapter/moe_lora_state.safetensors" | tee -a "$logfile" >&2
    return 1
  fi
  if [[ "$SKIP_EXISTING" == "1" ]] && _intent_result_complete "$resdir"; then
    echo "[$model/$arm] complete 500-example Intent result exists, eval SKIP: $resdir" | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$resdir"

  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    "PYTHONPATH=$REPO/eval_scripts/esft:$REPO"
    torchrun
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$port"
    "$REPO/eval_scripts/esft/eval_esft.py"
    --esft-dir "$DATA_ROOT/esft_eval"
    --task intent
    --mode all
    --template esft_raw
    --prompt-protocol auto
    --runtime-merge
    --runtime-merge-device-map auto
  )
  [[ "$model" == "gptoss" ]] && cmd+=(--flash-attn disabled)
  cmd+=(
    --hf-generate-batch-size "$EVAL_BATCH_SIZE"
    --tensor-parallel-size 1
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --model-name-or-path "$MODEL_BASE"
    --adapter-name-or-path "$adapter"
    --output-dir "$resdir"
  )

  echo "========== [$model/$arm] Intent evaluation ==========" | tee -a "$logfile"
  _run_array "$logfile" "${cmd[@]}"
}

run_zero_shot() {
  local model="$1" resdir="$2" port="$3" raw_log="$4"
  local logfile
  logfile="$(_prepare_log "$raw_log")"

  if [[ "$SKIP_EXISTING" == "1" ]] && _intent_result_complete "$resdir"; then
    echo "[$model/zeroshot] complete 500-example Intent result exists, eval SKIP: $resdir" | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$resdir"

  # Keep the zero-shot protocol identical to adapter evaluation, but omit
  # --adapter-name-or-path so eval_esft loads the untouched base weights.
  local -a cmd=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    "PYTHONPATH=$REPO/eval_scripts/esft:$REPO"
    torchrun
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$port"
    "$REPO/eval_scripts/esft/eval_esft.py"
    --esft-dir "$DATA_ROOT/esft_eval"
    --task intent
    --mode all
    --template esft_raw
    --prompt-protocol auto
    --runtime-merge
    --runtime-merge-device-map auto
  )
  [[ "$model" == "gptoss" ]] && cmd+=(--flash-attn disabled)
  cmd+=(
    --hf-generate-batch-size "$EVAL_BATCH_SIZE"
    --tensor-parallel-size 1
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --model-name-or-path "$MODEL_BASE"
    --output-dir "$resdir"
  )

  echo "========== [$model/zeroshot] Intent zero-shot evaluation ==========" | tee -a "$logfile"
  _run_array "$logfile" "${cmd[@]}"
}


run_all() {
  cd "$REPO" || return 1
  local port="$BASE_MASTER_PORT" fail_count=0 model arm out resdir logfile zero_resdir zero_log suffix
  local -a summary=()

  echo "=================================================================="
  echo " Intent-only budget-align for Gemma4 + GPT-OSS"
  echo " models=$MODELS"
  echo " arms=$ARMS"
  echo " epochs=$INTENT_EPOCHS learning_rate=$LEARNING_RATE GPUs=$GPUS"
  echo " attach_router=$ATTACH_ROUTER router_rank=$ROUTER_RANK router_alpha=$ROUTER_ALPHA run_tag=${RUN_TAG:-none}"
  echo " outputs=$OUT_ROOT and $RES_ROOT"
  echo " seed=$SEED data_seed=$DATA_SEED"
  echo " zero-shot evaluation=$RUN_ZERO_SHOT"
  echo "=================================================================="

  for model in $MODELS; do
    if ! model_meta "$model"; then
      summary+=("[FAIL] $model (unknown model)")
      fail_count=$((fail_count + 1))
      continue
    fi
    if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
      [[ -d "$MODEL_BASE" ]] || { echo "Missing model path: $MODEL_BASE" >&2; summary+=("[FAIL] $model (missing model)"); fail_count=$((fail_count + 1)); continue; }
      [[ -f "$MODEL_CONFIG" ]] || { echo "Missing train config: $MODEL_CONFIG" >&2; summary+=("[FAIL] $model (missing config)"); fail_count=$((fail_count + 1)); continue; }
    fi
    if [[ "$RUN_ZERO_SHOT" == "1" ]]; then
      model_root="$model${SEED_SUBDIR:+/$SEED_SUBDIR}"
      zero_resdir="$RES_ROOT/$model_root/zeroshot"
      zero_log="$RES_ROOT/$model_root/logs/zeroshot.log"
      if run_zero_shot "$model" "$zero_resdir" "$port" "$zero_log"; then
        summary+=("[OK]   $model/zeroshot intent")
      else
        summary+=("[FAIL] $model/zeroshot intent")
        fail_count=$((fail_count + 1))
      fi
      # Reserve a distinct port even when zero-shot is skipped via SKIP_EXISTING.
      port=$((port + 1))
    fi

    for arm in $ARMS; do
      if ! arm_args "$arm"; then
        summary+=("[FAIL] $model/$arm (unknown arm)")
        fail_count=$((fail_count + 1))
        continue
      fi
      suffix=""
      [[ -n "$RUN_TAG" ]] && suffix="_$RUN_TAG"
      model_root="$model${SEED_SUBDIR:+/$SEED_SUBDIR}"
      out="$OUT_ROOT/$model_root/${arm}${suffix}_ep${INTENT_EPOCHS}"
      resdir="$RES_ROOT/$model_root/${arm}${suffix}_ep${INTENT_EPOCHS}"
      logfile="$RES_ROOT/$model_root/logs/${arm}${suffix}_ep${INTENT_EPOCHS}.log"
      echo "[$model/$arm] effective ffn_rank=$MODEL_FFN_RANK (half=$((MODEL_FFN_RANK / 2)))" | tee -a "$(_prepare_log "$logfile")"

      if run_train "$model" "$arm" "$out" "$port" "$logfile"; then
        if run_eval "$model" "$arm" "$out" "$resdir" "$((port + 1))" "$logfile"; then
          summary+=("[OK]   $model/$arm train+intent (lr=$LEARNING_RATE)")
        else
          summary+=("[FAIL] $model/$arm intent eval")
          fail_count=$((fail_count + 1))
        fi
      else
        summary+=("[FAIL] $model/$arm training")
        fail_count=$((fail_count + 1))
      fi
      port=$((port + 2))
    done
  done

  echo
  echo "========== Intent budget-align summary =========="
  printf '  %s\n' "${summary[@]}"
  echo "Adapters: $OUT_ROOT/<gemma4|gptoss>/<arm>${RUN_TAG:+_$RUN_TAG}_ep${INTENT_EPOCHS}/"
  echo "Results:  $RES_ROOT/<gemma4|gptoss>/<arm>${RUN_TAG:+_$RUN_TAG}_ep${INTENT_EPOCHS}/"
  [[ "$RUN_ZERO_SHOT" == "1" ]] && echo "Zero-shot: $RES_ROOT/<gemma4|gptoss>/zeroshot/"
  (( fail_count == 0 ))
}

if [[ "$MODE" == "inside" || "$MODE" == "run" ]]; then
  run_all
  exit $?
fi
