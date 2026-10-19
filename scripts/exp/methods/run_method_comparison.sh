#!/usr/bin/env bash
# Granularity-aligned PEFT comparison: one model, both domain tasks, four GPUs.
#
# Default methods are the ten methods in the controlled Section 4.3 grid. MoE^2-LoRA is kept in scripts/exp/moe2 and is
# intentionally excluded from the controlled paper grid.

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"

MODEL="${MODEL:?Set MODEL=qwen3|gemma4|gptoss}"

# Position ablation: ATTACH_ATTN=0 drops the attention branch of every method
# (FFN+Router only). Outputs are isolated under *_noattn roots so the
# with-attention runs are never touched.
ATTACH_ATTN="${ATTACH_ATTN:-0}"
case "$ATTACH_ATTN" in
  0) ATTACH_ATTN_BOOL=false; NOATTN_SUFFIX="_noattn" ;;
  1) ATTACH_ATTN_BOOL=true; NOATTN_SUFFIX="" ;;
  *) echo "ATTACH_ATTN must be 0 or 1; got '$ATTACH_ATTN'." >&2; exit 2 ;;
esac

TASKS="${TASKS:-intent math}"
METHODS="${METHODS:-weight_lora expert_lora weight_dora weight_oft weight_fourier weight_ia3 expert_moe_lora_n8 layer_lora das_lora partial_last}"
PHASE="${PHASE:-full}"
GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
SEED_SUBDIR="${SEED_SUBDIR:-seed$SEED}"
MASTER_PORT="${MASTER_PORT:-29640}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-0}"
UNIFIED_LAYOUT="${UNIFIED_LAYOUT:-0}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
OUT_ROOT="${OUT_ROOT:-$REPO/saves/peft_compare_granularity$NOATTN_SUFFIX}"
RES_ROOT="${RES_ROOT:-$REPO/results/peft_compare_granularity$NOATTN_SUFFIX}"
MATH_EPOCHS="${MATH_EPOCHS:-1}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-80000}"
MATH_EVAL_BATCH_SIZE="${MATH_EVAL_BATCH_SIZE:-32}"
INTENT_EVAL_BATCH_SIZE="${INTENT_EVAL_BATCH_SIZE:-8}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
MATH500_EXPECTED_N="${MATH500_EXPECTED_N:-500}"
LR_TRIALS="${LR_TRIALS:-}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"
RUN_UPSTREAM="${RUN_UPSTREAM:-0}"
UPSTREAM_TASKS="${UPSTREAM_TASKS:-mmlu_pro ifeval}"
UPSTREAM_ENGINE="${UPSTREAM_ENGINE:-hf}"
UPSTREAM_MMLU_LIMIT="${UPSTREAM_MMLU_LIMIT:-0}"
UPSTREAM_IFEVAL_LIMIT="${UPSTREAM_IFEVAL_LIMIT:-0}"
DAS_MAXSAMPLES="${DAS_MAXSAMPLES:-200}"
DAS_GENERAL="${DAS_GENERAL:-$DATA_ROOT/general/commonsense_test/merged.json}"
DAS_JSON_OVERRIDE="${DAS_JSON:-}"
BUDGET_CHECK="${BUDGET_CHECK:-$REPO/scripts/exp/methods/check_budget.py}"

DEFAULT_METHODS=(
  weight_lora expert_lora weight_dora weight_oft weight_fourier weight_ia3 expert_moe_lora_n8 layer_lora das_lora partial_last
)
ALL_METHODS=(weight_lora expert_lora weight_dora weight_oft weight_fourier weight_ia3
  expert_moe_lora_n8 expert_moe2_lora layer_lora das_lora partial_last)

usage() {
  cat <<'EOF'
Usage:
  MODEL=qwen3 bash scripts/exp/methods/run_method_comparison.sh
  MODEL=gemma4 TASKS="intent math" PHASE=train bash scripts/exp/methods/run_method_comparison.sh
  MODEL=gptoss METHODS="weight_lora expert_lora" PHASE=full bash scripts/exp/methods/run_method_comparison.sh

Environment: TASKS, METHODS, LR_TRIALS, PHASE=full|train|eval,
RUNNER_DRY_RUN=1, SKIP_EXISTING=1, SMOKE_MAX_STEPS, SMOKE_MAX_SAMPLES,
SMOKE_EVAL_LIMIT.
RUN_UPSTREAM=1 optionally runs MMLU-Pro and IFEval through run_upstream_eval.sh (HF engine).
EOF
}

case "$PHASE" in
  full|train|eval) ;;
  *) echo "PHASE must be full, train, or eval; got '$PHASE'." >&2; exit 2 ;;
esac
case "$RUNNER_DRY_RUN:$SKIP_EXISTING" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "RUNNER_DRY_RUN and SKIP_EXISTING must each be 0 or 1." >&2; exit 2 ;;
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
      CONFIG_DIR="examples/peft_compare"
      MATH_TEMPLATE="qwen3"
      MODEL_EXPERTS=128
      MODEL_TOP_K=8
      DAS_TOPK=8
      ;;
    gemma4)
      MODEL_PATH="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT_ROOT/gemma-4-26B-A4B}}"
      CONFIG_DIR="examples/peft_compare_gemma4"
      MATH_TEMPLATE="default"
      MODEL_EXPERTS=128
      MODEL_TOP_K=8
      DAS_TOPK=8
      ;;
    gptoss)
      MODEL_PATH="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT_ROOT/gpt-oss-20b}}"
      CONFIG_DIR="examples/peft_compare_gptoss"
      MATH_TEMPLATE="default"
      MODEL_EXPERTS=32
      MODEL_TOP_K=4
      DAS_TOPK=4
      ;;
    *) echo "MODEL must be qwen3, gemma4, or gptoss; got '$MODEL'." >&2; return 1 ;;
  esac
}

set_lr_meta() {
  case "$1" in
    lr1e-3|1e-3|1.0e-3) LEARNING_RATE=1.0e-3; LR_TAG=lr1e-3 ;;
    lr1e-4|1e-4|1.0e-4) LEARNING_RATE=1.0e-4; LR_TAG=lr1e-4 ;;
    lr1e-5|1e-5|1.0e-5) LEARNING_RATE=1.0e-5; LR_TAG=lr1e-5 ;;
    *)
      echo "Unknown LR trial '$1' (use lr1e-3/lr1e-4/lr1e-5)." >&2
      return 1
      ;;
  esac
}

task_meta() {
  TASK="$1"
  case "$TASK" in
    math)
      DATASET=metamathqa_r1
      TEMPLATE="$MATH_TEMPLATE"
      CUTOFF_LEN=2048
      EPOCHS="$MATH_EPOCHS"
      TRAIN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
      EVAL_BATCH_SIZE="$MATH_EVAL_BATCH_SIZE"
      TASK_FLAGS=()
      DAS_DOMAIN="$DATA_ROOT/metamathqa_r1_clean.json"
      ;;
    intent)
      DATASET=esft_intent
      TEMPLATE=esft_raw
      CUTOFF_LEN=4096
      EPOCHS="$INTENT_EPOCHS"
      if [[ "$MODEL" == "gemma4" ]]; then
        TRAIN_BATCH=(per_device_train_batch_size=4 gradient_accumulation_steps=9)
      else
        TRAIN_BATCH=(per_device_train_batch_size=6 gradient_accumulation_steps=6)
      fi
      EVAL_BATCH_SIZE="$INTENT_EVAL_BATCH_SIZE"
      TASK_FLAGS=(packing=false)
      [[ "$MODEL" == "gptoss" ]] && TASK_FLAGS+=(flash_attn=disabled)
      DAS_DOMAIN="$DATA_ROOT/esft/intent_train.jsonl"
      ;;
    *) echo "Unknown task '$TASK' (use intent or math)." >&2; return 1 ;;
  esac

  DAS_SUFFIX=""
  [[ -n "$SMOKE_MAX_STEPS" ]] && DAS_SUFFIX="_smoke"
  if [[ -n "$DAS_JSON_OVERRIDE" ]]; then
    DAS_JSON="$DAS_JSON_OVERRIDE"
  else
    DAS_JSON="$REPO/eval_results/peft_compare_granularity/das_${MODEL}_${TASK}${DAS_SUFFIX}.json"
  fi
}

method_supported() {
  case "$1" in
    weight_lora|expert_lora|weight_dora|weight_oft|weight_fourier|weight_ia3|expert_moe_lora_n8|expert_moe2_lora|layer_lora|das_lora|partial_last) ;;
    *) echo "Unknown method '$1'. Allowed: ${ALL_METHODS[*]}" >&2; return 1 ;;
  esac
}

set_method_flags() {
  METHOD="$1"
  METHOD_CONFIG="$CONFIG_DIR/$METHOD.yaml"
  METHOD_FLAGS=()
  # LoRA-family methods use standard LoRA on Attention/Router in addition to
  # their method-specific FFN LoRA structure. These are method-local flags, not
  # adapters shared with Weight PEFT or Partial FT. ATTACH_ATTN=0 drops the
  # attention branch (position ablation); the attn rank flags then sit unused.
  LORA_ALL_POSITION_FLAGS=(
    moe_lora_attach_ffn=true
    "moe_lora_attach_attn=$ATTACH_ATTN_BOOL"
    moe_lora_attach_router=true
    moe_lora_attn_targets=auto
    moe_lora_attn_rank=32
    moe_lora_attn_alpha=64
    moe_lora_router_rank=8
    moe_lora_router_alpha=16
    moe_lora_target_layers=all
  )
  MOE_LORA_STRUCTURE_FLAGS=(
    moe_lora_pool_share=per_layer
    moe_lora_w_share=per_layer
    moe_lora_balance_loss_coef=0.0
  )
  case "$METHOD" in
    # These reproduce the two FFN-granularity reference arms exactly.
    weight_lora)
      METHOD_CONFIG="$CONFIG_DIR/expert_lora.yaml"
      METHOD_FLAGS=(
        finetuning_type=moe_lora
        moe_lora_ffn_granularity=expert_weight
        moe_lora_routing_mode=follow_moe
        "moe_lora_n_experts=$MODEL_EXPERTS"
        "moe_lora_top_k=$MODEL_TOP_K"
        moe_lora_rank=1
        moe_lora_alpha=2
        "${LORA_ALL_POSITION_FLAGS[@]}"
        "${MOE_LORA_STRUCTURE_FLAGS[@]}"
      )
      ;;
    expert_lora)
      METHOD_CONFIG="$CONFIG_DIR/expert_lora.yaml"
      METHOD_FLAGS=(
        finetuning_type=moe_lora
        moe_lora_ffn_granularity=bypass
        moe_lora_routing_mode=follow_moe
        "moe_lora_n_experts=$MODEL_EXPERTS"
        "moe_lora_top_k=$MODEL_TOP_K"
        moe_lora_rank=2
        moe_lora_alpha=4
        "${LORA_ALL_POSITION_FLAGS[@]}"
        "${MOE_LORA_STRUCTURE_FLAGS[@]}"
      )
      ;;
    layer_lora)
      case "$MODEL" in
        qwen3|gemma4) LAYER_RANK=256 ;;
        gptoss) LAYER_RANK=64 ;;
      esac
      METHOD_CONFIG="$CONFIG_DIR/expert_lora.yaml"
      METHOD_FLAGS=(
        finetuning_type=moe_lora
        moe_lora_ffn_granularity=bypass
        moe_lora_routing_mode=independent
        moe_lora_n_experts=1
        moe_lora_top_k=1
        "moe_lora_rank=$LAYER_RANK"
        "moe_lora_alpha=$((2 * LAYER_RANK))"
        "${LORA_ALL_POSITION_FLAGS[@]}"
        "${MOE_LORA_STRUCTURE_FLAGS[@]}"
      )
      ;;
    weight_dora)
      METHOD_FLAGS=(
        finetuning_type=weight_peft
        weight_peft_type=dora
        weight_peft_rank=1
        weight_peft_alpha=2
        weight_peft_attn_rank=32
        weight_peft_attn_alpha=64
        weight_peft_router_rank=8
        weight_peft_router_alpha=16
        "weight_peft_attach_attn=$ATTACH_ATTN_BOOL"
        weight_peft_attach_ffn=true
        weight_peft_attach_router=true
      )
      ;;
    weight_oft)
      case "$MODEL" in
        qwen3) OFT_BLOCK=4; ATTN_OFT_BLOCK=128; ROUTER_OFT_BLOCK=128 ;;
        gemma4) OFT_BLOCK=8; ATTN_OFT_BLOCK=128; ROUTER_OFT_BLOCK=128 ;;
        gptoss) OFT_BLOCK=6; ATTN_OFT_BLOCK=64; ROUTER_OFT_BLOCK=32 ;;
      esac
      METHOD_FLAGS=(
        finetuning_type=weight_peft
        weight_peft_type=oft
        "weight_peft_oft_block_size=$OFT_BLOCK"
        "weight_peft_attn_oft_block_size=$ATTN_OFT_BLOCK"
        "weight_peft_router_oft_block_size=$ROUTER_OFT_BLOCK"
        "weight_peft_attach_attn=$ATTACH_ATTN_BOOL"
        weight_peft_attach_ffn=true
        weight_peft_attach_router=true
      )
      ;;
    weight_fourier)
      case "$MODEL" in
        qwen3) FOURIER_N_FREQ=4096; ATTN_FOURIER_N_FREQ=139264; ROUTER_FOURIER_N_FREQ=17408 ;;
        gemma4) FOURIER_N_FREQ=5632; ATTN_FOURIER_N_FREQ=199814; ROUTER_FOURIER_N_FREQ=23552 ;;
        gptoss) FOURIER_N_FREQ=5760; ATTN_FOURIER_N_FREQ=165888; ROUTER_FOURIER_N_FREQ=23296 ;;
      esac
      METHOD_FLAGS=(
        finetuning_type=weight_peft
        weight_peft_type=fourier
        "weight_peft_fourier_n_frequencies=$FOURIER_N_FREQ"
        "weight_peft_attn_fourier_n_frequencies=$ATTN_FOURIER_N_FREQ"
        "weight_peft_router_fourier_n_frequencies=$ROUTER_FOURIER_N_FREQ"
        "weight_peft_attach_attn=$ATTACH_ATTN_BOOL"
        weight_peft_attach_ffn=true
        weight_peft_attach_router=true
      )
      ;;
    weight_ia3)
      METHOD_FLAGS=(
        finetuning_type=weight_peft
        weight_peft_type=ia3
        "weight_peft_attach_attn=$ATTACH_ATTN_BOOL"
        weight_peft_attach_ffn=true
        weight_peft_attach_router=true
      )
      ;;
    expert_moe_lora_n8)
      case "$MODEL" in
        qwen3|gemma4) MOE_RANK=32 ;;
        gptoss) MOE_RANK=8 ;;
      esac
      METHOD_FLAGS=(
        moe_lora_n_experts=8
        "moe_lora_rank=$MOE_RANK"
        "moe_lora_alpha=$((2 * MOE_RANK))"
        moe_lora_top_k=2
        moe_lora_routing_mode=independent
        "${LORA_ALL_POSITION_FLAGS[@]}"
        "${MOE_LORA_STRUCTURE_FLAGS[@]}"
      )
      ;;
    expert_moe2_lora)
      case "$MODEL" in
        qwen3|gemma4) MOE2_RANK=24 ;;
        gptoss) MOE2_RANK=1 ;;
      esac
      METHOD_FLAGS=(
        moe_lora_n_experts=8
        moe_lora_n_groups=1
        "moe_lora_rank=$MOE2_RANK"
        "moe_lora_alpha=$((2 * MOE2_RANK))"
        moe_lora_top_k=2
        moe_lora_routing_mode=learned
        moe_lora_hidden_bottleneck_dim=128
        "${LORA_ALL_POSITION_FLAGS[@]}"
        "${MOE_LORA_STRUCTURE_FLAGS[@]}"
      )
      ;;
    das_lora)
      case "$MODEL" in
        qwen3|gemma4) DAS_RANK=32 ;;
        gptoss) DAS_RANK=16 ;;
      esac
      METHOD_FLAGS=(
        "das_lora_selected_experts_path=$DAS_JSON"
        "das_lora_rank=$DAS_RANK"
        "das_lora_alpha=$((2 * DAS_RANK))"
        "${LORA_ALL_POSITION_FLAGS[@]}"
      )
      ;;
    partial_last)
      if [[ "$ATTACH_ATTN" == "1" ]]; then
        METHOD_FLAGS=(partial_ft_target=last partial_ft_modules=attn,expert,router)
      else
        METHOD_FLAGS=(partial_ft_target=last partial_ft_modules=expert,router)
      fi
      ;;
  esac
}

print_command() {
  printf '+'
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

adapter_state_file() {
  local method="$1" output_dir="$2"
  case "$method" in
    weight_dora|weight_oft|weight_fourier|weight_ia3) printf '%s/weight_peft_state.safetensors' "$output_dir" ;;
    das_lora) printf '%s/das_lora_state.safetensors' "$output_dir" ;;
    partial_last) printf '%s/partial_ft_state.safetensors' "$output_dir" ;;
    *) printf '%s/moe_lora_state.safetensors' "$output_dir" ;;
  esac
}

adapter_complete() {
  [[ -f "$(adapter_state_file "$1" "$2")" ]]
}

math500_complete() {
  local result_file="$1" expected="$2"
  [[ -s "$result_file" ]] || return 1
  "$PYTHON_BIN" - "$result_file" "$expected" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
try:
    with open(path, encoding="utf-8") as handle:
        indices = [int(json.loads(line)["idx"]) for line in handle if line.strip()]
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(indices) == expected and sorted(indices) == list(range(expected)) else 1)
PY
}

intent_complete() {
  local result_dir="$1" expected="$2"
  local predictions="$result_dir/intent_predictions.jsonl"
  [[ -f "$predictions" && -f "$result_dir/intent_metrics.json" ]] || return 1
  local count
  count="$(wc -l < "$predictions" 2>/dev/null || echo 0)"
  (( count >= expected ))
}

ensure_das() {
  local logfile="$1"
  if [[ -s "$DAS_JSON" ]]; then
    echo "[das] reuse $DAS_JSON" | tee -a "$logfile"
    return 0
  fi
  local first_gpu
  first_gpu="${GPUS%%,*}"
  local -a command=(
    env
    "CUDA_VISIBLE_DEVICES=$first_gpu"
    "$PYTHON_BIN" "$REPO/eval_scripts/compute_das.py"
    --base_model "$MODEL_PATH"
    --domain_dataset "$DAS_DOMAIN"
    --general_dataset "$DAS_GENERAL"
    --general_split test
    --top_k "$DAS_TOPK"
    --max_samples "$DAS_MAXSAMPLES"
    --output "$DAS_JSON"
  )
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$(dirname "$DAS_JSON")"
  echo "[das] precompute selected experts -> $DAS_JSON" | tee -a "$logfile"
  run_logged "$logfile" "${command[@]}" || return 1
  [[ "$RUNNER_DRY_RUN" == "1" || -s "$DAS_JSON" ]]
}

train_method() {
  local method="$1" output_dir="$2" logfile="$3" port="$4"
  set_method_flags "$method"
  [[ -f "$REPO/$METHOD_CONFIG" ]] || { echo "Missing config: $REPO/$METHOD_CONFIG" >&2; return 1; }
  if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && adapter_complete "$method" "$output_dir"; then
    echo "[$MODEL/$LR_TAG/$TASK/$method] existing adapter found; training skipped." | tee -a "$logfile"
    return 0
  fi
  if [[ "$method" == "das_lora" ]] && ! ensure_das "$logfile"; then
    echo "[$MODEL/$LR_TAG/$TASK/$method] DAS precompute failed." | tee -a "$logfile" >&2
    return 1
  fi

  local max_samples="${SMOKE_MAX_SAMPLES:-$MAX_SAMPLES}"
  local -a command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    FORCE_TORCHRUN=1
    "MASTER_PORT=$port"
    llamafactory-cli train "$METHOD_CONFIG"
    "model_name_or_path=$MODEL_PATH"
    "deepspeed=$DS_CONFIG"
    "dataset_dir=$DATA_ROOT"
    "dataset=$DATASET"
    "template=$TEMPLATE"
    "cutoff_len=$CUTOFF_LEN"
    "max_samples=$max_samples"
    "num_train_epochs=$EPOCHS"
    "learning_rate=$LEARNING_RATE"
    "seed=$SEED"
    "data_seed=$DATA_SEED"
  )
  command+=("${TRAIN_BATCH[@]}" "${TASK_FLAGS[@]}" "${METHOD_FLAGS[@]}")
  [[ -n "$SMOKE_MAX_STEPS" ]] && command+=("max_steps=$SMOKE_MAX_STEPS")
  # weight_peft writes its compact adapter state explicitly at the end of
  # run_sft().  Disable Trainer's full-model checkpoint here; otherwise every
  # training run attempts to duplicate the multi-GB base model under
  # checkpoint-* before the adapter-only save can finish.
  case "$method" in
    weight_dora|weight_oft|weight_fourier|weight_ia3)
      # Preserve literal quotes for OmegaConf; bare `no` becomes boolean False.
      command+=('save_strategy="no"')
      ;;
  esac
  command+=(
    report_to=none
    overwrite_output_dir=true
    "run_name=${MODEL}_${LR_TAG}_${TASK}_${method}"
    "output_dir=$output_dir"
  )
  run_logged "$logfile" "${command[@]}"
}

eval_method() {
  local method="$1" output_dir="$2" result_dir="$3" logfile="$4" port="$5"
  if [[ "$RUNNER_DRY_RUN" != "1" ]] && ! adapter_complete "$method" "$output_dir"; then
    echo "[$MODEL/$LR_TAG/$TASK/$method] adapter is missing; evaluation skipped." | tee -a "$logfile" >&2
    return 1
  fi

  if [[ "$TASK" == "math" ]]; then
    local result_file="$result_dir/math500.jsonl"
    local expected="${SMOKE_EVAL_LIMIT:-$MATH500_EXPECTED_N}"
    if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && math500_complete "$result_file" "$expected"; then
      echo "[$MODEL/$LR_TAG/$TASK/$method] complete MATH-500 result found; evaluation skipped." | tee -a "$logfile"
      return 0
    fi
    [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir"
    local -a command=(
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
    [[ -n "$SMOKE_EVAL_LIMIT" ]] && command+=(--limit "$SMOKE_EVAL_LIMIT")
    run_logged "$logfile" "${command[@]}"
    return
  fi

  local expected="${SMOKE_EVAL_LIMIT:-500}"
  if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && intent_complete "$result_dir" "$expected"; then
    echo "[$MODEL/$LR_TAG/$TASK/$method] complete Intent result found; evaluation skipped." | tee -a "$logfile"
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir"
  local -a command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    "PYTHONPATH=$REPO/src:$REPO/eval_scripts/esft:$REPO${PYTHONPATH:+:$PYTHONPATH}"
    torchrun
    --nproc_per_node=4
    "--master_port=$((port + 1))"
    "$REPO/eval_scripts/esft/eval_esft.py"
    --task intent
    --mode all
    --template esft_raw
    --prompt-protocol auto
    --runtime-merge
    --runtime-merge-device-map auto
    --hf-generate-batch-size "$EVAL_BATCH_SIZE"
    --tensor-parallel-size 1
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
    --model-name-or-path "$MODEL_PATH"
    --adapter-name-or-path "$output_dir"
    --output-dir "$result_dir"
  )
  [[ "$MODEL" == "gptoss" ]] && command+=(--flash-attn disabled)
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && command+=(--debug --debug-samples "$SMOKE_EVAL_LIMIT")
  run_logged "$logfile" "${command[@]}"
}

upstream_method() {
  local method="$1" output_dir="$2" result_dir="$3" logfile="$4"
  [[ "$RUN_UPSTREAM" == "1" ]] || return 0
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir/upstream"
  local -a command=(
    env MODEL="$MODEL" ADAPTER="$output_dir" OUT_DIR="$result_dir/upstream"
    GPUS="$GPUS" TASKS="$UPSTREAM_TASKS" ENGINE="$UPSTREAM_ENGINE"
    REPO="$REPO" BASE_MODEL="$MODEL_PATH" MAX_LEN=4096 IFEVAL_MAXNEW=1280
    "MMLU_LIMIT=$UPSTREAM_MMLU_LIMIT" "IFEVAL_LIMIT=$UPSTREAM_IFEVAL_LIMIT"
    bash "$REPO/scripts/reproduce/run_upstream_eval.sh"
  )
  run_logged "$logfile" "${command[@]}"
}


model_meta || exit 2
case "$MODEL" in
  qwen3|gemma4) DEFAULT_LR_TRIALS=(lr1e-4 lr1e-3 lr1e-5) ;;
  gptoss) DEFAULT_LR_TRIALS=(lr1e-3 lr1e-4 lr1e-5) ;;
esac
if [[ -n "$LR_TRIALS" ]]; then
  read -r -a SELECTED_LR_TRIALS <<< "$LR_TRIALS"
else
  SELECTED_LR_TRIALS=("${DEFAULT_LR_TRIALS[@]}")
fi
(( ${#SELECTED_LR_TRIALS[@]} > 0 )) || { echo "LR_TRIALS must not be empty." >&2; exit 2; }
for lr_trial in "${SELECTED_LR_TRIALS[@]}"; do set_lr_meta "$lr_trial" || exit 2; done

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" != "4" ]]; then
  echo "This protocol requires exactly four GPUs; GPUS='$GPUS'." >&2
  exit 2
fi
if [[ ! "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
  echo "MASTER_PORT must be numeric; got '$MASTER_PORT'." >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$BUDGET_CHECK" ]]; then
  echo "Missing model config or budget checker." >&2
  exit 2
fi

if (( $# > 0 )); then
  SELECTED_METHODS=("$@")
elif [[ "$METHODS" == "all" ]]; then
  SELECTED_METHODS=("${DEFAULT_METHODS[@]}")
else
  read -r -a SELECTED_METHODS <<< "$METHODS"
fi
(( ${#SELECTED_METHODS[@]} > 0 )) || { usage >&2; exit 2; }
for method in "${SELECTED_METHODS[@]}"; do method_supported "$method" || exit 2; done

for task in $TASKS; do
  case "$task" in intent|math) ;; *) echo "Unknown TASKS entry '$task'." >&2; exit 2 ;; esac
done

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
[[ -n "$PYTHON_BIN" ]] || { echo "No Python interpreter found." >&2; exit 2; }
"$PYTHON_BIN" "$BUDGET_CHECK" --model "$MODEL" --config "$MODEL_PATH/config.json" || exit 2

echo "=================================================================="
echo "Granularity-aligned PEFT comparison"
echo "  model=$MODEL tasks=$TASKS methods=${SELECTED_METHODS[*]}"
if [[ "$ATTACH_ATTN" == "1" ]]; then
  echo "  protocol=same-method adaptation on FFN + attention + router"
  echo "  anchor=attention LoRA r32/a64; router LoRA r8/a16 (capacity reference only)"
else
  echo "  protocol=same-method adaptation on FFN + router (NO attention, position ablation)"
  echo "  anchor=router LoRA r8/a16 (capacity reference only)"
fi
echo "  train=lr_trials=${SELECTED_LR_TRIALS[*]} phase=$PHASE GPUs=$GPUS skip_existing=$SKIP_EXISTING dry_run=$RUNNER_DRY_RUN"
echo "  seed=$SEED data_seed=$DATA_SEED"
echo "  outputs=$OUT_ROOT/<method>/<task>/$MODEL/$SEED_SUBDIR/<lr>"
echo "  results=$RES_ROOT/<method>/<task>/$MODEL/$SEED_SUBDIR/<lr>"
echo "  paper reference (main.tex:187-204): Weight-LoRA=68.7%, DoRA=68.3%, FourierFT=44.2%, IA3=41.2%, Partial-FT=32.7%, MoE-specific=64.1--65.8%."
echo "=================================================================="

declare -a SUMMARY=()
failures=0
port="$MASTER_PORT"
for lr_trial in "${SELECTED_LR_TRIALS[@]}"; do
  set_lr_meta "$lr_trial" || exit 2
  echo "========== $MODEL/$LR_TAG (learning_rate=$LEARNING_RATE) =========="
  for task in $TASKS; do
    task_meta "$task" || exit 2
    for method in "${SELECTED_METHODS[@]}"; do
      if [[ "$UNIFIED_LAYOUT" == "1" ]]; then
        output_dir="$OUT_ROOT/$method/$TASK/$MODEL/$SEED_SUBDIR/$LR_TAG"
        result_dir="$RES_ROOT/$method/$TASK/$MODEL/$SEED_SUBDIR/$LR_TAG"
        logfile="$(prepare_log "$RES_ROOT/$method/$TASK/$MODEL/$SEED_SUBDIR/logs/${method}_${LR_TAG}.log")"
      else
        output_dir="$OUT_ROOT/$MODEL/$LR_TAG/$TASK/$method"
        result_dir="$RES_ROOT/$MODEL/$LR_TAG/$TASK/$method"
        logfile="$(prepare_log "$RES_ROOT/$MODEL/$LR_TAG/$TASK/logs/$method.log")"
      fi
      echo "[$MODEL/$LR_TAG/$TASK/$method] output=$output_dir result=$result_dir port=$port" | tee -a "$logfile"
      status=0
      if [[ "$PHASE" == "train" || "$PHASE" == "full" ]]; then
        train_method "$method" "$output_dir" "$logfile" "$port" || status=1
      fi
      if [[ "$status" == "0" && ( "$PHASE" == "eval" || "$PHASE" == "full" ) ]]; then
        eval_method "$method" "$output_dir" "$result_dir" "$logfile" "$port" || status=1
      fi
      if [[ "$status" == "0" && "$PHASE" == "full" ]]; then
        upstream_method "$method" "$output_dir" "$result_dir" "$logfile" || status=1
      fi
      if [[ "$status" == "0" ]]; then
        SUMMARY+=("[OK]   $MODEL/$LR_TAG/$TASK/$method")
      else
        SUMMARY+=("[FAIL] $MODEL/$LR_TAG/$TASK/$method")
        failures=$((failures + 1))
      fi
      port=$((port + 2))
    done
  done
done

echo
echo "========== Granularity PEFT summary =========="
printf '  %s\n' "${SUMMARY[@]}"
echo "Adapters: $OUT_ROOT/<method>/<task>/$MODEL/$SEED_SUBDIR/<lr>/"
echo "Results:  $RES_ROOT/<method>/<task>/$MODEL/$SEED_SUBDIR/<lr>/(math500.jsonl | intent_{predictions,metrics}.json*)"
(( failures == 0 ))
