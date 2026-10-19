#!/usr/bin/env bash
# MoE^2-LoRA official-structure pilot for the method x LR chart (qwen3/intent).
#
# Base config is the official examples/peft_compare/expert_moe2_lora.yaml
# (global pool, n_groups=2, top_k=8, hidden_bottleneck=128, w_share=per_layer,
# routing_mode=learned). To fit the method_lr noattn budget band (~51M,
# expert_lora noattn = 51.17M) we drop attention LoRA (router LoRA is already
# absent in the official yaml) and shrink the FFN pool three ways:
#
#   setting    n_experts  rank/alpha   FFN pool     W_l        total
#   n280_r32   280 (140/g)  32/64      36.70M    14.30M     51.00M
#   n512_r17   512 (256/g)  17/34      35.65M    15.73M     51.38M
#   n512_r16   512 (256/g)  16/32      33.55M    15.73M     49.28M
#
# (W_l = h_proj 48x2048x128 = 12.58M fixed + proj 48x256xN_per_group, which
#  scales with the per-group expert count; verified against the official
#  checkpoint saves/peft_compare/expert_moe2_lora = 84.51M.)
#
# Training/eval protocol matches run_peft_compare_granularity.sh arms:
#   intent: esft_intent, template esft_raw, cutoff 4096, 3 epochs, pbs6 x gas6,
#           packing=false; eval via eval_scripts/esft/eval_esft.py runtime-merge.
#   math:   metamathqa_r1, template qwen3, cutoff 2048, 1 epoch, pbs6 x gas6;
#           eval via eval_scripts/eval_math500.py (500 problems).
#
# Usage:
#   bash scripts/exp/moe2/run_moe2_official_pilot.sh
#   TASK=math SETTINGS="n512_r16" bash .../run_moe2_official_pilot.sh
#   SETTINGS="n280_r32" LR_TRIALS="lr1e-4" PHASE=train bash .../run_moe2_official_pilot.sh
#   RUNNER_DRY_RUN=1 bash .../run_moe2_official_pilot.sh   # print commands only

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"

MODEL="${MODEL:-qwen3}"                     # qwen3 | gemma4 | gptoss
TASK="${TASK:-intent}"                      # space-separated list: "intent", "math", or "intent math"
SETTINGS="${SETTINGS:-}"                     # empty -> per-model default (see below)
LR_TRIALS="${LR_TRIALS:-lr1e-4 lr1e-3}"
PHASE="${PHASE:-full}"                      # full | train | eval
GPUS="${GPUS:-0,1,2,3}"
# Reference protocol is 4 GPUs x pbs6 x gas6 (effective batch 144). When
# running on fewer GPUs, raise TRAIN_GAS to keep the effective batch at 144
# (e.g. 2 GPUs -> TRAIN_GAS=12). Defaults reproduce the 4-GPU protocol exactly.
TRAIN_GAS="${TRAIN_GAS:-6}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-30880}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-1}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
OUT_ROOT="${OUT_ROOT:-$REPO/saves/moe2_official_lr}"
RES_ROOT="${RES_ROOT:-$REPO/results/moe2_official_lr}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
MATH_EPOCHS="${MATH_EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-80000}"
INTENT_EVAL_BATCH_SIZE="${INTENT_EVAL_BATCH_SIZE:-8}"
MATH_EVAL_BATCH_SIZE="${MATH_EVAL_BATCH_SIZE:-32}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
INTENT_EXPECTED_N="${INTENT_EXPECTED_N:-500}"
MATH500_EXPECTED_N="${MATH500_EXPECTED_N:-500}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"
ATTN_RANK="${ATTN_RANK:-0}"
ATTN_ALPHA="${ATTN_ALPHA:-}"
BALANCE_LOSS_COEF="${BALANCE_LOSS_COEF:-0.0}"
BALANCE_LOSS_SCHEDULE="${BALANCE_LOSS_SCHEDULE:-linear_decay}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"

[[ "$SEED" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer; got '$SEED'." >&2
  exit 2
}
[[ "$DATA_SEED" =~ ^[0-9]+$ ]] || {
  echo "DATA_SEED must be a non-negative integer; got '$DATA_SEED'." >&2
  exit 2
}

case "$PHASE" in
  full|train|eval) ;;
  *) echo "PHASE must be full, train, or eval; got '$PHASE'." >&2; exit 2 ;;
esac
[[ "$ATTN_RANK" =~ ^[0-9]+$ ]] || {
  echo "ATTN_RANK must be a non-negative integer; got '$ATTN_RANK'." >&2
  exit 2
}
if [[ -n "$ATTN_ALPHA" && ! "$ATTN_ALPHA" =~ ^[0-9]+$ ]]; then
  echo "ATTN_ALPHA must be empty or a non-negative integer; got '$ATTN_ALPHA'." >&2
  exit 2
fi
if (( ATTN_RANK > 0 )) && [[ "$ATTN_ALPHA" == "0" ]]; then
  echo "ATTN_ALPHA must be positive when ATTN_RANK is enabled." >&2
  exit 2
fi
if [[ ! "$BALANCE_LOSS_COEF" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "BALANCE_LOSS_COEF must be a non-negative number; got '$BALANCE_LOSS_COEF'." >&2
  exit 2
fi
case "$BALANCE_LOSS_SCHEDULE" in
  linear_decay|constant) ;;
  *) echo "BALANCE_LOSS_SCHEDULE must be linear_decay or constant; got '$BALANCE_LOSS_SCHEDULE'." >&2; exit 2 ;;
esac

# Per-model paths / official moe2 base config / math template, aligned with
# run_peft_compare_granularity.sh model_meta.
case "$MODEL" in
  qwen3)
    MODEL_PATH="${MODEL_PATH:-${QWEN3_BASE:-$CKPT_ROOT/Qwen3-30B-A3B-Base}}"
    BASE_CFG="examples/peft_compare/expert_moe2_lora.yaml"
    MATH_TEMPLATE=qwen3
    MODEL_TOP_K=8
    ;;
  gemma4)
    MODEL_PATH="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT_ROOT/gemma-4-26B-A4B}}"
    BASE_CFG="examples/peft_compare_gemma4/expert_moe2_lora.yaml"
    MATH_TEMPLATE=default
    MODEL_TOP_K=8
    ;;
  gptoss)
    MODEL_PATH="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT_ROOT/gpt-oss-20b}}"
    BASE_CFG="examples/peft_compare_gptoss/expert_moe2_lora.yaml"
    MATH_TEMPLATE=default
    MODEL_TOP_K=4
    ;;
  *) echo "MODEL must be qwen3, gemma4, or gptoss; got '$MODEL'." >&2; exit 2 ;;
esac

# gpt-oss must run with flash-attn disabled, both for training and intent eval
# (aligned with run_peft_compare_granularity.sh lines 147 & 535). math eval
# (eval_math500.py) handles gpt-oss internally and takes no flash-attn flag.
MODEL_TRAIN_FLAGS=()
[[ "$MODEL" == "gptoss" ]] && MODEL_TRAIN_FLAGS+=(flash_attn=disabled)

# Task protocol, aligned with run_peft_compare_granularity.sh task_meta.
# Sets CUR_TASK-scoped globals for the currently running task.
set_task_meta() {
  CUR_TASK="$1"
  case "$CUR_TASK" in
    intent)
      DATASET=esft_intent
      TEMPLATE=esft_raw
      CUTOFF_LEN=4096
      EPOCHS="$INTENT_EPOCHS"
      TASK_FLAGS=(packing=false)
      ;;
    math)
      DATASET=metamathqa_r1
      TEMPLATE="$MATH_TEMPLATE"
      CUTOFF_LEN=2048
      EPOCHS="$MATH_EPOCHS"
      TASK_FLAGS=()
      ;;
    *) echo "TASK must be intent or math; got '$CUR_TASK'." >&2; return 1 ;;
  esac
}

set_lr_meta() {
  case "$1" in
    lr1e-3|1e-3|1.0e-3) LEARNING_RATE=1.0e-3; LR_TAG=lr1e-3 ;;
    lr1e-4|1e-4|1.0e-4) LEARNING_RATE=1.0e-4; LR_TAG=lr1e-4 ;;
    lr1e-5|1e-5|1.0e-5) LEARNING_RATE=1.0e-5; LR_TAG=lr1e-5 ;;
    lr2e-3|2e-3|2.0e-3) LEARNING_RATE=2.0e-3; LR_TAG=lr2e-3 ;;
    lr3e-3|3e-3|3.0e-3) LEARNING_RATE=3.0e-3; LR_TAG=lr3e-3 ;;
    lr5e-3|5e-3|5.0e-3) LEARNING_RATE=5.0e-3; LR_TAG=lr5e-3 ;;
    *) echo "Unknown LR trial '$1' (use lr1e-5/lr1e-4/lr1e-3/lr2e-3/lr3e-3/lr5e-3)." >&2; return 1 ;;
  esac
}

# Setting names begin with "n<experts>_r<rank>". A trailing "_g<number>"
# overrides the number of global pool groups (for example,
# "n256_r20_g4_wp"). The group suffix is accepted before the optional "_wp"
# suffix; the runner validates that the number of LoRA experts is divisible by
# the requested group count before launching training.
#
# GPT-OSS budget settings may
# add "_hb<dim>[_wla][_tk<top_k>]"; weight-pool settings may instead add
# "_hbu<gate_up_dim>_hbd<down_dim>" for independent per-stage routing. The terminal "_wp" suffix selects the
# routed gate_up/down weight pool instead of the historical whole-layer bypass.
# The other established suffix forms remain supported below.
# -> N_EXPERTS / MOE2_RANK
# (alpha = 2*rank) and an OPTIONAL router LoRA of rank <router_rank>
# (alpha = 2*router_rank). Without the "_rt" suffix, router LoRA stays OFF
# (moe_lora_attach_router=false), so all existing setting names are unchanged.
# The optional "_wla" suffix keeps router LoRA off and initializes W_l with
# the pretrained MoE routing anchor (moe_lora_wl_init=anchor).
# The "_hb<dim>" suffix overrides the hidden-state bottleneck. Adding "_wla"
# additionally initializes W_l from the pretrained MoE routing anchor; without
# "_wla", W_l keeps the default initialization. All other settings retain
# bottleneck128 unchanged.
#   qwen3:  n280_r32 51.0M | n512_r17 51.4M | n512_r16 49.3M   (band 51.2M)
#           n256_r32 47.7M (author-official structure)
#           n256_r20_wp 46.9M (parameter-matched routed weight pool)
#           n256_r32_rt8 = n256_r32 + router LoRA r8/a16 (+~0.84M)
#   gemma4: n256_r22 43.5M                                     (band 44.0M)
#   gptoss: n256_r1 10.8M (drop rank) | n10_r32 10.7M (drop experts)  (band 9.4M,
#           route floor h_proj=8.85M forces >band; matches existing 10.5M point)
set_setting_flags() {
  SETTING="$1"
  N_GROUPS_OVERRIDE=
  TOP_K_OVERRIDE=
  NORMALIZE_TOPK_PROB=false
  WP_DOWN_ROUTE_ON_H=false
  WP_GATE_UP_STAGE=gate_up
  HIDDEN_BOTTLENECK_GATE_UP_DIM=
  HIDDEN_BOTTLENECK_DOWN_DIM=
  FFN_GRANULARITY=bypass
  B_INIT=normal
  ALPHA_OVERRIDE=
  SETTING_CORE="$SETTING"
  # "_bz" -> zero-initialize LoRA B (identity at step 0). The parsed
  # granularity determines whether this applies to bypass or weight_pool.
  if [[ "$SETTING_CORE" == *_bz ]]; then
    B_INIT=zero
    SETTING_CORE="${SETTING_CORE%_bz}"
  fi
  # "_a<alpha>" -> explicit alpha, overriding the default alpha = 2*rank.
  # Used to test alpha ~ sqrt(rank) / 2*sqrt(rank) scalings.
  if [[ "$SETTING_CORE" =~ _a([0-9]+)$ ]]; then
    ALPHA_OVERRIDE="${BASH_REMATCH[1]}"
    SETTING_CORE="${SETTING_CORE%_a$ALPHA_OVERRIDE}"
  fi
  if [[ "$SETTING_CORE" == *_ntk ]]; then
    NORMALIZE_TOPK_PROB=true
    SETTING_CORE="${SETTING_CORE%_ntk}"
  fi
  # "_drh" -> route the down adapter from the post-gate activation h_in instead of
  # the block input. Requires the _wp granularity; validated after _wp is parsed.
  if [[ "$SETTING_CORE" == *_drh ]]; then
    WP_DOWN_ROUTE_ON_H=true
    SETTING_CORE="${SETTING_CORE%_drh}"
  fi
  # "_up" -> adapt only the up half of the fused gate_up projection (leave the gate
  # half at its frozen base value). Halves the gate_up B params. Requires _wp;
  # validated after _wp is parsed.
  if [[ "$SETTING_CORE" == *_up ]]; then
    WP_GATE_UP_STAGE=up
    SETTING_CORE="${SETTING_CORE%_up}"
  fi
  if [[ "$SETTING_CORE" == *_wp ]]; then
    FFN_GRANULARITY=weight_pool
    SETTING_CORE="${SETTING_CORE%_wp}"
  fi

  # Keep the group suffix outside the setting-specific regexes so every
  # existing form (plain, _hb, _wla, _rt, and _tk) can opt into any positive
  # group count without duplicating parser branches. The canonical form is
  # n<experts>_r<rank>_g<number>[_wp].
  if [[ "$SETTING_CORE" =~ _g([0-9]+)$ ]]; then
    N_GROUPS_OVERRIDE="${BASH_REMATCH[1]}"
    SETTING_CORE="${SETTING_CORE%_g$N_GROUPS_OVERRIDE}"
  fi
  # Also accept the intuitive alternate spelling n<...>_wp_g<number>.
  if [[ "$SETTING_CORE" == *_wp ]]; then
    FFN_GRANULARITY=weight_pool
    SETTING_CORE="${SETTING_CORE%_wp}"
  fi

  if [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hbu([0-9]+)_hbd([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_GATE_UP_DIM="${BASH_REMATCH[3]}"
    HIDDEN_BOTTLENECK_DOWN_DIM="${BASH_REMATCH[4]}"
    HIDDEN_BOTTLENECK_DIM=
    ROUTER_RANK=0
    WL_INIT=default
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hb([0-9]+)_wla_tk([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_DIM="${BASH_REMATCH[3]}"
    ROUTER_RANK=0
    WL_INIT=anchor
    TOP_K_OVERRIDE="${BASH_REMATCH[4]}"
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hb([0-9]+)_tk([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_DIM="${BASH_REMATCH[3]}"
    ROUTER_RANK=0
    WL_INIT=default
    TOP_K_OVERRIDE="${BASH_REMATCH[4]}"
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hb([0-9]+)_wla$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_DIM="${BASH_REMATCH[3]}"
    ROUTER_RANK=0
    WL_INIT=anchor
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hb([0-9]+)_wla_rt([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_DIM="${BASH_REMATCH[3]}"
    ROUTER_RANK="${BASH_REMATCH[4]}"
    WL_INIT=anchor
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    ROUTER_RANK=0
    WL_INIT=default
    HIDDEN_BOTTLENECK_DIM=
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_rt([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    ROUTER_RANK="${BASH_REMATCH[3]}"
    WL_INIT=default
    HIDDEN_BOTTLENECK_DIM=
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_hb([0-9]+)$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    HIDDEN_BOTTLENECK_DIM="${BASH_REMATCH[3]}"
    ROUTER_RANK=0
    WL_INIT=default
  elif [[ "$SETTING_CORE" =~ ^n([0-9]+)_r([0-9]+)_wla$ ]]; then
    N_EXPERTS="${BASH_REMATCH[1]}"
    MOE2_RANK="${BASH_REMATCH[2]}"
    ROUTER_RANK=0
    WL_INIT=anchor
    HIDDEN_BOTTLENECK_DIM=
  else
    echo "Unknown setting '$1' (expected n<experts>_r<rank>[_g<number>|_rt<router_rank>|_wla|_hb<dim>[_wla][_rt<router_rank>|_tk<top_k>]|_hbu<dim>_hbd<dim>][_wp[_up]][_ntk][_a<alpha>][_bz])." >&2
    return 1
  fi
  if [[ -n "$HIDDEN_BOTTLENECK_GATE_UP_DIM" && "$FFN_GRANULARITY" != "weight_pool" ]]; then
    echo "Setting '$1': _hbu/_hbd requires the terminal _wp suffix." >&2
    return 1
  fi
  if [[ -n "$N_GROUPS_OVERRIDE" ]] &&
    ! [[ "$N_GROUPS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Setting '$1': n_groups must be a positive integer, got '$N_GROUPS_OVERRIDE'." >&2
    return 1
  fi
  if [[ "$B_INIT" == "zero" && "$FFN_GRANULARITY" == "expert_weight" ]]; then
    echo "Setting '$1': _bz (zero B init) is not supported for expert_weight." >&2
    return 1
  fi
  if [[ "$WP_DOWN_ROUTE_ON_H" == "true" && "$FFN_GRANULARITY" != "weight_pool" ]]; then
    echo "Setting '$1': _drh (down route on h_in) requires the terminal _wp suffix." >&2
    return 1
  fi
  if [[ "$WP_GATE_UP_STAGE" == "up" && "$FFN_GRANULARITY" != "weight_pool" ]]; then
    echo "Setting '$1': _up (up-only gate_up adaptation) requires the terminal _wp suffix." >&2
    return 1
  fi
  if [[ -n "$ALPHA_OVERRIDE" ]]; then
    if (( ALPHA_OVERRIDE < 1 )); then
      echo "Setting '$1': alpha must be >= 1, got '$ALPHA_OVERRIDE'." >&2
      return 1
    fi
    MOE2_ALPHA="$ALPHA_OVERRIDE"
  else
    MOE2_ALPHA=$(( MOE2_RANK * 2 ))
  fi
  ROUTER_ALPHA=$(( ROUTER_RANK * 2 ))
  local n_groups="${N_GROUPS_OVERRIDE:-2}"
  if (( N_EXPERTS % n_groups != 0 )); then
    echo "Setting '$1': n_experts=$N_EXPERTS must be divisible by n_groups=$n_groups." >&2
    return 1
  fi
  if [[ -n "$TOP_K_OVERRIDE" ]] && (( TOP_K_OVERRIDE < 1 || TOP_K_OVERRIDE > N_EXPERTS / ${N_GROUPS_OVERRIDE:-2} )); then
    echo "Setting '$1': top_k=$TOP_K_OVERRIDE must be between 1 and the per-group expert count $((N_EXPERTS / ${N_GROUPS_OVERRIDE:-2}))." >&2
    return 1
  fi
}

default_settings_for_model() {
  case "$MODEL" in
    qwen3)  echo "n280_r32 n512_r17 n512_r16" ;;
    gemma4) echo "n256_r22" ;;
    gptoss) echo "n256_r1 n10_r32" ;;
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
  mkdir -p "$(dirname "$logfile")" || {
    echo "Unable to create log directory for $logfile" >&2
    return 1
  }
  touch "$logfile" || {
    echo "Unable to write log file $logfile" >&2
    return 1
  }
  printf '%s' "$logfile"
}

log_line() {
  local logfile="$1"
  shift
  printf '%s\n' "$*" | tee -a "$logfile"
}

run_logged() {
  local logfile="$1"
  shift
  print_command "$@" | tee -a "$logfile" || return 1
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  "$@" 2>&1 | tee -a "$logfile"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  (( pipeline_status[0] != 0 )) && return "${pipeline_status[0]}"
  return "${pipeline_status[1]}"
}

adapter_complete() {
  [[ -f "$1/moe_lora_state.safetensors" ]]
}

intent_complete() {
  local result_dir="$1" expected="$2"
  local predictions="$result_dir/intent_predictions.jsonl"
  [[ -f "$predictions" && -f "$result_dir/intent_metrics.json" ]] || return 1
  local count
  count="$(wc -l < "$predictions" 2>/dev/null || echo 0)"
  (( count >= expected ))
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

train_arm() {
  local output_dir="$1" logfile="$2" port="$3"
  if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && adapter_complete "$output_dir"; then
    log_line "$logfile" "[$MODEL/$LR_TAG/$CUR_TASK/$SETTING] existing adapter found; training skipped." || return 1
    return 0
  fi
  local max_samples="${SMOKE_MAX_SAMPLES:-$MAX_SAMPLES}"
  local -a command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    "CUDA_HOME=${CUDA_HOME:-}"
    "CUDA_PATH=${CUDA_PATH:-}"
    "NPROC_PER_NODE=$NGPU"
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    FORCE_TORCHRUN=1
    "MASTER_PORT=$port"
    llamafactory-cli train "$BASE_CFG"
    "model_name_or_path=$MODEL_PATH"
    "deepspeed=$DS_CONFIG"
    "dataset_dir=$DATA_ROOT"
    "dataset=$DATASET"
    "template=$TEMPLATE"
    "cutoff_len=$CUTOFF_LEN"
    "max_samples=$max_samples"
    "num_train_epochs=$EPOCHS"
    per_device_train_batch_size=6
    "gradient_accumulation_steps=$TRAIN_GAS"
    "learning_rate=$LEARNING_RATE"
    "seed=$SEED"
    "data_seed=$DATA_SEED"
    # official-structure knobs kept from the yaml: n_groups=2, top_k=8,
    # pool_share=global, w_share=per_layer, hidden_bottleneck_dim=128.
    # Attention LoRA off unless ATTN_RANK>0; router LoRA off unless the setting has an
    # "_rt<rank>" suffix (ROUTER_RANK>0).
    "moe_lora_balance_loss_coef=$BALANCE_LOSS_COEF"
    "moe_lora_balance_loss_schedule=$BALANCE_LOSS_SCHEDULE"
    "moe_lora_ffn_granularity=$FFN_GRANULARITY"
    "moe_lora_n_experts=$N_EXPERTS"
    "moe_lora_rank=$MOE2_RANK"
    "moe_lora_alpha=$MOE2_ALPHA"
    report_to=none
    overwrite_output_dir=true
    "run_name=${MODEL}_${LR_TAG}_${CUR_TASK}_moe2official_${SETTING}_seed${SEED}"
    "output_dir=$output_dir"
  )
  if (( ${ATTN_RANK:-0} > 0 )); then
    command+=(
      moe_lora_attach_attn=true
      moe_lora_attn_targets=auto
      "moe_lora_attn_rank=$ATTN_RANK"
      "moe_lora_attn_alpha=${ATTN_ALPHA:-$(( ATTN_RANK * 2 ))}"
    )
  else
    command+=(moe_lora_attach_attn=false)
  fi
  if (( ROUTER_RANK > 0 )); then
    command+=(
      moe_lora_attach_router=true
      "moe_lora_router_rank=$ROUTER_RANK"
      "moe_lora_router_alpha=$ROUTER_ALPHA"
    )
  else
    command+=(moe_lora_attach_router=false)
  fi
  [[ "$WL_INIT" == "anchor" ]] && command+=(moe_lora_wl_init=anchor)
  if [[ "$B_INIT" == "zero" ]]; then
    case "$FFN_GRANULARITY" in
      bypass) command+=(moe_lora_b_init=zero) ;;
      weight_pool) command+=(moe_lora_wp_b_init=zero) ;;
    esac
  fi
  [[ -n "$HIDDEN_BOTTLENECK_DIM" ]] && command+=("moe_lora_hidden_bottleneck_dim=$HIDDEN_BOTTLENECK_DIM")
  [[ -n "$HIDDEN_BOTTLENECK_GATE_UP_DIM" ]] && command+=(
    "moe_lora_hidden_bottleneck_gate_up_dim=$HIDDEN_BOTTLENECK_GATE_UP_DIM"
    "moe_lora_hidden_bottleneck_down_dim=$HIDDEN_BOTTLENECK_DOWN_DIM"
  )
  [[ -n "$N_GROUPS_OVERRIDE" ]] && command+=("moe_lora_n_groups=$N_GROUPS_OVERRIDE")
  [[ -n "$TOP_K_OVERRIDE" ]] && command+=("moe_lora_top_k=$TOP_K_OVERRIDE")
  [[ "$NORMALIZE_TOPK_PROB" == "true" ]] && command+=(moe_lora_normalize_topk_prob=true)
  [[ "$WP_DOWN_ROUTE_ON_H" == "true" ]] && command+=(moe_lora_wp_down_route_on_h=true)
  [[ "$WP_GATE_UP_STAGE" == "up" ]] && command+=(moe_lora_wp_gate_up_stage=up)
  command+=("${TASK_FLAGS[@]}")
  command+=("${MODEL_TRAIN_FLAGS[@]}")
  [[ -n "$SMOKE_MAX_STEPS" ]] && command+=("max_steps=$SMOKE_MAX_STEPS")
  run_logged "$logfile" "${command[@]}"
}

eval_arm() {
  local output_dir="$1" result_dir="$2" logfile="$3" port="$4"
  if [[ "$RUNNER_DRY_RUN" != "1" ]] && ! adapter_complete "$output_dir"; then
    log_line "$logfile" "[$MODEL/$LR_TAG/$CUR_TASK/$SETTING] adapter is missing; evaluation skipped." >&2 || return 1
    return 1
  fi

  if [[ "$CUR_TASK" == "math" ]]; then
    local result_file="$result_dir/math500.jsonl"
    local expected="${SMOKE_EVAL_LIMIT:-$MATH500_EXPECTED_N}"
    if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && math500_complete "$result_file" "$expected"; then
      log_line "$logfile" "[$MODEL/$LR_TAG/$CUR_TASK/$SETTING] complete MATH-500 result found; evaluation skipped." || return 1
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
      "--nproc_per_node=$NGPU"
      "--master_port=$((port + 1))"
      "$REPO/eval_scripts/eval_math500.py"
      --base_model "$MODEL_PATH"
      --adapter_path "$output_dir"
      --batch_size "$MATH_EVAL_BATCH_SIZE"
      --max_new_tokens "$EVAL_MAX_NEW_TOKENS"
      --save_path "$result_file"
    )
    [[ -n "$SMOKE_EVAL_LIMIT" ]] && command+=(--limit "$SMOKE_EVAL_LIMIT")
    run_logged "$logfile" "${command[@]}"
    return
  fi

  local expected="${SMOKE_EVAL_LIMIT:-$INTENT_EXPECTED_N}"
  if [[ "$SKIP_EXISTING" == "1" && "$RUNNER_DRY_RUN" != "1" ]] && intent_complete "$result_dir" "$expected"; then
    log_line "$logfile" "[$MODEL/$LR_TAG/$CUR_TASK/$SETTING] complete Intent result found; evaluation skipped." || return 1
    return 0
  fi
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$result_dir"
  local -a command=(
    env
    "CUDA_VISIBLE_DEVICES=$GPUS"
    HF_HUB_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    "PYTHONPATH=$REPO/eval_scripts/esft:$REPO"
    torchrun
    "--nproc_per_node=$NGPU"
    "--master_port=$((port + 1))"
    "$REPO/eval_scripts/esft/eval_esft.py"
    --task intent
    --mode all
    --template esft_raw
    --prompt-protocol auto
    --runtime-merge
    --runtime-merge-device-map auto
    --hf-generate-batch-size "$INTENT_EVAL_BATCH_SIZE"
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

[[ -z "$SETTINGS" ]] && SETTINGS="$(default_settings_for_model)"
for lr_trial in $LR_TRIALS; do set_lr_meta "$lr_trial" || exit 2; done
for setting in $SETTINGS; do set_setting_flags "$setting" || exit 2; done
for t in $TASK; do set_task_meta "$t" || exit 2; done

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NGPU="${#GPU_IDS[@]}"
if (( NGPU < 1 )); then
  echo "GPUS must list at least one GPU; got '$GPUS'." >&2
  exit 2
fi
if (( NGPU * 6 * TRAIN_GAS != 144 )); then
  echo "WARNING: effective batch = ${NGPU}x6x${TRAIN_GAS} = $((NGPU * 6 * TRAIN_GAS)) != 144 (reference protocol)." >&2
fi

# Activate a compatible environment before invoking this script.
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
  [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "No executable Python interpreter found." >&2; exit 2; }
  if [[ "$PHASE" != "eval" ]] && ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "llamafactory-cli is missing in the active environment." >&2
    exit 2
  fi
fi
[[ -f "$REPO/$BASE_CFG" ]] || { echo "Missing base config: $REPO/$BASE_CFG" >&2; exit 2; }
if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
  [[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config: $MODEL_PATH" >&2; exit 2; }
  [[ -f "$DATA_ROOT/dataset_info.json" ]] || { echo "Missing dataset registration: $DATA_ROOT/dataset_info.json" >&2; exit 2; }
fi

# DeepSpeed 0.17 probes nvcc during import whenever CUDA devices are visible.
# Plain rlaunch shells often expose the driver but omit CUDA_HOME from the
# environment, so select a toolkit matching this PyTorch build before torchrun.
# shellcheck disable=SC1091
source "$REPO/scripts/exp/moe2/local_cuda_env.sh"
require_cuda_toolkit=0
if [[ "$RUNNER_DRY_RUN" != "1" && "$PHASE" != "eval" ]]; then
  require_cuda_toolkit=1
fi
if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
setup_local_cuda_env "$PYTHON_BIN" "$require_cuda_toolkit" || exit 2
fi

if [[ "$RUNNER_DRY_RUN" != "1" ]]; then
  if ! CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" - "$NGPU" <<'PY'
import sys

import torch

expected = int(sys.argv[1])
available = torch.cuda.is_available()
actual = torch.cuda.device_count()
print(f"CUDA preflight: available={available} visible_devices={actual} expected={expected}")
raise SystemExit(0 if available and actual == expected else 1)
PY
  then
    echo "CUDA preflight failed for GPUS='$GPUS'. Run inside the GPU allocation shell and verify nvidia-smi first." >&2
    exit 2
  fi
fi

echo "=================================================================="
echo "MoE^2-LoRA official-structure pilot ($MODEL / tasks: $TASK)"
echo "  base=$BASE_CFG (global pool, n_groups=2, bottleneck=128)"
echo "  attention_rank=$ATTN_RANK attention_alpha=${ATTN_ALPHA:-auto} (rank 0 disables attention LoRA)"
echo "  balance_loss_coef=$BALANCE_LOSS_COEF balance_loss_schedule=$BALANCE_LOSS_SCHEDULE"
echo "  router LoRA follows each setting"
echo "  settings=$SETTINGS"
echo "  lr_trials=$LR_TRIALS phase=$PHASE GPUs=$GPUS seed=$SEED data_seed=$DATA_SEED skip_existing=$SKIP_EXISTING dry_run=$RUNNER_DRY_RUN"
echo "  cuda_home=${CUDA_HOME:-not-required}"
echo "  outputs=$OUT_ROOT/$MODEL/<lr>/<task>/<setting>"
echo "  results=$RES_ROOT/$MODEL/<lr>/<task>/<setting>"
echo "=================================================================="

declare -a SUMMARY=()
failures=0
port="$BASE_MASTER_PORT"
for cur_task in $TASK; do
  set_task_meta "$cur_task" || exit 2
  echo "########## task=$CUR_TASK dataset=$DATASET template=$TEMPLATE cutoff=$CUTOFF_LEN epochs=$EPOCHS ##########"
  for lr_trial in $LR_TRIALS; do
    set_lr_meta "$lr_trial" || exit 2
    echo "========== $MODEL/$CUR_TASK/$LR_TAG (learning_rate=$LEARNING_RATE) =========="
    for setting in $SETTINGS; do
      set_setting_flags "$setting" || exit 2
      output_dir="$OUT_ROOT/$MODEL/$LR_TAG/$CUR_TASK/moe2official_$SETTING"
      result_dir="$RES_ROOT/$MODEL/$LR_TAG/$CUR_TASK/moe2official_$SETTING"
      logfile="$(prepare_log "$RES_ROOT/$MODEL/$LR_TAG/$CUR_TASK/logs/moe2official_$SETTING.log")" || exit 2
      log_line "$logfile" "[$MODEL/$LR_TAG/$CUR_TASK/$SETTING] seed=$SEED data_seed=$DATA_SEED ffn_granularity=$FFN_GRANULARITY n_experts=$N_EXPERTS rank=$MOE2_RANK alpha=$MOE2_ALPHA attention_rank=$ATTN_RANK attention_alpha=${ATTN_ALPHA:-auto} balance_loss_coef=$BALANCE_LOSS_COEF balance_loss_schedule=$BALANCE_LOSS_SCHEDULE router_rank=$ROUTER_RANK router_alpha=$ROUTER_ALPHA hidden_bottleneck=${HIDDEN_BOTTLENECK_DIM:-128} hidden_bottleneck_gate_up=${HIDDEN_BOTTLENECK_GATE_UP_DIM:-shared} hidden_bottleneck_down=${HIDDEN_BOTTLENECK_DOWN_DIM:-shared} n_groups=${N_GROUPS_OVERRIDE:-2} top_k=${TOP_K_OVERRIDE:-$MODEL_TOP_K} normalize_topk=$NORMALIZE_TOPK_PROB b_init=$B_INIT wp_down_route_on_h=$WP_DOWN_ROUTE_ON_H wp_gate_up_stage=$WP_GATE_UP_STAGE alpha_override=${ALPHA_OVERRIDE:-none} output=$output_dir port=$port" || exit 2
      status=0
      if [[ "$PHASE" == "train" || "$PHASE" == "full" ]]; then
        train_arm "$output_dir" "$logfile" "$port" || status=1
      fi
      if [[ "$status" == "0" && ( "$PHASE" == "eval" || "$PHASE" == "full" ) ]]; then
        eval_arm "$output_dir" "$result_dir" "$logfile" "$port" || status=1
      fi
      if [[ "$status" == "0" ]]; then
        SUMMARY+=("$MODEL/$LR_TAG/$CUR_TASK/$SETTING: OK")
      else
        SUMMARY+=("$MODEL/$LR_TAG/$CUR_TASK/$SETTING: FAIL")
        failures=$((failures + 1))
      fi
      port=$((port + 10))
    done
  done
done

echo "========== summary =========="
printf '  %s\n' "${SUMMARY[@]:-(none)}"
echo "failures=$failures"
exit "$((failures > 0 ? 1 : 0))"
