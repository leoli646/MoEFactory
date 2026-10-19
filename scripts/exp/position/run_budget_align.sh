#!/usr/bin/env bash
# 预算拉齐(budget-aligned)实验驱动: Qwen3-30B-A3B-Base × math/intent。
#
# 论文第 4.1 节的四个适配位置；每种设置可关闭或启用 router rank 8。
#   attn_r64: attention r64
#   dense_r256: 单个整层 FFN 旁路 r256
#   attn32_dense128: attention r32 + 整层 FFN 旁路 r128
#   attn64_dense256: attention r64 + 整层 FFN 旁路 r256
#
# dense_r256 实现说明(不改 Python 代码):
#   moe_lora_routing_mode=independent + moe_lora_n_experts=1 + moe_lora_top_k=1
#   → IndependentRouter 对 1 类做 softmax 恒得 1.0,LoRAPool 恒选 expert 0 权重 1.0,
#     数学上 == 整个 MoE FFN block 旁一条 dense LoRA (scaling·B(A(h)) 加到 block 输出)。
#   已 CPU 验证: 输出与手工 dense LoRA 逐元素一致,fwd+bwd 干净;每层 gate(2048 参数)
#   梯度恒 0,是无害死参数(48 层共 98K,不影响预算对齐结论)。
#   balance loss coef 默认 0,n=1 的 aux 恒为 1 也不会进 loss。
#   attn64_dense256/attn32_dense128 复用同一技巧: attn 侧 LinearLoRAAdapter 只读
#   moe_lora_attn_rank/alpha,与 routing_mode/n_experts/top_k 无关,两分支正交叠加。
#
# 4 个 arm 的结构和预算设定在 math/intent 之间完全共享。
# math 训练协议与 pos_ab qwen3 math 完全一致:
#   examples/train_moe_lora_qwen3/baseline1_qwen3_math_base_run.yaml
#   (metamathqa_r1 80k, 1 epoch, lr 6e-4 cosine, ds_z2 覆盖 yaml 的 z3)
#   batch: attn-only 臂沿用 yaml 的 PBS12×GAS3;带 FFN 分支的臂(dense/expert 池)降为
#   PBS6×GAS6；四卡有效 batch 为 144。
# intent 只切换数据协议: esft_intent + esft_raw + packing=false, 默认 3 epoch;
# 只评测 Intent 500 条程序化精确匹配，不启动任何通用能力评测。
#
# 输出:
#   math:   saves/budget_align/math/qwen3_<arm>_ep1/
#           results/budget_align/math/qwen3/<arm>_ep1/adapter_math500.jsonl
#   intent: saves/budget_align/intent/qwen3_<arm>_ep3/
#           results/budget_align/intent/qwen3/<arm>_ep3/intent_{predictions.jsonl,metrics.json}
#
# 用法:
#   bash scripts/exp/position/run_budget_align.sh                  # math: 4 臂顺序 train+eval
#   DOMAIN=intent bash scripts/exp/position/run_budget_align.sh    # intent: 4 臂顺序 train+eval
#   ARMS="attn_r64" bash scripts/exp/position/run_budget_align.sh  # 单臂
#   DOMAIN=intent ARMS="attn_r64" PHASE=eval GPUS=0,1,2,3 bash scripts/exp/position/run_budget_align.sh
#   PHASE=train|eval|full  EVALS="math500"  GPUS=0,1,2,3
#   smoke: SMOKE_MAX_STEPS=10 SMOKE_MAX_SAMPLES=512 SMOKE_EVAL_LIMIT=16 (输出用 *_smoke 根)

set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-${DATA_ROOT:-$REPO/data}}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${CKPT_ROOT:-$REPO/models}"
QWEN3="${MODEL_PATH:-${QWEN3_BASE:-$CKPT/Qwen3-30B-A3B-Base}}"
MCFG="examples/train_moe_lora_qwen3/baseline1_qwen3_math_base_run.yaml"

GPUS="${GPUS:-0,1,2,3}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-$SEED}"
PHASE="${PHASE:-full}"                 # full | train | eval
ARMS="${ARMS:-attn_r64 dense_r256 attn64_dense256 attn32_dense128}"
DOMAIN="${DOMAIN:-math}"               # math | intent
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
MASTER_PORT="${MASTER_PORT:-29820}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
# Optional overrides used by learning-rate sweeps. Empty values preserve the
# original output layout and the learning rate from MCFG (6e-4 for Qwen3).
LEARNING_RATE="${LEARNING_RATE:-}"
RUN_TAG="${RUN_TAG:-}"
MODEL_SUBDIR="${MODEL_SUBDIR:-}"
# Optional router-gate LoRA add-on. Defaults preserve the original six arms.
ATTACH_ROUTER="${ATTACH_ROUTER:-false}"
ROUTER_RANK="${ROUTER_RANK:-8}"
ROUTER_ALPHA="${ROUTER_ALPHA:-$((ROUTER_RANK * 2))}"
RUNNER_DRY_RUN="${RUNNER_DRY_RUN:-${DRY_RUN:-0}}"
# smoke / 调试(为空=全量)。设了任一 SMOKE_* 时输出根自动切到 *_smoke,绝不污染正式产物。
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"

case "$PHASE" in
  full|train|eval) ;;
  *) echo "UNKNOWN_PHASE:$PHASE（用 full/train/eval）" >&2; exit 2 ;;
esac

case "$ATTACH_ROUTER" in
  true|false) ;;
  *) echo "ATTACH_ROUTER must be true or false, got '$ATTACH_ROUTER'." >&2; exit 2 ;;
esac
if ! [[ "$SEED" =~ ^[0-9]+$ && "$DATA_SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED and DATA_SEED must be non-negative integers; got SEED=$SEED DATA_SEED=$DATA_SEED." >&2
  exit 2
fi
case "$RUNNER_DRY_RUN" in
  0|1) ;;
  *) echo "RUNNER_DRY_RUN must be 0 or 1, got '$RUNNER_DRY_RUN'." >&2; exit 2 ;;
esac
if ! [[ "$ROUTER_RANK" =~ ^[1-9][0-9]*$ && "$ROUTER_ALPHA" =~ ^[1-9][0-9]*$ ]]; then
  echo "ROUTER_RANK and ROUTER_ALPHA must be positive integers." >&2
  exit 2
fi

ROUTER_FLAGS="moe_lora_attach_router=$ATTACH_ROUTER"
if [[ "$ATTACH_ROUTER" == "true" ]]; then
  ROUTER_FLAGS="$ROUTER_FLAGS moe_lora_router_rank=$ROUTER_RANK moe_lora_router_alpha=$ROUTER_ALPHA"
fi

case "$DOMAIN" in
  math)
    RUN_EPOCHS=1
    EVALS="${EVALS:-math500}"
    EVAL_BSZ="${EVAL_BSZ:-32}"
    EVAL_MAXNEW="${EVAL_MAXNEW:-512}"
    TRAIN_PROTOCOL="max_samples=${SMOKE_MAX_SAMPLES:-80000} num_train_epochs=1"
    ;;
  intent)
    RUN_EPOCHS="$INTENT_EPOCHS"
    EVALS="${EVALS:-intent}"
    EVAL_BSZ="${EVAL_BSZ:-8}"
    EVAL_MAXNEW="${EVAL_MAXNEW:-512}"
    # Intent 样本很短，packing 会把每 epoch 压到约 1 个 step，必须关闭。
    TRAIN_PROTOCOL="dataset=esft_intent template=esft_raw cutoff_len=4096 packing=false max_samples=${SMOKE_MAX_SAMPLES:-80000} num_train_epochs=$INTENT_EPOCHS"
    ;;
  *) echo "UNKNOWN_DOMAIN:$DOMAIN（用 math/intent）" >&2; exit 2 ;;
esac

if [[ -n "$SMOKE_MAX_STEPS$SMOKE_MAX_SAMPLES$SMOKE_EVAL_LIMIT" ]]; then
  OUT_ROOT="${OUT_ROOT:-saves/budget_align_smoke/$DOMAIN}"
  RES_ROOT="${RES_ROOT:-results/budget_align_smoke/$DOMAIN}"
else
  OUT_ROOT="${OUT_ROOT:-saves/budget_align/$DOMAIN}"
  RES_ROOT="${RES_ROOT:-results/budget_align/$DOMAIN}"
fi

# arm_flags <arm> -> 该臂的 llamafactory 覆盖参数串(全部 α/r=2)
arm_flags() {
  case "$1" in
    attn_r64)     # attn-only, rank 拉到 64 (53.48M, 对齐 expert 池 50.33M, 2 的幂次)
      echo "moe_lora_attach_ffn=false moe_lora_attach_attn=true $ROUTER_FLAGS" \
           "moe_lora_attn_targets=auto moe_lora_attn_rank=64 moe_lora_attn_alpha=128" ;;
    dense_r256)   # 整个 MoE FFN 旁 dense LoRA r=256 (50.33M, 与 expert 池逐字节同预算)
      echo "moe_lora_attach_ffn=true moe_lora_attach_attn=false $ROUTER_FLAGS" \
           "moe_lora_routing_mode=independent moe_lora_n_experts=1 moe_lora_top_k=1" \
           "moe_lora_rank=256 moe_lora_alpha=512" ;;
    attn64_dense256)  # 组合臂 C: attn r64 + dense FFN 旁路 r256 (103.81M, 2×预算)
      echo "moe_lora_attach_ffn=true moe_lora_attach_attn=true $ROUTER_FLAGS" \
           "moe_lora_attn_targets=auto moe_lora_attn_rank=64 moe_lora_attn_alpha=128" \
           "moe_lora_routing_mode=independent moe_lora_n_experts=1 moe_lora_top_k=1" \
           "moe_lora_rank=256 moe_lora_alpha=512" ;;
    attn32_dense128)  # 组合臂 D: attn r32 + dense FFN 旁路 r128 (51.90M, ≈1×预算)
      echo "moe_lora_attach_ffn=true moe_lora_attach_attn=true $ROUTER_FLAGS" \
           "moe_lora_attn_targets=auto moe_lora_attn_rank=32 moe_lora_attn_alpha=64" \
           "moe_lora_routing_mode=independent moe_lora_n_experts=1 moe_lora_top_k=1" \
           "moe_lora_rank=128 moe_lora_alpha=256" ;;
    *) echo "UNKNOWN_ARM:$1（用 attn_r64/dense_r256/attn64_dense256/attn32_dense128）" >&2; return 1 ;;
  esac
}

# arm_batch <arm> -> 该臂的 batch 覆盖(空串 = 沿用 yaml 的 PBS12×GAS3)。
# 带 FFN 分支使用 PBS6×GAS6，保持四卡有效 batch 为 144。
arm_batch() {
  case "$1" in
    attn_r64) echo "" ;;
    *)        echo "per_device_train_batch_size=${FFN_PBS:-6} gradient_accumulation_steps=${FFN_GAS:-6}" ;;
  esac
}

_launch() {
  local _g; IFS=',' read -ra _g <<< "$GPUS"; local ngpu=${#_g[@]}
  if (( ngpu > 1 )); then echo "torchrun --nproc_per_node=$ngpu --master_port=$MASTER_PORT"; else printf '%q\n' "${PYTHON_BIN:-python}"; fi
}

_run() {
  echo "+ $*" | tee -a "${LOGFILE:-/dev/null}"
  [[ "$RUNNER_DRY_RUN" == "1" ]] && return 0
  if [[ -n "${LOGFILE:-}" ]]; then eval "$*" 2>&1 | tee -a "$LOGFILE"; else eval "$*"; fi
}

# _eval_math_one <arm> <bench:math500> <adapter_dir> <res_dir>
_eval_math_one() {
  local arm="$1" bench="$2" adapter="$3" resdir="$4"
  local launch; launch="$(_launch)"
  local script expected out lim="" tag=""
  case "$bench" in
    math500) script=eval_scripts/eval_math500.py; expected=500 ;;
    *) echo "UNKNOWN_EVAL:$bench" >&2; return 1 ;;
  esac
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && { lim="--limit $SMOKE_EVAL_LIMIT"; tag=".limit${SMOKE_EVAL_LIMIT}"; expected=$SMOKE_EVAL_LIMIT; }
  out="$resdir/adapter_${bench}${tag}.jsonl"
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$resdir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
    if (( n >= expected )); then
      echo "[$arm/$bench] $out 已有 $n 行(>=$expected), SKIP" | tee -a "$LOGFILE"; return 0
    fi
  fi
  _run "CUDA_VISIBLE_DEVICES=$GPUS $launch $script --base_model ${QWEN3@Q} --adapter_path ${adapter@Q} \
        --batch_size $EVAL_BSZ --max_new_tokens $EVAL_MAXNEW $lim --save_path ${out@Q}"
}

# _eval_intent_one <arm> <adapter_dir> <res_dir>
_eval_intent_one() {
  local arm="$1" adapter="$2" resdir="$3"
  local launch; launch="$(_launch)"
  local expected=500 debug_args=""
  local predictions="$resdir/intent_predictions.jsonl"
  local metrics="$resdir/intent_metrics.json"
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && { expected="$SMOKE_EVAL_LIMIT"; debug_args="--debug --debug-samples $SMOKE_EVAL_LIMIT"; }
  [[ "$RUNNER_DRY_RUN" == "1" ]] || mkdir -p "$resdir"
  if [[ "$SKIP_EXISTING" == "1" && -f "$predictions" && -f "$metrics" ]]; then
    local n; n=$(wc -l < "$predictions" 2>/dev/null || echo 0)
    if (( n >= expected )); then
      echo "[$arm/intent] $predictions 已有 $n 行(>=$expected), SKIP" | tee -a "$LOGFILE"; return 0
    fi
  fi
  local eval_pythonpath="$REPO/eval_scripts/esft:$REPO"
  _run "CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONPATH=${eval_pythonpath@Q} $launch eval_scripts/esft/eval_esft.py \
        --esft-dir ${DATA_ROOT@Q}/esft_eval --task intent --mode all --template esft_raw --prompt-protocol auto \
        --runtime-merge --runtime-merge-device-map auto --hf-generate-batch-size $EVAL_BSZ \
        --tensor-parallel-size 1 --max-new-tokens $EVAL_MAXNEW $debug_args \
        --model-name-or-path ${QWEN3@Q} --adapter-name-or-path ${adapter@Q} --output-dir ${resdir@Q}"
}

# _eval_one <arm> <bench> <adapter_dir> <res_dir>
_eval_one() {
  local arm="$1" bench="$2" adapter="$3" resdir="$4"
  case "$DOMAIN:$bench" in
    math:math500) _eval_math_one "$arm" "$bench" "$adapter" "$resdir" ;;
    intent:intent) _eval_intent_one "$arm" "$adapter" "$resdir" ;;
    *) echo "UNKNOWN_EVAL:$bench for DOMAIN=$DOMAIN" >&2; return 1 ;;
  esac
}

# do_arm <arm>
do_arm() {
  local arm="$1"
  local flags; flags="$(arm_flags "$arm")" || return 1
  local ep="ep${RUN_EPOCHS}"
  local suffix=""
  [[ -n "$RUN_TAG" ]] && suffix="_${RUN_TAG}"
  local out resdir
  if [[ -n "$MODEL_SUBDIR" ]]; then
    out="$OUT_ROOT/$MODEL_SUBDIR/${arm}${suffix}_${ep}"
    resdir="$RES_ROOT/$MODEL_SUBDIR/${arm}${suffix}_${ep}"
  else
    out="$OUT_ROOT/qwen3_${arm}${suffix}_${ep}"
    resdir="$RES_ROOT/qwen3/${arm}${suffix}_${ep}"
  fi
  if [[ "$RUNNER_DRY_RUN" == "1" ]]; then
    LOGFILE=/dev/null
  else
    mkdir -p "$RES_ROOT"
    # Preserve prior/partial logs so SKIP_EXISTING or a resumed sweep does not
    # erase diagnostic evidence from an earlier RJob.
    if [[ -n "$MODEL_SUBDIR" ]]; then
      LOGFILE="$RES_ROOT/$MODEL_SUBDIR/logs/${arm}${suffix}_${ep}.log"
    else
      LOGFILE="$RES_ROOT/qwen3_${arm}${suffix}_${ep}.log"
    fi
    mkdir -p "$(dirname "$LOGFILE")"; touch "$LOGFILE"
  fi
  echo "[budget_align|qwen3/$DOMAIN/$arm] 日志: $REPO/$LOGFILE | 卡=$GPUS | flags: $flags" | tee -a "$LOGFILE"

  local rc=0
  if [[ "$PHASE" == "train" || "$PHASE" == "full" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -f "$out/moe_lora_state.safetensors" ]]; then
      echo "[$arm] $out/moe_lora_state.safetensors 已存在, 训练 SKIP" | tee -a "$LOGFILE"
    else
      echo -e "\n========== [$arm] 训练 ==========" | tee -a "$LOGFILE"
      local ttrain="$TRAIN_PROTOCOL"
      [[ -n "$SMOKE_MAX_STEPS" ]] && ttrain="$ttrain max_steps=$SMOKE_MAX_STEPS"
      local batch; batch="$(arm_batch "$arm")"
      local lr_override=""
      [[ -n "$LEARNING_RATE" ]] && lr_override="learning_rate=$LEARNING_RATE"
      _run "CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 \
            llamafactory-cli train ${MCFG@Q} model_name_or_path=${QWEN3@Q} deepspeed=${DS_CONFIG@Q} dataset_dir=${DATA_ROOT@Q} $ttrain $batch \
            seed=$SEED data_seed=$DATA_SEED $flags $lr_override \
            report_to=none overwrite_output_dir=true output_dir=${out@Q}" \
        || { echo "[$arm] 训练失败"; return 1; }
    fi
  fi

  if [[ "$PHASE" == "eval" || "$PHASE" == "full" ]]; then
    echo -e "\n========== [$arm] 评测 ==========" | tee -a "$LOGFILE"
    if [[ "$RUNNER_DRY_RUN" == "1" || -f "$out/moe_lora_state.safetensors" ]]; then
      for bench in $EVALS; do _eval_one "$arm" "$bench" "$out" "$resdir" || rc=1; done
    else
      echo "[$arm] 缺少 $out/moe_lora_state.safetensors, 跳过评测"; rc=1
    fi
  fi

  return $rc
}

echo "=================================================================="
echo " budget_align 启动: DOMAIN=$DOMAIN | ARMS=$ARMS | PHASE=$PHASE | EVALS=$EVALS | GPUS=$GPUS"
echo "   learning_rate=${LEARNING_RATE:-MCFG默认值} | run_tag=${RUN_TAG:-无}"
echo "   seed=$SEED | data_seed=$DATA_SEED"
echo "   attach_router=$ATTACH_ROUTER | router_rank=$ROUTER_RANK | router_alpha=$ROUTER_ALPHA"
echo "   输出根: $OUT_ROOT/ + $RES_ROOT/"
echo "=================================================================="

declare -a SUMMARY=()
fail_count=0
for arm in $ARMS; do
  if do_arm "$arm"; then SUMMARY+=("[OK]   $arm"); else SUMMARY+=("[FAIL] $arm"); fail_count=$((fail_count+1)); fi
done

echo -e "\n========== budget_align 汇总 =========="
printf '  %s\n' "${SUMMARY[@]:-（无）}"
if [[ -n "$MODEL_SUBDIR" ]]; then
  echo "ckpt: $OUT_ROOT/$MODEL_SUBDIR/<arm>${RUN_TAG:+_$RUN_TAG}_ep${RUN_EPOCHS}/ | 结果: $RES_ROOT/$MODEL_SUBDIR/<arm>${RUN_TAG:+_$RUN_TAG}_ep${RUN_EPOCHS}/"
else
  echo "ckpt: $OUT_ROOT/qwen3_<arm>${RUN_TAG:+_$RUN_TAG}_ep${RUN_EPOCHS}/ | 结果: $RES_ROOT/qwen3/<arm>${RUN_TAG:+_$RUN_TAG}_ep${RUN_EPOCHS}/"
fi
(( fail_count > 0 )) && exit 1
exit 0
