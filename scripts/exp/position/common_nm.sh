#!/usr/bin/env bash
# 新模型(gemma-4-26B-A4B / gpt-oss-20b)实验公共库:被 run_peft_compare_nm.sh /
# run_pos_ab_nm.sh / eval_general_nm.sh 引用。
#
# 设计要点(与既有 qwen3 流程严格隔离):
#   - 只读取 examples/peft_compare_gemma4|gptoss/ 与 examples/train_moe_lora_newmodels/ 配置;
#   - 输出隔离在 saves/peft_compare_nm/ saves/pos_ab_nm/ 与 results/*_nm/ 下,
#     不写任何既有 qwen3 目录;
#   - 评测协议复用现有脚本(eval_math500/eval_esft/eval_mmlu_pro/
#     eval_targeted_general/eval_commonsense),不改其代码。
#
# 模型速查:
#   gemma4 : Gemma4ForConditionalGeneration(多模态壳), 30 层全 MoE, E=128 top8,
#            d=2816, vocab 262k。文本层在 model.language_model.layers(src 已支持)。
#   gptoss : GptOssForCausalLM, 24 层全 MoE, E=32 top4, d=2880, mxfp4 权重。
#            训练端 llamafactory 自动 Mxfp4Config(dequantize=True);
#            评测端 transformers 因缺 kernels 包自动回退 bf16 反量化(警告属预期)。

set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-${DATA_ROOT:-$REPO/data}}"
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR:-/tmp}/hf_datasets}"

CKPT="${CKPT_ROOT:-$REPO/models}"
GEMMA4_BASE="${MODEL_PATH:-${GEMMA4_BASE:-$CKPT/gemma-4-26B-A4B}}"
GPTOSS_BASE="${MODEL_PATH:-${GPTOSS_BASE:-$CKPT/gpt-oss-20b}}"

# ---- 全局可覆盖参数 ----
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29710}"
DS_CONFIG="${DS_CONFIG:-examples/deepspeed/ds_z2_config.json}"
EVAL_MAXNEW="${EVAL_MAXNEW:-512}"
INTENT_EPOCHS="${INTENT_EPOCHS:-3}"
# 冒烟: 限制训练步数/样本数/评测样本数(为空=全量)
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-}"

# model_meta <gemma4|gptoss> -> 设置 MTAG / MBASE / MCFG_DIR / MBASELINE_CFG /
#   MINTENT_BATCH(intent cutoff=4096 时的 batch 覆盖, 有效 batch 恒 =144)
model_meta() {
  case "$1" in
    gemma4)
      MTAG=gemma4; MBASE="$GEMMA4_BASE"
      MCFG_DIR="examples/peft_compare_gemma4"
      MBASELINE_CFG="examples/train_moe_lora_newmodels/baseline1_gemma4_math.yaml"
      # 26B + 262k vocab: intent(cutoff 4096) 压 batch 防 OOM
      MINTENT_BATCH="per_device_train_batch_size=4 gradient_accumulation_steps=9"
      MODEL_N_EXPERTS=128; MODEL_TOP_K=8; MODEL_FFN_RANK=2
      MDAS_TOPK=8            # das_lora 选专家数 = 模型 top_k
      ;;
    gptoss)
      MTAG=gptoss; MBASE="$GPTOSS_BASE"
      MCFG_DIR="examples/peft_compare_gptoss"
      MBASELINE_CFG="examples/train_moe_lora_newmodels/baseline1_gptoss_math.yaml"
      MINTENT_BATCH="per_device_train_batch_size=6 gradient_accumulation_steps=6"
      MODEL_N_EXPERTS=32; MODEL_TOP_K=4; MODEL_FFN_RANK=8
      MDAS_TOPK=4
      ;;
    *) echo "UNKNOWN_MODEL:$1 (用 gemma4|gptoss)" >&2; return 1 ;;
  esac
}

# task_meta <math|intent> -> TTRAIN_OVERRIDE / TEPOCHS / TEVAL_BSZ
#   math  : 配置自带 metamathqa_r1/template=default, 无需覆盖(除冒烟限额)
#   intent: 覆盖 dataset/template/cutoff/packing/epochs + per-model batch(与 qwen3 协议一致)
task_meta() {
  local task="$1"
  case "$task" in
    math)
      TEPOCHS="${MATH_EPOCHS:-1}"
      TTRAIN_OVERRIDE="num_train_epochs=$TEPOCHS"
      [[ -n "$SMOKE_MAX_SAMPLES" ]] && TTRAIN_OVERRIDE="$TTRAIN_OVERRIDE max_samples=$SMOKE_MAX_SAMPLES"
      TEVAL_BSZ="${EVAL_BSZ:-32}"
      ;;
    intent)
      TEPOCHS="$INTENT_EPOCHS"
      TTRAIN_OVERRIDE="dataset=esft_intent template=esft_raw cutoff_len=4096 packing=false num_train_epochs=$TEPOCHS $MINTENT_BATCH"
      # gptoss intent 强制 eager(2026-07-14): flex attention 在 intent/cutoff4096 上
      # rank 级无声 CUDA fault -> step 0 卡死(gloo monitoredBarrier 600s 超时)。
      # 07-12 全部 gptoss intent 方法都在 eager 下训完(6x6 无 OOM), 保持同一协议。
      # math 必须保留 flex(eager 在 math/attn-adapter 下 OOM, 见 attention.py 注释)。
      [[ "${MTAG:-}" == "gptoss" ]] && TTRAIN_OVERRIDE="$TTRAIN_OVERRIDE flash_attn=disabled"
      [[ -n "$SMOKE_MAX_SAMPLES" ]] && TTRAIN_OVERRIDE="$TTRAIN_OVERRIDE max_samples=$SMOKE_MAX_SAMPLES"
      TEVAL_BSZ="${EVAL_BSZ:-8}"
      ;;
    *) echo "UNKNOWN_TASK:$task (用 math|intent)" >&2; return 1 ;;
  esac
  [[ -n "$SMOKE_MAX_STEPS" ]] && TTRAIN_OVERRIDE="$TTRAIN_OVERRIDE max_steps=$SMOKE_MAX_STEPS"
  return 0
}

_launch() {  # python / torchrun 启动器(按 GPU 数)
  local _g; IFS=',' read -ra _g <<< "$GPUS"; local ngpu=${#_g[@]}
  if (( ngpu > 1 )); then echo "torchrun --nproc_per_node=$ngpu --master_port=$MASTER_PORT"; else printf '%q\n' "${PYTHON_BIN:-python}"; fi
}

_run() {  # 回显 + 执行 + 落日志
  echo "+ $*" | tee -a "${LOGFILE:-/dev/null}"
  [[ "${RUNNER_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]] && return 0
  if [[ -n "${LOGFILE:-}" ]]; then eval "$*" 2>&1 | tee -a "$LOGFILE"; else eval "$*"; fi
}

log() { echo -e "\n========== $* ==========" | tee -a "${LOGFILE:-/dev/null}"; }


# 分布式冒烟诊断(仅 多卡+冒烟 时启用, 与既有 peft_compare 脚本一致)
_diag_env() {
  local _gg; IFS=',' read -ra _gg <<< "$GPUS"; local _ng=${#_gg[@]}
  if [[ "$_ng" -gt 1 && -n "$SMOKE_MAX_STEPS" ]]; then
    # PARTIAL_FT_HANG_DUMP_SEC: partial_ft 卡步超时自动转储各 rank 堆栈(见 partial_ft.py)
    echo "TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_TRACE_BUFFER_SIZE=2000 TORCH_NCCL_DUMP_ON_TIMEOUT=1 NCCL_DEBUG=WARN PYTHONFAULTHANDLER=1 PARTIAL_FT_HANG_DUMP_SEC=${PARTIAL_FT_HANG_DUMP_SEC:-600}"
  fi
}
_timeout_wrap() {
  local _gg; IFS=',' read -ra _gg <<< "$GPUS"; local _ng=${#_gg[@]}
  if [[ "$_ng" -gt 1 && -n "$SMOKE_MAX_STEPS" ]]; then
    echo "timeout --signal=TERM --kill-after=30 ${SMOKE_TIMEOUT:-1800}"
  fi
}

# ---- domain 评测 ----

# MATH-500(math domain 统一口径)
# adapter 为空 = zero-shot base; adapter 类型由 eval_scripts/_common.py 自动识别。
# 第三个参数可覆盖结果文件名; peft_method 默认保留 math500.jsonl,
# position ablation 使用 adapter_math500.jsonl / zeroshot_math500.jsonl。
do_eval_math500() {
  local adapter="$1" outdir="$2" save_name="${3:-math500.jsonl}"
  local launch; launch="$(_launch)"
  [[ "${RUNNER_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]] || mkdir -p "$outdir"
  local ad="" lim=""
  [[ -n "$adapter" ]] && printf -v ad -- '--adapter_path %q' "$adapter"
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && lim="--limit $SMOKE_EVAL_LIMIT"
  _run "CUDA_VISIBLE_DEVICES=$GPUS $launch eval_scripts/eval_math500.py \
        --base_model ${MBASE@Q} $ad --batch_size $TEVAL_BSZ --max_new_tokens $EVAL_MAXNEW $lim \
        --save_path ${outdir@Q}/${save_name@Q}"
}


# intent(esft 程序化打分, runtime-merge + HF 推理, 与 qwen3 完全一致)
# gptoss: --flash-attn disabled 强制 eager。flex attention 在 intent 长 prompt 下
# illegal memory access(2026-07-14, zeroshot 与 train 双现场); custom adapter 路径
# 本就 AutoModel 默认(eager), 该开关让 zeroshot(load_model 路径)与其对齐。
do_eval_intent() {
  local adapter="$1" outdir="$2"
  local launch; launch="$(_launch)"
  [[ "${RUNNER_DRY_RUN:-${DRY_RUN:-0}}" == "1" ]] || mkdir -p "$outdir"
  local ad="" dbg="" attn=""
  [[ -n "$adapter" ]] && printf -v ad -- '--adapter-name-or-path %q' "$adapter"
  [[ -n "$SMOKE_EVAL_LIMIT" ]] && dbg="--debug --debug-samples $SMOKE_EVAL_LIMIT"
  [[ "${MTAG:-}" == "gptoss" ]] && attn="--flash-attn disabled"
  local eval_pythonpath="$REPO/eval_scripts/esft:$REPO"
  local eenv="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=${eval_pythonpath@Q}"
  _run "CUDA_VISIBLE_DEVICES=$GPUS $eenv $launch eval_scripts/esft/eval_esft.py \
        --esft-dir ${DATA_ROOT@Q}/esft_eval --task intent --mode all --template esft_raw --prompt-protocol auto \
        --runtime-merge --runtime-merge-device-map auto $attn \
        --hf-generate-batch-size $TEVAL_BSZ --tensor-parallel-size 1 \
        --max-new-tokens $EVAL_MAXNEW $dbg \
        --model-name-or-path ${MBASE@Q} $ad --output-dir ${outdir@Q}"
}

# do_eval_domain <task> <adapter|""> <outdir>
do_eval_domain_peft() {
  local task="$1" adapter="$2" outdir="$3"
  case "$task" in
    math)   do_eval_math500 "$adapter" "$outdir" ;;
    intent) do_eval_intent "$adapter" "$outdir" ;;
    *) echo "UNKNOWN_TASK:$task" >&2; return 1 ;;
  esac
}
