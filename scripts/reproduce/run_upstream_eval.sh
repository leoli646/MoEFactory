#!/usr/bin/env bash
set -euo pipefail

# 单个权重（或 base）的上游数据集评测：MMLU-Pro (logprob 遗忘探针口径) + IFEval。
# **自动按权重结构选推理引擎**（探测靠 vllm_moe_lora.is_vllm_supported，CPU 读权重名，
# parse_overlay fail-fast）：
#   vLLM（overlay 静态合并进底座权重，与 token/路由无关的权重变换才可合并）：
#     - attn LoRA / router LoRA（位置×lr 臂含 attn_r64/follow_moe）
#     - n1top1-independent dense 旁路（layer_bypass_r256）
#     - expert_weight_rX（moe_lora _wp_adapter，逐专家 LoRA；含 aux_coef 链）
#     - weight_peft: dora / oft / ia3 / lora（逐专家 + router 权重变换）
#     - partial_last（partial_ft，逐专家全权重替换）
#   HF（_common.load 吃全部结构；增量依赖 token 路由或谱支撑复现脆弱，无法静态合并）：
#     - lora_pool n_experts>1（expert_bypass_r2 / moe_n8）、learned 路由（moe2）、das_lora
#     - weight_peft: fourier（谱支撑靠训练注入顺序 seed 复现）
#     - gemma4 一律 HF（vLLM transformers backend 不支持其异构 KV 头）
#     - gptoss 的 weight_peft/expert_weight（底座专家 mxfp4 量化，merge 需反量化-再量化有损）
#
# 用法（单个模型或 adapter）:
#   MODEL=qwen3 ADAPTER=saves/.../xxx OUT_DIR=results/upstream_eval/.../xxx \
#     GPUS=0,1,2,3 bash scripts/reproduce/run_upstream_eval.sh
#   MODEL=qwen3 ARM=base OUT_DIR=results/upstream_eval/base/qwen3 \
#     bash scripts/reproduce/run_upstream_eval.sh   # base（无 adapter）
#
# 变量: MODEL(必填) ADAPTER(权重目录; 空=base) OUT_DIR(必填) GPUS(默认 0,1,2,3)
#       TASKS(默认 "mmlu_pro ifeval") ENGINE(auto|vllm|hf, 默认 auto) SKIP_EXISTING
#       MMLU_LIMIT IFEVAL_LIMIT MAX_LEN

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-qwen3}"
ADAPTER="${ADAPTER:-}"
OUT_DIR="${OUT_DIR:?need OUT_DIR}"
GPUS="${GPUS:-0,1,2,3}"
TASKS="${TASKS:-mmlu_pro ifeval}"
ENGINE="${ENGINE:-auto}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"
MAX_LEN="${MAX_LEN:-4096}"
MMLU_LIMIT="${MMLU_LIMIT:-0}"
IFEVAL_LIMIT="${IFEVAL_LIMIT:-0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
IFEVAL_MAXNEW="${IFEVAL_MAXNEW:-1280}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MOEFACTORY_DATA_ROOT="$DATA_ROOT"
export NLTK_DATA="${NLTK_DATA:-$DATA_ROOT/nltk_data}"
# HF-engine weight_pool grouped forward can transiently spike (per-token top_k
# gather); default to expandable_segments to reclaim the reserved-but-unallocated
# fragmentation that OOM'd a long-prompt MMLU-Pro shard.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() { echo "[$(date +%F\ %T)] $*"; }

# ---- base model 解析 ----
case "$MODEL" in
  qwen3)  BASE_DIR=Qwen3-30B-A3B-Base ;;
  gemma4) BASE_DIR=gemma-4-26B-A4B ;;
  gptoss) BASE_DIR=gpt-oss-20b ;;
  *) echo "Unsupported MODEL=$MODEL" >&2; exit 2 ;;
esac
case "$MODEL" in
  qwen3) MODEL_DEFAULT="${QWEN3_BASE:-$CKPT_ROOT/$BASE_DIR}" ;;
  gemma4) MODEL_DEFAULT="${GEMMA4_BASE:-$CKPT_ROOT/$BASE_DIR}" ;;
  gptoss) MODEL_DEFAULT="${GPTOSS_BASE:-$CKPT_ROOT/$BASE_DIR}" ;;
esac
BASE_MODEL="${BASE_MODEL:-${MODEL_PATH:-$MODEL_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$BASE_MODEL/config.json" ]] || { echo "missing base model: $BASE_MODEL" >&2; exit 2; }
fi
# Dry previews choose the universally supported HF path without importing torch.
[[ "$DRY_RUN" == 1 && "$ENGINE" == auto ]] && ENGINE=hf
run_eval() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  [[ "$DRY_RUN" == 1 ]] || "$@"
}

# ---- 引擎选择：gemma4 强制 HF；否则 auto 探测（base 走 vLLM/HF 均可，默认 vLLM）----
if [[ "$ENGINE" == "auto" ]]; then
  if [[ "$MODEL" == gemma4 ]]; then
    ENGINE="hf"
  elif [[ -z "$ADAPTER" ]]; then
    ENGINE="vllm"   # base
  else
    if "$PYTHON_BIN" -c 'import sys;sys.path.insert(0,"eval_scripts");import vllm_moe_lora as v;sys.exit(0 if v.is_vllm_supported(sys.argv[1]) else 1)' "$ADAPTER" 2>/dev/null; then
      ENGINE="vllm"
    else
      ENGINE="hf"
    fi
  fi
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NSHARD="${#GPU_ARR[@]}"
[[ "$DRY_RUN" == 1 ]] || mkdir -p "$OUT_DIR/logs"
log "MODEL=$MODEL ENGINE=$ENGINE shards=$NSHARD adapter=${ADAPTER:-BASE}"
log "base=$BASE_MODEL out=$OUT_DIR"

adapter_flag=(); [[ -n "$ADAPTER" ]] && adapter_flag=(--adapter_path "$ADAPTER")

run_task() {
  local task="$1"
  local out_sub="$OUT_DIR/$task"
  if [[ "$SKIP_EXISTING" == "1" && -s "$out_sub/summary.json" ]]; then
    log "skip $task (已有 $out_sub/summary.json)"; return 0
  fi
  [[ "$DRY_RUN" == 1 ]] || mkdir -p "$out_sub"

  if [[ "$task" == "mmlu_pro" ]]; then
    # MMLU-Pro logprob（遗忘探针口径），用 eval_vllm_mmlu_logprob（含 HF 退路）
    local lim=(); [[ "$MMLU_LIMIT" != "0" ]] && lim=(--limit "$MMLU_LIMIT")
    log "task=mmlu_pro (logprob, engine=$ENGINE): $NSHARD shards"
    local pids=()
    for i in "${!GPU_ARR[@]}"; do
      run_eval env "CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]}" "$PYTHON_BIN" eval_scripts/eval_vllm_mmlu_logprob.py \
        --engine "$ENGINE" --base_model "$BASE_MODEL" "${adapter_flag[@]}" \
        --out_dir "$out_sub" --shard_idx "$i" --num_shards "$NSHARD" \
        --max_length "$MAX_LEN" --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --hf_batch_size "${HF_BSZ:-4}" "${lim[@]}" \
        > >(if [[ "$DRY_RUN" == 1 ]]; then cat; else cat > "$OUT_DIR/logs/mmlu_pro.shard${i}.log"; fi) 2>&1 &
      pids+=($!)
    done
    local fail=0
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "mmlu_pro shard$i FAILED"; fail=1; }; done
    [[ "$fail" == "1" ]] && { tail -n 25 -- "$OUT_DIR/logs/mmlu_pro".shard*.log >&2; return 1; }
    run_eval "$PYTHON_BIN" eval_scripts/eval_vllm_mmlu_logprob.py --merge_shards \
      --base_model "$BASE_MODEL" "${adapter_flag[@]}" \
      --out_dir "$out_sub" --num_shards "$NSHARD" 2>&1 | if [[ "$DRY_RUN" == 1 ]]; then cat; else tee "$OUT_DIR/logs/mmlu_pro.merge.log"; fi

  elif [[ "$task" == "ifeval" ]]; then
    # IFEval（generation, strict/loose），用 eval_vllm_general（含 HF 退路）
    local lim=(); [[ "$IFEVAL_LIMIT" != "0" ]] && lim=(--limit "$IFEVAL_LIMIT")
    log "task=ifeval (engine=$ENGINE): $NSHARD shards"
    local pids=()
    for i in "${!GPU_ARR[@]}"; do
      run_eval env "CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]}" "$PYTHON_BIN" eval_scripts/eval_vllm_general.py \
        --task ifeval --engine "$ENGINE" --base_model "$BASE_MODEL" "${adapter_flag[@]}" \
        --out_dir "$out_sub" --shard_idx "$i" --num_shards "$NSHARD" \
        --gen_max_new_tokens "$IFEVAL_MAXNEW" --max_model_len "$MAX_LEN" \
        --gpu_memory_utilization "$GPU_MEM_UTIL" --hf_batch_size "${HF_BSZ:-8}" "${lim[@]}" \
        > >(if [[ "$DRY_RUN" == 1 ]]; then cat; else cat > "$OUT_DIR/logs/ifeval.shard${i}.log"; fi) 2>&1 &
      pids+=($!)
    done
    local fail=0
    for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "ifeval shard$i FAILED"; fail=1; }; done
    [[ "$fail" == "1" ]] && { tail -n 25 -- "$OUT_DIR/logs/ifeval".shard*.log >&2; return 1; }
    run_eval "$PYTHON_BIN" eval_scripts/eval_vllm_general.py --task ifeval --merge_shards \
      --base_model "$BASE_MODEL" "${adapter_flag[@]}" \
      --out_dir "$out_sub" --num_shards "$NSHARD" 2>&1 | if [[ "$DRY_RUN" == 1 ]]; then cat; else tee "$OUT_DIR/logs/ifeval.merge.log"; fi
  else
    echo "unknown task: $task" >&2; return 2
  fi
}

fail=0
for task in $TASKS; do run_task "$task" || fail=1; done
[[ "$fail" == "0" ]] && log "DONE $MODEL ${ADAPTER:-BASE} -> $OUT_DIR" || { log "有失败项"; exit 1; }
