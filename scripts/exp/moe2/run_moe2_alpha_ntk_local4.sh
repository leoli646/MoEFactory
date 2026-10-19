#!/usr/bin/env bash
# MoE2-LoRA weight_pool: alpha x normalize_topk 二维格补全（本地 4 卡顺序跑）
#
# 起因：P0 五臂上游评测出现一个干净的赢家
#   wp-ntk-a128 (alpha=128, ntk=true)  Intent .818  dMMLU -0.31  dIF +1.66  39.6M
#   BASELINE n256_r32 (无 attn)        Intent .840  dMMLU -6.24  dIF -16.27 51.0M
# 但 alpha=128 与 normalize_topk 是同时引入的，只有一个组合点，无法判断
#   (a) 是 ntk 单独修好的、alpha 只是搭车，还是
#   (b) 两者必须配合。
# 且 alpha=128 是否为平台/尖峰未知（尖峰会被 reviewer 质疑脆弱）。
#
# 本脚本默认补两个格子（均为 qwen3 / intent / lr1e-3，与 P0 口径一致）：
#   A1  n128_r32_wp_a128       alpha=128 ntk=false  -> 补 2x2 设计第四格
#   A2  n128_r32_wp_ntk_a96    alpha=96  ntk=true   -> alpha 下侧
#   A3  n128_r32_wp_ntk_a192   alpha=192 ntk=true   -> 可选，不再默认运行
#
# 已有的两格（无需重跑）：
#   alpha=64  ntk=true  -> wp-ntk       (saves/moe2_wp_lr_alpha_binit/.../n128_r32_wp_ntk)
#   alpha=128 ntk=true  -> wp-ntk-a128  (saves/moe2_wp_lr_alpha_binit/.../n128_r32_wp_ntk_a128)
#   alpha=64  ntk=false -> 原始 wp      (saves/moe2_official_lr_fixed_init/.../n128_r32_wp)
#
# 后缀解析已用 set_setting_flags 静态验证，三者均正确剥离，无需改代码。
#
# 用法：
#   DRY_RUN=1 bash scripts/exp/moe2/run_moe2_alpha_ntk_local4.sh          # 只打印命令
#   DRY_RUN=0 bash scripts/exp/moe2/run_moe2_alpha_ntk_local4.sh          # 真跑（训练+下游+上游）
#   DRY_RUN=0 ARMS="n128_r32_wp_a128" bash scripts/...           # 只跑一个臂
#   DRY_RUN=0 MODEL=gemma4 SKIP_UPSTREAM=1 bash scripts/...       # 只跑 a128 + Intent
#   DRY_RUN=0 MODEL=gptoss SKIP_UPSTREAM=1 bash scripts/...       # 只跑 a128 + Intent
#   DRY_RUN=0 SMOKE=1 bash scripts/...                           # 限样本快验通路
#   DRY_RUN=0 SKIP_TRAIN=1 bash scripts/...                      # 只补上游（ckpt 已存在时）
#
# 变量：GPUS(0,1,2,3) BSZ(4，仅下游/上游评测；训练固定 pbs=6) DRY_RUN(1) SMOKE(0)
#       SKIP_TRAIN(0) SKIP_UPSTREAM(0) SKIP_EXISTING(1)
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"

# Use the caller's activated environment.
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL="${MODEL:-qwen3}"
TASK="${TASK:-intent}"
LR="${LR:-lr1e-3}"
GPUS="${GPUS:-0,1,2,3}"
TRAIN_GAS="${TRAIN_GAS:-6}"
DRY_RUN="${DRY_RUN:-1}"
SMOKE="${SMOKE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_UPSTREAM="${SKIP_UPSTREAM:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

case "$MODEL" in
  qwen3|gemma4|gptoss) ;;
  *) echo "MODEL 必须是 qwen3、gemma4 或 gptoss，当前: $MODEL" >&2; exit 2 ;;
esac

# 上游评测口径：与 P0 (run_moe2_redesign_local4.sh:31-35) 逐项一致
ENGINE="${ENGINE:-hf}"
MAX_LEN="${MAX_LEN:-4096}"
IFEVAL_MAXNEW="${IFEVAL_MAXNEW:-1280}"
MMLU_LIMIT="${MMLU_LIMIT:-0}"
IFEVAL_LIMIT="${IFEVAL_LIMIT:-0}"

# 下游 Intent / 上游 HF batch：默认 4。不继承调用者环境，避免污染。
BSZ="${BSZ:-4}"
if [[ -n "${HF_BSZ:-}" && "${HF_BSZ}" != "${BSZ}" ]]; then
  echo "[warn] 忽略继承的 HF_BSZ=${HF_BSZ}，请用 BSZ=<n> 指定（当前 BSZ=${BSZ}）" >&2
fi

if (( SMOKE )); then
  # Never let a 3-step smoke adapter satisfy SKIP_EXISTING for a full run.
  OUT_ROOT="${SMOKE_OUT_ROOT:-$REPO/saves/moe2_alpha_ntk_grid_smoke}"
  RES_ROOT="${SMOKE_RES_ROOT:-$REPO/results/moe2_alpha_ntk_grid_smoke}"
  RES_UP="${SMOKE_RES_UP:-$REPO/results/upstream_moe2_redesign_smoke/$MODEL}"
else
  OUT_ROOT="${OUT_ROOT:-$REPO/saves/moe2_alpha_ntk_grid}"
  RES_ROOT="${RES_ROOT:-$REPO/results/moe2_alpha_ntk_grid}"
  # 上游结果并入 P0 同一根，方便 collect 脚本一次出全表
  RES_UP="${RES_UP:-$REPO/results/upstream_moe2_redesign/$MODEL}"
fi

if [[ -z "${ARMS:-}" ]]; then
  case "$MODEL" in
    qwen3)
      ARMS="n128_r32_wp_a128
n128_r32_wp_ntk_a96"
      ;;
    gemma4|gptoss)
      ARMS="n128_r32_wp_ntk_a128"
      ;;
  esac
fi

# Configure the same NVIDIA runtime paths used by the working RJob wrapper and
# make CUDA_HOME available to DeepSpeed and later HF upstream evaluation.
# shellcheck disable=SC1091
source "$REPO/scripts/exp/moe2/local_cuda_env.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
[[ -n "$PYTHON_BIN" ]] || { echo "未找到 Python 解释器。" >&2; exit 2; }
require_cuda_toolkit=0
(( DRY_RUN == 0 && SKIP_TRAIN == 0 )) && require_cuda_toolkit=1
(( DRY_RUN )) || setup_local_cuda_env "$PYTHON_BIN" "$require_cuda_toolkit"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if (( DRY_RUN )); then
    case "${*: -1}" in
      scripts/exp/moe2/run_moe2_official_pilot.sh|scripts/reproduce/run_upstream_eval.sh) DRY_RUN=1 "$@" ;;
    esac
  else
    "$@"
  fi
}

# 上游评测：MMLU-Pro(logprob) + IFEval。口径与 P0 逐项一致（run_moe2_redesign_local4.sh）
eval_upstream() {
  local label="$1" adapter="$2" outdir="$3"
  local mmlu_lim="$MMLU_LIMIT" if_lim="$IFEVAL_LIMIT"
  if (( SMOKE )); then mmlu_lim=16; if_lim=16; fi
  run env \
    "MODEL=$MODEL" \
    "ADAPTER=$adapter" \
    "OUT_DIR=$outdir" \
    "GPUS=$GPUS" \
    "TASKS=mmlu_pro ifeval" \
    "ENGINE=$ENGINE" \
    "SKIP_EXISTING=$SKIP_EXISTING" \
    "MAX_LEN=$MAX_LEN" \
    "IFEVAL_MAXNEW=$IFEVAL_MAXNEW" \
    "HF_BSZ=$BSZ" \
    "MMLU_LIMIT=$mmlu_lim" \
    "IFEVAL_LIMIT=$if_lim" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    bash scripts/reproduce/run_upstream_eval.sh
}

echo "===== MoE2 alpha x normalize_topk 格子补全 ====="
echo "  model=$MODEL task=$TASK lr=$LR gpus=$GPUS eval_bsz=$BSZ train_pbs=6"
echo "  out=$OUT_ROOT"
echo "  upstream_out=$RES_UP"
(( DRY_RUN )) && echo "  [DRY_RUN=1 只打印，不执行]"
echo

i=0
failures=0
n=$(grep -c . <<<"$(echo "$ARMS" | sed '/^\s*$/d')")
while read -r setting; do
  [[ -z "$setting" ]] && continue
  i=$((i+1))
  label="${setting#n128_r32_wp}"; label="wp${label:-}"   # n128_r32_wp_ntk_a96 -> wp_ntk_a96
  ck="$OUT_ROOT/$MODEL/$LR/$TASK/moe2official_$setting"
  echo "---------- arm $i/$n: $setting  (label=$label) ----------"

  if (( SKIP_TRAIN )); then
    echo "  [SKIP_TRAIN=1 跳过训练]"
  else
    train_command=(
      env
      "MODEL=$MODEL"
      "TASK=$TASK"
      "SETTINGS=$setting"
      "LR_TRIALS=$LR"
      "RUNNER_DRY_RUN=$DRY_RUN"
      "GPUS=$GPUS"
      "TRAIN_GAS=$TRAIN_GAS"
      PHASE=full
      "SKIP_EXISTING=$SKIP_EXISTING"
      "INTENT_EVAL_BATCH_SIZE=$BSZ"
      "OUT_ROOT=$OUT_ROOT"
      "RES_ROOT=$RES_ROOT"
    )
    if (( SMOKE )); then
      train_command+=(SMOKE_MAX_STEPS=3 SMOKE_MAX_SAMPLES=256 SMOKE_EVAL_LIMIT=16)
    fi
    train_command+=(bash scripts/exp/moe2/run_moe2_official_pilot.sh)
    # PHASE=full: 训练 + 下游 intent 评测一次做完
    run "${train_command[@]}" \
      || { echo "!! $setting 训练失败，跳过其上游评测" >&2; failures=$((failures + 1)); continue; }
  fi

  if (( SKIP_UPSTREAM )); then
    echo "  [SKIP_UPSTREAM=1 跳过 MMLU-Pro/IFEval 上游评测]"
  elif (( DRY_RUN )) || [[ -f "$ck/moe_lora_config.json" ]]; then
    eval_upstream "$label" "$ck" "$RES_UP/$label" \
      || { echo "!! $setting 上游评测失败，继续下一个" >&2; failures=$((failures + 1)); }
  else
    echo "!! 未找到 ckpt，跳过上游: $ck" >&2
    failures=$((failures + 1))
  fi
  echo
done <<< "$(echo "$ARMS" | sed '/^\s*$/d')"

echo "===== 全部完成。汇总： ====="
echo "  failures=$failures"
echo "  python3 scripts/exp/moe2/collect_moe2_redesign.py --upstream-root '$RES_UP' \\"
echo "    --out results/upstream_moe2_redesign/summary_$MODEL.csv --model $MODEL"
exit "$((failures > 0 ? 1 : 0))"
