# 原生实验脚本：训练、下游评测与学习率选择

本指南以源仓库 `MoEFactory` 的 shell runner 为准，保留各模型、任务、适配方法的原始配置和训练/评测流程。这里直接调用 `scripts/exp/` 中的四组实验与 `scripts/exp/moe2/` 下的脚本。环境安装见 [环境说明](../../docs/environment.md)，数据准备与注册见 [数据说明](../../docs/data.md)，实现与论文设置的差异见 [实现核对](../../docs/implementation_audit.md)。脚本提供执行入口，不代表已经获得论文结果。

## 运行前统一设置

在公开仓库根目录执行以下设置；后面的每个代码块都在同一个 shell 中运行。默认仅预览，预览也需要模型的真实 `config.json` 来检查结构和参数预算，但不需要加载模型权重。

```bash
set -euo pipefail
export REPO="$PWD"
export MOEFACTORY_DATA_ROOT="${MOEFACTORY_DATA_ROOT:-$REPO/data}"
export CKPT_ROOT="${CKPT_ROOT:-$REPO/models}"
export SAVES="$REPO/saves/four_experiments_native"
export RESULTS="$REPO/results/four_experiments_native"
export GPUS=0,1,2,3
export PYTHON_BIN=python3
export RUNNER_DRY_RUN=1
export LRS="lr1e-5 lr1e-4 lr1e-3"
cd "$REPO"
```

模型目录默认是 `$CKPT_ROOT/Qwen3-30B-A3B-Base`、`$CKPT_ROOT/gemma-4-26B-A4B` 和 `$CKPT_ROOT/gpt-oss-20b`。也可分别设置 `QWEN3_BASE`、`GEMMA4_BASE`、`GPTOSS_BASE`。运行单模型时可以用 `MODEL_PATH`；跨模型循环不要全局设置同一个 `MODEL_PATH`。`MOEFACTORY_DATA_ROOT` 必须与已准备的数据目录一致；训练集名称仍由 `data/dataset_info.json` 注册。

Math 使用 `metamathqa_r1` 训练、MATH-500 测试；Intent 使用注册的 Intent 训练集和 500 条测试样本。默认 Math 训练 1 epoch，Intent 训练 3 epochs。完整网格覆盖 `qwen3 gemma4 gptoss`、两个任务、`42 0 1` 三个 seed，以及 `{1e-5, 1e-4, 1e-3}` 三个学习率。下列循环顺序执行，避免在同一组 GPU 上并发占用。

确认预览、数据、模型和依赖后，执行 `export RUNNER_DRY_RUN=0`，再运行所需代码块。四组 runner 的执行默认值通常是 `0`，因此不要省略本节的显式预览设置。`SKIP_EXISTING=1` 是跳过脚本认定已完成的输出，不保证从优化器 checkpoint 恢复中断训练。

## 1. 方法对比（methods）

默认运行论文第 4.3 节的十个方法，METHODS=all 也选择这十个方法。MoE²-LoRA 使用后面的独立入口。

```bash
METHODS_FULL="weight_lora expert_lora weight_dora weight_oft weight_fourier weight_ia3 expert_moe_lora_n8 layer_lora das_lora partial_last"
for model in qwen3 gemma4 gptoss; do
  for seed in 42 0 1; do
    MODEL="$model" TASKS="math intent" METHODS="$METHODS_FULL" \
      SEED="$seed" DATA_SEED="$seed" SEED_SUBDIR="seed$seed" \
      LR_TRIALS="$LRS" PHASE=full ATTACH_ATTN=0 RUN_UPSTREAM=0 \
      UNIFIED_LAYOUT=1 OUT_ROOT="$SAVES/methods" RES_ROOT="$RESULTS/methods" \
      bash scripts/exp/methods/run_method_comparison.sh
  done
done
```

这里 `weight_lora` 对应专家权重内 LoRA（专家 rank 1、router rank 8），`expert_lora` 是专家旁路，`layer_lora` 是整层旁路；不要仅凭名字将两者混同。此入口关闭 attention，使用各方法实际支持的 router 分支。MoE²-LoRA 的 pilot/redesign 使用独立入口和目录。

DAS 的通用参考集默认是 `$MOEFACTORY_DATA_ROOT/general/commonsense_test/merged.json`；可通过 `DAS_GENERAL=/path/to/general.json` 指定自己的文件。领域参考集分别使用 Math、Intent 的训练数据。`DAS_MAXSAMPLES=200` 控制评分采样数。默认评分缓存位于 `eval_results/peft_compare_granularity/das_${MODEL}_${TASK}.json`，在 seed/LR 之间复用；需要使用既有评分时可设置 `DAS_JSON`。改变参考数据或评分设置后，应指定新的缓存路径，避免误用旧评分。这个 general 集是路由评分参考，不是 MMLU-Pro/IFEval 测试集。

默认 `RUN_UPSTREAM=0` 只做训练和下游评测。只有明确要额外执行 MMLU-Pro/IFEval 且准备好数据时才设置为 `1`；上游评测不是选择下游最佳 LR 的依据。

## 2. FFN 粒度对比（granularity）

四个原生 arm 是整层旁路 `layer_bypass_r256`、专家旁路 `expert_bypass_r2`、专家权重内 `expert_weight_r1` 和 `expert_weight_r2`。模型间的实际参数预算由各模型配置和预算检查器确定。

```bash
for task in math intent; do
  for model in qwen3 gemma4 gptoss; do
    for seed in 42 0 1; do
      MODEL="$model" \
        ARMS="layer_bypass_r256 expert_bypass_r2 expert_weight_r1 expert_weight_r2" \
        SEED="$seed" DATA_SEED="$seed" SEED_SUBDIR="seed$seed" \
        LR_TRIALS="$LRS" PHASE=full \
        OUT_ROOT="$SAVES/granularity/$task" RES_ROOT="$RESULTS/granularity/$task" \
        bash "scripts/exp/granularity/run_${task}_ffn_granularity.sh"
    done
  done
done
```

两个脚本都读取 `RUNNER_DRY_RUN=1`；Math 额外支持 `dry-run` 位置参数，Intent 无需这个位置参数。这里统一使用环境变量预览。

## 3. 辅助损失系数对比（loss）

使用 `expert_weight_r1`，固定系数调度为 `constant`，扫描 `aux0 aux1e-4 aux1e-3 aux1e-2 aux1e-1`。源脚本默认 LR 只有 `lr1e-4 lr1e-3`；下面显式传入完整三个学习率。`REUSE_COEF0=0` 使每个候选在本组输出下独立训练，避免把其他目录的 aux0 结果当作当前 seed 的产物。

```bash
for task in math intent; do
  for model in qwen3 gemma4 gptoss; do
    for seed in 42 0 1; do
      MODEL="$model" TASK="$task" ARM=expert_weight_r1 \
        COEF_TRIALS="aux0 aux1e-4 aux1e-3 aux1e-2 aux1e-1" \
        SEED="$seed" DATA_SEED="$seed" SEED_SUBDIR="seed$seed" \
        LR_TRIALS="$LRS" PHASE=full REUSE_COEF0=0 UNIFIED_LAYOUT=1 \
        OUT_ROOT="$SAVES/loss" RES_ROOT="$RESULTS/loss" \
        bash scripts/exp/loss/run_auxiliary_loss.sh
    done
  done
done
```

## 4. 适配位置与 router 对比（position）

四个 arm 是 `attn_r64 dense_r256 attn64_dense256 attn32_dense128`，每个都对比无 router 和 router rank 8。专家适配使用单个整层旁路：专家单独适配 rank 256，联合适配 rank 128，高预算联合适配 rank 256；每种设置分别关闭或启用 rank-8 router，共八个 arm。

Qwen 使用核心入口 `run_budget_align.sh`，显式传入 LR、输出标签和任务评测项，避免使用 wrapper 中不完整的默认 LR 网格。

```bash
POSITION_ARMS="attn_r64 dense_r256 attn64_dense256 attn32_dense128"
for task in math intent; do
  if [[ "$task" == math ]]; then evals=math500; else evals=intent; fi
  for seed in 42 0 1; do
    for lr in $LRS; do
      for router in false true; do
        tag="$lr"
        if [[ "$router" == true ]]; then tag="router_r8_$lr"; fi
        DOMAIN="$task" ARMS="$POSITION_ARMS" EVALS="$evals" PHASE=full \
          SEED="$seed" DATA_SEED="$seed" MODEL_SUBDIR="qwen3/seed$seed" \
          LEARNING_RATE="${lr#lr}" RUN_TAG="$tag" ATTACH_ROUTER="$router" \
          ROUTER_RANK=8 ROUTER_ALPHA=16 \
          OUT_ROOT="$SAVES/position/$task" RES_ROOT="$RESULTS/position/$task" \
          bash scripts/exp/position/run_budget_align.sh
      done
    done
  done
done

# Gemma / GPT-OSS Math：入口一次覆盖 base 与 router 两种 variant。
for seed in 42 0 1; do
  for lr in $LRS; do
    MODELS="gemma4 gptoss" ARMS="$POSITION_ARMS" VARIANTS="base router" \
      SEED="$seed" DATA_SEED="$seed" SEED_SUBDIR="seed$seed" \
      LEARNING_RATE="${lr#lr}" ROUTER_RANK=8 ROUTER_ALPHA=16 \
      OUT_ROOT="$SAVES/position/math" RES_ROOT="$RESULTS/position/math" \
      bash scripts/exp/position/run_budget_align_position_nm_math.sh
  done
done

# Gemma / GPT-OSS Intent：分别运行无 router / router。
for seed in 42 0 1; do
  for lr in $LRS; do
    for router in false true; do
      tag="$lr"
      if [[ "$router" == true ]]; then tag="router_r8_$lr"; fi
      MODELS="gemma4 gptoss" ARMS="$POSITION_ARMS" RUN_ZERO_SHOT=0 \
        SEED="$seed" DATA_SEED="$seed" SEED_SUBDIR="seed$seed" \
        LEARNING_RATE="${lr#lr}" RUN_TAG="$tag" ATTACH_ROUTER="$router" \
        ROUTER_RANK=8 ROUTER_ALPHA=16 \
        OUT_ROOT="$SAVES/position/intent" RES_ROOT="$RESULTS/position/intent" \
        bash scripts/exp/position/run_budget_align_nm_intent.sh
    done
  done
done
```

Gemma/GPT 两个入口按原流程训练后评测，不使用 `PHASE` 切换。Intent 的 `RUN_ZERO_SHOT=0` 避免在 adapter 网格中重复执行 base 评测；单独需要零样本基线时设置为 `1`。其预览同样使用 `RUNNER_DRY_RUN=1`。

## 输出布局与三 seed 学习率选择

以上命令统一使用 `$SAVES` 和 `$RESULTS` 两个根目录。二者以下的相对路径一致，Math 的 `ep` 为 1，Intent 为 3：

| 实验组 | 根目录下的相对路径 |
| --- | --- |
| methods | `methods/<method>/<task>/<model>/seed<seed>/<lr>/` |
| granularity | `granularity/<task>/<model>/seed<seed>/<arm>_<lr>_ep<ep>/` |
| loss | `loss/<coef>/<task>/<model>/seed<seed>/expert_weight_r1_<lr>_<coef>_ep<ep>/` |
| position（无 router） | `position/<task>/<model>/seed<seed>/<arm>_<lr>_ep<ep>/` |
| position（router） | `position/<task>/<model>/seed<seed>/<arm>_router_r8_<lr>_ep<ep>/` |

常规 Math 结果为 `math500.jsonl`，position 是 `adapter_math500.jsonl`；Intent 结果是 `intent_predictions.jsonl` 与 `intent_metrics.json`。训练目录还需保存方法对应的 adapter 权重、配置、`trainer_state.json` 和 `train_results.json`。

**完成实际训练和评测后**再运行下面的汇总命令，预览不会产生可汇总的结果。选择器检查三个 seed、三个 LR、完成的训练元数据和每份评测恰好 500 个唯一样本；缺失、未训练完成或结果不一致都会报错。对每个模型/任务/setting 按三个 seed 的下游准确率均值选择一个 LR，并记录均值、标准差和产物路径。

```bash
for family in methods granularity loss position; do
  python3 scripts/reproduce/select_best_lr.py \
    --layout native --family "$family" \
    --result-root "$RESULTS" --output-root "$SAVES" \
    --output "$RESULTS/selection_${family}.json"
done
```

可用 `--models`、`--tasks`、`--settings` 限定已经完成的子集；完整三 LR 与三 seed 仍是必要条件。native methods 默认只汇总上面的 10 个论文对比方法，不包含独立 MoE² pilot/redesign setting。上游评测应在确定所需 adapter 后单独执行，保持原有 token budget 与 timeout。

## 独立 MoE² 原生流程

这些入口位于 `scripts/exp/moe2/`，保留源仓库的结构探索逻辑；输出不交给上面的 `--layout native` 四组选择器。MoE² 的 `normalize_topk_prob` 不应根据名称推断：普通 setting 与带 `_ntk` 的归一化变体是不同配置，详情以预览的训练参数和保存配置为准。

### 官方 pilot

`TASK` 虽是单数变量，接受空格分隔的任务列表。模型默认 setting 为：Qwen `n280_r32 n512_r17 n512_r16`；Gemma `n256_r22`；GPT-OSS `n256_r1 n10_r32`。默认 LR 也只有两个，下面补齐三个。pilot 不会自动给路径增加 seed，所以必须让根目录按 seed 隔离。

```bash
for model in qwen3 gemma4 gptoss; do
  for seed in 42 0 1; do
    MODEL="$model" TASK="math intent" SEED="$seed" DATA_SEED="$seed" \
      LR_TRIALS="$LRS" PHASE=full RUNNER_DRY_RUN="$RUNNER_DRY_RUN" \
      OUT_ROOT="$SAVES/moe2_pilot/seed$seed" \
      RES_ROOT="$RESULTS/moe2_pilot/seed$seed" \
      bash scripts/exp/moe2/run_moe2_official_pilot.sh
  done
done
```

此处相对路径是 `<model>/<lr>/<task>/moe2official_<setting>/`。可用 `SETTINGS` 显式指定模型支持的 setting。默认 attention rank 0、balance loss 系数 0；这些实际值不自动构成论文配置一致性的证明。

### redesign：P0 / P1 / P2

`run_moe2_redesign_local4.sh` 使用 `DRY_RUN`，默认 `1`。P0 现在从头训练五个单因素设置，不依赖私有历史 checkpoint：`n128_r32_wp_a11`、`n128_r32_wp_bz`、`n128_r32_wp_ntk`、`n128_r32_wp_ntk_a128`、`n128_r32_wp_up`。默认 P0 LR 为 `lr1e-3`。P1 组合设置为 `n128_r32_wp_up_ntk_a11_bz`，默认比较 `lr1e-3 lr1e-4`；P2 调用 `collect_moe2_redesign.py` 汇总已测结果，不生成假定分数。

默认任务为 Intent，执行 P0/P1 还会运行上游评测，因此与仅做下游的四组网格分开使用。下面示例扩展 P1 为完整三 LR，并为 seed、model 隔离所有根目录；`PHASES` 可以限制阶段。只想先检查命令时保留 `RUNNER_DRY_RUN=1`，执行时本块把它传给 `DRY_RUN`。

```bash
for model in qwen3 gemma4 gptoss; do
  for seed in 42 0 1; do
    MODEL="$model" SEED="$seed" DATA_SEED="$seed" \
      DRY_RUN="$RUNNER_DRY_RUN" PHASES="P0 P1 P2" P1_LRS="$LRS" \
      OUT_ROOT_P0="$SAVES/moe2_redesign/seed$seed/p0" \
      RES_ROOT_P0="$RESULTS/moe2_redesign/seed$seed/p0" \
      OUT_ROOT_P1="$SAVES/moe2_redesign/seed$seed/p1" \
      RES_ROOT_P1="$RESULTS/moe2_redesign/seed$seed/p1" \
      RES_UP="$RESULTS/moe2_redesign/seed$seed/upstream/$model" \
      bash scripts/exp/moe2/run_moe2_redesign_local4.sh
  done
done
```

这是结构探索入口，默认 P0 固定一个 LR；它不等价于对所有 setting 完成主实验三 LR 搜索。`SMOKE=1` 会将短跑输出隔离到 smoke 根目录，不能替代完整结果。P2 要求 Intent 500、MMLU-Pro 12,032、IFEval 541 条；短跑结果会保持 `pending` 并返回非零，因此做 smoke 时应选 `PHASES="P0 P1"`，随后单独查看短跑产物。

汇总器会检查 MMLU-Pro 与 IFEval 的底座、adapter 路径一致。需要相对底座的性能变化时，用 `BASE_UPSTREAM_DIR` 指定本次实际测得的 base 上游结果目录；该目录必须没有 adapter，且底座与实验一致。未提供则漂移列留空，不使用历史基线分数。

### alpha / top-k 归一化探索

`run_moe2_alpha_ntk_local4.sh` 默认 Intent、单个 `LR=lr1e-3`。Qwen 默认比较 `n128_r32_wp_a128` 与 `n128_r32_wp_ntk_a96`；Gemma/GPT 默认 `n128_r32_wp_ntk_a128`。`ARMS` 可覆盖设置，`SKIP_TRAIN` 可用于既有 adapter，`SKIP_UPSTREAM=1` 只做下游。下面明确关闭上游，按 seed 隔离；需要上游时在准备好数据后改为 `0`。

```bash
for model in qwen3 gemma4 gptoss; do
  for seed in 42 0 1; do
    MODEL="$model" TASK=intent LR=lr1e-3 SEED="$seed" DATA_SEED="$seed" \
      DRY_RUN="$RUNNER_DRY_RUN" SKIP_UPSTREAM=1 \
      OUT_ROOT="$SAVES/moe2_alpha/seed$seed" \
      RES_ROOT="$RESULTS/moe2_alpha/seed$seed" \
      RES_UP="$RESULTS/moe2_alpha/seed$seed/upstream/$model" \
      bash scripts/exp/moe2/run_moe2_alpha_ntk_local4.sh
  done
done
```

预览只验证参数与路径展开；完整可运行性仍取决于模型权重、数据、CUDA/依赖和实际训练评测。
