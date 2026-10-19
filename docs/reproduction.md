# Source implementation reproduction protocol

This public workflow exposes the experiment protocol implemented in the source `MoEFactory` checkout. It does not claim exact paper compliance where the source implementation and a paper description differ. In particular, the method runner uses FFN+router with attention disabled by default, and the two MoE² entries make top-k normalization explicit rather than silently changing the source default.

## Original four experiment groups

The original shell organization is preserved under `scripts/exp/position/`,
`granularity/`, `methods/`, and `loss/`, including their parameter-budget
checkers. Use the [original launch guide](../scripts/exp/README.md) as the
main entry point. It gives complete three-model, two-task, three-seed,
three-learning-rate commands and the output-root conventions needed by
`select_best_lr.py --layout native`.

Activate the [documented environment](environment.md) first. Original shell
runners use the current environment and execute by default; set
`RUNNER_DRY_RUN=1` for a preview. Their defaults retain the original source
subsets: methods defaults to the ten paper methods,
and loss defaults to two LRs. The complete grids in the launch guide set these
lists explicitly. The separate MoE² pilot/redesign/alpha scripts default to
preview mode and preserve the source suffix parser and model-specific flags.

The Python tools described below remain available for individual command
inspection, supplementary matrix generation, and result selection. They do
not replace the original shell runners.

## Data and model layout

Prepare the public datasets through `scripts/data/prepare_metamathqa_r1.py` and `scripts/data/prepare_esft.py`. The resulting data root must contain the registered `metamathqa_r1` and `esft_intent` datasets plus `esft_eval/intent.jsonl` for public Intent evaluation. Supply model paths explicitly; no private checkpoint, Conda, container, scheduler, or node path is embedded in these scripts.

The short model identifiers used by these runners are protocol aliases, not Hugging Face repository names. With `--model-root /models`, the matrix generator expects local directories (or symlinks) `/models/qwen3`, `/models/gemma4`, and `/models/gptoss`. Point each alias at the actual downloaded base-model directory, or use `run_protocol.py --model-path` to provide the canonical directory directly.

### Offline evaluation data

`run_protocol.py` enables offline Hugging Face flags for downstream evaluation, and `run_upstream_eval.sh` defaults to offline mode. Before using `--execute` or `--run-upstream`, place the evaluation data below the same `--data-root` / `MOEFACTORY_DATA_ROOT`:

```text
<data-root>/
├── math/MATH-500/test.jsonl
├── MMLU-Pro/data/test-00000-of-00001.parquet
├── MMLU-Pro/data/validation-00000-of-00001.parquet
├── general/IFEval/ifeval_input_data.jsonl
└── nltk_data/tokenizers/punkt/english.pickle
```

- **MATH-500:** `eval_math500.py` first reads `math/MATH-500/test.jsonl`; only when that file is absent does it call `load_dataset("HuggingFaceH4/MATH-500")`, which cannot satisfy the default offline execution. The local file must contain exactly 500 nonempty JSONL rows. Each row needs string fields `problem` and `answer`; the canonical export also contains `solution`, `subject`, integer `level`, and string `unique_id`.
- **MMLU-Pro:** the source loader reads every `<data-root>/MMLU-Pro/data/test-*.parquet` file with the datasets Parquet loader. The canonical test split has 12,032 rows. Required evaluation fields are `question`, list-of-string `options`, `category`, and either a nonempty letter `answer` or integer `answer_index`; the canonical schema also includes integer `question_id`, string `cot_content`, and string `src`. The canonical validation split has 70 rows and the same schema. Current upstream runs are 0-shot and therefore do not read validation, but retain it for `--n_shot` experiments and source compatibility.
- **IFEval:** `eval_vllm_general.py` directly opens `general/IFEval/ifeval_input_data.jsonl`; there is no download fallback. The canonical file has 541 nonempty JSONL rows with integer `key`, string `prompt`, list-of-string `instruction_id_list`, and list-of-object `kwargs`. Each `kwargs` entry corresponds positionally to one instruction ID.
- **NLTK data:** the vendored IFEval checker loads `nltk:tokenizers/punkt/english.pickle`. `run_upstream_eval.sh` exports `NLTK_DATA=<data-root>/nltk_data`. Include both the English `punkt` model shown above and `tokenizers/punkt_tab/english/` for newer NLTK releases that redirect this loader to the tabular resource. Prepare these resources while online, then copy the resulting `nltk_data` directory; the public runner does not download them during evaluation.

Acquire these datasets from their canonical public distributions and preserve the layouts and fields above. No benchmark data is copied into this repository.

### Evaluation dependencies

The base package already declares `datasets`, which supplies the JSON/Parquet loading path, and `requirements/vllm.txt` supplies vLLM for supported models. Gemma4 is intentionally routed through the Hugging Face engine. Install `requirements/evaluation.txt` for the dependencies imported directly by the vendored IFEval checker: NLTK, `langdetect`, and `immutabledict`. The checker replaced its upstream `absl.logging` use with Python's standard `logging`, so `absl-py` is not required by this copy.

## Preview one run

`scripts/reproduce/run_protocol.py` is preview-only unless `--execute` is
supplied. This previews a complete Math method cell:

```bash
python3 scripts/reproduce/run_protocol.py \
  --model qwen3 --model-path /models/qwen3 --data-root /data/moefactory \
  --family methods --setting weight_lora --task math --lr lr1e-4 --seed 42
```

For direct inspection, `examples/paper/weight_lora_<model>_<task>_lr1e-5_seed42.yaml` contains six complete snapshots (three models × Math/Intent) generated from the same protocol builder. They use portable `models/<model>` and `data` paths, explicitly set seed 42 and LR `1e-5`, and set `overwrite_output_dir: false`. Regenerate them with `python3 scripts/reproduce/generate_paper_configs.py`; the generator and source runners remain authoritative.

Add `--execute` only when you deliberately want to start training/evaluation. Execution refuses to write into an existing non-empty adapter/result target unless `--force` is given. `--phase train|eval|full` controls the local phase. `--run-upstream` appends MMLU-Pro and IFEval commands using the source budgets (`4096`, and IFEval `1280` new tokens). DAS-LoRA additionally requires `--das-general /path/to/general.json`; its preview includes the source `compute_das.py` call before training. The DAS saver copies that selection into each adapter as `das_selected_experts.json`, and the saved `das_lora_config.json` refers to this relative file; both are required to reload the adapter. Matrix generation also requires `--das-general` when DAS-LoRA is selected.

## Supplementary Python matrix generation

The main source selection protocol uses seeds `42 0 1` and learning rates `{1e-5, 1e-4, 1e-3}`. Generate repeatable preview commands with:

```bash
python3 scripts/exp/run_family.py \
  --family granularity --model-root /models --data-root /data/moefactory
```

Use `--settings`, `--models`, `--tasks`, `--lrs`, or `--seeds` to select a bounded subset. Generated commands remain previews; append `--execute` to a reviewed individual command to run it. There are no cluster submission wrappers in the public workflow.

The families cover the source position, granularity, methods, and auxiliary-loss workflows. The method family includes MoE-LoRA, Weight-PEFT variants, DAS-LoRA, Partial FT, the source comparison `expert_moe2_lora` setting, and bounded variants extracted from `run_moe2_official_pilot.sh`: Qwen `n280_r32/n512_r17/n512_r16`, Gemma `n256_r22`, and GPT-OSS `n256_r1/n10_r32`. These pilot entries expose narrower exploratory grids found in the source scripts; their presence in the public matrix generator is not a claim that every pilot variant was originally run over the full three-LR, three-seed main protocol. `moe2_pilot_default` preserves the per-model pilot default and `moe2_pilot_default_ntk` adds top-k normalization. `moe2_sall_n128_r32_wp_up_ntk_a11_bz` exposes the Qwen final redesign combination from `run_moe2_redesign_local4.sh` (weight pool, up-only gate/up adaptation, normalized top-k, alpha 11, zero B init). Pilot/final settings explicitly disable attention and router adapters and set balance-loss coefficient `0.0` with the source `linear_decay` schedule. Pool sharing, model-specific ranks, DAS top-k, and task batch overrides are emitted in every preview.

## Select a common learning rate

For outputs from the original shell commands in the launch guide, use:

```bash
python3 scripts/reproduce/select_best_lr.py \
  --layout native --family methods \
  --result-root results/four_experiments_native \
  --output-root saves/four_experiments_native \
  --output results/four_experiments_native/methods_selected.json
```

Repeat for `position`, `granularity`, and `loss`. Native layout means the
explicit root and seed conventions in the launch guide; it is not a search
over arbitrary historical directories. Original MoE² pilot/redesign outputs
use their own roots and collector. The default native method selection covers only the ten paper methods; the
explicit expert_moe2_lora compatibility setting remains available separately.

For outputs generated with the supplementary Python protocol, select the LR
independently for each model/task/setting with the default `protocol` layout:

```bash
python3 scripts/reproduce/select_best_lr.py \
  --family granularity \
  --result-root results/source_protocol --output-root saves/source_protocol \
  --output results/source_protocol/granularity_selected.json
```

The selector requires every requested LR and all three seeds, validates exactly 500 unique downstream examples, checks Intent metrics against predictions, and requires nonempty adapter weights, an object-valued adapter config, and finite completed training metadata. Missing or malformed results are written to `errors` and cause a nonzero exit. It never substitutes zeros or paper values. Selection uses the highest downstream three-seed mean for one common LR; exact ties choose the smaller LR.

This is a structural completion check, not cryptographic provenance verification. It does not hash target datasets, prompts, model weights, or configs, and therefore does not by itself prove that independently produced artifacts used an identical protocol.

The output `rows` record the selected checkpoints: `experiment`, `model`, `domain`, `setting`, `lr`, and three `entries` with `seed` and `adapter`. Adapter paths are relative to the selection JSON directory when possible.

## Native MoE² improvement workflow

`scripts/exp/moe2/run_moe2_redesign_local4.sh` preserves the P0/P1/P2 sequence.
P0 now trains all five single-factor variants before downstream and upstream
evaluation, removing its dependence on author-existing checkpoints. P1 trains
`n128_r32_wp_up_ntk_a11_bz` at the original two learning rates. P2 reads actual
Intent predictions/metrics and MMLU-Pro/IFEval summaries through
`collect_moe2_redesign.py`. Missing or inconsistent results produce a nonzero
exit; historical scores are not used as substitutes. Base-relative drift
columns require `BASE_UPSTREAM_DIR` containing newly measured base summaries.

`run_moe2_alpha_ntk_local4.sh` supplies the separate alpha/normalization grid;
`run_moe2_official_pilot.sh` accepts the original pool/rank/initialization,
weight-pool, up-only, and `_ntk` suffix combinations. These exploratory grids
remain distinct from the full three-LR, three-seed method comparison. To run
them for multiple seeds, give each seed its own adapter, downstream, and
upstream roots as documented in the launch guide.

## Source mapping

- method flags and DAS preprocessing: scripts/exp/methods/run_method_comparison.sh
- granularity flags: scripts/exp/granularity/run_{math,intent}_ffn_granularity.sh
- position arms: scripts/exp/position/run_budget_align*.sh
- auxiliary loss: scripts/exp/loss/run_auxiliary_loss.sh
- seed/LR protocol: scripts/reproduce/protocol.py and examples/paper/source_protocol.json
- downstream LR selection: scripts/reproduce/select_best_lr.py
- upstream evaluation: scripts/reproduce/run_upstream_eval.sh
- MoE²-LoRA: scripts/exp/moe2/
