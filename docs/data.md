# Data preparation and ESFT evaluation

This repository contains dataset registrations and conversion utilities, but
does not redistribute source datasets, generated predictions, caches, or model
weights. Run all commands from the repository root.

## Registered training datasets

`data/dataset_info.json` registers the datasets used by the included training
configurations:

| Name | Generated file | Format |
| --- | --- | --- |
| `metamathqa_r1` | `data/metamathqa_r1_clean.json` | JSON array with ShareGPT `messages` |
| `esft_intent` | `data/esft/intent_train.jsonl` | ShareGPT JSONL |

Each ShareGPT sample has a `messages` list. Messages use `role` (`user` or
`assistant`) and `content` string fields. To keep datasets outside the repository,
pass their parent directory as `--dataset_dir`; it must contain the same
`dataset_info.json` and relative files shown above. The ESFT preparation command
copies this registry into an external `--output-root` automatically.

## MetaMathQA-R1

Obtain `oumi-ai/MetaMathQA-R1` from its public provider under the provider's license and
export the training split locally as JSON or JSONL. Each source record must
contain `prompt` and `response`; optional `metadata.complete=false` records are
skipped. The experiment protocol removes complete think-tag blocks and trains
on the final response:

    python scripts/data/prepare_metamathqa_r1.py \
      --input /path/to/local/metamathqa_r1.jsonl \
      --output data/metamathqa_r1_clean.json

Use `--keep-think` only for a deliberate reasoning-trace variant. The normal
`metamathqa_r1` registration refers to the cleaned output.

For a Math-only external data root, also copy the public registry there:

```bash
mkdir -p /path/to/data
cp data/dataset_info.json /path/to/data/dataset_info.json
```

The MetaMath converter writes the selected output file; it does not install the
registry. The ESFT preparation command below copies the registry automatically.

## General samples for DAS

The original method runner computes DAS before the `das_lora` arm. In addition
to the task's training data, place the source general calibration samples at
`<data-root>/general/commonsense_test/merged.json`, or set
`DAS_GENERAL=/path/to/general.json`. This input is not distributed here.

The loader accepts a JSON array, JSONL records (also when the suffix is `.json`),
or an object containing a `test` or `data` list. Supported text records include
`input`/`output`, `instruction`/optional `output`, `text`, and ShareGPT
`messages`. Preserve the intended general calibration corpus when reproducing
the source selection; a different corpus changes the selected experts.
The native runner uses at most 200 domain/general records and per-layer
selection top-k 8 for Qwen3/Gemma4, 4 for GPT-OSS. `DAS_MAXSAMPLES` and
`DAS_JSON` expose the source overrides. The cache and saved
`das_selected_experts.json` are generated artifacts required by later loading.

## ESFT Intent

Use a local checkout/export of the public `deepseek-ai/ESFT` release with this source layout:

    ESFT/
      datasets/train/intent.jsonl
      datasets/eval/intent.jsonl

Training rows contain ShareGPT `messages`. Evaluation rows contain `idx`, a
fully formatted `prompt`, and non-empty `raw_answers`. Preserve the released
train/eval split and prompt text exactly; the referenced intent evaluation set
has 500 examples.

    python scripts/data/prepare_esft.py \
      --source-root /path/to/ESFT \
      --output-root data

This creates training files under `data/esft/` and evaluation files under
`data/esft_eval/`. Only Intent is exported; --tasks intent is accepted for an explicit selection.

## Intent evaluation

The Intent command below uses the HF runtime installed with the core project.
For the optional vLLM generation backend, install requirements/esft.txt.
Intent scoring uses the included exact-match metric and needs no judge service.

Generate and score intent predictions in one command:

    python eval_scripts/esft/eval_esft.py \
      --task intent \
      --mode all \
      --model-name-or-path /path/to/base-model \
      --adapter-name-or-path /path/to/adapter \
      --template esft_raw \
      --prompt-protocol auto \
      --runtime-merge \
      --runtime-merge-device-map auto \
      --hf-generate-batch-size 8 \
      --tensor-parallel-size 1 \
      --max-new-tokens 512 \
      --output-dir results/intent

The released protocol uses deterministic generation, `cutoff_len=3584`, and
`max_new_tokens=512`. The default `esft_raw` prompt protocol consumes the
released prompt verbatim. Intent scoring extracts the first balanced JSON
object and requires exact intent and slot equality; list slot order is ignored
and spaces in predicted slot values are normalized as in the source evaluator.
Custom DAS, Partial-FT, and custom MoE-LoRA checkpoints require the shown
`--runtime-merge` path; the default vLLM adapter path cannot load those states.

By default, evaluation reads `data/esft_eval`. Set `MOEFACTORY_DATA_ROOT` or
pass `--data-root /path/to/data` to read `/path/to/data/esft_eval`. The legacy
`--esft-dir` option remains an explicit direct-directory override.

## Tokenizer compatibility

`eval_scripts/esft/tokenizer_compat.py` supplies
`all_special_tokens_extended` when a local Qwen tokenizer is loaded through the
slow `Qwen2Tokenizer` class but vLLM expects the newer property. The patch is
conditional and is a no-op for compatible versions and non-Qwen tokenizers.
