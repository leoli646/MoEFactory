# MoEFactory

Research code for *An Empirical Study of Parameter-Efficient Fine-Tuning for
Mixture-of-Experts LLMs*, built on LLaMA Factory.

## Release scope

This release contains the four experimental studies in Section 4 and the
MoE²-LoRA experiments in Section 5 and Appendix D:

| Study | Entry points |
| --- | --- |
| Adaptation position | scripts/exp/position/ |
| Adaptation granularity | scripts/exp/granularity/ |
| PEFT method comparison | scripts/exp/methods/ |
| Auxiliary loss coefficients | scripts/exp/loss/ |
| MoE²-LoRA and expert-weight redesign | scripts/exp/moe2/ |

The three backbones are Qwen3-30B-A3B-Base, Gemma-4-26B-A4B, and GPT-OSS-20B.
Math uses MetaMathQA-R1 training and MATH-500 evaluation. Intent uses ESFT Intent
training and exact-match evaluation. MMLU-Pro and IFEval measure retained general
capability. ESFT exports and evaluation support **Intent only**.

The controlled method comparison contains ten methods: weight-level LoRA, DoRA,
OFT, FourierFT, IA3, expert-level LoRA, MoE-LoRA, module-level LoRA, DAS-LoRA,
and partial fine-tuning. MoE²-LoRA has separate experiment entry points.
Representation/geometry analysis and paper-figure scripts are outside this
release. Training loss logging and loss-curve plotting remain available.

## Installation

Use Python 3.11 and Bash 4.4 or newer on Linux for the GPU experiment runners.
Follow [environment setup](docs/environment.md) for the pinned training profile:

    python -m pip install -r requirements/source_training.txt
    python -m pip install -e .
    python -m pip install -r requirements/evaluation.txt

vLLM is optional for supported evaluation paths. MoE²-LoRA and other
routing-dependent adapters use the HF/runtime-merge path.

## Models and data

Download model weights separately and set model paths explicitly, or use:

    models/Qwen3-30B-A3B-Base
    models/gemma-4-26B-A4B
    models/gpt-oss-20b

Prepare the two registered training datasets using [data preparation](docs/data.md).
Set MOEFACTORY_DATA_ROOT when using an external data directory; that directory
must contain dataset_info.json. Standalone training accepts dataset_dir explicitly.
Datasets, model weights, adapters, predictions, logs and result artifacts are not
included and are ignored by .gitignore.

## Reproduce experiments

The [experiment launch guide](scripts/exp/README.md) documents the four study
groups, three seeds, learning-rate selection, and the MoE²-LoRA workflows.
[Reproduction details](docs/reproduction.md) cover the data layout, supplementary
Python command previews, and selection of the best common learning rate.

Preview a paper method comparison before launching training:

    MODEL=qwen3 TASKS="math intent" METHODS="weight_lora weight_dora" \
      LR_TRIALS="lr1e-5 lr1e-4 lr1e-3" RUNNER_DRY_RUN=1 \
      bash scripts/exp/methods/run_method_comparison.sh

The method runner defaults to the ten paper methods. METHODS=all selects that
same controlled grid. The separate MoE²-LoRA pilot previews with:

    RUNNER_DRY_RUN=1 bash scripts/exp/moe2/run_moe2_official_pilot.sh

The four study runners execute by default unless RUNNER_DRY_RUN=1 is set.
MoE² wrapper scripts default to previews; inspect their documented execution flags.

Training loss curves use the existing plot_loss: true YAML option. The plotting
implementation is src/llamafactory/extras/ploting.py and remains connected to SFT.

## Repository layout

    src/llamafactory/       Shared training runtime and adapter implementations
    src/train.py            Training entry point
    examples/               Training templates and DeepSpeed settings
    scripts/data/           Math and Intent preparation
    scripts/exp/            Four study groups and MoE²-LoRA experiments
    scripts/reproduce/      Command previews, LR selection and upstream evaluation
    eval_scripts/           MATH-500, Intent, MMLU-Pro, IFEval and adapter loaders
    data/dataset_info.json   Portable Math and Intent registrations
    tests/                  Data, protocol, shell and adapter regression checks
    docs/                   Environment, data and reproduction instructions

## Validation and implementation notes

See [validation commands](docs/runtime_validation.md),
[implementation boundaries](docs/implementation_audit.md), and
[release scope checks](docs/public_release_check.md). Template YAML defaults and
launcher overrides differ; inspect the effective command before training.
DAS-LoRA retains the source scoring/adapter behavior described in the audit.

## License

Apache-2.0. The LLaMA Factory source and third-party components retain their
copyright notices. Model and dataset licenses are separate.
