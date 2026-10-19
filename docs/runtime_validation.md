# Runtime validation

## Dependency-light checks

Use Python 3.11+ with PyYAML, and Bash 4.4+ available as bash on PATH:

    python -B -m unittest discover -s tests -p 'test_public*.py'
    python -B -m unittest discover -s tests -p test_reproduction_protocol.py
    python -B -m unittest discover -s tests -p 'test_native_*.py'

These tests validate the Math/Intent registry, conversion, Intent scoring,
experiment command generation, seed/LR selection, shell failure propagation,
MoE²-LoRA workflows and measured result collection. Native runner tests use tiny
fixtures and fake training/evaluation executables. They do not run GPU training.
The public release tests also enforce removal of analysis/figure directories and
continued connection of the SFT loss plotter.

## Adapter CPU checks

In an environment containing the pinned training dependencies:

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
      python tests/run_cpu_tests.py

This imports the actual adapters from this checkout and checks routed LoRA,
DAS-LoRA, partial fine-tuning, checkpoint boundaries and exact OFT rotations.
Only the methods included in this release are tested.

## Validation limits

Static checks and shell fixtures do not establish end-to-end large-model or
multi-GPU correctness. Full model injection, tokenizer/data pipelines,
DeepSpeed/ZeRO, distributed kernels, training/resume and full benchmark runs
require the prepared Linux/CUDA environment, model weights and datasets.
See [environment setup](environment.md) and [data preparation](data.md).
