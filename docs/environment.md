# Training environment

Use the source experiment dependency profile for the Qwen3, Gemma4 and GPT-OSS
workflows. The selected source environment was inspected with Python 3.11.15,
PyTorch 2.8.0+cu128 and Transformers 5.6.0. The public package metadata and runtime
check now allow Transformers through 5.6.0, so this profile does not require
`DISABLE_VERSION_CHECK`. Earlier package metadata capped Transformers at 5.2.0;
the source experiment wrappers explicitly bypassed that stale check.

## Install

Use a fresh Linux environment with Python 3.11. The commands below use a virtual
environment; an already activated equivalent Conda environment also works.
Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# The audited PyTorch wheels use CUDA 12.8. Select a compatible NVIDIA driver.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Apply the pinned profile together with the project's remaining dependencies.
# DS_BUILD_OPS=0 avoids building all optional DeepSpeed extensions at install time.
DS_BUILD_OPS=0 python -m pip install -r requirements/source_training.txt -e .
python -m pip check
unset DISABLE_VERSION_CHECK
```

`requirements/source_training.txt` pins the selected core training packages. It
is not a complete transitive lockfile. These installation commands have not been
executed in a fresh environment as part of this audit; the evidence below comes
from the existing source environment and public code. A clean installation and
GPU smoke run remain necessary to validate the full environment.

The scripts use the caller's activated environment and `PATH`; `PYTHON_BIN` can
select an interpreter for their Python preflights and direct Python evaluations.
Ensure `python`, `torchrun` and `llamafactory-cli` all resolve to that same
environment. Run `python -m pip check` after adding optional packages.

DeepSpeed may compile extensions at runtime. The MoE² training wrappers check for
a CUDA toolkit whose major version matches `torch.version.cuda`; install a
CUDA 12.x toolkit with `nvcc` and set `CUDA_HOME` if detection cannot find it.
The CUDA runtime included with PyTorch wheels does not supply the complete
compiler toolkit. Model/data preparation is separate; use the public data
documentation and the runners' `MOEFACTORY_DATA_ROOT` and model path overrides.

## Inference engine

The native MoE² wrappers default to `ENGINE=hf`. This follows the source behavior
for adapters whose learned routes depend on each token. vLLM is optional and
does not replace the HF execution path for these adapters.

The inspected source environment also contains `vllm==0.11.0`. Its package
metadata requires torch 2.8.0, torchaudio 2.8.0 and torchvision 0.23.0, matching
the pinned profile. If a supported evaluation path needs it, install with the
same constraints and check the environment again:

```bash
python -m pip install -c requirements/source_training.txt vllm==0.11.0
python -m pip check
```

Do not infer vLLM model/adapter support from package installation. The evaluation
code checks adapter structure and falls back to HF where needed. `kernels` was
not installed in the inspected source environment, and the model classes below
still imported successfully. Optional optimized kernels and quantized GPT-OSS
execution have separate hardware and runtime requirements.

## Audited versions and CPU preflight

| Package | Inspected version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| transformers | 5.6.0 |
| peft | 0.18.1 |
| accelerate | 1.11.0 |
| datasets | 4.0.0 |
| trl | 0.24.0 |
| tokenizers | 0.22.2 |
| torchdata | 0.11.0 |
| deepspeed | 0.18.4 |
| vllm, optional | 0.11.0 |
| kernels | not installed |

Run this bounded import check without downloading or instantiating a model:

```bash
env -u DISABLE_VERSION_CHECK CUDA_VISIBLE_DEVICES='' \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import importlib.metadata as metadata
import sys

print("Python", sys.version.split()[0])
for name in ("torch", "transformers", "peft", "accelerate", "datasets",
             "deepspeed", "trl", "tokenizers", "torchdata"):
    print(name, metadata.version(name))

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration, Gemma4TextExperts, Gemma4TextModel,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM, Qwen3MoeSparseMoeBlock,
)
from llamafactory.hparams import parser

print("Model classes and public training parser imported; dependency checks enabled.")
PY

env -u DISABLE_VERSION_CHECK CUDA_VISIBLE_DEVICES='' \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python tests/run_cpu_tests.py
```

Gemma4's ZeRO-3 support directly imports `Gemma4TextExperts` from Transformers.
The audited 5.6.0 environment provides it; the minimum Transformers release that
provides every required Gemma4 API was not established by this local audit.
For this reason, reproduce that workflow with the pinned profile rather than
assuming every version in the broad package range supports every model.

The CPU checks exercise adapter tensor operations, checkpoint boundaries, public
dataset registration and real argument parsing. They do not certify full model
loading, tokenizer/data execution, DeepSpeed optimizer state, distributed GPU
training, quantized experts, or complete downstream/upstream evaluations. No
fresh installation, model download, GPU training or GPU evaluation was performed
for this environment audit. See [runtime_validation.md](runtime_validation.md)
for the bounded adapter tests and their limits.
