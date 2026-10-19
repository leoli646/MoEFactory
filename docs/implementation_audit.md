# Source implementation and release scope

This public release follows the local MoEFactory implementation and its
experiment launchers. Baseline YAML defaults, launcher overrides, and recorded
experiment results are distinct sources of evidence. A configuration name
alone is not proof of agreement with every paper equation.

| Question | Actual source behavior | Release interpretation |
| --- | --- | --- |
| DAS expert scoring | `eval_scripts/compute_das.py` averages dense base-router softmax probabilities and subtracts general scores from domain scores, then selects experts per layer. | This is the source's simplified DAS scoring, not the cumulative CDAS procedure described in the manuscript. |
| DAS adapter location | `DASLoRAPool` applies a LoRA transform to hidden inputs; patched expert/MoE forward adds its routed output to the original output. | This is a parallel selected-expert branch, not three LoRA updates inside gate/up/down projections. Exporting files does not remove this structural difference. |
| DAS router training | `model/adapter.py::_setup_das_lora` first injects DAS and then calls `inject_common_lora`. | The common router anchor enabled by the method runner is trainable. Reading the freezing loop in `inject_das_lora` alone misses the later injection. |
| Main method settings | The source method runner defaults to attention off and enables router adaptation. Its `weight_lora` arm uses `finetuning_type=moe_lora`, `expert_weight`, expert rank 1/alpha 2, router rank 8/alpha 16. DoRA uses Weight-PEFT rank 1/alpha 2 with router rank 8/alpha 16. | The old `weight_lora.yaml` rank 2 / attention-on / learning-rate defaults are templates, not the effective main-run configuration. |
| MoE2 top-k normalization | `LoRAPool.forward` renormalizes selected probabilities only when `normalize_topk_prob` is true. Source pilot/redesign launchers use setting suffixes including `_ntk` to enable it. | Preserve both source variants and record the actual setting. The baseline YAML's global 512-entry pool and attention settings do not describe every later pilot. |
| Seed/LR selection | The source paper planner requires all three seeds at all three learning rates before selection, uses downstream mean, and applies an explicit LR-order tie break. Additional MoE2 pilots have narrower, explicitly named grids. | Do not choose from an incomplete grid or describe a pilot as the complete main protocol. |

## Runtime boundaries

SFT setup, checkpoint callbacks, final saving, and HF evaluation dispatch exist
for MoE-LoRA, Weight-PEFT, DAS-LoRA, and Partial FT. Partial FT saves only selected
base parameters and its configuration; it is not a low-rank adapter. The shared
HF evaluator detects each format by its checkpoint marker. Dedicated vLLM
overlays cover supported MoE-LoRA and Weight-PEFT paths; their presence does not
imply that every adapter or model family supports every vLLM backend. Use the
HF/runtime-merge path for DAS/Partial FT where selected by the source runner.

The release retains core numerical implementation behavior. It does not silently
replace simplified DAS with CDAS or change top-k normalization for historical
checkpoints. Paper-level algorithm alignment would be a separate change and
would require new numerical and experiment validation.

## Provenance of restored files

- ESFT evaluator and tokenizer compatibility: source `eval_scripts/esft/`.
- Training dataset registration/preparation: dataset locations referenced by
  the source runners; the public registry contains portable file names only.
- Local experiment runners: source `scripts/exp/` and source experiment
  launch logic; machine-specific setup and cluster submission are excluded.
- `examples/deepspeed/ds_z3_config.json`: the existing matching configuration
  from the source project's companion tree (`lora_moe-main`), also byte-identical
  in the companion LlamaFactory tree. The original MoEFactory YAML referenced
  this file but did not bundle it.

Model weights, private/local datasets, result snapshots, generated caches,
scheduler state, and cluster submission wrappers are not public source inputs.
Local CPU tests and static validation do not establish large-model multi-GPU
training or vLLM end-to-end correctness; see `runtime_validation.md` for the
actual validation scope.

## Scope after cleanup

Only the Section 4 study entry points and the MoE²-LoRA experiment workflows are
published. Shared LLaMA Factory infrastructure remains available to support
training, checkpoint loading, evaluation and loss logging/plotting. Research
analysis, non-Intent ESFT tasks, MoLA, training probes, router-LR/gradient experiments and unpublished method
variants have been removed. The model/data/checkpoint files are external inputs.

No new GPU training or full-model evaluation is implied by this cleanup.
