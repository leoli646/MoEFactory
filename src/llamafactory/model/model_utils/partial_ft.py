"""Partial fine-tuning: train a contiguous subset of decoder layers (full weights).

Unlike the adapter methods, this is *plain* fine-tuning restricted to a few layers.
Following the peft_compare convention, only the **attention q/k/v/o** weights and the
**fused MoE expert** weights (gate_up_proj + down_proj) inside the selected layers are
made trainable; everything else (router, layernorms, embeddings, lm_head, all other
layers) stays frozen. ``partial_ft_modules`` can widen this to {router, norm}.

Layer selection (``partial_ft_target``):
  first / middle / last                     -> a single layer (index 0 / L//2 / L-1)
  first_third / middle_third / last_third   -> a contiguous third of the layers
  custom                                    -> explicit indices in ``partial_ft_layers``

Memory note (4x H200, Qwen3-30B-A3B, 48 layers, 128 experts):
  one layer's experts ≈ 0.6B params -> single-layer presets train ≈0.62B (attn+expert),
  comfortably within a frozen-30B + ZeRO-2 budget. A *third* (16 layers) trains ≈9.9B
  params; feasible on 4x H200 with ZeRO-2 but needs a smaller batch (see the configs).

Save/load mirrors ``weight_peft``: only the trainable (selected) base weights are written
to ``partial_ft_state.safetensors`` (their names ARE the base parameter names), plus a
``partial_ft_config.json``; load just overwrites them via ``load_state_dict(strict=False)``.
Auto-detected by ``eval_scripts/_common.py`` and ``scripts/esft/eval_esft.py``.

MoE experts implementation: when ``expert`` is trained, ``setup_partial_ft`` forces the model's
``_experts_implementation`` to ``eager``. transformers defaults to ``grouped_mm``
(``torch._grouped_mm``), whose *backward* -- only reached when the fused expert weights are
trainable, i.e. only here -- deadlocks on the first DeepSpeed multi-GPU step. ``eager`` is a plain
per-expert ``F.linear`` loop (slower, correct, hang-free). Override with
``PARTIAL_FT_KEEP_GROUPED_MM=1``; set ``PARTIAL_FT_HANG_DUMP_SEC=<sec>`` to auto-dump per-rank
tracebacks if a step ever stalls.
"""

from __future__ import annotations

import faulthandler
import glob
import json
import os
import signal
from typing import TYPE_CHECKING, List, Optional

import safetensors.torch
import torch
import torch.nn as nn
from transformers import TrainerCallback

from ...extras.logging import get_logger
from .moe_lora import (
    _get_router_linear,
    _locate_decoder_layers,
    _resolve_attn_targets,
    common_lora_config,
    inject_common_lora,
)
from .weight_peft import _find_experts_module

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


def _select_layer_ids(num_layers: int, target: str, custom: str) -> List[int]:
    t1, t2 = num_layers // 3, (2 * num_layers) // 3
    mapping = {
        "first": [0],
        "middle": [num_layers // 2],
        "last": [num_layers - 1],
        "first_third": list(range(0, max(1, t1))),
        "middle_third": list(range(t1, max(t1 + 1, t2))),
        "last_third": list(range(t2, num_layers)),
    }
    if target == "custom":
        ids = sorted({int(x) for x in custom.replace(" ", "").split(",") if x != ""})
        if not ids:
            raise ValueError("partial_ft_target=custom requires partial_ft_layers (e.g. '0,1,2').")
        bad = [i for i in ids if i < 0 or i >= num_layers]
        if bad:
            raise ValueError(f"partial_ft_layers out of range [0,{num_layers}): {bad}")
        return ids
    if target not in mapping:
        raise ValueError(f"Unknown partial_ft_target: {target!r}")
    return mapping[target]


def _collect_trainable_params(model: "PreTrainedModel", finetuning_args: "FinetuningArguments"):
    """Return (selected_layer_ids, set(id(param)) to train, n_layers)."""
    layers_container = _locate_decoder_layers(model)
    if layers_container is None:
        raise ValueError("partial_ft: cannot locate decoder layers.")
    layers = layers_container.layers
    num_layers = len(layers)
    sel = _select_layer_ids(num_layers, finetuning_args.partial_ft_target, finetuning_args.partial_ft_layers)
    modules = [m.strip() for m in finetuning_args.partial_ft_modules.split(",") if m.strip()]

    train_ids = set()
    sel_set = set(sel)
    for li, layer in enumerate(layers):
        if li not in sel_set:
            continue
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        moe_block = layer if hasattr(layer, "experts") and hasattr(layer, "router") else mlp

        if "attn" in modules and attn is not None:
            for pname in _resolve_attn_targets(attn, "auto"):
                proj = getattr(attn, pname)
                train_ids.add(id(proj.weight))
                if getattr(proj, "bias", None) is not None:
                    train_ids.add(id(proj.bias))

        if "expert" in modules and moe_block is not None:
            experts = _find_experts_module(moe_block)
            if experts is not None:
                train_ids.add(id(experts.gate_up_proj))
                train_ids.add(id(experts.down_proj))
                for bname in ("gate_up_proj_bias", "down_proj_bias"):
                    b = getattr(experts, bname, None)
                    if isinstance(b, nn.Parameter):
                        train_ids.add(id(b))

        if "router" in modules and moe_block is not None:
            router_linear = _get_router_linear(moe_block)
            if router_linear is not None and isinstance(getattr(router_linear, "weight", None), nn.Parameter):
                train_ids.add(id(router_linear.weight))
                if isinstance(getattr(router_linear, "bias", None), nn.Parameter):
                    train_ids.add(id(router_linear.bias))

        if "norm" in modules:
            for nname, p in layer.named_parameters():
                if "norm" in nname.lower():
                    train_ids.add(id(p))

    return sel, train_ids, num_layers


def force_eager_experts(model: "PreTrainedModel", context: str) -> None:
    """Switch the MoE experts to the ``eager`` forward whenever the fused experts are trainable.

    Why: transformers' default ``_experts_implementation`` is ``grouped_mm`` (``torch._grouped_mm``
    on SM80+ GPUs). Its *forward* is fine (it's what every adapter method uses at eval), but its
    *backward* is only ever exercised when the fused ``gate_up_proj`` / ``down_proj`` weights
    themselves are trainable -- in this repo that is ``partial_ft`` with ``expert`` in
    ``partial_ft_modules`` and *full* fine-tuning. Under DeepSpeed multi-GPU that backward
    deadlocks on the very first optimization step (the stuck CUDA API also freezes the NCCL
    watchdog thread, which then aborts all ranks after 480s with "watchdog got stuck").
    The ``eager`` implementation is a plain per-expert ``F.linear`` loop whose backward is
    ordinary autograd: slower, but correct and hang-free. Set ``PARTIAL_FT_KEEP_GROUPED_MM=1``
    to opt out and keep grouped_mm.
    """
    if os.environ.get("PARTIAL_FT_KEEP_GROUPED_MM", "0") == "1":
        return

    cfg = getattr(model, "config", None)
    cur = getattr(cfg, "_experts_implementation", None) if cfg is not None else None
    if cur in (None, "eager"):
        return  # nothing to switch (model has no dispatchable experts, or already eager)
    if not hasattr(model, "set_experts_implementation"):
        logger.warning_rank0(
            f"{context} trains the fused experts but the model exposes no set_experts_implementation(); "
            f"leaving experts_implementation={cur!r}. If training hangs on step 0, this is why."
        )
        return
    try:
        model.set_experts_implementation("eager")
        logger.info_rank0(
            f"{context}: switched experts_implementation {cur!r} -> 'eager' for the trainable MoE "
            "experts (grouped_mm backward deadlocks under DeepSpeed). Set PARTIAL_FT_KEEP_GROUPED_MM=1 to override."
        )
    except Exception as e:  # noqa: BLE001 - never let a diagnostics/perf switch break training setup
        logger.warning_rank0(f"{context}: could not switch experts_implementation to 'eager' ({e}); training may hang.")


def _force_eager_experts_if_trainable(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """partial_ft entry: only needed when ``expert`` is among the trainable module groups."""
    modules = {m.strip() for m in finetuning_args.partial_ft_modules.split(",") if m.strip()}
    if "expert" not in modules:
        return  # not training the fused experts -> grouped_mm backward is never triggered
    force_eager_experts(model, context="partial_ft")


def _install_hang_debugger() -> None:
    """Make multi-GPU stalls diagnosable: dump every thread's Python traceback when a step stalls.

    - Always registers SIGUSR1 so ``kill -USR1 <pid>`` dumps a live traceback for that rank.
    - If ``PARTIAL_FT_HANG_DUMP_SEC`` (seconds) is set, also auto-dumps all threads when any single
      stretch exceeds that budget (repeat=True, exit=False) -- harmless during normal training since
      individual steps are short, but it captures the exact stuck frame on every rank if a step hangs.
    """
    try:
        faulthandler.enable()
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
        timeout = float(os.environ.get("PARTIAL_FT_HANG_DUMP_SEC", "0") or 0)
        if timeout > 0:
            faulthandler.dump_traceback_later(timeout, repeat=True, exit=False)
            logger.info_rank0(f"partial_ft: hang debugger armed (auto-dump tracebacks after {timeout:.0f}s stalls).")
    except Exception:  # noqa: BLE001 - diagnostics must never break training
        pass


def setup_partial_ft(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """Freeze everything, then unfreeze the selected layers' attn+expert (+optional) weights."""
    _force_eager_experts_if_trainable(model, finetuning_args)
    _install_hang_debugger()
    sel, train_ids, num_layers = _collect_trainable_params(model, finetuning_args)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        if id(p) in train_ids:
            p.requires_grad = True

    def _real_numel(p):
        return getattr(p, "ds_numel", None) or p.numel()

    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    logger.info_rank0(
        f"partial_ft setup complete | target={finetuning_args.partial_ft_target} "
        f"layers={sel} (of {num_layers}) modules={finetuning_args.partial_ft_modules} | "
        f"trainable: {trainable / 1e6:.2f}M ({100 * trainable / max(total, 1):.3f}%) | "
        f"total: {total / 1e6:.2f}M"
    )
    if trainable == 0:
        raise ValueError("partial_ft selected no trainable parameters; check partial_ft_modules / layers.")


def _selected_param_names(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def save_partial_ft_state(model: nn.Module, save_dir: str,
                          finetuning_args: "FinetuningArguments") -> None:
    """Save only the (selected) trainable base weights + config. Names == base param names."""
    os.makedirs(save_dir, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import deepspeed
        gathered_ctx = deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0)
    except Exception:
        gathered_ctx = None

    state = {}
    if gathered_ctx is not None:
        with gathered_ctx:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state[name] = param.detach().cpu().clone()
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu()

    if not state:
        logger.warning_rank0("No trainable params, skip saving partial_ft state.")
        return
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    safetensors.torch.save_file(state, os.path.join(save_dir, "partial_ft_state.safetensors"))
    config = {
        "partial_ft_target": finetuning_args.partial_ft_target,
        "partial_ft_layers": finetuning_args.partial_ft_layers,
        "partial_ft_modules": finetuning_args.partial_ft_modules,
        **common_lora_config(finetuning_args),
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
    }
    with open(os.path.join(save_dir, "partial_ft_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info_rank0(f"Saved partial_ft state ({len(state)} tensors) to {save_dir}")


def load_partial_ft_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    """Overwrite the saved (selected) base weights back onto a freshly loaded base model."""
    from ...hparams import FinetuningArguments

    state_path = os.path.join(load_dir, "partial_ft_state.safetensors")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing partial_ft_state.safetensors in {load_dir}")

    config_path = os.path.join(load_dir, "partial_ft_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        meta = {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}
        finetuning_args = FinetuningArguments(finetuning_type="partial_ft", **config)
        inject_common_lora(model, finetuning_args)
    else:
        meta = {}

    state = safetensors.torch.load_file(state_path)
    # Cast each saved tensor to the matching base param dtype before loading (fp32 ckpt -> bf16 base).
    model_state = model.state_dict()
    fixed = {}
    for k, v in state.items():
        if k in model_state:
            fixed[k] = v.to(model_state[k].dtype)
        else:
            fixed[k] = v
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    unexpected_ours = [n for n in unexpected if n in state]
    assert not unexpected_ours, f"Unexpected partial_ft weights not in model: {unexpected_ours[:5]}"

    if meta:
        base_meta = meta.get("_meta_base_model")
        actual_base = getattr(model.config, "name_or_path", "unknown")
        if base_meta and base_meta != actual_base:
            logger.warning_rank0(
                f"Base model mismatch: loaded {actual_base} but checkpoint trained on {base_meta}."
            )
    logger.info_rank0(f"Loaded partial_ft state from {load_dir} ({len(state)} tensors)")
    return model


class PartialFTSaveCallback(TrainerCallback):
    """Save partial_ft state at each checkpoint; drop the redundant full-model shards."""

    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_partial_ft_state(model, ckpt_dir, self.finetuning_args)
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
