"""DAS-LoRA: Selective LoRA on DAS-identified domain experts.

Adapted from "Exploring Expert Concentration" (Anonymous, ICLR 2026 submission).
Simplifications vs. original paper:
  - Skip Stage 1 (attention + router fine-tuning); compute DAS directly on
    pre-trained routing.
  - Use a parallel LoRA branch tied to each selected MoE expert (activated
    via expert-id matching against MoE router top-k indices), rather than
    fine-tuning the expert MLP weights directly. This avoids the need to
    un-fuse OLMoE's grouped expert tensors while preserving the
    expert-selection paradigm.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING, List, Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

from ...extras.logging import get_logger
from .moe_lora import (
    LoRAExpert,
    _extract_experts_routing,
    _find_moe_blocks,
    _get_config_value,
    _get_expert_adapter_host,
    _get_moe_top_k,
    _uses_experts_host,
    common_lora_config,
    inject_common_lora,
    is_common_lora_parameter,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class DASLoRAPool(nn.Module):
    """Per-layer pool of K LoRA modules, each tied to one DAS-selected MoE expert.

    Forward pass: for each token, if any of its top-k MoE experts is in the
    selected set, apply the corresponding LoRA branch (weighted by routing score).
    """

    def __init__(self, selected_expert_ids: List[int], d_model: int, rank: int, alpha: int):
        super().__init__()
        self.selected_ids = list(selected_expert_ids)
        self.n_selected = len(self.selected_ids)
        # ID → local index buffer (for vectorized lookup; -1 means not selected)
        max_id = max(self.selected_ids) if self.selected_ids else 0
        id_map = torch.full((max_id + 1,), -1, dtype=torch.long)
        for local_idx, expert_id in enumerate(self.selected_ids):
            id_map[expert_id] = local_idx
        self.register_buffer("id_to_local", id_map, persistent=False)

        self.experts = nn.ModuleList([
            LoRAExpert(d_model, rank, alpha) for _ in self.selected_ids
        ])

    def forward(self, h_flat: torch.Tensor,
                top_k_indices: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        h_flat:        [n_tokens, d_model]
        top_k_indices: [n_tokens, top_k] (MoE expert IDs from router)
        top_k_weights: [n_tokens, top_k] (softmax routing weights)
        Returns:       [n_tokens, d_model]
        """
        out = torch.zeros_like(h_flat)
        weight_dtype = top_k_weights.dtype

        for local_idx, expert_id in enumerate(self.selected_ids):
            mask = (top_k_indices == expert_id)  # [n_tokens, top_k]
            if not mask.any():
                continue
            # Aggregate routing weight per token (sum over top_k positions
            # where this expert was selected — usually 0 or 1 position)
            weight_per_token = (top_k_weights * mask.to(weight_dtype)).sum(dim=-1)
            active = weight_per_token > 0
            if not active.any():
                continue
            h_active = h_flat[active]
            lora_out = self.experts[local_idx](h_active)
            w = weight_per_token[active].unsqueeze(-1).to(lora_out.dtype)
            out = out.index_add(0, active.nonzero(as_tuple=True)[0], w * lora_out)

        # --- 保持各 DP rank 的梯度桶一致 (多卡 ZeRO 必需) ---
        # MoE 稀疏激活: 不同 rank 的本地 batch 路由到不同的被选专家子集, 于是"本步产生
        # 梯度的参数集合"每张卡不同。DeepSpeed ZeRO-1/2 只把有梯度的参数打包 all_reduce,
        # 桶大小就会跨 rank 不一致 -> "Detected mismatch between collectives on ranks"。
        # 这里让每个被选专家每步都进入计算图: 对 1 个零 token 跑一遍所有专家, 再乘 0 加回。
        # 前向加 0 (B 作用在零输入上输出为 0), 反向给未被路由命中的专家补零梯度 ——
        # 数学上是 no-op, 只为保证所有 rank 规约的可训练参数集合完全相同。
        if self.training:
            probe = h_flat.new_zeros((1, h_flat.shape[-1]))
            touch = sum(e(probe).sum() for e in self.experts)
            out = out + 0.0 * touch
        return out


def patched_das_moe_forward(self, hidden_states: torch.Tensor):
    """Forward patch: original MoE output + DAS-LoRA branch on selected experts."""
    original_result = self._original_forward(hidden_states)
    if isinstance(original_result, tuple):
        moe_output, *rest = original_result
    else:
        moe_output = original_result
        rest = []

    if hidden_states.dim() == 3:
        b, t, d = hidden_states.shape
        h_flat = hidden_states.reshape(-1, d)
    else:
        h_flat = hidden_states
        b = t = None

    # Compute MoE router top-k from raw logits (mirrors moe_lora's approach)
    gate_bias = getattr(self.gate, "bias", None)
    router_logits = F.linear(h_flat, self.gate.weight, gate_bias)
    routing_probs = F.softmax(router_logits.float(), dim=-1).to(h_flat.dtype)
    top_k_weights, top_k_indices = routing_probs.topk(self._das_top_k_moe, dim=-1)

    lora_output = self.das_lora_pool(h_flat, top_k_indices, top_k_weights)

    if b is not None:
        lora_output = lora_output.reshape(b, t, d)

    final_output = moe_output + lora_output
    if rest:
        return (final_output, *rest)
    return final_output


def patched_das_experts_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
    """DAS branch for experts modules that receive the model's real top-k dispatch."""
    base_output = self._original_forward(hidden_states, *args, **kwargs)
    top_k_indices, top_k_weights = _extract_experts_routing(args, kwargs)
    original_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    lora_output = self.das_lora_pool(h_flat, top_k_indices, top_k_weights).reshape(original_shape)
    return base_output + lora_output.to(base_output.dtype)


def _resolve_selected_path(finetuning_args: "FinetuningArguments") -> str:
    p = finetuning_args.das_lora_selected_experts_path
    if not p or not os.path.exists(p):
        raise FileNotFoundError(
            f"das_lora_selected_experts_path not found: {p}. "
            f"Run eval_scripts/compute_das.py first to generate the JSON."
        )
    return p


def inject_das_lora(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """Inject DAS-LoRA: per-layer LoRA branches tied to DAS-selected experts."""
    selected_path = _resolve_selected_path(finetuning_args)
    with open(selected_path) as f:
        das_data = json.load(f)
    selected_per_layer = das_data["selected_experts_per_layer"]

    rank = finetuning_args.das_lora_rank
    alpha = finetuning_args.das_lora_alpha

    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype

    d_model = _get_config_value(model.config, "hidden_size")
    top_k_moe = _get_moe_top_k(model.config)
    if d_model is None or top_k_moe is None:
        raise RuntimeError(
            "Could not determine hidden_size / MoE top-k from model.config "
            "(tried num_experts_per_tok and top_k_experts)."
        )

    all_moe_blocks = _find_moe_blocks(model)
    if len(selected_per_layer) != len(all_moe_blocks):
        raise ValueError(
            f"DAS file has {len(selected_per_layer)} layers, "
            f"but model has {len(all_moe_blocks)} MoE layers"
        )

    n_per_layer = [len(s) for s in selected_per_layer]
    total_selected = sum(n_per_layer)
    logger.info_rank0(
        f"DAS-LoRA injecting into {len(all_moe_blocks)} MoE layers | "
        f"selected experts per layer: {n_per_layer[0] if all(k == n_per_layer[0] for k in n_per_layer) else n_per_layer} | "
        f"total {total_selected} LoRA experts | rank={rank}, alpha={alpha}"
    )

    for layer_idx, moe_block in enumerate(all_moe_blocks):
        selected_ids = selected_per_layer[layer_idx]
        if not selected_ids:
            logger.warning_rank0(f"Layer {layer_idx}: no experts selected, skipping")
            continue
        adapter_host = _get_expert_adapter_host(moe_block)
        adapter_host.das_lora_pool = DASLoRAPool(
            selected_ids, d_model, rank, alpha
        ).to(device=device, dtype=dtype)
        adapter_host._original_forward = adapter_host.forward
        if _uses_experts_host(moe_block):
            adapter_host.forward = patched_das_experts_forward.__get__(adapter_host)
        else:
            adapter_host._das_top_k_moe = top_k_moe
            adapter_host.forward = patched_das_moe_forward.__get__(adapter_host)

    # Reset requires_grad
    for name, p in model.named_parameters():
        if "das_lora_pool" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 兼容 DeepSpeed ZeRO-3:用 ds_numel 拿全量(否则 numel 只能拿本地 shard)
    def _real_numel(p):
        return getattr(p, "ds_numel", None) or p.numel()
    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    logger.info_rank0(
        f"DAS-LoRA setup complete | trainable: {trainable / 1e6:.2f}M "
        f"({100 * trainable / total:.3f}%) | total: {total / 1e6:.2f}M"
    )
    if not (0.0001 < trainable / total < 0.05):
        # ZeRO-3 早期 base model 可能未 materialize,降级为 warning
        logger.warning(
            f"DAS-LoRA trainable ratio {trainable / total:.4f} out of expected range. "
            "Under ZeRO-3 this may be due to params not yet materialized; verify after first step."
        )


def save_das_lora_state(model: nn.Module, save_dir: str,
                        finetuning_args: "FinetuningArguments") -> None:
    """Save trainable params + config + DAS selected experts JSON."""
    os.makedirs(save_dir, exist_ok=True)

    # ZeRO-3 兼容: 在 save 前 gather 所有 trainable params 的全量副本
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import deepspeed
        gathered_ctx = deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0)
    except Exception:
        # 非 ZeRO-3 训练(普通 DDP / 单卡):无需 gather
        gathered_ctx = None

    state = {}
    if gathered_ctx is not None:
        with gathered_ctx:
            # 只在 rank 0 真正构造 state(其他 rank 拿空 dict),safetensors.save_file 在所有 rank 都跑会冲突
            for name, param in model.named_parameters():
                if param.requires_grad:
                    state[name] = param.detach().cpu().clone()
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().cpu()

    if not state:
        logger.warning_rank0("No trainable params, skip saving DAS-LoRA state.")
        return
    # 只 rank 0 写入,避免多 rank 写同一文件
    is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_rank0:
        safetensors.torch.save_file(state, os.path.join(save_dir, "das_lora_state.safetensors"))

        # Copy DAS selection JSON to checkpoint dir
        src = finetuning_args.das_lora_selected_experts_path
        dst = os.path.join(save_dir, "das_selected_experts.json")
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)

        config = {
            "das_lora_selected_experts_path": "das_selected_experts.json",
            "das_lora_rank": finetuning_args.das_lora_rank,
            "das_lora_alpha": finetuning_args.das_lora_alpha,
            **common_lora_config(finetuning_args),
            "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
        }
        with open(os.path.join(save_dir, "das_lora_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info_rank0(f"Saved DAS-LoRA state ({len(state)} tensors) to {save_dir}")


def load_das_lora_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    """Load DAS-LoRA: rebuild structure from config + weights."""
    from ...hparams import FinetuningArguments

    config_path = os.path.join(load_dir, "das_lora_config.json")
    state_path = os.path.join(load_dir, "das_lora_state.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing das_lora_config.json or das_lora_state.safetensors in {load_dir}"
        )

    with open(config_path) as f:
        config = json.load(f)
    {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}

    # Resolve relative path of selected experts file
    sel_path = config["das_lora_selected_experts_path"]
    if not os.path.isabs(sel_path):
        sel_path = os.path.join(load_dir, sel_path)
    config["das_lora_selected_experts_path"] = sel_path

    finetuning_args = FinetuningArguments(finetuning_type="das_lora", **config)
    inject_das_lora(model, finetuning_args)
    inject_common_lora(model, finetuning_args)

    state = safetensors.torch.load_file(state_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, f"Unexpected weights: {unexpected[:5]}"
    unloaded = [n for n in missing if "das_lora_pool" in n or is_common_lora_parameter(n)]
    assert not unloaded, f"Failed to load DAS-LoRA weights: {unloaded[:5]}"

    logger.info_rank0(f"Loaded DAS-LoRA state from {load_dir}")
    return model


class DASLoRASaveCallback(TrainerCallback):
    """Save DAS-LoRA state at each checkpoint."""

    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_das_lora_state(model, ckpt_dir, self.finetuning_args)
        # Remove redundant base model files saved by Trainer
        import glob
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                    logger.info_rank0(f"Removed redundant base model file: {f}")
                except FileNotFoundError:
                    pass
