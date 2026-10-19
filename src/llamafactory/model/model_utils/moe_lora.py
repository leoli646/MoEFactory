# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE-LoRA Routing Compression: 在 MoE Transformer 主干上引入 LoRA 分支，
LoRA 的 routing 不独立学习，而是直接复用 MoE 的 router_logits 投影压缩而来:

    p_L = softmax(W @ router_logits),  W: N_E -> N_L

"""

import json
import math
import os
from typing import TYPE_CHECKING, List, Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from transformers import TrainerCallback

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


SUPPORTED_MOE_BLOCK_NAMES = {
    "OlmoeSparseMoeBlock",        # OLMoE
    "Qwen3MoeSparseMoeBlock",     # Qwen3-MoE
    "DeepseekV2MoE",              # DeepSeek-V2 (custom modeling_deepseek.py)
    "DeepseekV2Moe",              # DeepSeek-V2 (transformers 5.x native, 注意大小写)
    "DeepseekV3MoE",              # DeepSeek-V3 (custom)
    "DeepseekV3Moe",              # DeepSeek-V3 (transformers 5.x native)
    "Qwen3_5MoeSparseMoeBlock",   # Qwen3.5-MoE (multimodal, transformers 5.2+)
    "GptOssMLP",                  # GPT-OSS (router + transposed fused experts)
    "Gemma4TextDecoderLayer",     # Gemma 4 MoE lives directly on the decoder layer
}

EXPERTS_HOST_MOE_BLOCK_NAMES = {
    "GptOssMLP",
    "Gemma4TextDecoderLayer",
}


# ---------------------------------------------------------------------------
# 核心组件
# ---------------------------------------------------------------------------


class RoutingProjection(nn.Module):
    """投影 raw router_logits 到 LoRA routing space。

    输入: router_logits in R^{..., N_E}（稠密、未 softmax）
    流程: Linear(N_E -> N_L) -> softmax
    输出: p_L in R^{..., N_L}

    可选: hidden_bottleneck_dim > 0 时,从 hidden state 抽 token-specific 信号
    与 router_logits 拼接后喂给 W_l(突破 64-d 瓶颈)。

    内置 entropy 监控 buffer，forward 时累加（no_grad 不影响训练）。
    """

    def __init__(self, n_experts: int, n_lora: int,
                 d_model: int = 0, hidden_bottleneck_dim: int = 0,
                 das_init: Optional[torch.Tensor] = None,
                 das_init_strength: float = 1.0,
                 wl_init: str = "default"):
        super().__init__()
        self.hidden_bottleneck_dim = hidden_bottleneck_dim
        if hidden_bottleneck_dim > 0:
            assert d_model > 0, "d_model required when hidden_bottleneck_dim > 0"
            # 从 hidden state 抽 token-specific 信号
            self.h_proj = nn.Linear(d_model, hidden_bottleneck_dim, bias=False)
            input_dim = n_experts + hidden_bottleneck_dim
        else:
            self.h_proj = None
            input_dim = n_experts
        self.proj = nn.Linear(input_dim, n_lora, bias=False)

        # DAS-informed initialization (optional, takes precedence over wl_init):
        # hard-partition the N_E base experts by DAS rank into N_L groups,
        # assign each group to a distinct LoRA expert via strong positive weight.
        # 其余位置保留 Kaiming default init,确保训练后仍可演化。
        if das_init is not None:
            assert das_init.shape[0] == n_experts, \
                f"das_init dim {das_init.shape[0]} != n_experts {n_experts}"
            with torch.no_grad():
                sorted_idx = das_init.argsort(descending=True)  # high to low
                if n_lora <= n_experts:
                    # 多对一: 每个 LoRA expert 拥有 (N_E / N_L) 个 base expert
                    n_per_group = n_experts // n_lora
                    for lora_i in range(n_lora):
                        start = lora_i * n_per_group
                        end = start + n_per_group if lora_i < n_lora - 1 else n_experts
                        group = sorted_idx[start:end]
                        self.proj.weight.data[lora_i, group] = das_init_strength
                else:
                    # n_lora > n_experts: 每 base expert 被多个 LoRA "共享"(round-robin)
                    # LoRA i 映射到 DAS 排名 (i mod N_E) 的 base expert
                    for lora_i in range(n_lora):
                        base_idx = sorted_idx[lora_i % n_experts].item()
                        self.proj.weight.data[lora_i, base_idx] = das_init_strength
        elif wl_init == "anchor":
            # Anchor init (DAS-free): identity-like on z-channel, zero on h-channel.
            # 让 LoRA routing 在 step 0 镜像 base routing 的一个 chunked 版本——
            # 利用 base router 的 pretrained 结构作为"锚",避免随机 W_l 在 day 1
            # 触发 routing collapse;training 中 W_l 渐进偏移学 task-specific routing。
            with torch.no_grad():
                self.proj.weight.data.zero_()
                if n_lora <= n_experts:
                    n_per_group = n_experts // n_lora
                    for lora_i in range(n_lora):
                        start = lora_i * n_per_group
                        end = start + n_per_group if lora_i < n_lora - 1 else n_experts
                        self.proj.weight.data[lora_i, start:end] = 1.0
                else:
                    for lora_i in range(n_lora):
                        base_idx = lora_i % n_experts
                        self.proj.weight.data[lora_i, base_idx] = 1.0
                # h-channel (columns n_experts:) 已 zero,无需再设

        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换）
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, router_logits: torch.Tensor,
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.h_proj is not None:
            assert hidden_state is not None, (
                "hidden_state required when hidden_bottleneck_dim > 0"
            )
            h_c = self.h_proj(hidden_state)
            routing_input = torch.cat([router_logits, h_c], dim=-1)
        else:
            routing_input = router_logits
        p_L = torch.softmax(self.proj(routing_input), dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class IndependentRouter(nn.Module):
    """独立 router (Baseline2 / LoRAMoE-style).

    直接从 hidden_states 学路由，不依赖 MoE 的 router_logits。
    输入: h in R^{..., d_model}
    输出: p_L in R^{..., n_lora} (softmax 概率)
    """

    def __init__(self, d_model: int, n_lora: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_lora, bias=False)
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        p_L = torch.softmax(self.gate(h), dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class ResidualRouter(nn.Module):
    """残差叠加路由：W1 @ router_logits + W2 @ h

    结合 MoE routing 先验（64 维）和 hidden state 信息（2048 维）。
    训练后通过 W1/W2 的 norm 比值可量化两路信号的相对贡献。
    """

    def __init__(self, n_experts: int, d_model: int, n_lora: int):
        super().__init__()
        self.proj_moe = nn.Linear(n_experts, n_lora, bias=False)
        self.proj_h = nn.Linear(d_model, n_lora, bias=False)
        self.register_buffer("entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("entropy_count", torch.zeros(1, dtype=torch.long), persistent=False)

    def forward(self, router_logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj_moe(router_logits) + self.proj_h(h)
        p_L = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            ent = -(p_L * torch.log(p_L + 1e-8)).sum(dim=-1).mean()
            self.entropy_sum += ent.detach()
            self.entropy_count += 1
        return p_L


class LoRAExpert(nn.Module):
    """单个 LoRA adapter
    A: d -> r（Kaiming 初始化）, B: r -> d（可配置初始化）
    forward(h) -> (alpha / rank) * B(A(h))
    """

    def __init__(self, d_model: int, rank: int, alpha: int, b_init: str = "normal"):
        super().__init__()
        if b_init not in ("normal", "zero"):
            raise ValueError(f"b_init must be 'normal' or 'zero', got {b_init!r}.")
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, d_model, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        if b_init == "zero":
            nn.init.zeros_(self.B.weight)
        else:
            nn.init.normal_(self.B.weight, std=1e-4)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.B(self.A(h))


def _normalize_selected_topk(topk_vals: torch.Tensor) -> torch.Tensor:
    denominator = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return topk_vals / denominator


class LoRAPool(nn.Module):
    """N_L 个 LoRAExpert 的池
    forward(h, p_L) -> top-k 选择 + 加权和

    h:    [..., d_model]      MoE block 入口的 hidden_states
    p_L:  [..., n_experts]    routing 概率分布
    返回: [..., d_model]      LoRA 分支的输出（与 h 同形状）

    内置激活计数 buffer，监控每个 expert 的使用频率（防 dead expert）。
    """

    def __init__(
        self,
        n_experts: int,
        d_model: int,
        rank: int,
        alpha: int,
        top_k: int,
        normalize_topk_prob: bool = False,
        b_init: str = "normal",
    ):
        super().__init__()
        if b_init not in ("normal", "zero"):
            raise ValueError(f"b_init must be 'normal' or 'zero', got {b_init!r}.")
        self.n_experts = n_experts
        self.top_k = top_k
        self.normalize_topk_prob = normalize_topk_prob
        self.rank = rank
        self.scaling = alpha / rank
        self.experts = nn.ModuleList(
            [LoRAExpert(d_model, rank, alpha, b_init=b_init) for _ in range(n_experts)]
        )
        # torch 2.8 无 F.grouped_mm，历史上这里静默落回 _forward_loop（大池如
        # n_experts=128 时每层逐 expert Python 循环，~6x 慢）。现在默认走
        # weight_peft._SafeGroupedLinearFn：forward 用 grouped 内核，backward 的
        # dW 用逐 expert 小 mm（绝不触发 grouped_mm 自带 autograd 的死锁 dW 内核）。
        # MOE_LORA_FORCE_LOOP=1 还原旧循环路径。
        self._use_grouped_mm = os.environ.get("MOE_LORA_FORCE_LOOP", "0") != "1"
        # 监控用 buffer（long 类型不会被 .to(dtype=bfloat16) 转换，避免精度丢失）
        self.register_buffer("activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long), persistent=False)
        # 全生命周期累计（不被 on_log 清零，用于训练后剪枝 dead experts）
        self.register_buffer("lifetime_activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)

    def forward(self, h: torch.Tensor, p_L: torch.Tensor) -> torch.Tensor:
        p_L_orig = p_L
        p_L = p_L.to(h.dtype)
        topk_vals, topk_idx = p_L.topk(self.top_k, dim=-1)  # [..., k]
        if self.normalize_topk_prob:
            topk_vals = _normalize_selected_topk(topk_vals)

        with torch.no_grad():
            flat_idx = topk_idx.reshape(-1)
            ones = torch.ones_like(flat_idx)
            self.activation_count.scatter_add_(0, flat_idx, ones)
            self.lifetime_activation_count.scatter_add_(0, flat_idx, ones)
            n_tokens = 1
            for s in h.shape[:-1]:
                n_tokens *= s
            self.total_tokens += n_tokens

        # load balancing loss: L = N * Σ(f_i * p_i)
        # 注意: 多层共享同一 LoRAPool 时(global / grouped pool 默认),后续层会
        # 覆盖前面层的 _aux_loss,实际只反映"本 batch 最后一层"的 balance 信号。
        # 这是历史行为(OLMoE 训练时如此),保留以便复现。
        # 累加版会让 balance loss 数量级 ≈ layers_per_pool× 放大,需要相应调整 coef。
        if self.training:
            n_tokens_f = float(h.shape[0]) if h.dim() == 2 else float(h.shape[0] * h.shape[1])
            one_hot = torch.zeros(int(n_tokens_f), self.n_experts, device=h.device, dtype=p_L_orig.dtype)
            one_hot.scatter_(1, topk_idx.reshape(int(n_tokens_f), self.top_k), 1.0)
            f = one_hot.mean(dim=0)
            p = p_L_orig.reshape(int(n_tokens_f), -1).mean(dim=0)
            self._aux_loss = self.n_experts * (f * p).sum()
        else:
            self._aux_loss = None

        if self._use_grouped_mm:
            return self._forward_grouped_mm(h, topk_vals, topk_idx)
        else:
            return self._forward_loop(h, topk_vals, topk_idx)

    def dispatch(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """跳过内部 top-K, 直接用外部传入的 (topk_vals, topk_idx) 做 expert dispatch.

        用于 joint top-K 场景(per-layer + cross-layer global pool 联合竞争).
        topk_idx 必须在 [0, self.n_experts) 范围内; 没选中的 slot 应填 idx=0,
        vals=0(权重 0 时输出 = 0 * expert(h) = 0, 不污染结果).

        不更新 activation_count 也不算 balance loss (joint 模式由外部 forward 统筹).
        """
        topk_vals = topk_vals.to(h.dtype)
        if self._use_grouped_mm:
            return self._forward_grouped_mm(h, topk_vals, topk_idx)
        else:
            return self._forward_loop(h, topk_vals, topk_idx)

    def _forward_grouped_mm(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """高效实现：token 按 expert 分组后用 grouped 内核跑两级低秩投影。

        经 weight_peft._SafeGroupedLinearFn（forward 用 grouped 内核，backward 的
        grad_input 用同款内核、grad_weight 用逐 expert 小 mm），可训练的 A/B 不会
        触发 grouped_mm 自带 autograd 的 dW 内核（DeepSpeed 首步死锁根源）。
        rank padding 到 8 的倍数（16 字节对齐）由 _grouped_lowrank 处理。"""
        from .weight_peft import _grouped_lowrank, _make_safe_grouped_linear

        orig_shape = h.shape
        d = h.shape[-1]
        h_flat = h.reshape(-1, d)  # [N, d]
        N = h_flat.shape[0]
        k = topk_idx.shape[-1]   # 从 topk_idx 推断, 兼容 dispatch 模式(K 可能 != self.top_k)

        # 展开: 每个 token 复制 k 份，对应 k 个选中的 expert
        h_expanded = h_flat.unsqueeze(1).expand(N, k, d).reshape(N * k, d)  # [N*k, d]
        expert_ids = topk_idx.reshape(N * k)  # [N*k]
        weights = topk_vals.reshape(N * k)  # [N*k]

        # 按 expert_id 排序，grouped 内核要求同组 token 连续
        sorted_order = expert_ids.argsort(stable=True)
        h_sorted = h_expanded[sorted_order].contiguous()  # [N*k, d]
        expert_ids_sorted = expert_ids[sorted_order]
        weights_sorted = weights[sorted_order]  # [N*k]

        # 计算每个 expert 的 token 数量 -> cumulative offsets
        counts = torch.zeros(self.n_experts, dtype=torch.int32, device=h.device)
        counts.scatter_add_(0, expert_ids_sorted.to(torch.int64), torch.ones(N * k, dtype=torch.int32, device=h.device))
        offs = counts.cumsum(0).to(torch.int32)  # [n_experts]

        # 堆叠所有 expert 的权重（logical F.linear 朝向: A [E, r, d], B [E, d, r]）
        A_weight = torch.stack([e.A.weight for e in self.experts])
        B_weight = torch.stack([e.B.weight for e in self.experts])

        gl = _make_safe_grouped_linear(None)
        output_sorted = _grouped_lowrank(gl, h_sorted, A_weight, B_weight, offs, self.scaling)

        # 应用 routing weight
        output_sorted = weights_sorted.unsqueeze(-1).to(output_sorted.dtype) * output_sorted

        # 反排序 + 聚合 k 个 expert 的输出
        output_expanded = torch.zeros(N * k, d, device=h.device, dtype=output_sorted.dtype)
        output_expanded[sorted_order] = output_sorted
        out = output_expanded.reshape(N, k, d).sum(dim=1)  # [N, d]

        return out.to(h.dtype).reshape(orig_shape)

    def _forward_loop(self, h: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Fallback: 逐 expert for 循环（兼容旧版 PyTorch）。"""
        out = torch.zeros_like(h)
        for i in range(self.n_experts):
            expert_mask = topk_idx == i
            if not expert_mask.any():
                continue
            weight = (topk_vals * expert_mask).sum(dim=-1, keepdim=True)
            active = (weight.squeeze(-1) > 0)
            if not active.any():
                continue
            out_i = self.experts[i](h[active])
            update = (weight[active] * out_i).to(out.dtype)
            out[active] = out[active] + update

        # ---- DeepSpeed ZeRO 同步修复 ----
        # 稀疏路由下,某个 rank 的 micro-batch 可能完全没有 token 选中某些 expert,
        # 这些 expert 的 A/B 参数本步在该 rank 上拿不到梯度。DeepSpeed ZeRO-1/2 按
        # "实际产生梯度的参数"来组 allreduce/reduce-scatter 的 bucket,各 rank 的
        # 可训练参数梯度集合不一致 -> 张量形状对不上 -> 报
        # "Detected mismatch between collectives on ranks"。
        # 这里加一个 0 权重的 touch 项,让每个 expert 的参数每步都进入计算图
        # (梯度为 0 但 .grad 被填充),保证各 rank 的 bucket 一致。
        # 对前向数值无影响(加的是 0);grouped_mm 路径天然 stack 全部 expert 权重,无此问题。
        if self.training:
            touch = None
            for p in self.experts.parameters():
                if not p.requires_grad:
                    continue
                s = p.sum()
                touch = s if touch is None else touch + s
            if touch is not None:
                out = out + (0.0 * touch).to(out.dtype)
        return out


def _is_moe_lora_pool(module: nn.Module) -> bool:
    return isinstance(module, LoRAPool) or getattr(module, "_is_moe_lora_pool", False)


# ---------------------------------------------------------------------------
# Position-ablation 组件: Attention LoRA + Router(gate) LoRA
# ---------------------------------------------------------------------------


class LinearLoRAAdapter(nn.Module):
    """标准 LoRA 旁路,挂在单个 nn.Linear(如 attention 的 q/k/v/o_proj)旁。

    A: in_features -> rank (Kaiming), B: rank -> out_features (近零初始化)。
    forward(x) -> (alpha / rank) * B(A(x))   —— 与原 Linear 输出相加。
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.normal_(self.B.weight, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.B(self.A(x))


def _scaled_alpha(base_rank: int, base_alpha: int, layer_rank: int) -> int:
    """Preserve alpha/rank when a budget-matched layer uses a different rank."""
    scaled = base_alpha * layer_rank
    if scaled % base_rank:
        raise ValueError(
            "moe_lora_layer_ranks cannot preserve the configured alpha/rank ratio exactly: "
            f"base rank/alpha={base_rank}/{base_alpha}, layer rank={layer_rank}."
        )
    return scaled // base_rank


def _resolve_layer_ranks(layer_ranks: str, default_rank: int, n_layers: int) -> List[int]:
    """Resolve optional per-layer ranks in selected-MoE-layer order."""
    value = (layer_ranks or "").strip()
    if not value:
        return [default_rank] * n_layers
    try:
        ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("moe_lora_layer_ranks must be a comma-separated list of positive integers.") from exc
    if len(ranks) != n_layers:
        raise ValueError(
            f"moe_lora_layer_ranks has {len(ranks)} entries but {n_layers} selected MoE layers were found."
        )
    if any(rank <= 0 for rank in ranks):
        raise ValueError("moe_lora_layer_ranks entries must all be positive integers.")
    return ranks


def _patched_linear_forward(self, x, *args, **kwargs):
    """注入到 attention proj 上的 forward: 原输出 + LoRA 旁路。"""
    base = self._lora_orig_forward(x, *args, **kwargs)
    return base + self._lora_adapter(x.to(self._lora_adapter.A.weight.dtype)).to(base.dtype)


class LoRAWeightDelta(nn.Module):
    """对一个权重矩阵 W 施加低秩增量的 parametrization:

        W_eff = W + (alpha / rank) * (B @ A)

    用于 router/gate: gate 的有效权重变成 base + 低秩增量,从而改变 base MoE 的
    expert 选择。用 torch.nn.utils.parametrize 注册到 gate.weight 上,对所有"读取
    gate.weight"的调用方(OLMoE/Qwen3 router.forward 内的 F.linear、DeepSeek block
    内直接 gate.weight)都透明生效。

    A: [rank, in_features](Kaiming),B: [out_features, rank](近零初始化)。
    """

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.B, std=1e-4)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (self.B @ self.A)
        return weight + delta.to(weight.dtype)


def _resolve_attn_targets(attn_module: nn.Module, targets_cfg: str) -> List[str]:
    """解析 attention LoRA 的目标 Linear 子模块名。

    'auto' -> 该 attention 下所有 nn.Linear 直接子模块(完整注意力参数化)。
        OLMoE/Qwen3: q_proj,k_proj,v_proj,o_proj
        DeepSeek-V2 MLA: q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
    否则按逗号列表过滤(只保留确实存在且为 Linear 的子模块)。
    """
    linear_children = [
        name for name, m in attn_module.named_children() if isinstance(m, nn.Linear)
    ]
    targets_cfg = (targets_cfg or "auto").strip()
    if targets_cfg == "auto":
        return linear_children
    wanted = [t.strip() for t in targets_cfg.split(",") if t.strip()]
    return [t for t in wanted if t in linear_children]


def _inject_attention_lora(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments",
    target_layer_indices, device, dtype,
) -> int:
    """在选定 decoder 层的 attention 投影上注入标准 LoRA。返回挂载的 adapter 数。"""
    rank = finetuning_args.moe_lora_attn_rank or finetuning_args.moe_lora_rank
    alpha = finetuning_args.moe_lora_attn_alpha or finetuning_args.moe_lora_alpha
    targets_cfg = getattr(finetuning_args, "moe_lora_attn_targets", "auto")

    layers_container = _locate_decoder_layers(model)
    if layers_container is None:
        raise ValueError("Cannot locate decoder layers for attention LoRA injection.")
    layers = layers_container.layers

    n_attached = 0
    chosen_names = None
    for layer_idx, layer in enumerate(layers):
        if layer_idx not in target_layer_indices:
            continue
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is None:
            continue
        proj_names = _resolve_attn_targets(attn, targets_cfg)
        if not proj_names:
            logger.warning_rank0(
                f"MoE-LoRA[attn]: layer {layer_idx} resolved 0 target linears from "
                f"moe_lora_attn_targets='{targets_cfg}'. Available: "
                f"{[n for n, m in attn.named_children() if isinstance(m, nn.Linear)]}"
            )
        if chosen_names is None:
            chosen_names = proj_names
        for pname in proj_names:
            proj = getattr(attn, pname)
            adapter = LinearLoRAAdapter(
                proj.in_features, proj.out_features, rank, alpha
            ).to(device=device, dtype=dtype)
            proj._lora_adapter = adapter
            proj._lora_orig_forward = proj.forward
            proj.forward = _patched_linear_forward.__get__(proj)
            n_attached += 1

    logger.info_rank0(
        f"MoE-LoRA[attn]: injected LoRA on {n_attached} attention proj linears "
        f"({len(target_layer_indices)} layers x {chosen_names}) | rank={rank}, alpha={alpha}"
    )
    return n_attached


def _inject_router_lora(
    moe_blocks: List[nn.Module], finetuning_args: "FinetuningArguments", device, dtype,
) -> int:
    """在每个目标 MoE block 的 router projection 上注册低秩增量。"""
    rank = finetuning_args.moe_lora_router_rank or finetuning_args.moe_lora_rank
    alpha = finetuning_args.moe_lora_router_alpha or finetuning_args.moe_lora_alpha

    n_attached = 0
    for moe_block in moe_blocks:
        router_linear = _get_router_linear(moe_block)
        if router_linear is None:
            continue
        w = router_linear.weight
        if w.dim() != 2:
            continue
        out_f, in_f = w.shape
        delta = LoRAWeightDelta(out_f, in_f, rank, alpha).to(device=device, dtype=dtype)
        parametrize.register_parametrization(router_linear, "weight", delta)
        n_attached += 1

    logger.info_rank0(
        f"MoE-LoRA[router]: injected low-rank delta on {n_attached} MoE router weights "
        f"| rank={rank}, alpha={alpha}"
    )
    return n_attached


def _apply_lora_with_optional_cross_pool(moe_block, h_input, p_L):
    """LoRA dispatch helper:
    - 无 cross-layer pool: 走原 LoRAPool.forward 路径(单 pool 自己 top-K + balance loss).
    - 有 cross-layer pool:
        * Default (top_k_local/global=0): joint top-K (over n_local+n_global combined),
          按 idx<n_local 拆分 dispatch 到 per-layer 和 cross-layer pool.
        * Split mode (top_k_local>0 AND top_k_global>0): per-layer 和 global pool 各自
          独立 top-K (互不竞争). 给两个 pool 保证最低 quota.
      Joint/split 模式都不算 balance loss(由 LoRAPool.dispatch 跳过).
    """
    cross_pool = getattr(moe_block, "cross_layer_lora_pool", None)
    if cross_pool is None:
        return moe_block.lora_pool(h_input, p_L)

    n_local = moe_block.lora_pool.n_experts
    p_L_h = p_L.to(h_input.dtype)

    # Split top-K mode: each pool selects independently from its own logits slice
    fargs = getattr(moe_block, "_finetuning_args", None)
    k_local = getattr(fargs, "moe_lora_top_k_local", 0) if fargs is not None else 0
    k_global = getattr(fargs, "moe_lora_top_k_global", 0) if fargs is not None else 0
    normalize_topk_prob = (
        getattr(fargs, "moe_lora_normalize_topk_prob", False) if fargs is not None else False
    )

    if k_local > 0 and k_global > 0:
        p_local = p_L_h[..., :n_local]                            # (..., n_local)
        p_global = p_L_h[..., n_local:]                           # (..., n_global)
        local_vals, local_idx = p_local.topk(k_local, dim=-1)     # (..., k_local)
        global_vals, global_idx = p_global.topk(k_global, dim=-1) # (..., k_global)
        if normalize_topk_prob:
            denominator = (
                local_vals.sum(dim=-1, keepdim=True) + global_vals.sum(dim=-1, keepdim=True)
            ).clamp_min(1e-12)
            local_vals = local_vals / denominator
            global_vals = global_vals / denominator
        out_local = moe_block.lora_pool.dispatch(h_input, local_vals, local_idx)
        out_global = cross_pool.dispatch(h_input, global_vals, global_idx)
        return out_local + out_global

    # Default: joint top-K over combined logits
    K = moe_block.lora_pool.top_k
    topk_vals, topk_idx = p_L_h.topk(K, dim=-1)                   # (..., K)
    if normalize_topk_prob:
        topk_vals = _normalize_selected_topk(topk_vals)
    local_mask = topk_idx < n_local                                # bool, (..., K)
    local_mask_f = local_mask.to(topk_vals.dtype)

    local_idx = topk_idx.masked_fill(~local_mask, 0)
    local_vals = topk_vals * local_mask_f
    global_idx = (topk_idx - n_local).masked_fill(local_mask, 0)
    global_vals = topk_vals * (1.0 - local_mask_f)

    out_local = moe_block.lora_pool.dispatch(h_input, local_vals, local_idx)
    out_global = cross_pool.dispatch(h_input, global_vals, global_idx)
    return out_local + out_global


def patched_moe_forward(self, hidden_states: torch.Tensor):
    """注入到 OLMoE / Qwen3-MoE 等 MoE block 上的 forward。

    依赖以下属性（由 inject_moe_lora 在 monkey-patch 时挂上）：
        self._original_forward    原始 MoE forward
        self._finetuning_args     训练超参（用于读 detach_p_e 等开关）
        self.routing_projection   RoutingProjection 实例
        self.lora_pool            LoRAPool 实例
    """
    # 防御性: 仅当挂了 lora_pool(attach_ffn=True)时才走 LoRA 分支,否则原样返回。
    if getattr(self, "lora_pool", None) is None:
        return self._original_forward(hidden_states)
    # 1. 原 MoE 计算 + 取出 router_logits
    #    旧版 transformers: forward 返回 (final_hidden_states, router_logits)
    #    新版 transformers (OLMoE): forward 只返回 final_hidden_states，
    #      需要额外调用 self.gate 拿 router_logits。注意 OlmoeTopKRouter.forward
    #      返回三元组 (router_logits, router_scores, router_indices)。
    original_result = self._original_forward(hidden_states)
    if isinstance(original_result, tuple):
        moe_output, router_logits, *rest = original_result
    else:
        moe_output = original_result
        rest = []
        # 自己跑 raw logits，绕开 OlmoeTopKRouter 内部的 float32 softmax
        # （否则 router_logits 会变成 float32 概率，下游和 bfloat16 的 h 不兼容）
        if hidden_states.dim() == 3:
            h_flat_for_gate = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            h_flat_for_gate = hidden_states
        gate_bias = getattr(self.gate, "bias", None)
        # detach gate.weight 以避免 DeepSpeed ZeRO 把它误认为被 trainable 两次访问。
        # 在 PEFT 场景下 gate.weight 是 frozen 的,这次 F.linear 只是"读取"它来计算
        # router_logits 给 RoutingProjection,不需要梯度流经它。detach 保证 DeepSpeed
        # 的 grad reduce hook 不会被错误地触发两次。
        router_logits = F.linear(
            h_flat_for_gate, self.gate.weight.detach(),
            gate_bias.detach() if gate_bias is not None else None,
        )

    # 2. LoRA 分支
    detach = getattr(self._finetuning_args, "moe_lora_detach_p_e", False)
    logits_for_lora = router_logits.detach() if detach else router_logits
    routing_mode = getattr(self._finetuning_args, "moe_lora_routing_mode", "learned")

    # OLMoE 内部把 hidden_states reshape 成 [B*T, D] 后再调 gate，
    # 所以 router_logits 形状是 [B*T, N_E]，而 hidden_states 仍是 [B, T, D]。
    # 让 LoRA 分支也用 flatten 形式，避免形状不匹配。
    if router_logits.dim() == 2 and hidden_states.dim() == 3:
        b, t, d = hidden_states.shape
        h_flat = hidden_states.reshape(-1, d)
        if routing_mode == "follow_moe":
            p_L = _follow_moe_p_L(self, h_flat, logits_for_lora)
        elif routing_mode == "independent":
            p_L = self.independent_router(h_flat)
        elif routing_mode == "residual":
            p_L = self.residual_router(logits_for_lora, h_flat)
        else:
            # learned: optional hidden state bottleneck
            if getattr(self.routing_projection, "h_proj", None) is not None:
                p_L = self.routing_projection(logits_for_lora, h_flat)
            else:
                p_L = self.routing_projection(logits_for_lora)     # [B*T, N_L (+ N_global)]
        lora_output = _apply_lora_with_optional_cross_pool(self, h_flat, p_L)   # [B*T, D]
        lora_output = lora_output.reshape(b, t, d)
    else:
        if routing_mode == "follow_moe":
            p_L = _follow_moe_p_L(self, hidden_states, logits_for_lora)
        elif routing_mode == "independent":
            p_L = self.independent_router(hidden_states)
        elif routing_mode == "residual":
            p_L = self.residual_router(logits_for_lora, hidden_states)
        else:
            if getattr(self.routing_projection, "h_proj", None) is not None:
                p_L = self.routing_projection(logits_for_lora, hidden_states)
            else:
                p_L = self.routing_projection(logits_for_lora)
        lora_output = _apply_lora_with_optional_cross_pool(self, hidden_states, p_L)

    # 3. 相加，保持原签名
    final_output = moe_output + lora_output

    if rest:
        return (final_output, *rest)
    return final_output


def _extract_experts_routing(args, kwargs):
    """Return the top-k indices/weights passed to a fused experts module."""
    if len(args) >= 2:
        return args[0], args[1]
    indices = kwargs.get("top_k_index", kwargs.get("router_indices"))
    weights = kwargs.get("top_k_weights", kwargs.get("routing_weights"))
    if indices is None or weights is None:
        raise RuntimeError("MoE experts forward did not receive top-k routing indices and weights.")
    return indices, weights


def _follow_external_moe_p_L(
    expert_host: nn.Module,
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Build the sparse follow-MoE distribution from the router's actual dispatch."""
    n_experts = expert_host._moe_lora_n_experts_moe
    valid = (top_k_indices >= 0) & (top_k_indices < n_experts)
    safe_indices = top_k_indices.clamp(0, n_experts - 1)
    safe_weights = top_k_weights * valid.to(top_k_weights.dtype)
    p_l = top_k_weights.new_zeros(*top_k_weights.shape[:-1], n_experts)
    p_l.scatter_add_(-1, safe_indices, safe_weights)
    return p_l


def patched_experts_moe_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
    """Add the routed LoRA branch at the experts host used by Gemma 4 and GPT-OSS.

    Both architectures pass their already-selected expert ids and normalized routing
    weights to ``experts.forward``. Reusing those values avoids reimplementing either
    model's dispatch semantics (Gemma's per-expert scale and GPT-OSS's top-k softmax).
    """
    base_output = self._original_forward(hidden_states, *args, **kwargs)
    if getattr(self, "lora_pool", None) is None:
        return base_output

    top_k_indices, top_k_weights = _extract_experts_routing(args, kwargs)
    original_shape = hidden_states.shape
    h_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    routing_mode = getattr(self._finetuning_args, "moe_lora_routing_mode", "learned")

    if routing_mode == "follow_moe":
        p_l = _follow_external_moe_p_L(self, top_k_indices, top_k_weights)
    elif routing_mode == "independent":
        p_l = self.independent_router(h_flat)
    else:
        router_logits = getattr(self, "_moe_lora_cached_router_logits", None)
        self._moe_lora_cached_router_logits = None
        if router_logits is None or router_logits.shape[0] != h_flat.shape[0]:
            router_logits = _compute_router_logits(self._moe_lora_router, h_flat)
        if getattr(self._finetuning_args, "moe_lora_detach_p_e", False):
            router_logits = router_logits.detach()
        if routing_mode == "residual":
            p_l = self.residual_router(router_logits, h_flat)
        elif getattr(self.routing_projection, "h_proj", None) is not None:
            p_l = self.routing_projection(router_logits, h_flat)
        else:
            p_l = self.routing_projection(router_logits)

    lora_output = _apply_lora_with_optional_cross_pool(self, h_flat, p_l).reshape(original_shape)
    return base_output + lora_output.to(base_output.dtype)


def _follow_moe_p_L(moe_block, h_flat: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    """对照组路由：直接复用原 MoE router 的 top_k 选择，跳过 RoutingProjection。

    输出 p_L: [n_tokens, n_experts_moe] 稀疏权重矩阵，只在原 router 选中的 top_k
    位置上有值（= router 的 softmax 权重），其他位置为 0。下游 LoRAPool 会按这个
    分布做 top-k（n_lora == n_experts_moe，top_k_lora 通常等于原 MoE 的 top_k）。

    实现说明：直接对 router_logits 做 softmax + topk，**等价于** OlmoeTopKRouter 内部
    那一套（softmax → topk → 可选 norm_topk_prob）。这样就完全避开了
    RoutingProjection，参数也不再有 W。
    """
    n_experts = router_logits.shape[-1]
    top_k = moe_block.lora_pool.top_k
    probs = torch.softmax(router_logits, dim=-1, dtype=router_logits.dtype)
    top_vals, top_idx = probs.topk(top_k, dim=-1)
    p_L = torch.zeros_like(probs)
    p_L.scatter_(1, top_idx, top_vals)
    return p_L


def _uses_experts_host(moe_block: nn.Module) -> bool:
    return type(moe_block).__name__ in EXPERTS_HOST_MOE_BLOCK_NAMES


def _expert_weight_router_aux_hook(module, inputs, output):
    """Per-layer load-balancing loss for expert_weight granularity, from the base router.

    expert_weight arms have no LoRAPool (the LBL carrier of the bypass arms), so the
    signal is computed at the router itself: aux = N_E * Σ_i f_i · P_i with
    f_i = hard dispatch frequency from the router's own top-k indices (no grad) and
    P_i = mean *dense* softmax probability (grad flows through the parametrized router
    weight into the router LoRA delta, and through hidden states into earlier layers).
    Same 口径 as LoRAPool (per-layer value, ideal = top_k; collected as per-layer mean).

    All three supported routers return ``(dense, top_k_weights, top_k_indices)``;
    Gemma-style routers (identified by ``.proj``) return probabilities as ``dense``,
    Qwen3/GPT-OSS return raw logits.
    """
    if not (module.training and torch.is_grad_enabled()):
        module._moe_lora_aux_loss = None
        return
    if not (isinstance(output, tuple) and len(output) >= 3):
        return
    dense, _, top_idx = output[0], output[1], output[2]
    if getattr(module, "proj", None) is not None:
        probs = dense.float()
    else:
        probs = torch.softmax(dense.float(), dim=-1)
    probs = probs.reshape(-1, probs.shape[-1])
    n_experts = probs.shape[-1]
    idx = top_idx.reshape(-1, top_idx.shape[-1]).to(torch.long)
    one_hot = torch.zeros(idx.shape[0], n_experts, device=probs.device, dtype=probs.dtype)
    one_hot.scatter_(1, idx, 1.0)
    f = one_hot.mean(dim=0)
    p = probs.mean(dim=0)
    module._moe_lora_aux_loss = n_experts * (f * p).sum()


def _install_expert_weight_balance_hooks(moe_blocks: List[nn.Module]) -> None:
    """Attach the router-side LBL hooks for expert_weight granularity."""
    n_hooked = 0
    for block_index, moe_block in enumerate(moe_blocks):
        router = _get_router_module(moe_block)
        if router is None:
            raise ValueError(
                f"balance_loss_coef > 0 with expert_weight granularity, but MoE block "
                f"{block_index} ({type(moe_block).__name__}) has no gate/router module."
            )
        router._moe_lora_aux_loss = None
        router.register_forward_hook(_expert_weight_router_aux_hook)
        n_hooked += 1
    logger.info_rank0(
        f"MoE-LoRA[expert_weight] balance loss: router-side LBL hooks on {n_hooked} layers "
        "(dense-softmax P_i x hard-dispatch f_i, per-layer mean aggregation)."
    )


def _get_router_module(moe_block: nn.Module) -> Optional[nn.Module]:
    """Return a model's MoE router without changing legacy ``.gate`` lookup."""
    return getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)


def _get_router_linear(moe_block: nn.Module) -> Optional[nn.Module]:
    """Return the 2-D router projection carrying the expert logits weight."""
    router = _get_router_module(moe_block)
    if router is None:
        return None
    if isinstance(getattr(router, "weight", None), torch.Tensor):
        return router
    projection = getattr(router, "proj", None)
    if isinstance(getattr(projection, "weight", None), torch.Tensor):
        return projection
    return None


def _get_expert_adapter_host(moe_block: nn.Module) -> nn.Module:
    """Return the module whose forward receives hidden states plus top-k routing."""
    if _uses_experts_host(moe_block):
        experts = getattr(moe_block, "experts", None)
        if experts is None:
            raise ValueError(f"{type(moe_block).__name__} has no experts module.")
        return experts
    return moe_block


def _get_config_value(config, *names):
    """Read a field from a top-level config or its nested text config."""
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)
    for cfg in configs:
        for name in names:
            value = getattr(cfg, name, None)
            if value is not None:
                return value
    return None


def _get_moe_top_k(config) -> Optional[int]:
    return _get_config_value(config, "num_experts_per_tok", "top_k_experts")


def _compute_router_logits(router: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute dense pre-top-k router logits for GPT-OSS and Gemma 4.

    Gemma's router applies an unscaled RMS norm and two learned scales before its
    ``proj`` linear. GPT-OSS exposes ``weight``/``bias`` directly. The fallback is
    intentionally generic for future routers that return logits as their first item.
    """
    projection = getattr(router, "proj", None)
    if projection is not None and isinstance(getattr(projection, "weight", None), torch.Tensor):
        router_input = hidden_states
        norm = getattr(router, "norm", None)
        if norm is not None:
            router_input = norm(router_input)
        scale = getattr(router, "scale", None)
        if scale is not None:
            router_input = router_input * scale
        scalar_root_size = getattr(router, "scalar_root_size", None)
        if scalar_root_size is not None:
            router_input = router_input * scalar_root_size
        return projection(router_input)

    if isinstance(getattr(router, "weight", None), torch.Tensor):
        return F.linear(hidden_states, router.weight, getattr(router, "bias", None))

    result = router(hidden_states)
    if isinstance(result, tuple):
        return result[0]
    return result


def _install_router_logits_cache(moe_block: nn.Module, expert_host: nn.Module) -> None:
    """Cache the exact parent-router signal consumed immediately by experts.forward."""
    router = _get_router_module(moe_block)
    if router is None:
        raise ValueError(f"{type(moe_block).__name__} has no gate/router module.")
    object.__setattr__(expert_host, "_moe_lora_router", router)

    def _capture_router_logits(module, inputs, output):
        if isinstance(output, tuple) and getattr(module, "proj", None) is None:
            logits = output[0]
        else:
            logits = _compute_router_logits(module, inputs[0])
        expert_host._moe_lora_cached_router_logits = logits

    handle = router.register_forward_hook(_capture_router_logits)
    object.__setattr__(expert_host, "_moe_lora_router_hook_handle", handle)


def _resolve_target_layer_indices(target_layers: str, n_layers: int) -> List[int]:
    """解析 moe_lora_target_layers 配置成具体层号列表。

    支持: "all" / "first_half" / "last_half" / "last_third" / "0,5,10,15"
    """
    target_layers = target_layers.strip()
    if target_layers == "all":
        return list(range(n_layers))
    if target_layers == "first_half":
        return list(range(n_layers // 2))
    if target_layers == "last_half":
        return list(range(n_layers // 2, n_layers))
    if target_layers == "last_third":
        return list(range(2 * n_layers // 3, n_layers))
    # 显式层号列表，逗号分隔
    return [int(s) for s in target_layers.split(",") if s.strip()]


def _locate_decoder_layers(model: "PreTrainedModel") -> Optional[nn.Module]:
    """定位 decoder 层列表，兼容纯文本 LM 和多模态 LM。

    优先级:
    1. model.model.layers          (OLMoE / Qwen3-MoE / DeepSeek-V2 等纯文本)
    2. model.model.text_model.layers / model.model.language_model.layers  (Qwen3.5-VL 等多模态)
    3. model.language_model.model.layers  (其他多模态 wrapper)
    返回 layers 容器 (有 .layers 属性的 module)，找不到返回 None。
    """
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base
    # 多模态: model.model.text_model / language_model
    if base is not None:
        for sub_name in ("text_model", "language_model"):
            sub = getattr(base, sub_name, None)
            if sub is not None and hasattr(sub, "layers"):
                return sub
            # 再下一层 wrapper: text_model.model.layers
            if sub is not None:
                inner = getattr(sub, "model", None)
                if inner is not None and hasattr(inner, "layers"):
                    return inner
    # 顶层 language_model wrapper
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner
    return None


def _find_moe_blocks(model: "PreTrainedModel") -> List[nn.Module]:
    """遍历 decoder layers，找到所有 MoE block。"""
    moe_blocks = []
    layers = _locate_decoder_layers(model)
    if layers is None:
        raise ValueError(
            "Cannot locate decoder layers. Tried model.model.layers, "
            "model.model.text_model.layers, model.model.language_model.layers, "
            "model.language_model.(model.)layers."
        )
    for layer in layers.layers:
        # Gemma 4 owns router/experts directly on the decoder layer. Other supported
        # models keep their MoE block under ``mlp`` or ``feed_forward``.
        if type(layer).__name__ == "Gemma4TextDecoderLayer" and hasattr(layer, "experts"):
            block = layer
        else:
            block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is not None and type(block).__name__ in SUPPORTED_MOE_BLOCK_NAMES:
            moe_blocks.append(block)
    if not moe_blocks:
        raise ValueError(
            f"No MoE block found in model. Supported types: {SUPPORTED_MOE_BLOCK_NAMES}. "
            f"Got layer types: {[type(l).__name__ for l in layers.layers[:3]]}"
        )
    return moe_blocks


_COMMON_LORA_CONFIG_KEYS = (
    "moe_lora_attach_attn",
    "moe_lora_attach_router",
    "moe_lora_attn_targets",
    "moe_lora_attn_rank",
    "moe_lora_attn_alpha",
    "moe_lora_router_rank",
    "moe_lora_router_alpha",
    "moe_lora_target_layers",
    "moe_lora_pool_share",
    "moe_lora_w_share",
    "moe_lora_balance_loss_coef",
)

_COMMON_LORA_CONFIG_DEFAULTS = {
    "moe_lora_attach_attn": False,
    "moe_lora_attach_router": False,
    "moe_lora_attn_targets": "auto",
    "moe_lora_attn_rank": 0,
    "moe_lora_attn_alpha": 0,
    "moe_lora_router_rank": 0,
    "moe_lora_router_alpha": 0,
    "moe_lora_target_layers": "all",
    "moe_lora_pool_share": "per_layer",
    "moe_lora_w_share": "per_layer",
    "moe_lora_balance_loss_coef": 0.0,
}


def common_lora_is_enabled(finetuning_args: "FinetuningArguments") -> bool:
    """Return whether a legacy-compatible method-local Attention/Router LoRA is requested."""
    return bool(
        getattr(finetuning_args, "moe_lora_attach_attn", False)
        or getattr(finetuning_args, "moe_lora_attach_router", False)
    )


def common_lora_config(finetuning_args: "FinetuningArguments") -> dict:
    """Serialize legacy-compatible Attention/Router LoRA fields into a checkpoint config."""
    return {
        key: getattr(finetuning_args, key, _COMMON_LORA_CONFIG_DEFAULTS[key])
        for key in _COMMON_LORA_CONFIG_KEYS
    }


def is_common_lora_parameter(name: str) -> bool:
    return (
        "._lora_adapter." in name
        or ("parametrizations.weight" in name and "parametrizations.weight.original" not in name)
    )


def inject_common_lora(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """Add a LoRA method's Attention/Router branches while preserving its FFN branch.

    The historical function name is retained for checkpoint compatibility; these
    adapters are method-local and are not a public/shared adapter across methods.
    """
    if not common_lora_is_enabled(finetuning_args):
        return

    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype
    target_layers = getattr(finetuning_args, "moe_lora_target_layers", "all")
    attach_attn = getattr(finetuning_args, "moe_lora_attach_attn", False)
    attach_router = getattr(finetuning_args, "moe_lora_attach_router", False)

    if attach_attn:
        layers_container = _locate_decoder_layers(model)
        if layers_container is None:
            raise ValueError("Cannot locate decoder layers for common attention LoRA injection.")
        target_indices = set(_resolve_target_layer_indices(target_layers, len(layers_container.layers)))
        _inject_attention_lora(model, finetuning_args, target_indices, device, dtype)

    if attach_router:
        all_moe_blocks = _find_moe_blocks(model)
        target_indices = set(_resolve_target_layer_indices(target_layers, len(all_moe_blocks)))
        _inject_router_lora(
            [block for index, block in enumerate(all_moe_blocks) if index in target_indices],
            finetuning_args,
            device,
            dtype,
        )

    for name, param in model.named_parameters():
        if is_common_lora_parameter(name):
            param.requires_grad = True

    logger.info_rank0(
        "Method-local Attention/Router LoRA enabled | "
        f"attention={attach_attn}, router={attach_router}, target_layers={target_layers}"
    )


def inject_moe_lora(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """主入口: 在所有 MoE block 旁注入 LoRA 分支。

    1. 收集所有 MoE block
    2. 按 target_layers 过滤
    3. 按 pool_share / w_share 创建 RoutingProjection + LoRAPool（per-layer 或 global）
    4. cast 到 base model 的 device + dtype
    5. monkey-patch forward
    6. 重置 requires_grad（base model 全 freeze，新模块全 trainable）
    7. 校验 trainable 比例
    """
    n_lora = finetuning_args.moe_lora_n_experts
    rank = finetuning_args.moe_lora_rank
    alpha = finetuning_args.moe_lora_alpha
    top_k = finetuning_args.moe_lora_top_k
    pool_share = finetuning_args.moe_lora_pool_share
    w_share = finetuning_args.moe_lora_w_share
    target_layers = finetuning_args.moe_lora_target_layers
    routing_mode = getattr(finetuning_args, "moe_lora_routing_mode", "learned")
    ffn_granularity = getattr(finetuning_args, "moe_lora_ffn_granularity", "bypass")
    if ffn_granularity not in ("bypass", "expert_weight", "weight_pool"):
        raise ValueError(
            "moe_lora_ffn_granularity must be 'bypass', 'expert_weight', or 'weight_pool', "
            f"got {ffn_granularity!r}."
        )
    b_init = getattr(finetuning_args, "moe_lora_b_init", "normal")
    if b_init not in ("normal", "zero"):
        raise ValueError(f"moe_lora_b_init must be 'normal' or 'zero', got {b_init!r}.")
    if b_init != "normal" and ffn_granularity != "bypass":
        raise ValueError(
            f"moe_lora_b_init={b_init!r} only applies to "
            "moe_lora_ffn_granularity='bypass'; it would be silently ignored otherwise."
        )
    wp_b_init = getattr(finetuning_args, "moe_lora_wp_b_init", "normal")
    if wp_b_init not in ("normal", "zero"):
        raise ValueError(f"moe_lora_wp_b_init must be 'normal' or 'zero', got {wp_b_init!r}.")
    if wp_b_init != "normal" and ffn_granularity != "weight_pool":
        raise ValueError(
            f"moe_lora_wp_b_init={wp_b_init!r} only applies to "
            "moe_lora_ffn_granularity='weight_pool'; it would be silently ignored otherwise."
        )
    wp_gate_up_stage = getattr(finetuning_args, "moe_lora_wp_gate_up_stage", "gate_up")
    if wp_gate_up_stage not in ("gate_up", "up"):
        raise ValueError(f"moe_lora_wp_gate_up_stage must be 'gate_up' or 'up', got {wp_gate_up_stage!r}.")
    if wp_gate_up_stage != "gate_up" and ffn_granularity != "weight_pool":
        raise ValueError(
            f"moe_lora_wp_gate_up_stage={wp_gate_up_stage!r} only applies to "
            "moe_lora_ffn_granularity='weight_pool'; it would be silently ignored otherwise."
        )

    # Position ablation 开关: 控制 LoRA 注入到 {attn, ffn, router} 哪些位置。
    # attach_ffn 默认 True → 与历史行为(纯 FFN expert pool baseline)完全一致。
    attach_ffn = getattr(finetuning_args, "moe_lora_attach_ffn", True)
    attach_attn = getattr(finetuning_args, "moe_lora_attach_attn", False)
    attach_router = getattr(finetuning_args, "moe_lora_attach_router", False)
    if not (attach_ffn or attach_attn or attach_router):
        raise ValueError(
            "MoE-LoRA position ablation: at least one of moe_lora_attach_ffn / "
            "moe_lora_attach_attn / moe_lora_attach_router must be True."
        )
    logger.info_rank0(
        f"MoE-LoRA position ablation: attach_ffn={attach_ffn}, "
        f"attach_attn={attach_attn}, attach_router={attach_router}"
    )

    # 取 base model 的 device 和 dtype（为 cast 用）
    sample_param = next(model.parameters())
    device, dtype = sample_param.device, sample_param.dtype

    d_model = _get_config_value(model.config, "hidden_size")
    if d_model is None:
        raise RuntimeError("Could not determine hidden_size from model.config (or text_config).")

    # transformers 4.x: num_experts; transformers 5.x: num_local_experts (Qwen3-MoE);
    # DeepSeek-V2 / V3: n_routed_experts (shared experts 不参与 routing,不算 N_E)
    n_experts_moe = (
        _get_config_value(model.config, "num_experts")
        or _get_config_value(model.config, "num_local_experts")
        or _get_config_value(model.config, "n_routed_experts")
    )
    if n_experts_moe is None:
        fields = list(vars(model.config).keys())[:30]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            fields += ["text_config." + k for k in list(vars(text_config).keys())[:30]]
        raise RuntimeError(
            "Could not determine N_E from config. Tried num_experts, num_local_experts, n_routed_experts "
            f"(in both top-level and text_config). Available config fields: {fields}"
        )

    # follow_moe 模式：复用原 MoE router，n_lora 必须等于 n_experts_moe，
    # top_k 通常等于原 MoE 的 top_k（OLMoE 是 num_experts_per_tok）。
    # 仅 FFN 分支(attach_ffn)才用到 n_lora/follow_moe 约束;attn/router-only 时跳过。
    if attach_ffn and routing_mode == "follow_moe":
        if n_lora != n_experts_moe:
            raise ValueError(
                f"routing_mode='follow_moe' requires moe_lora_n_experts == n_experts_moe ({n_experts_moe}), "
                f"got {n_lora}. Set moe_lora_n_experts={n_experts_moe} in your yaml."
            )
        # w_share 在 follow_moe 下没有意义（不创建 W），强制 per_layer 防误解
        w_share = "per_layer"

    # Hidden state bottleneck(可选,默认 0 = 关闭,vanilla V2)
    hidden_bottleneck_dim = getattr(finetuning_args, "moe_lora_hidden_bottleneck_dim", 0)
    hidden_bottleneck_gate_up_dim = getattr(
        finetuning_args, "moe_lora_hidden_bottleneck_gate_up_dim", None
    )
    hidden_bottleneck_down_dim = getattr(
        finetuning_args, "moe_lora_hidden_bottleneck_down_dim", None
    )
    separate_weight_routing = (
        hidden_bottleneck_gate_up_dim is not None or hidden_bottleneck_down_dim is not None
    )
    if separate_weight_routing:
        if hidden_bottleneck_gate_up_dim is None or hidden_bottleneck_down_dim is None:
            raise ValueError(
                "Separate weight_pool routing requires both "
                "moe_lora_hidden_bottleneck_gate_up_dim and moe_lora_hidden_bottleneck_down_dim."
            )
        if hidden_bottleneck_gate_up_dim < 0 or hidden_bottleneck_down_dim < 0:
            raise ValueError("Separate weight_pool hidden bottleneck dimensions must be non-negative.")
        if not attach_ffn or ffn_granularity != "weight_pool":
            raise ValueError("Separate gate_up/down routing is only supported for weight_pool FFN granularity.")
        if w_share != "per_layer":
            raise ValueError("Separate gate_up/down routing currently requires moe_lora_w_share='per_layer'.")
    normalize_topk_prob = getattr(finetuning_args, "moe_lora_normalize_topk_prob", False)

    wp_down_route_on_h = getattr(finetuning_args, "moe_lora_wp_down_route_on_h", False)
    if wp_down_route_on_h:
        if ffn_granularity != "weight_pool" or not attach_ffn:
            raise ValueError(
                "moe_lora_wp_down_route_on_h=True only applies to "
                "moe_lora_ffn_granularity='weight_pool'; it would be silently ignored otherwise."
            )
        if separate_weight_routing:
            raise ValueError(
                "moe_lora_wp_down_route_on_h is incompatible with the separate gate_up/down "
                "hidden-bottleneck routing (moe_lora_hidden_bottleneck_gate_up_dim/_down_dim); "
                "pick one down-routing scheme."
            )

    # Hierarchical/Grouped Pool(可选,默认 1 = 单一 global pool)
    n_groups = getattr(finetuning_args, "moe_lora_n_groups", 1)

    # DAS-informed initialization for RoutingProjection (可选,默认关闭)
    # 提供 das_qwen3_math.json / das_qwen3_code.json 等文件,按每层 DAS rank 把
    # base experts hard-partition 给 N_L 个 LoRA expert,防止训练初期 expert collapse。
    das_init_path = getattr(finetuning_args, "moe_lora_das_init_path", None)
    das_init_strength = getattr(finetuning_args, "moe_lora_das_init_strength", 1.0)
    wl_init = getattr(finetuning_args, "moe_lora_wl_init", "default")
    n_global = getattr(finetuning_args, "moe_lora_n_global", 0)
    das_per_layer_tensor = None
    if das_init_path:
        import json as _json
        with open(das_init_path) as f:
            das_data = _json.load(f)
        das_per_layer_tensor = torch.tensor(das_data["das_per_layer"], dtype=torch.float32)
        logger.info_rank0(
            f"Loaded DAS init from {das_init_path}: "
            f"shape={tuple(das_per_layer_tensor.shape)}, strength={das_init_strength}"
        )

    # 1. 收集 MoE block + 按 target_layers 过滤
    all_moe_blocks = _find_moe_blocks(model)
    target_indices = set(_resolve_target_layer_indices(target_layers, len(all_moe_blocks)))
    moe_blocks = [b for i, b in enumerate(all_moe_blocks) if i in target_indices]
    layer_ranks = _resolve_layer_ranks(
        getattr(finetuning_args, "moe_lora_layer_ranks", ""), rank, len(moe_blocks)
    )
    if len(set(layer_ranks)) > 1 and pool_share != "per_layer":
        raise ValueError("Non-uniform moe_lora_layer_ranks requires moe_lora_pool_share='per_layer'.")
    if ffn_granularity == "expert_weight" and attach_ffn:
        if routing_mode != "follow_moe":
            raise ValueError("moe_lora_ffn_granularity='expert_weight' requires moe_lora_routing_mode='follow_moe'.")
        if n_lora != n_experts_moe or top_k != _get_moe_top_k(model.config):
            raise ValueError(
                "moe_lora_ffn_granularity='expert_weight' reuses the base MoE dispatch and requires "
                f"moe_lora_n_experts={n_experts_moe} and moe_lora_top_k={_get_moe_top_k(model.config)}."
            )
        if pool_share != "per_layer" or w_share != "per_layer":
            raise ValueError("moe_lora_ffn_granularity='expert_weight' requires per-layer pool/w sharing.")
        if n_global > 0 or n_groups > 1:
            raise ValueError("expert_weight FFN granularity is incompatible with global/grouped LoRA pools.")
    if ffn_granularity == "weight_pool" and attach_ffn:
        if routing_mode != "learned":
            raise ValueError("moe_lora_ffn_granularity='weight_pool' requires moe_lora_routing_mode='learned'.")
        if n_global > 0:
            raise ValueError("moe_lora_ffn_granularity='weight_pool' does not support moe_lora_n_global > 0.")
        if getattr(finetuning_args, "moe_lora_balance_loss_coef", 0.0) > 0:
            raise ValueError("moe_lora_ffn_granularity='weight_pool' currently requires balance_loss_coef=0.")

    # n_groups 校验
    if n_groups > 1:
        if pool_share != "global":
            raise ValueError(f"n_groups={n_groups} requires pool_share='global', got '{pool_share}'.")
        if n_lora % n_groups != 0:
            raise ValueError(f"n_lora={n_lora} must be divisible by n_groups={n_groups}.")
        if len(moe_blocks) % n_groups != 0:
            logger.warning_rank0(
                f"layer_count={len(moe_blocks)} not divisible by n_groups={n_groups}; "
                "groups will be slightly unbalanced."
            )

    n_lora_per_group = n_lora // n_groups  # = n_lora when n_groups=1

    if separate_weight_routing:
        bottleneck_str = (
            f", hidden_bottleneck_gate_up={hidden_bottleneck_gate_up_dim}"
            f", hidden_bottleneck_down={hidden_bottleneck_down_dim}"
        )
    else:
        bottleneck_str = f", hidden_bottleneck={hidden_bottleneck_dim}" if hidden_bottleneck_dim > 0 else ""
    groups_str = f", n_groups={n_groups} (n_lora_per_group={n_lora_per_group})" if n_groups > 1 else ""
    normalize_str = ", normalize_topk_prob=true" if normalize_topk_prob else ""
    if attach_ffn:
        logger.info_rank0(
            f"MoE-LoRA[ffn] injecting into {len(moe_blocks)}/{len(all_moe_blocks)} MoE layers | "
            f"N_E={n_experts_moe} -> N_L={n_lora} (rank={rank}, alpha={alpha}, top_k={top_k}) | "
            f"granularity={ffn_granularity}, pool_share={pool_share}, w_share={w_share}, "
            f"routing_mode={routing_mode}{bottleneck_str}{groups_str}{normalize_str}"
        )

    # 2. 全局共享时在 model 顶层挂 pool(支持 grouped pool: n_groups > 1 时挂多个)
    shared_pool = None         # n_groups=1 时使用
    grouped_pools = None       # n_groups>1 时使用,长度 n_groups
    shared_weight_pool = None
    grouped_weight_pools = None
    shared_proj = None
    shared_indep_router = None
    shared_residual_router = None

    if attach_ffn and ffn_granularity == "bypass" and pool_share == "global":
        if n_groups == 1:
            # 单一 global pool(原始行为)
            shared_pool = LoRAPool(
                n_lora,
                d_model,
                rank,
                alpha,
                top_k,
                normalize_topk_prob=normalize_topk_prob,
                b_init=b_init,
            ).to(device=device, dtype=dtype)
            model.add_module("global_lora_pool", shared_pool)
        else:
            # Grouped Pool: n_groups 个独立 LoRAPool,每个有 n_lora_per_group 个 expert
            grouped_pools = []
            for g in range(n_groups):
                pool = LoRAPool(
                    n_lora_per_group,
                    d_model,
                    rank,
                    alpha,
                    top_k,
                    normalize_topk_prob=normalize_topk_prob,
                    b_init=b_init,
                ).to(device=device, dtype=dtype)
                model.add_module(f"global_lora_pool_g{g}", pool)
                grouped_pools.append(pool)

    if attach_ffn and ffn_granularity == "weight_pool" and pool_share == "global":
        from .weight_peft import WeightLoRAPool, _find_experts_module

        if not moe_blocks:
            raise ValueError("weight_pool FFN granularity found no selected MoE layers.")

        def _new_weight_pool(sample_block, pool_experts):
            experts = _find_experts_module(sample_block)
            if experts is None:
                raise ValueError(
                    "weight_pool FFN granularity requires fused gate_up_proj/down_proj expert weights."
                )
            return WeightLoRAPool(
                experts.gate_up_proj,
                experts.down_proj,
                pool_experts,
                rank,
                alpha,
                top_k,
                normalize_topk_prob=normalize_topk_prob,
                is_transposed=getattr(experts, "is_transposed", False),
                b_init=wp_b_init,
                gate_up_stage=wp_gate_up_stage,
            ).to(device=device, dtype=dtype)

        if n_groups == 1:
            shared_weight_pool = _new_weight_pool(moe_blocks[0], n_lora)
            model.add_module("global_weight_lora_pool", shared_weight_pool)
        else:
            grouped_weight_pools = []
            for group_index in range(n_groups):
                sample_index = group_index * len(moe_blocks) // n_groups
                pool = _new_weight_pool(moe_blocks[sample_index], n_lora_per_group)
                model.add_module(f"global_weight_lora_pool_g{group_index}", pool)
                grouped_weight_pools.append(pool)

    # 2c. (可选) cross-layer global pool: 跨层共享一个 routed LoRA pool,
    # 跟 per-layer pool 联合 joint top-K 竞争(joint 模式不算 balance loss).
    # global pool 的 rank / alpha 可以跟 per-layer pool 不同 (通过 moe_lora_global_rank/alpha 设置)
    cross_layer_pool = None
    if attach_ffn and ffn_granularity == "bypass" and n_global > 0:
        if pool_share != "per_layer":
            raise ValueError(
                f"moe_lora_n_global={n_global} requires moe_lora_pool_share='per_layer' "
                f"(got '{pool_share}'). Cross-layer pool is meant to complement per-layer pool, "
                f"not stack on top of an existing global pool."
            )
        if routing_mode != "learned":
            raise ValueError(
                f"moe_lora_n_global={n_global} requires moe_lora_routing_mode='learned' "
                f"(got '{routing_mode}'). Joint top-K requires W_l to output joint logits."
            )
        if n_groups != 1:
            raise ValueError(f"moe_lora_n_global={n_global} requires n_groups=1, got {n_groups}.")
        global_rank = getattr(finetuning_args, "moe_lora_global_rank", 0) or rank
        global_alpha = getattr(finetuning_args, "moe_lora_global_alpha", 0) or alpha
        cross_layer_pool = LoRAPool(
            n_global,
            d_model,
            global_rank,
            global_alpha,
            top_k,
            normalize_topk_prob=normalize_topk_prob,
            b_init=b_init,
        ).to(device=device, dtype=dtype)
        model.add_module("cross_layer_lora_pool", cross_layer_pool)
        logger.info_rank0(
            f"MoE-LoRA: cross-layer global pool enabled with n_global={n_global} "
            f"(rank={global_rank}, alpha={global_alpha}). Per-layer W_l will output (N_local={n_lora}+N_global={n_global}) logits, "
            f"joint top-K={top_k} competes across both pools. Joint mode disables balance loss."
        )

    if (
        attach_ffn
        and ffn_granularity in ("bypass", "weight_pool")
        and w_share == "global"
        and routing_mode == "learned"
    ):
        # global w_share 不支持 grouped(每层 W_l 必须独立才能路由到自己组)
        if n_groups > 1:
            raise ValueError("w_share='global' is incompatible with n_groups>1; use w_share='per_layer'.")
        if das_per_layer_tensor is not None:
            logger.warning_rank0(
                "DAS init is set but w_share='global'; per-layer DAS cannot be applied to "
                "a shared W_l. Falling back to using layer-0 DAS as init prior."
            )
            global_das_init = das_per_layer_tensor[0]
        else:
            global_das_init = None
        shared_proj = RoutingProjection(
            n_experts_moe, n_lora, d_model=d_model, hidden_bottleneck_dim=hidden_bottleneck_dim,
            das_init=global_das_init, das_init_strength=das_init_strength,
            wl_init=wl_init,
        ).to(device=device, dtype=dtype)
        model.add_module("global_routing_projection", shared_proj)
    if attach_ffn and ffn_granularity == "bypass" and w_share == "global" and routing_mode == "independent":
        shared_indep_router = IndependentRouter(d_model, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_independent_router", shared_indep_router)
    if attach_ffn and ffn_granularity == "bypass" and w_share == "global" and routing_mode == "residual":
        shared_residual_router = ResidualRouter(n_experts_moe, d_model, n_lora).to(device=device, dtype=dtype)
        model.add_module("global_residual_router", shared_residual_router)

    # 3. 给每个 MoE block 挂 FFN LoRA 模块 + monkey-patch forward
    #    attach_ffn=False 时跳过整段(attn/router-only ablation)。
    if attach_ffn and ffn_granularity == "expert_weight":
        from .weight_peft import inject_expert_weight_lora

        inject_expert_weight_lora(model, moe_blocks, layer_ranks, rank, alpha)

    if attach_ffn and ffn_granularity == "weight_pool":
        from .weight_peft import WeightLoRAPool, _find_experts_module, _wp_experts_forward

        for layer_idx, moe_block in enumerate(moe_blocks):
            experts = _find_experts_module(moe_block)
            if experts is None:
                raise ValueError(
                    f"weight_pool FFN granularity could not find fused gate_up_proj/down_proj "
                    f"parameters in selected MoE layer {layer_idx} ({type(moe_block).__name__})."
                )
            group_idx = layer_idx * n_groups // len(moe_blocks)

            if separate_weight_routing:
                layer_das = (
                    das_per_layer_tensor[layer_idx]
                    if das_per_layer_tensor is not None and layer_idx < das_per_layer_tensor.shape[0]
                    else None
                )
                experts.routing_projection_gate_up = RoutingProjection(
                    n_experts_moe,
                    n_lora_per_group,
                    d_model=d_model,
                    hidden_bottleneck_dim=hidden_bottleneck_gate_up_dim,
                    das_init=layer_das,
                    das_init_strength=das_init_strength,
                    wl_init=wl_init,
                ).to(device=device, dtype=dtype)
                experts.routing_projection_down = RoutingProjection(
                    n_experts_moe,
                    n_lora_per_group,
                    d_model=d_model,
                    hidden_bottleneck_dim=hidden_bottleneck_down_dim,
                    das_init=layer_das,
                    das_init_strength=das_init_strength,
                    wl_init=wl_init,
                ).to(device=device, dtype=dtype)
            elif w_share == "per_layer":
                layer_das = (
                    das_per_layer_tensor[layer_idx]
                    if das_per_layer_tensor is not None and layer_idx < das_per_layer_tensor.shape[0]
                    else None
                )
                experts.routing_projection = RoutingProjection(
                    n_experts_moe,
                    n_lora_per_group,
                    d_model=d_model,
                    hidden_bottleneck_dim=hidden_bottleneck_dim,
                    das_init=layer_das,
                    das_init_strength=das_init_strength,
                    wl_init=wl_init,
                ).to(device=device, dtype=dtype)
            else:
                object.__setattr__(experts, "routing_projection", shared_proj)

            if pool_share == "per_layer":
                layer_rank = layer_ranks[layer_idx]
                layer_alpha = _scaled_alpha(rank, alpha, layer_rank)
                experts._weight_lora_pool = WeightLoRAPool(
                    experts.gate_up_proj,
                    experts.down_proj,
                    n_lora,
                    layer_rank,
                    layer_alpha,
                    top_k,
                    normalize_topk_prob=normalize_topk_prob,
                    is_transposed=getattr(experts, "is_transposed", False),
                    b_init=wp_b_init,
                    gate_up_stage=wp_gate_up_stage,
                ).to(device=device, dtype=dtype)
            elif n_groups > 1:
                object.__setattr__(experts, "_weight_lora_pool", grouped_weight_pools[group_idx])
            else:
                object.__setattr__(experts, "_weight_lora_pool", shared_weight_pool)

            if wp_down_route_on_h:
                # Route the down adapter from the intermediate activation it acts on
                # (h_in = act_fn(gate)*up, width = moe_intermediate_size) rather than
                # the block input. IndependentRouter maps that activation straight to
                # the pool's per-group logits. gate_up keeps the shared input route.
                pool = experts._weight_lora_pool
                intermediate_dim = pool.down.in_features
                experts.routing_projection_down_on_h = IndependentRouter(
                    intermediate_dim, pool.n_experts
                ).to(device=device, dtype=dtype)

            experts._finetuning_args = finetuning_args
            experts._moe_lora_cached_router_logits = None
            _install_router_logits_cache(moe_block, experts)
            experts._wp_orig_forward = experts.forward
            experts.forward = _wp_experts_forward.__get__(experts)

    for layer_idx, moe_block in enumerate(moe_blocks if attach_ffn and ffn_granularity == "bypass" else []):
        adapter_host = _get_expert_adapter_host(moe_block)
        # 计算该层属于哪个 group(n_groups=1 时永远是 0)
        group_idx = layer_idx * n_groups // len(moe_blocks)

        # RoutingProjection / IndependentRouter (follow_moe 模式下不创建任何 router)
        # 注意: grouped pool 时 W_l 输出维度是 n_lora_per_group(而非 n_lora)
        # 注意: n_global > 0 时 W_l 输出 (n_lora_per_group + n_global), joint top-K
        proj_out_dim = n_lora_per_group + n_global  # n_global=0 时退化为原行为

        if routing_mode == "learned":
            if w_share == "per_layer":
                # 取该层对应的 DAS 信号(per-layer init prior)
                layer_das = (
                    das_per_layer_tensor[layer_idx]
                    if das_per_layer_tensor is not None and layer_idx < das_per_layer_tensor.shape[0]
                    else None
                )
                proj = RoutingProjection(
                    n_experts_moe, proj_out_dim, d_model=d_model, hidden_bottleneck_dim=hidden_bottleneck_dim,
                    das_init=layer_das, das_init_strength=das_init_strength,
                    wl_init=wl_init,
                ).to(device=device, dtype=dtype)
                adapter_host.routing_projection = proj
            else:
                object.__setattr__(adapter_host, "routing_projection", shared_proj)
        elif routing_mode == "independent":
            if w_share == "per_layer":
                indep_router = IndependentRouter(d_model, proj_out_dim).to(device=device, dtype=dtype)
                adapter_host.independent_router = indep_router
            else:
                object.__setattr__(adapter_host, "independent_router", shared_indep_router)
        elif routing_mode == "residual":
            if w_share == "per_layer":
                res_router = ResidualRouter(n_experts_moe, d_model, proj_out_dim).to(device=device, dtype=dtype)
                adapter_host.residual_router = res_router
            else:
                object.__setattr__(adapter_host, "residual_router", shared_residual_router)

        # LoRAPool
        if pool_share == "per_layer":
            layer_rank = layer_ranks[layer_idx]
            layer_alpha = _scaled_alpha(rank, alpha, layer_rank)
            pool = LoRAPool(
                n_lora,
                d_model,
                layer_rank,
                layer_alpha,
                top_k,
                normalize_topk_prob=normalize_topk_prob,
                b_init=b_init,
            ).to(device=device, dtype=dtype)
            adapter_host.lora_pool = pool
        elif n_groups > 1:
            # Grouped pool: 该层指向所属 group 的 pool
            object.__setattr__(adapter_host, "lora_pool", grouped_pools[group_idx])
        else:
            # 单一 global pool
            object.__setattr__(adapter_host, "lora_pool", shared_pool)

        # (可选) cross-layer global pool: 每层 reference 同一个 model 级 pool
        if cross_layer_pool is not None:
            object.__setattr__(adapter_host, "cross_layer_lora_pool", cross_layer_pool)

        # monkey-patch forward
        adapter_host._original_forward = adapter_host.forward
        adapter_host._finetuning_args = finetuning_args
        if _uses_experts_host(moe_block):
            adapter_host._moe_lora_n_experts_moe = n_experts_moe
            adapter_host._moe_lora_cached_router_logits = None
            if routing_mode in ("learned", "residual"):
                _install_router_logits_cache(moe_block, adapter_host)
            adapter_host.forward = patched_experts_moe_forward.__get__(adapter_host)
        else:
            adapter_host.forward = patched_moe_forward.__get__(adapter_host)

    # 3b. Attention LoRA(标准 LoRA 挂在 self_attn 的 *_proj 上,按 target_layers 选层)
    if attach_attn:
        layers_container = _locate_decoder_layers(model)
        if layers_container is None:
            raise ValueError("Cannot locate decoder layers for attention LoRA injection.")
        n_decoder_layers = len(layers_container.layers)
        attn_target_indices = set(_resolve_target_layer_indices(target_layers, n_decoder_layers))
        _inject_attention_lora(model, finetuning_args, attn_target_indices, device, dtype)

    # 3c. Router LoRA(在已过滤的 MoE block 的 gate.weight 上注册低秩增量)
    if attach_router:
        _inject_router_lora(moe_blocks, finetuning_args, device, dtype)

    # 4. 重置 requires_grad（base model 全 freeze，注入的模块全 trainable）
    #    新增 position-ablation 标记: '_lora_adapter'(attn) / 'parametrizations.weight'(router gate)
    def _is_trainable_lora(name: str) -> bool:
        return (
            "routing_projection" in name
            or "lora_pool" in name
            or "independent_router" in name
            or "residual_router" in name
            or "_lora_adapter" in name                 # attention LoRA
            or "parametrizations.weight" in name        # router gate LoRA delta (A/B)
            or "._wp_adapter." in name                  # fused expert-weight LoRA
        )

    for name, p in model.named_parameters():
        if _is_trainable_lora(name) and "parametrizations.weight.original" not in name:
            # router 的 base gate 权重在 parametrize 后变成 ...parametrizations.weight.original,
            # 必须保持 frozen;只训练 parametrization 内的 A/B。
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 5. 校验 trainable 比例(兼容 DeepSpeed ZeRO-3:用 ds_numel 而非 numel)
    def _real_numel(p):
        # ZeRO-3 sharded params 使用 ds_numel 拿到全量,否则 numel() 只能拿到本地 shard
        return getattr(p, "ds_numel", None) or p.numel()
    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    ratio = trainable / total if total > 0 else 0.0
    if not (0.00001 < ratio < 0.10):
        # ZeRO-3 早期阶段 base model 可能还没 materialize,降级为 warning
        logger.warning(
            f"Trainable ratio {ratio:.5f} out of expected range [1e-5, 0.1]. "
            f"trainable={trainable}, total={total}. "
            "Under ZeRO-3 this may be due to params not yet materialized; verify after first step."
        )

    # 6. 注册 balance loss hook（coef > 0 时生效；schedule=linear_decay 线性衰减到 0，constant 全程恒定）
    balance_coef = getattr(finetuning_args, "moe_lora_balance_loss_coef", 0.0)
    if balance_coef > 0:
        # expert_weight 粒度没有 LoRAPool（bypass 的 LBL 载体），改在 base router 上挂
        # 逐层 LBL hook：f_i 用 router 自己的 top-k 硬派发频率(no grad)，P_i 用稠密
        # softmax 均值（梯度经 parametrize 的 router LoRA delta + hidden states 回传）。
        if attach_ffn and ffn_granularity == "expert_weight":
            _install_expert_weight_balance_hooks(moe_blocks)
        balance_schedule = getattr(finetuning_args, "moe_lora_balance_loss_schedule", "linear_decay")
        model._balance_loss_state = {"step": 0, "total_steps": 1}

        def _add_balance_loss_hook(module, input, output):
            if not module.training or not hasattr(output, "loss") or output.loss is None:
                return output
            # 注意: 必须是 0 维标量。zeros(1) 会把 loss 广播成 [1],
            # transformers>=5.x 的 trainer 里 `_tr_loss += tr_loss_step` 会报
            # "output with shape [] doesn't match the broadcast shape [1]"。
            aux = torch.zeros((), device=output.loss.device, dtype=output.loss.dtype)
            n_pools = 0
            for m in module.modules():
                if _is_moe_lora_pool(m) and getattr(m, "_aux_loss", None) is not None:
                    aux = aux + m._aux_loss.to(aux.dtype)
                    m._aux_loss = None
                    n_pools += 1
                elif getattr(m, "_moe_lora_aux_loss", None) is not None:
                    # expert_weight 粒度: router-side LBL(见 _expert_weight_router_aux_hook)
                    aux = aux + m._moe_lora_aux_loss.to(aux.dtype)
                    m._moe_lora_aux_loss = None
                    n_pools += 1
            # 逐层(pool)均值归一化,与 HF load_balancing_loss_func 的口径一致。
            # 若按 48 层求和,aux 量级 ~O(100),coef=1e-3 时 balance 项即达 LM loss
            # 的 ~6 倍(实测把 intent 域模型的输入条件化能力直接训没)。归一化后
            # coef 与文献常用值可比(Mixtral 2e-2 / OLMoE 1e-2 / Qwen 1e-3)。
            if n_pools > 0:
                aux = aux / n_pools
            if balance_schedule == "constant":
                effective_coef = balance_coef
            else:  # linear_decay（历史行为）
                state = module._balance_loss_state
                progress = state["step"] / max(state["total_steps"], 1)
                effective_coef = balance_coef * (1.0 - progress)
            # 防静默失效: reentrant gradient checkpointing 会在 no_grad 下重跑
            # decoder 层 forward,旁路张量 _aux_loss 因此没有梯度 —— balance loss
            # 只加数值不进梯度,消融全部无效。这里 fail fast 而不是浪费整次训练。
            if n_pools > 0 and torch.is_grad_enabled() and not aux.requires_grad:
                raise RuntimeError(
                    "MoE-LoRA balance loss has no gradient (aux.requires_grad=False). "
                    "This usually means reentrant gradient checkpointing severed the "
                    "side-channel graph. Set `use_reentrant_gc: false` (and ensure it "
                    "is not overridden by Trainer re-enabling GC) or disable "
                    "gradient checkpointing."
                )
            # hook 会消费掉各 pool 的 _aux_loss,在 model 上存一份(已是逐层均值)供 MoELoRAStatsCallback 记录
            module._last_balance_loss = aux.detach()
            output.loss = output.loss + effective_coef * aux
            return output

        model.register_forward_hook(_add_balance_loss_hook)
        logger.info_rank0(
            f"MoE-LoRA balance loss enabled (coef={balance_coef}, schedule={balance_schedule})"
        )


# ---------------------------------------------------------------------------
# 保存 / 加载
# ---------------------------------------------------------------------------


def save_moe_lora_state(
    model: nn.Module,
    save_dir: str,
    finetuning_args: "FinetuningArguments",
) -> None:
    """只保存 trainable 参数 + 配置到 save_dir。

    产出:
        moe_lora_state.safetensors   仅 trainable 权重
        moe_lora_config.json         重建结构所需的配置
    """
    os.makedirs(save_dir, exist_ok=True)

    # ZeRO-3 兼容: 在 save 前 gather 所有 trainable params 的全量副本
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import deepspeed
        gathered_ctx = deepspeed.zero.GatheredParameters(trainable_params, modifier_rank=0)
    except Exception:
        gathered_ctx = None

    # 1. 提取所有 trainable 参数(在 gather 上下文内)
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
        logger.warning_rank0("No trainable params found, skip saving moe_lora state.")
        return

    # 只 rank 0 写入磁盘,避免多 rank 同时写同一文件
    is_rank0 = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if not is_rank0:
        return
    safetensors.torch.save_file(state, os.path.join(save_dir, "moe_lora_state.safetensors"))

    # 2. 保存配置(重建模型结构必需)
    config = {
        "moe_lora_n_experts": finetuning_args.moe_lora_n_experts,
        "moe_lora_rank": finetuning_args.moe_lora_rank,
        "moe_lora_alpha": finetuning_args.moe_lora_alpha,
        "moe_lora_top_k": finetuning_args.moe_lora_top_k,
        "moe_lora_normalize_topk_prob": getattr(finetuning_args, "moe_lora_normalize_topk_prob", False),
        "moe_lora_pool_share": finetuning_args.moe_lora_pool_share,
        "moe_lora_w_share": finetuning_args.moe_lora_w_share,
        "moe_lora_target_layers": finetuning_args.moe_lora_target_layers,
        "moe_lora_ffn_granularity": getattr(finetuning_args, "moe_lora_ffn_granularity", "bypass"),
        "moe_lora_b_init": getattr(finetuning_args, "moe_lora_b_init", "normal"),
        "moe_lora_wp_b_init": getattr(finetuning_args, "moe_lora_wp_b_init", "normal"),
        "moe_lora_wp_gate_up_stage": getattr(finetuning_args, "moe_lora_wp_gate_up_stage", "gate_up"),
        "moe_lora_wp_down_route_on_h": getattr(finetuning_args, "moe_lora_wp_down_route_on_h", False),
        "moe_lora_layer_ranks": getattr(finetuning_args, "moe_lora_layer_ranks", ""),
        "moe_lora_detach_p_e": finetuning_args.moe_lora_detach_p_e,
        "moe_lora_routing_mode": getattr(finetuning_args, "moe_lora_routing_mode", "learned"),
        "moe_lora_balance_loss_coef": getattr(finetuning_args, "moe_lora_balance_loss_coef", 0.0),
        "moe_lora_balance_loss_schedule": getattr(finetuning_args, "moe_lora_balance_loss_schedule", "linear_decay"),
        "moe_lora_hidden_bottleneck_dim": getattr(finetuning_args, "moe_lora_hidden_bottleneck_dim", 0),
        "moe_lora_hidden_bottleneck_gate_up_dim": getattr(
            finetuning_args, "moe_lora_hidden_bottleneck_gate_up_dim", None
        ),
        "moe_lora_hidden_bottleneck_down_dim": getattr(
            finetuning_args, "moe_lora_hidden_bottleneck_down_dim", None
        ),
        "moe_lora_n_groups": getattr(finetuning_args, "moe_lora_n_groups", 1),
        "moe_lora_wl_init": getattr(finetuning_args, "moe_lora_wl_init", "default"),
        "moe_lora_n_global": getattr(finetuning_args, "moe_lora_n_global", 0),
        "moe_lora_global_rank": getattr(finetuning_args, "moe_lora_global_rank", 0),
        "moe_lora_global_alpha": getattr(finetuning_args, "moe_lora_global_alpha", 0),
        "moe_lora_top_k_local": getattr(finetuning_args, "moe_lora_top_k_local", 0),
        "moe_lora_top_k_global": getattr(finetuning_args, "moe_lora_top_k_global", 0),
        # position ablation 开关(重建结构必需)
        "moe_lora_attach_ffn": getattr(finetuning_args, "moe_lora_attach_ffn", True),
        "moe_lora_attach_attn": getattr(finetuning_args, "moe_lora_attach_attn", False),
        "moe_lora_attach_router": getattr(finetuning_args, "moe_lora_attach_router", False),
        "moe_lora_attn_targets": getattr(finetuning_args, "moe_lora_attn_targets", "auto"),
        "moe_lora_attn_rank": getattr(finetuning_args, "moe_lora_attn_rank", 0),
        "moe_lora_attn_alpha": getattr(finetuning_args, "moe_lora_attn_alpha", 0),
        "moe_lora_router_rank": getattr(finetuning_args, "moe_lora_router_rank", 0),
        "moe_lora_router_alpha": getattr(finetuning_args, "moe_lora_router_alpha", 0),
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
    }
    with open(os.path.join(save_dir, "moe_lora_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info_rank0(
        f"Saved MoE-LoRA state ({len(state)} tensors) to {save_dir}"
    )

    # 3. 保存全生命周期激活统计（用于训练后剪枝）
    pools = [(n, m) for n, m in model.named_modules() if _is_moe_lora_pool(m)]
    if pools:
        lifetime_stats = {}
        for name, pool in pools:
            counts = pool.lifetime_activation_count.cpu().tolist()
            dead = [i for i, c in enumerate(counts) if c == 0]
            lifetime_stats[name] = {
                "activation_count": counts,
                "dead_expert_indices": dead,
                "n_dead": len(dead),
                "n_total": len(counts),
            }
        with open(os.path.join(save_dir, "moe_lora_lifetime_stats.json"), "w") as f:
            json.dump(lifetime_stats, f, indent=2)
        total_dead = sum(s["n_dead"] for s in lifetime_stats.values())
        total_experts = sum(s["n_total"] for s in lifetime_stats.values())
        logger.info_rank0(
            f"Lifetime dead experts: {total_dead}/{total_experts}"
        )


def load_moe_lora_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    """先重建结构（注入），再加载权重。用于评估 / 续训。"""
    from ...hparams import FinetuningArguments

    config_path = os.path.join(load_dir, "moe_lora_config.json")
    state_path = os.path.join(load_dir, "moe_lora_state.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing moe_lora_config.json or moe_lora_state.safetensors in {load_dir}"
        )

    # 1. 读配置
    with open(config_path) as f:
        config = json.load(f)

    # 2. pop 元信息字段
    meta = {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}

    # 3. 用配置重建结构
    finetuning_args = FinetuningArguments(finetuning_type="moe_lora", **config)
    inject_moe_lora(model, finetuning_args)

    # 4. 加载权重
    state = safetensors.torch.load_file(state_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # 5. 双向校验（防 silent fail）
    # 注意: router gate parametrize 后,base 权重变成 ...parametrizations.weight.original,
    # 它是 frozen base(不在 checkpoint 里),必然出现在 missing —— 不应算作未加载的 LoRA。
    assert len(unexpected) == 0, f"Unexpected weights in checkpoint: {unexpected[:5]}"
    _lora_markers = (
        "routing_projection", "lora_pool", "independent_router",
        "residual_router", "_lora_adapter", "parametrizations.weight", "._wp_adapter.",
        "routing_projection_down_on_h",
    )
    unloaded_lora = [
        n for n in missing
        if any(m in n for m in _lora_markers) and "parametrizations.weight.original" not in n
    ]
    assert not unloaded_lora, (
        f"Failed to load LoRA params (naming mismatch?): {unloaded_lora[:5]}"
    )

    # 6. base model 一致性（warning, 不阻断）
    base_model_meta = meta.get("_meta_base_model")
    actual_base = getattr(model.config, "name_or_path", "unknown")
    if base_model_meta and base_model_meta != actual_base:
        logger.warning_rank0(
            f"Base model mismatch: loaded {actual_base} but checkpoint was trained on "
            f"{base_model_meta}. Behaviour may be unexpected."
        )

    logger.info_rank0(f"Loaded MoE-LoRA state from {load_dir} ({len(state)} tensors)")
    return model


# ---------------------------------------------------------------------------
# Trainer Callbacks
# ---------------------------------------------------------------------------


class MoELoRABalanceDecayCallback(TrainerCallback):
    """同步训练进度到 model._balance_loss_state，供 balance loss hook 做线性衰减。"""

    def on_step_begin(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is not None and hasattr(model, "_balance_loss_state"):
            model._balance_loss_state["step"] = state.global_step
            model._balance_loss_state["total_steps"] = state.max_steps


class MoELoRASaveCallback(TrainerCallback):
    """每次 trainer 触发 checkpoint 保存时，同步保存 MoE-LoRA 自定义状态。

    注意: trainer 默认 checkpoint 路径是 {output_dir}/checkpoint-{global_step}，
    这个 callback 跟随该约定。如果改了 save_strategy 自定义路径，需要相应调整。
    """

    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_moe_lora_state(model, ckpt_dir, self.finetuning_args)
        # 删除 Trainer 自动保存的完整基座模型,只保留 LoRA 状态
        import glob
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                    logger.info_rank0(f"Removed redundant base model file: {f}")
                except FileNotFoundError:
                    # 多进程可能已被其他 rank 删除
                    pass


class MoELoRAStatsCallback(TrainerCallback):
    """收集 MoE-LoRA 监控指标（每 logging_steps 触发），写入 trainer log
    （Trainer 会自动转发到 wandb / tensorboard / mlflow）。

    监控指标:
        moe_lora/{block}/activation_max         expert 最大激活频率
        moe_lora/{block}/activation_min         expert 最小激活频率
        moe_lora/{block}/dead_experts           频率 < 1e-4 的 expert 数
        moe_lora/{block}/activation_imbalance   max_freq / ideal_uniform 比值
        moe_lora/{block}/W_fro_norm             投影矩阵 W 的 Frobenius 范数
        moe_lora/{block}/p_L_entropy            p_L 分布熵（按 batch 平均）
        moe_lora/{*}_avg                         以上指标的全局平均
    """

    def on_log(self, args, state, control, model: Optional[nn.Module] = None,
               logs: Optional[dict] = None, **kwargs):
        if logs is None or model is None:
            return

        pools = [(n, m) for n, m in model.named_modules() if _is_moe_lora_pool(m)]
        projs = [(n, m) for n, m in model.named_modules() if isinstance(m, RoutingProjection)]
        indep_routers = [(n, m) for n, m in model.named_modules() if isinstance(m, IndependentRouter)]

        # ① 激活频率
        # 跨 rank 聚合 activation_count / total_tokens,否则每个 rank 只看到 1/N batch,
        # 真实非 dead 的 expert 可能在本 rank 上看起来 dead → dead_experts 严重高估
        try:
            import torch.distributed as dist
            _dist_ok = dist.is_available() and dist.is_initialized()
        except Exception:
            _dist_ok = False

        for name, pool in pools:
            if _dist_ok:
                # all_reduce sum 到所有 rank(in-place),拿到全局真实计数
                # 用 clone 避免改 buffer 本身(后续还要 zero_)
                act_global = pool.activation_count.clone().to(torch.long)
                tot_global = pool.total_tokens.clone().to(torch.long)
                dist.all_reduce(act_global, op=dist.ReduceOp.SUM)
                dist.all_reduce(tot_global, op=dist.ReduceOp.SUM)
            else:
                act_global = pool.activation_count
                tot_global = pool.total_tokens

            if tot_global.item() == 0:
                continue
            freq = act_global.float() / tot_global.float()
            ideal = pool.top_k / pool.n_experts
            logs[f"moe_lora/{name}/activation_max"] = freq.max().item()
            logs[f"moe_lora/{name}/activation_min"] = freq.min().item()
            logs[f"moe_lora/{name}/dead_experts"] = (freq < 1e-4).sum().item()
            logs[f"moe_lora/{name}/activation_imbalance"] = (freq.max() / (ideal + 1e-8)).item()
            # wandb histogram：每个 expert 的激活频率分布
            try:
                import wandb
                if wandb.run is not None:
                    # wandb.Histogram 在所有值相同时(早期训练 freq 全 0)会崩
                    # → 用 try 兜底,数据无效就跳过 histogram
                    try:
                        wandb.log({
                            f"moe_lora/{name}/activation_freq": wandb.Histogram(freq.cpu().numpy()),
                            f"moe_lora/{name}/lifetime_activation": wandb.Histogram(
                                pool.lifetime_activation_count.float().cpu().numpy()
                            ),
                        }, commit=False)
                    except (ValueError, RuntimeError):
                        pass
            except ImportError:
                pass
            pool.activation_count.zero_()
            pool.total_tokens.zero_()

        # ② W 的 Frobenius 范数
        w_norms = []
        for name, proj in projs:
            w_norm = proj.proj.weight.detach().float().norm().item()
            logs[f"moe_lora/{name}/W_fro_norm"] = w_norm
            w_norms.append(w_norm)
        if w_norms:
            logs["moe_lora/W_fro_norm_avg"] = sum(w_norms) / len(w_norms)

        # ③ p_L entropy（直接读 RoutingProjection / IndependentRouter 内置 buffer）
        entropies = []
        for name, proj in projs:
            if proj.entropy_count.item() == 0:
                continue
            avg_ent = (proj.entropy_sum / proj.entropy_count).item()
            logs[f"moe_lora/{name}/p_L_entropy"] = avg_ent
            entropies.append(avg_ent)
            proj.entropy_sum.zero_()
            proj.entropy_count.zero_()
        for name, router in indep_routers:
            if router.entropy_count.item() == 0:
                continue
            avg_ent = (router.entropy_sum / router.entropy_count).item()
            logs[f"moe_lora/{name}/p_L_entropy"] = avg_ent
            entropies.append(avg_ent)
            router.entropy_sum.zero_()
            router.entropy_count.zero_()
        if entropies:
            logs["moe_lora/p_L_entropy_avg"] = sum(entropies) / len(entropies)

        # ④ balance loss（读取最近一次 forward 的值）
        aux_losses = []
        for name, pool in pools:
            if getattr(pool, "_aux_loss", None) is not None:
                aux_losses.append(pool._aux_loss.item())
        if aux_losses:
            logs["moe_lora/balance_loss"] = sum(aux_losses) / len(aux_losses)
        elif getattr(model, "_last_balance_loss", None) is not None:
            # coef > 0 时 balance loss hook 已消费掉各 pool 的 _aux_loss,
            # 读取 hook 存下的均值,保证消融实验中该指标仍然可见
            logs["moe_lora/balance_loss"] = model._last_balance_loss.item()
