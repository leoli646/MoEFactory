"""Weight-granularity PEFT adapters (LoRA/DoRA/OFT/FourierFT/IA3) for MoE models.

Motivation
----------
On Qwen3-MoE the experts are *fused* 3D ``nn.Parameter`` (``experts.gate_up_proj``
``[E, 2*inter, hidden]`` and ``experts.down_proj`` ``[E, hidden, inter]``, both in
``[E, out, in]`` ``F.linear`` orientation). GPT-OSS stores the same logical expert
weights transposed as ``[E, in, out]``. PEFT cannot cleanly adapt these
(``ParamWrapper`` wraps only one param per module; ``use_dora`` raises; OFT has no
``target_parameters``), so we implement canonical weight-level adapters directly.

Implementation note (important)
-------------------------------
We do NOT replace the fused expert weight via ``torch.nn.utils.parametrize`` -- on
H200 the runtime uses ``grouped_mm_experts_forward`` -> ``torch._grouped_mm``, and
feeding it a *computed* (non-leaf) weight hangs the first step. Instead we keep the
base ``grouped_mm`` on the **original leaf weights** (same as the working baseline
MoE-LoRA) and inject the adapter as cheap *grouped low-rank* ops at the gate_up and
down stages -- so the giant weights are never materialised/duplicated:

  * LoRA : out += (alpha/r) * groupedB(groupedA(x))
  * DoRA : out  = (m / ||W + (alpha/r)BA||_col) * (out + lora)   (norm detached)
  * OFT  : out += groupedW((R - I) x)   (HF-peft convention: block-diagonal orthogonal
           R over the INPUT features, exact Cayley -- y = W(Rx); ``oft_out`` keeps the
           legacy output-side rotation out = blockR @ out for old checkpoints)

Compute path: every grouped call in the adapted experts forward (frozen base weights and
trainable adapter tensors alike) goes through ``_SafeGroupedLinearFn``: the fast grouped
kernel in forward, hand-written backward (grouped kernel for grad_input + small
per-expert dense matmuls for grad_weight), so the grouped weight-gradient kernel that
deadlocks on the first DeepSpeed step is never invoked. ``WEIGHT_PEFT_FORCE_LOOP=1``
restores the historical pure per-expert loop (``_wp_experts_forward_loop``), which is
the numerical reference but ~6x slower.

Attention (small ``nn.Linear`` weights) still uses ``parametrize`` -- that path is
standard ``F.linear`` and works fine; the hang was specific to grouped_mm.

Math mirrors the canonical definitions (and PEFT where PEFT supports the layer); see
``tests/test_weight_peft_variants.py``. Save/load mirrors ``das_lora`` (own
``weight_peft_state.safetensors`` + ``weight_peft_config.json``; auto-detected by
``eval_scripts/_common.py``).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from transformers import TrainerCallback

from ...extras.logging import get_logger
from .moe_lora import (
    _compute_router_logits,
    _find_moe_blocks,
    _get_router_linear,
    _locate_decoder_layers,
    _resolve_attn_targets,
    _scaled_alpha,
    common_lora_config,
    inject_common_lora,
    is_common_lora_parameter,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


# ``torch.nn.functional.grouped_mm`` is autograd-safe; the private ``torch._grouped_mm``
# fallback (used by ``transformers._grouped_linear`` when the functional API is missing)
# HANGS in the backward of the first step on H200 -- but only when a *weight* operand of the
# grouped matmul requires grad (its weight-gradient kernel deadlocks under DeepSpeed; the
# input-gradient path is exercised by every frozen-base run in this repo and is fine).
#
# The historical mitigation was a per-expert Python loop (``_wp_experts_forward_loop``),
# which is ~6x slower per step. The current default is ``_wp_experts_forward`` +
# ``_SafeGroupedLinearFn``: forward uses the fast grouped kernel, backward computes
# grad_input with the same forward-style kernel and grad_weight with small per-expert
# dense matmuls -- the deadlocking grouped weight-gradient kernel is never invoked.
# Set WEIGHT_PEFT_FORCE_LOOP=1 to restore the loop path exactly.
_HAS_FUNC_GROUPED_MM = hasattr(F, "grouped_mm")
_HAS_PRIVATE_GROUPED_MM = hasattr(torch, "_grouped_mm")


def _wp_grouped_enabled() -> bool:
    if os.environ.get("WEIGHT_PEFT_FORCE_LOOP", "0") == "1":
        return False
    try:
        from transformers.integrations import moe  # noqa: F401
    except Exception:
        return False
    return True


def _gmm_forward(x2d: torch.Tensor, w_kn: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Forward-only grouped matmul: (S,K) tokens sorted by expert @ (E,K,N) -> (S,N).

    On CUDA this is the same fast kernel the frozen base experts use; elsewhere (or when
    the kernel is unusable) it is transformers' registered fallback op, whose loop matches
    the historical per-expert path. Callers own the backward; this function must never be
    asked to record grads for the weight."""
    from transformers.integrations.moe import _can_use_grouped_mm, _grouped_mm

    if x2d.device.type == "cuda" and _can_use_grouped_mm(x2d, w_kn, offs):
        return _grouped_mm(x2d, w_kn, offs)
    return torch.ops.transformers.grouped_mm_fallback(x2d, w_kn, offs)


class _SafeGroupedLinearFn(torch.autograd.Function):
    """Grouped linear whose 3D weight may require grad, without grouped_mm's autograd.

    forward: fast grouped kernel (runs under the custom-Function no-grad context).
    backward: grad_input via the same forward-style kernel (2d @ 3d -- the path every
    frozen-base experiment already exercises); grad_weight via a per-expert loop of small
    dense matmuls. The grouped weight-gradient kernel that deadlocks on the first
    DeepSpeed multi-GPU step is never invoked.

    ``weight`` follows transformers' ``_grouped_linear`` layout convention:
    ``[E, out, in]`` normally, ``[E, in, out]`` when ``is_transposed``.
    ``counts`` is the per-expert token count as a host-side list (avoids a device sync in
    backward); pass None to derive it from ``offs`` on demand."""

    @staticmethod
    def forward(ctx, x, weight, offs, is_transposed, counts):
        w_kn = weight if is_transposed else weight.transpose(-2, -1)
        out = _gmm_forward(x, w_kn, offs)
        ctx.save_for_backward(x, weight, offs)
        ctx.is_transposed = is_transposed
        ctx.counts = counts
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, offs = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            w_nk = weight.transpose(-2, -1) if ctx.is_transposed else weight
            grad_x = _gmm_forward(grad_out, w_nk, offs)
        if ctx.needs_input_grad[1]:
            counts = ctx.counts
            if counts is None:
                boundaries = offs.to(torch.long).tolist()
                counts = [end - start for start, end in zip([0] + boundaries[:-1], boundaries)]
            # Full-size zero grad keeps DeepSpeed's per-rank gradient buckets identical even
            # when an expert received no tokens on this rank (mirrors the loop path's "touch").
            grad_w = torch.zeros_like(weight)
            start = 0
            for e, c in enumerate(counts):
                if c:
                    seg = x[start:start + c].transpose(0, 1) @ grad_out[start:start + c]  # [K,N]
                    grad_w[e] = seg if ctx.is_transposed else seg.transpose(0, 1)
                start += c
        return grad_x, grad_w, None, None, None


def _make_safe_grouped_linear(counts):
    """Build a ``_grouped_linear``-compatible callable routed through ``_SafeGroupedLinearFn``.

    Used for every grouped call in the adapted experts forward (frozen base weights and
    trainable adapter tensors alike), so no grouped op ever engages grouped_mm's own
    autograd. ``bias`` is the pre-gathered per-token bias, added after the matmul exactly
    like transformers' ``_grouped_linear`` does."""

    def safe_grouped_linear(x, weight, offs, bias=None, is_transposed=False):
        out = _SafeGroupedLinearFn.apply(x, weight, offs, is_transposed, counts)
        if bias is not None:
            out = out + bias
        return out

    return safe_grouped_linear


# ---------------------------------------------------------------------------
# Fused-expert LoRA initialisation
# ---------------------------------------------------------------------------

def _init_fused_lora_A(parameter: torch.Tensor) -> None:
    """Kaiming-initialize a fused LoRA ``A`` tensor per logical adapter.

    A fused expert adapter is stored as ``[E, rank, in_features]`` but each
    ``parameter[e]`` is an independent 2-D LoRA matrix.  Calling
    :func:`torch.nn.init.kaiming_uniform_` on the whole 3-D tensor makes
    PyTorch infer ``fan_in = rank * in_features`` and therefore shrinks the
    initialization by ``1 / sqrt(rank)``.  Initializing each expert slice
    separately preserves the ordinary LoRA fan-in of ``in_features`` (and is
    numerically identical to the old behavior when ``rank == 1``).
    """
    if parameter.ndim != 3:
        raise ValueError(
            f"Fused LoRA A must be a 3-D tensor [experts, rank, in_features], "
            f"got {tuple(parameter.shape)}."
        )
    for expert_parameter in parameter.unbind(dim=0):
        nn.init.kaiming_uniform_(expert_parameter, a=5 ** 0.5)


# ---------------------------------------------------------------------------
# Shared OFT helper: build block-diagonal orthogonal R from skew-free params.
# skew: [..., n_free] (n_free = b*(b-1)/2); returns R: [..., n_blocks, b, b]
# ---------------------------------------------------------------------------

def _grouped_lowrank(gl, x_g, A, B, offsets, scaling):
    """Grouped low-rank delta: scaling * groupedB(groupedA(x)), with A:[E,r,in], B:[E,out,r].

    torch._grouped_mm requires 16-byte (== 8 bf16 elems) aligned strides; the rank dim
    (usually < 8) appears as a grouped_mm operand dim in BOTH the forward and the backward
    of both matmuls. So we pad the rank dim of A and B to a multiple of 8 with zeros up
    front (same trick as moe_lora); the padded rows/cols contribute nothing and their
    gradients are discarded by the slice."""
    cdt = x_g.dtype
    Ac = A.to(cdt)            # [E, r, in]
    Bc = B.to(cdt)            # [E, out, r]
    r = Ac.shape[1]
    if r % 8 != 0:
        rp = ((r + 7) // 8) * 8
        Ac = F.pad(Ac, (0, 0, 0, rp - r))   # pad rank dim of [E,r,in] -> [E,rp,in]
        Bc = F.pad(Bc, (0, rp - r))         # pad rank dim of [E,out,r] -> [E,out,rp]
    ax = gl(x_g, Ac.contiguous(), offsets, is_transposed=False)        # (S, rp)
    delta = gl(ax.contiguous(), Bc.contiguous(), offsets, is_transposed=False)  # (S, out)
    return scaling * delta


class _RoutedLoRAStage(nn.Module):
    """One logical routed low-rank projection shared by all native MoE experts.

    ``A`` and ``B`` use the logical ``F.linear`` orientation. Routing happens over
    the first dimension of both tensors, independently of the native expert that
    processes the token. This lets a WeightLoRAPool reuse one adapter selection at
    the fused ``gate_up`` and ``down`` projections without materialising a dense
    expert-weight delta.
    """

    def __init__(
        self,
        n_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        b_init: str = "normal",
        active_out_features: Optional[int] = None,
        out_offset: int = 0,
    ):
        super().__init__()
        if b_init not in ("normal", "zero"):
            raise ValueError(f"_RoutedLoRAStage b_init must be 'normal' or 'zero', got '{b_init}'.")
        # The logical output width is ``out_features``; the adapter may learn only a contiguous
        # ``active_out_features`` slice starting at ``out_offset`` (the rest of the delta stays
        # zero). This drives the up-only gate_up mode: learn the up half of the fused
        # [gate | up] output and leave the gate half untouched, halving the B parameters.
        if active_out_features is None:
            active_out_features = out_features
        if active_out_features < 1 or out_offset < 0 or out_offset + active_out_features > out_features:
            raise ValueError(
                f"_RoutedLoRAStage active slice [{out_offset}, {out_offset + active_out_features}) "
                f"is out of bounds for out_features={out_features}."
            )
        self.n_experts = n_experts
        self.in_features = in_features
        self.out_features = out_features
        self.active_out_features = active_out_features
        self.out_offset = out_offset
        self.rank = rank
        self.scaling = alpha / rank
        self.b_init = b_init
        self.A = nn.Parameter(torch.empty(n_experts, rank, in_features))
        self.B = nn.Parameter(torch.empty(n_experts, active_out_features, rank))
        _init_fused_lora_A(self.A)
        if b_init == "zero":
            # Standard LoRA convention: the adapter is an exact identity at step 0.
            nn.init.zeros_(self.B)
        else:
            # Historical MoE^2-LoRA pools use a tiny non-zero B initialization rather
            # than identity-at-start zero B. Keep that as the default so existing
            # checkpoints and every other granularity are untouched.
            nn.init.normal_(self.B, std=1e-4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                f"WeightLoRAPool stages expect [tokens, hidden], got {tuple(hidden_states.shape)}."
            )
        if topk_values.shape != topk_indices.shape or topk_values.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "WeightLoRAPool routing shape must be [tokens, top_k] and match hidden states."
            )
        if _wp_grouped_enabled():
            return self._forward_grouped(hidden_states, topk_values, topk_indices)
        return self._forward_loop(hidden_states, topk_values, topk_indices)

    def _to_full_width(self, active_delta: torch.Tensor) -> torch.Tensor:
        """Place an ``active_out_features``-wide delta into the full ``out_features`` output.

        No-op when the whole output is adapted; otherwise the un-adapted region stays zero
        (e.g. the gate half of a fused [gate | up] projection in up-only mode)."""
        if self.active_out_features == self.out_features:
            return active_delta
        left = self.out_offset
        right = self.out_features - self.out_offset - self.active_out_features
        return F.pad(active_delta, (left, right))

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_tokens, top_k = topk_indices.shape
        token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        expert_ids = topk_indices.reshape(-1)
        values = topk_values.to(hidden_states.dtype).reshape(-1)

        perm = torch.argsort(expert_ids)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        expert_ids_g = expert_ids[perm]
        hidden_states_g = hidden_states[token_idx[perm]]

        histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
        tokens_per_expert = torch.histc(
            histc_input, bins=self.n_experts, min=0, max=self.n_experts - 1
        )
        offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
        counts = None
        if torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.parameters()):
            counts = tokens_per_expert.to(torch.long).tolist()
        grouped_linear = _make_safe_grouped_linear(counts)
        delta_g = _grouped_lowrank(grouped_linear, hidden_states_g, self.A, self.B, offsets, self.scaling)
        weighted = delta_g * values[perm].unsqueeze(-1)
        weighted = weighted[inv_perm]
        summed = weighted.view(num_tokens, top_k, self.active_out_features).sum(dim=1)
        return self._to_full_width(summed)

    def _forward_loop(
        self,
        hidden_states: torch.Tensor,
        topk_values: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        topk_values = topk_values.to(hidden_states.dtype)
        output = hidden_states.new_zeros(hidden_states.shape[0], self.active_out_features)
        for expert_index in range(self.n_experts):
            expert_mask = topk_indices == expert_index
            if not expert_mask.any():
                continue
            weights = (topk_values * expert_mask).sum(dim=-1)
            active = weights != 0
            if not active.any():
                continue
            active_hidden = hidden_states[active]
            delta = F.linear(F.linear(active_hidden, self.A[expert_index].to(hidden_states.dtype)),
                             self.B[expert_index].to(hidden_states.dtype))
            output[active] = output[active] + weights[active].unsqueeze(-1) * self.scaling * delta

        # Sparse adapter routing can leave one logical expert inactive on a DDP
        # rank. Preserve a zero gradient edge for every parameter in the loop path.
        if self.training:
            touch = None
            for parameter in self.parameters():
                if parameter.requires_grad:
                    value = parameter.sum()
                    touch = value if touch is None else touch + value
            if touch is not None:
                output = output + (0.0 * touch).to(output.dtype)
        return self._to_full_width(output)


class WeightLoRAPool(nn.Module):
    """Routed LoRA pool at the two fused MoE expert projection weights.

    One logical adapter expert contains an aligned pair of ``gate_up`` and
    ``down`` low-rank components. The caller may reuse one route for both stages
    or select each stage independently. Native MoE expert dispatch is deliberately
    outside this module and remains unchanged.
    """

    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        n_experts: int,
        rank: int,
        alpha: int,
        top_k: int,
        normalize_topk_prob: bool = False,
        is_transposed: bool = False,
        b_init: str = "normal",
        gate_up_stage: str = "gate_up",
    ):
        super().__init__()
        if n_experts < 1 or rank < 1 or top_k < 1 or top_k > n_experts:
            raise ValueError(
                "WeightLoRAPool requires n_experts, rank, and top_k >= 1 with top_k <= n_experts."
            )
        if gate_up_stage not in ("gate_up", "up"):
            raise ValueError(f"gate_up_stage must be 'gate_up' or 'up', got '{gate_up_stage}'.")
        gate_up_out, gate_up_in = self._logical_shape(gate_up_weight, is_transposed)
        down_out, down_in = self._logical_shape(down_weight, is_transposed)
        if gate_up_in != down_out or gate_up_out != 2 * down_in:
            raise ValueError(
                "WeightLoRAPool expects fused gated expert weights with gate_up [2m, d] and down [d, m]."
            )
        self.n_experts = n_experts
        self.top_k = top_k
        self.rank = rank
        self.normalize_topk_prob = normalize_topk_prob
        self.is_transposed = is_transposed
        self._is_moe_lora_pool = True
        self.b_init = b_init
        self.gate_up_stage = gate_up_stage
        # In up-only mode the gate_up adapter learns just the up half of the fused
        # [gate | up] output (the second contiguous chunk of width ``down_in``), so the
        # multiplicative gate input stays at its frozen base value and the B params halve.
        if gate_up_stage == "up":
            gate_up_active = down_in
            gate_up_offset = down_in
        else:
            gate_up_active = gate_up_out
            gate_up_offset = 0
        self.gate_up = _RoutedLoRAStage(
            n_experts, gate_up_in, gate_up_out, rank, alpha, b_init=b_init,
            active_out_features=gate_up_active, out_offset=gate_up_offset,
        )
        self.down = _RoutedLoRAStage(n_experts, down_in, down_out, rank, alpha, b_init=b_init)
        self.register_buffer("activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("lifetime_activation_count", torch.zeros(n_experts, dtype=torch.long), persistent=False)

    @staticmethod
    def _logical_shape(weight: torch.Tensor, is_transposed: bool) -> tuple[int, int]:
        if weight.dim() != 3:
            raise ValueError(f"Expected a fused 3-D expert weight, got {tuple(weight.shape)}.")
        _, dim_1, dim_2 = weight.shape
        return (dim_2, dim_1) if is_transposed else (dim_1, dim_2)

    def select(self, probabilities: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if probabilities.shape[-1] != self.n_experts:
            raise ValueError(
                f"WeightLoRAPool expected {self.n_experts} routing logits, got {probabilities.shape[-1]}."
            )
        probabilities = probabilities.to(dtype)
        topk_values, topk_indices = probabilities.topk(self.top_k, dim=-1)
        if self.normalize_topk_prob:
            topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        with torch.no_grad():
            flat_indices = topk_indices.reshape(-1)
            ones = torch.ones_like(flat_indices)
            self.activation_count.scatter_add_(0, flat_indices, ones)
            self.lifetime_activation_count.scatter_add_(0, flat_indices, ones)
            self.total_tokens += probabilities.reshape(-1, self.n_experts).shape[0]
        return topk_values, topk_indices


def _cayley_R(skew: torch.Tensor, block_size: int, triu_r, triu_c, dtype):
    if block_size <= 1:
        raise ValueError(f"OFT block_size must be greater than 1, got {block_size}")
    b = block_size
    lead = skew.shape[:-1]
    Q = torch.zeros(*lead, b, b, device=skew.device, dtype=torch.float32)
    Q[..., triu_r, triu_c] = skew.float()
    Q = Q - Q.transpose(-1, -2)
    eye = torch.eye(b, device=skew.device, dtype=torch.float32).expand_as(Q)
    # Match HF PEFT's OFTRotationModule._cayley_batch.  ``left=False`` solves
    # X @ (I + Q) = I - Q, which is the orientation used by the OFT merge rule.
    R = torch.linalg.solve(eye + Q, eye - Q, left=False)  # exact Cayley -> orthogonal
    return R.to(dtype)






# ---------------------------------------------------------------------------
# Shared FourierFT helpers (DFT-based weight delta; mirrors HF peft FourierFT).
#
# A weight delta is parameterised by n_freq *real* learnable coefficients placed at a
# FIXED random spectral support {(row,col)}; the delta is ``ifft2(sparse_spectrum).real``
# times a scaling. The support is regenerated deterministically from (out,in,n_freq,seed)
# so it is never stored -- only the seed is saved, exactly recovering the locations on load.
# ---------------------------------------------------------------------------

def _fourier_indices(out: int, in_: int, n_freq: int, seed: int):
    """Reproducible spectral support: up to n_freq distinct (row,col) positions in [out,in]."""
    total = out * in_
    n = min(int(n_freq), total)
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    flat = torch.randperm(total, generator=g)[:n]
    rows = (flat // in_).long()
    cols = (flat % in_).long()
    return rows, cols, n


def _fourier_delta(spectrum: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                   out: int, in_: int, scaling: float) -> torch.Tensor:
    """spectrum: [..., n] real coeffs -> dense [..., out, in] delta = scaling * ifft2(spectrum).real.

    Computation is in float32 for FFT accuracy; caller casts to the weight dtype. ``...`` is the
    optional expert lead-dim ([E] for fused experts, empty for attention)."""
    lead = spectrum.shape[:-1]
    dense = torch.zeros(*lead, out, in_, dtype=torch.complex64, device=spectrum.device)
    dense[..., rows, cols] = spectrum.to(torch.float32).to(torch.complex64)
    return torch.fft.ifft2(dense, dim=(-2, -1)).real * scaling


# ===========================================================================
# 2D projection adapters (attention/router nn.Linear weights, via parametrize)
# ===========================================================================

class WeightLoRA(nn.Module):
    """Canonical LoRA weight delta on a 2D weight [out,in]: W + (alpha/r) B@A."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: int):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be greater than 0, got {rank}")
        out, in_ = weight.shape
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_))
        self.B = nn.Parameter(torch.zeros(out, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, weight):
        return weight + (self.scaling * (self.B @ self.A)).to(weight.dtype)




class WeightDoRA(nn.Module):
    """Canonical DoRA on a 2D weight [out,in] (norm over in-dim, detached)."""

    def __init__(self, weight: torch.Tensor, rank: int, alpha: int):
        super().__init__()
        if rank < 1:
            raise ValueError(f"DoRA rank must be greater than 0, got {rank}")
        out, in_ = weight.shape
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_))
        self.B = nn.Parameter(torch.zeros(out, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.magnitude = nn.Parameter(torch.linalg.norm(weight.detach().float(), dim=1).to(weight.dtype))

    def forward(self, weight):
        v = weight + (self.scaling * (self.B @ self.A)).to(weight.dtype)
        n = torch.linalg.norm(v.float(), dim=1, keepdim=True).to(weight.dtype).detach()
        return (self.magnitude.unsqueeze(-1) / n) * v


class WeightOFT(nn.Module):
    """OFT on a 2D weight [out,in], HF-peft convention: rotate the INPUT features.

    HF peft applies ``y = base_layer(R(x))`` (merge: ``W <- W @ R^T``) with the
    block-diagonal orthogonal ``R`` built over ``in_features``. We mirror that as a
    weight parametrization: ``W_eff = W @ blockdiag(R)^T``.
    """

    def __init__(self, weight: torch.Tensor, block_size: int):
        super().__init__()
        if block_size <= 1:
            raise ValueError(f"OFT block_size must be greater than 1, got {block_size}")
        out, in_ = weight.shape
        if in_ % block_size != 0:
            raise ValueError(f"OFT block_size={block_size} must divide in_features={in_}")
        self.block_size = block_size
        self.n_blocks = in_ // block_size
        b = block_size
        self.skew = nn.Parameter(torch.zeros(self.n_blocks, b * (b - 1) // 2))
        triu = torch.triu_indices(b, b, offset=1)
        self.register_buffer("_triu_r", triu[0], persistent=False)
        self.register_buffer("_triu_c", triu[1], persistent=False)

    def forward(self, weight):
        R = _cayley_R(self.skew, self.block_size, self._triu_r, self._triu_c, weight.dtype)
        out, in_ = weight.shape
        w = weight.view(out, self.n_blocks, self.block_size)
        # (W R^T)[o, n, c] = sum_k W[o, n, k] R[n, c, k]
        return torch.einsum("onk,nck->onc", w, R).reshape(out, in_)






class WeightOFTOut(nn.Module):
    """Legacy OFT on a 2D weight [out,in]: W_eff = R @ W (rotate OUTPUT features).

    Historical implementation kept for loading/evaluating checkpoints trained before
    the HF alignment (``weight_peft_type=oft`` configs without the
    ``weight_peft_oft_convention`` marker are remapped to this class on load).
    """

    def __init__(self, weight: torch.Tensor, block_size: int):
        super().__init__()
        if block_size <= 1:
            raise ValueError(f"OFT block_size must be greater than 1, got {block_size}")
        out, in_ = weight.shape
        if out % block_size != 0:
            raise ValueError(f"OFT(out) block_size={block_size} must divide out_features={out}")
        self.block_size = block_size
        self.n_blocks = out // block_size
        b = block_size
        self.skew = nn.Parameter(torch.zeros(self.n_blocks, b * (b - 1) // 2))
        triu = torch.triu_indices(b, b, offset=1)
        self.register_buffer("_triu_r", triu[0], persistent=False)
        self.register_buffer("_triu_c", triu[1], persistent=False)

    def forward(self, weight):
        R = _cayley_R(self.skew, self.block_size, self._triu_r, self._triu_c, weight.dtype)
        out, in_ = weight.shape
        w = weight.view(self.n_blocks, self.block_size, in_)
        return torch.einsum("nij,njk->nik", R, w).reshape(out, in_)


class WeightFourier(nn.Module):
    """Canonical FourierFT on a 2D weight [out,in]: W + scaling * ifft2(sparse_spectrum).real.

    ``spectrum`` (zero-init -> identity at start) holds n real coefficients at a fixed random
    spectral support; mirrors HF peft ``FourierFTLayer.get_delta_weight``."""

    def __init__(self, weight: torch.Tensor, n_freq: int, scaling: float, seed: int):
        super().__init__()
        out, in_ = weight.shape
        rows, cols, n = _fourier_indices(out, in_, n_freq, seed)
        self.out_features, self.in_features, self.scaling = out, in_, float(scaling)
        self.spectrum = nn.Parameter(torch.zeros(n))
        self.register_buffer("_rows", rows, persistent=False)
        self.register_buffer("_cols", cols, persistent=False)

    def forward(self, weight):
        delta = _fourier_delta(self.spectrum, self._rows, self._cols,
                               self.out_features, self.in_features, self.scaling)
        return weight + delta.to(weight.dtype)


class WeightIA3(nn.Module):
    """Canonical IA3 output rescaling on a 2D weight [out,in].

    IA3 multiplies the layer's output activations by a learned vector ``l`` (ones-init):
    ``y = l ⊙ (W x) = (diag(l) W) x``, i.e. scale the rows of W. Used on attention k/v
    projections (canonical IA3 rescales keys & values)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        out, _ = weight.shape
        self.l = nn.Parameter(torch.ones(out))

    def forward(self, weight):
        return self.l.to(weight.dtype).unsqueeze(1) * weight


class _BiasIA3(nn.Module):
    """Scale a projection bias with the same IA3 vector that scales its weight rows."""

    def __init__(self, adapter: WeightIA3):
        super().__init__()
        # The weight parametrization owns/registers the learned vector. Keeping a
        # raw module reference here avoids duplicate state-dict keys while still
        # following the parameter if model/device conversion replaces it.
        object.__setattr__(self, "_adapter", adapter)

    def forward(self, bias):
        return self._adapter.l.to(bias.dtype) * bias


def _make_projection_adapter(ptype, weight, fa, position: str, seed: int = 0):
    """Build the selected method for an attention or router 2-D projection.

    Position-specific capacity fields use ``0`` as a backwards-compatible fallback
    to the historical global FFN value.
    """
    rank = getattr(fa, f"weight_peft_{position}_rank", 0) or fa.weight_peft_rank
    alpha = (
        getattr(fa, f"weight_peft_{position}_alpha", 0)
        or fa.weight_peft_alpha
        or (2 * rank)
    )
    block_size = (
        getattr(fa, f"weight_peft_{position}_oft_block_size", 0)
        or fa.weight_peft_oft_block_size
    )
    n_freq = (
        getattr(fa, f"weight_peft_{position}_fourier_n_frequencies", 0)
        or fa.weight_peft_fourier_n_frequencies
    )
    if ptype == "lora":
        return WeightLoRA(weight, rank, alpha)
    if ptype == "dora":
        return WeightDoRA(weight, rank, alpha)
    if ptype == "oft":
        return WeightOFT(weight, block_size)
    if ptype == "oft_out":
        return WeightOFTOut(weight, block_size)
    if ptype == "fourier":
        return WeightFourier(weight, n_freq,
                             fa.weight_peft_fourier_scaling, seed)
    if ptype == "ia3":
        return WeightIA3(weight)
    raise ValueError(f"Unknown weight_peft_type: {ptype!r}")


def _register_projection_adapter(projection: nn.Module, adapter: nn.Module, ptype: str) -> None:
    """Register a weight adapter and preserve true IA3 output scaling when bias exists."""
    parametrize.register_parametrization(projection, "weight", adapter)
    bias = getattr(projection, "bias", None)
    if ptype == "ia3" and isinstance(bias, nn.Parameter):
        parametrize.register_parametrization(projection, "bias", _BiasIA3(adapter))


# ===========================================================================
# Expert adapters (3D fused weight [E,out,in] or transposed [E,in,out]) — injected via grouped ops,
# base grouped_mm stays on the leaf weight (no full-weight materialisation).
# ===========================================================================

class _StageAdapter(nn.Module):
    """Adapter params for one expert projection stage (gate_up or down).

    Adapter parameters always use logical ``F.linear`` orientation, while the frozen
    expert weight may be stored as ``[E,out,in]`` (Qwen3) or ``[E,in,out]`` (GPT-OSS).
    """

    def __init__(self, weight: torch.Tensor, ptype: str, rank: int, alpha: int, block_size: int,
                 stage: str = "gate_up", fourier_n_freq: int = 0, fourier_scaling: float = 150.0,
                 fourier_seed: int = 0, is_transposed: bool = False):
        super().__init__()
        e, dim_1, dim_2 = weight.shape
        out, in_ = (dim_2, dim_1) if is_transposed else (dim_1, dim_2)
        if rank < 1:
            raise ValueError(f"Stage adapter rank must be greater than 0, got {rank}")
        self.ptype = ptype
        self.stage = stage
        self.is_transposed = is_transposed
        self.scaling = alpha / rank
        if ptype in ("lora", "dora"):
            self.A = nn.Parameter(torch.empty(e, rank, in_))
            self.B = nn.Parameter(torch.zeros(e, out, rank))
            _init_fused_lora_A(self.A)
            if ptype == "dora":
                logical_weight = weight.transpose(-2, -1) if is_transposed else weight
                self.magnitude = nn.Parameter(
                    torch.linalg.norm(logical_weight.detach().float(), dim=2).to(weight.dtype)  # [E,out]
                )
        elif ptype in ("oft", "oft_out"):
            # "oft" (HF-peft convention): block-diagonal R over the logical INPUT features;
            # forward rotates the stage input (y = W(Rx), merge W <- W R^T).
            # "oft_out" (legacy): R over the logical OUTPUT features; y = R(Wx).
            ref = in_ if ptype == "oft" else out
            ref_name = "in" if ptype == "oft" else "out"
            if block_size <= 1:
                raise ValueError(f"OFT block_size must be greater than 1, got {block_size}")
            if ref % block_size != 0:
                raise ValueError(f"OFT block_size={block_size} must divide {ref_name}={ref}")
            b = block_size
            self.block_size = b
            self.n_blocks = ref // b
            self.skew = nn.Parameter(torch.zeros(e, self.n_blocks, b * (b - 1) // 2))
            triu = torch.triu_indices(b, b, offset=1)
            self.register_buffer("_triu_r", triu[0], persistent=False)
            self.register_buffer("_triu_c", triu[1], persistent=False)
        elif ptype == "fourier":
            # Per-expert real spectrum at a shared (per-stage) random support; zero-init -> identity.
            rows, cols, n = _fourier_indices(out, in_, fourier_n_freq, fourier_seed)
            self.out_features, self.in_features, self.fourier_scaling = out, in_, float(fourier_scaling)
            self.spectrum = nn.Parameter(torch.zeros(e, n))
            self.register_buffer("_rows", rows, persistent=False)
            self.register_buffer("_cols", cols, persistent=False)
        elif ptype == "ia3":
            # IA3 rescales the feedforward (down-projection) *input* by a per-expert learned
            # vector l (ones-init). gate_up stage is a no-op (canonical IA3 has one ff vector).
            if stage == "down":
                self.l_ff = nn.Parameter(torch.ones(e, in_))  # [E, inter]
        else:
            raise ValueError(f"Unknown weight_peft_type: {ptype!r}")

    def train(self, mode: bool = True):
        # The eval-time Fourier delta cache is only valid while the spectrum is frozen.
        if mode:
            self.__dict__.pop("_fourier_eval_delta", None)
        return super().train(mode)

    def stage_delta(self, weight):
        """Full Fourier delta in the expert weight's physical storage orientation.

        Built in expert-chunks so the dense complex spectrum transient stays bounded. The loop
        path does NOT use this -- it computes each expert's delta on the fly (see apply_chunk).
        Returns None for every non-fourier type (they need no precompute).

        Eval-time cache: at eval the spectrum is frozen, so the delta is identical on every
        forward -- yet autoregressive generation calls this once per decode step (the ifft2
        over [E,out,in] dominated eval wall-clock ~60x). The delta is therefore cached per
        adapter while ``self.training`` is False, guarded by a free-memory margin so DP evals
        that load the full model per GPU never over-commit (uncached stages just recompute).
        ``WEIGHT_PEFT_FOURIER_EVAL_CACHE=0`` disables caching; the cached tensor is the exact
        tensor a recompute would produce, so results are bit-identical either way."""
        if self.ptype != "fourier":
            return None
        cache_ok = not self.training and not torch.is_grad_enabled()
        if cache_ok:
            cached = self.__dict__.get("_fourier_eval_delta")
            if cached is not None:
                return cached
        E = self.spectrum.shape[0]
        chunk = 32
        if self.is_transposed:
            out_features, in_features = self.in_features, self.out_features
            rows, cols = self._cols, self._rows
        else:
            out_features, in_features = self.out_features, self.in_features
            rows, cols = self._rows, self._cols
        if E <= chunk:
            delta = _fourier_delta(
                self.spectrum, rows, cols, out_features, in_features, self.fourier_scaling
            ).to(weight.dtype)
        else:
            parts = [
                _fourier_delta(
                    self.spectrum[i:i + chunk], rows, cols, out_features, in_features, self.fourier_scaling
                ).to(weight.dtype)
                for i in range(0, E, chunk)
            ]
            delta = torch.cat(parts, dim=0)
        if cache_ok and self._fourier_cache_allowed(delta):
            self._fourier_eval_delta = delta.detach()
        return delta

    @staticmethod
    def _fourier_cache_allowed(delta: torch.Tensor) -> bool:
        if os.environ.get("WEIGHT_PEFT_FOURIER_EVAL_CACHE", "1") != "1":
            return False
        if delta.device.type != "cuda":
            return True  # CPU tests / CPU eval: no VRAM pressure concern
        try:
            free_bytes, _ = torch.cuda.mem_get_info(delta.device)
        except Exception:  # noqa: BLE001 - never let the cache heuristic break eval
            return False
        margin_gb = float(os.environ.get("WEIGHT_PEFT_FOURIER_CACHE_MARGIN_GB", "24"))
        # The delta is already allocated; keep the margin free for KV cache/activations.
        return free_bytes > margin_gb * (1 << 30)

    def apply_grouped(self, grouped_linear, x_g, base_out, weight, expert_ids_g, offsets, pre=None):
        """x_g: (S,in) tokens sorted by expert; base_out: (S,out); weight: fused leaf weight.

        NOTE: do NOT name this ``apply`` -- that shadows ``nn.Module.apply(fn)``, which
        DeepSpeed calls during engine init (set_z3_leaf_modules) and would crash with a
        signature mismatch (only manifests under distributed/deepspeed, not single-GPU).

        NOTE: torch._grouped_mm requires bf16 and is NOT autocast-enabled, while the
        trainable adapter params may be upcast to fp32. So we cast A/B to the compute
        dtype (x_g.dtype) for the grouped matmuls, and keep the DoRA scale in that dtype
        too (otherwise bf16*fp32 promotes the activation to fp32 and the next grouped_mm
        rejects it)."""
        cdt = x_g.dtype
        if self.ptype == "lora":
            A = self.A
            return base_out + _grouped_lowrank(grouped_linear, x_g, A, self.B, offsets, self.scaling)
        if self.ptype == "dora":
            out_full = base_out + _grouped_lowrank(grouped_linear, x_g, self.A, self.B, offsets, self.scaling)
            with torch.no_grad():
                logical_weight = weight.transpose(-2, -1) if self.is_transposed else weight
                v = logical_weight.float() + self.scaling * torch.bmm(self.B.float(), self.A.float())
                norm = torch.linalg.norm(v, dim=2)                                   # [E,out]
            scale = (self.magnitude.float() / norm).to(cdt)                          # [E,out]
            return out_full * scale[expert_ids_g]
        if self.ptype == "fourier":
            # ``pre`` follows the base weight's physical orientation.
            return base_out + grouped_linear(
                x_g, pre.to(cdt), offsets, is_transposed=self.is_transposed
            )
        if self.ptype == "ia3":
            if self.stage != "down":
                return base_out                                                      # gate_up: no-op
            scaled = (self.l_ff[expert_ids_g].to(cdt) - 1.0) * x_g                    # W(l⊙x)=Wx + W((l-1)⊙x)
            return base_out + grouped_linear(
                scaled, weight, offsets, is_transposed=self.is_transposed
            )
        if self.ptype == "oft":
            # HF-peft convention: rotate the stage INPUT: y = W(Rx) = Wx + W((R-I)x).
            R = _cayley_R(self.skew, self.block_size, self._triu_r, self._triu_c, cdt)
            S, in_ = x_g.shape
            xb = x_g.view(S, self.n_blocks, self.block_size)
            Rg = R[expert_ids_g]                                                      # (S,nb,b,b)
            # HF: x_rot = R^T x per block (W_eff = W blockdiag(R)^T, same as WeightOFT 2D)
            rx = torch.einsum("snji,snj->sni", Rg, xb).reshape(S, in_)
            return base_out + grouped_linear(
                rx - x_g, weight, offsets, is_transposed=self.is_transposed
            )
        # oft_out (legacy): rotate the stage OUTPUT: y = R(Wx)
        R = _cayley_R(self.skew, self.block_size, self._triu_r, self._triu_c, base_out.dtype)  # [E,nb,b,b]
        S, out = base_out.shape
        xb = base_out.view(S, self.n_blocks, self.block_size)
        Rg = R[expert_ids_g]                                                          # (S,nb,b,b)
        ob = torch.einsum("snij,snj->sni", Rg, xb)
        return ob.reshape(S, out)

    def apply_chunk(self, x_c, base_c, weight, e, pre=None):
        """Per-expert 2D variant of ``apply_grouped`` for the loop fallback.

        x_c: (s,in) tokens routed to expert ``e``; base_c: (s,out) == F.linear(x_c, weight[e]);
        weight: leaf 3D fused weight in either supported orientation. Returns the adapted
        (s,out). Uses only plain matmuls, never grouped_mm. Math is identical to
        ``apply_grouped`` restricted to one expert."""
        cdt = x_c.dtype
        if self.ptype == "lora":
            A = self.A[e]
            delta = (x_c @ A.to(cdt).t()) @ self.B[e].to(cdt).t()             # (s,out)
            return base_c + self.scaling * delta
        if self.ptype == "dora":
            delta = (x_c @ self.A[e].to(cdt).t()) @ self.B[e].to(cdt).t()
            out_full = base_c + self.scaling * delta
            with torch.no_grad():
                logical_weight = weight[e].t() if self.is_transposed else weight[e]
                v = logical_weight.float() + self.scaling * (self.B[e].float() @ self.A[e].float())
                norm = torch.linalg.norm(v, dim=1)                                   # [out]
            scale = (self.magnitude[e].float() / norm).to(cdt)                       # [out] broadcast over tokens
            return out_full * scale
        if self.ptype == "fourier":
            # Compute this expert's delta on the fly ([out,in] transient, active experts only) --
            # far cheaper in memory than materialising the full [E,out,in] for the loop path.
            delta_e = pre[e] if pre is not None else _fourier_delta(
                self.spectrum[e], self._rows, self._cols,
                self.out_features, self.in_features, self.fourier_scaling)
            if pre is not None and self.is_transposed:
                delta_e = delta_e.t()
            return base_c + F.linear(x_c, delta_e.to(cdt))                           # delta_e: [out,in]
        if self.ptype == "ia3":
            if self.stage != "down":
                return base_c                                                        # gate_up: no-op
            logical_weight = weight[e].t() if self.is_transposed else weight[e]
            return base_c + F.linear(
                (self.l_ff[e].to(cdt) - 1.0) * x_c, logical_weight
            )                                                                         # W(l⊙x)
        if self.ptype == "oft":
            # HF-peft convention: rotate the expert INPUT: y = W(Rx) = base + W((R-I)x).
            R = _cayley_R(self.skew[e], self.block_size, self._triu_r, self._triu_c, cdt)
            s, in_ = x_c.shape
            xb = x_c.view(s, self.n_blocks, self.block_size)
            # HF: x_rot = R^T x per block (W_eff = W blockdiag(R)^T, same as WeightOFT 2D)
            rx = torch.einsum("nji,snj->sni", R, xb).reshape(s, in_)
            logical_weight = weight[e].t() if self.is_transposed else weight[e]
            return base_c + F.linear(rx - x_c, logical_weight)
        # oft_out (legacy): rotate the expert OUTPUT: y = R(Wx)
        R = _cayley_R(self.skew[e], self.block_size, self._triu_r, self._triu_c, base_c.dtype)  # [nb,b,b]
        s, out = base_c.shape
        xb = base_c.view(s, self.n_blocks, self.block_size)
        ob = torch.einsum("nij,snj->sni", R, xb)
        return ob.reshape(s, out)


def _weight_pool_routes(
    self, hidden_states: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Select gate_up/down adapters from shared or independent routing heads."""
    pool = self._weight_lora_pool
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    router_logits = getattr(self, "_moe_lora_cached_router_logits", None)
    self._moe_lora_cached_router_logits = None
    if router_logits is not None:
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])
    if router_logits is None or router_logits.shape[0] != hidden_states.shape[0]:
        router_logits = _compute_router_logits(self._moe_lora_router, hidden_states)
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])
    if getattr(self._finetuning_args, "moe_lora_detach_p_e", False):
        router_logits = router_logits.detach()

    def project(routing_projection):
        if getattr(routing_projection, "h_proj", None) is not None:
            return routing_projection(router_logits, hidden_states)
        return routing_projection(router_logits)

    gate_up_projection = getattr(self, "routing_projection_gate_up", None)
    down_projection = getattr(self, "routing_projection_down", None)
    if gate_up_projection is not None or down_projection is not None:
        if gate_up_projection is None or down_projection is None:
            raise RuntimeError("Separate weight_pool routing requires both gate_up and down projections.")
        gate_up_route = pool.select(project(gate_up_projection), hidden_states.dtype)
        down_route = pool.select(project(down_projection), hidden_states.dtype)
        return gate_up_route, down_route

    shared_route = pool.select(project(self.routing_projection), hidden_states.dtype)
    return shared_route, shared_route


def _wp_down_route_on_h(self, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the down adapter from the intermediate activation it acts on.

    ``h_in`` is ``act_fn(gate)*up`` in the expert-sorted grouped layout, so the returned
    (values, indices) are already aligned row-for-row with ``h_in`` -- no token remap.
    """
    pool = self._weight_lora_pool
    p_L = self.routing_projection_down_on_h(h_in)
    return pool.select(p_L, h_in.dtype)


def _wp_uses_down_route_on_h(self) -> bool:
    return getattr(self, "routing_projection_down_on_h", None) is not None


def _wp_experts_forward_loop(self, hidden_states, top_k_index, top_k_weights):
    """Per-expert loop replacement for grouped_mm_experts_forward (no grouped_mm).

    Used when ``torch.nn.functional.grouped_mm`` is unavailable, since the private
    ``torch._grouped_mm`` fallback hangs in backward. Tokens are sorted by expert (so each
    expert's tokens are contiguous) and each expert is processed with a single ``F.linear``
    over its slice -- every token is touched exactly once per stage, so the FLOPs match the
    grouped path; only the Python-level loop over (active) experts is added. Outputs are
    re-assembled by concatenating the per-expert slices in expert order (== sorted order),
    which keeps the whole thing autograd-friendly (no in-place index writes)."""
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    invalid_mask = expert_ids >= self.num_experts
    expert_ids = expert_ids.clamp(0, self.num_experts - 1)

    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    x_g = hidden_states[token_idx[perm]]

    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    counts = tokens_per_expert.to(torch.long).tolist()  # one host sync; reused for both stages

    has_bias = getattr(self, "has_bias", False)
    is_transposed = getattr(self, "is_transposed", False)
    wp = getattr(self, "_wp_adapter", None)
    weight_pool = getattr(self, "_weight_lora_pool", None)
    if (wp is None) == (weight_pool is None):
        raise RuntimeError("Fused experts require exactly one static adapter or WeightLoRAPool.")
    down_on_h = weight_pool is not None and _wp_uses_down_route_on_h(self)
    if weight_pool is not None:
        (gate_up_values, gate_up_indices), (down_values, down_indices) = _weight_pool_routes(self, hidden_states)
        if not down_on_h:
            down_values_g = down_values[token_idx[perm]]
            down_indices_g = down_indices[token_idx[perm]]
        gate_up_delta_g = weight_pool.gate_up(hidden_states, gate_up_values, gate_up_indices)[token_idx[perm]]

    # ---- gate_up stage (loop over experts that actually received tokens) ----
    # NOTE: fourier computes its per-expert delta inside apply_chunk (pre stays None) so the
    # dense [out,in] transient is bounded and inactive experts are skipped.
    gu_w = self.gate_up_proj
    gu_bias = self.gate_up_proj_bias if has_bias else None
    # HF DoRA never scales the bias -> defer it past the adapter (same as grouped path).
    gu_defer_bias = is_transposed or (wp is not None and wp["gate_up"].ptype == "dora")
    outs, start = [], 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        xc = x_g[start:start + c]
        logical_weight = gu_w[e].t() if is_transposed else gu_w[e]
        base_bias = None if gu_defer_bias else (gu_bias[e] if has_bias else None)
        base_c = F.linear(xc, logical_weight, base_bias)
        if wp is not None:
            adapted_c = wp["gate_up"].apply_chunk(xc, base_c, gu_w, e)
        else:
            adapted_c = base_c + gate_up_delta_g[start:start + c]
        if has_bias and gu_defer_bias:
            adapted_c = adapted_c + gu_bias[e]
        outs.append(adapted_c)
        start += c
    proj_out = torch.cat(outs, dim=0)

    # gating / activation
    if getattr(self, "has_gate", True):
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # ---- down stage ----
    dn_w = self.down_proj
    dn_bias = self.down_proj_bias if has_bias else None
    dn_defer_bias = is_transposed or (wp is not None and wp["down"].ptype == "dora")
    h_in = proj_out
    if weight_pool is not None:
        if down_on_h:
            down_values_g, down_indices_g = _wp_down_route_on_h(self, h_in)
        down_delta_g = weight_pool.down(h_in, down_values_g, down_indices_g)
    outs, start = [], 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        hc = h_in[start:start + c]
        logical_weight = dn_w[e].t() if is_transposed else dn_w[e]
        base_bias = None if dn_defer_bias else (dn_bias[e] if has_bias else None)
        base_c = F.linear(hc, logical_weight, base_bias)
        if wp is not None:
            adapted_c = wp["down"].apply_chunk(hc, base_c, dn_w, e)
        else:
            adapted_c = base_c + down_delta_g[start:start + c]
        if has_bias and dn_defer_bias:
            adapted_c = adapted_c + dn_bias[e]
        outs.append(adapted_c)
        start += c
    proj_out = torch.cat(outs, dim=0)

    # Sparse dispatch can leave an expert inactive on one data-parallel rank. Touch every
    # adapter with a zero coefficient so ZeRO/DDP sees the same gradient-bearing parameter
    # set on all ranks without changing the forward value.
    if self.training and wp is not None:
        touch = None
        for parameter in wp.parameters():
            if not parameter.requires_grad:
                continue
            value = parameter.sum()
            touch = value if touch is None else touch + value
        if touch is not None:
            proj_out = proj_out + (0.0 * touch).to(proj_out.dtype)

    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
    weighted_out.masked_fill_(invalid_mask[perm].unsqueeze(-1), 0.0)
    weighted_out = weighted_out[inv_perm]
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


def _wp_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Drop-in replacement for grouped_mm_experts_forward with weight-level adapters
    on the leaf weights (no full-weight materialisation).

    Every grouped call (frozen base weights and trainable adapter tensors alike) goes
    through ``_SafeGroupedLinearFn``, so grouped_mm's own autograd -- whose
    weight-gradient kernel deadlocks on the first DeepSpeed step -- is never engaged."""
    if not _wp_grouped_enabled():
        return _wp_experts_forward_loop(self, hidden_states, top_k_index, top_k_weights)

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    invalid_mask = expert_ids >= self.num_experts
    expert_ids = expert_ids.clamp(0, self.num_experts - 1)

    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    x_g = hidden_states[token_idx[perm]]

    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    has_bias = getattr(self, "has_bias", False)
    is_transposed = getattr(self, "is_transposed", False)
    wp = getattr(self, "_wp_adapter", None)
    weight_pool = getattr(self, "_weight_lora_pool", None)
    if (wp is None) == (weight_pool is None):
        raise RuntimeError("Fused experts require exactly one static adapter or WeightLoRAPool.")
    down_on_h = weight_pool is not None and _wp_uses_down_route_on_h(self)
    if weight_pool is not None:
        (gate_up_values, gate_up_indices), (down_values, down_indices) = _weight_pool_routes(self, hidden_states)
        if not down_on_h:
            down_values_g = down_values[token_idx[perm]]
            down_indices_g = down_indices[token_idx[perm]]

    # Host-side per-expert counts, needed only when adapter weight grads will be computed
    # (one sync per layer, same as the loop path; skipped entirely at eval/no-grad).
    counts = None
    if wp is not None and torch.is_grad_enabled() and any(p.requires_grad for p in wp.parameters()):
        counts = tokens_per_expert.to(torch.long).tolist()
    gl = _make_safe_grouped_linear(counts)

    # gate_up
    gu_bias = self.gate_up_proj_bias[expert_ids_g] if has_bias else None
    # HF DoRA never scales the bias: defer it past the adapter so the m/||v|| factor
    # multiplies only Wx (+ the low-rank delta), exactly like the transposed path.
    gu_defer_bias = is_transposed or (wp is not None and wp["gate_up"].ptype == "dora")
    base_bias = None if gu_defer_bias else gu_bias
    proj_out = gl(
        x_g, self.gate_up_proj, offsets, bias=base_bias, is_transposed=is_transposed
    )
    if wp is not None:
        pre_gu = wp["gate_up"].stage_delta(self.gate_up_proj)
        proj_out = wp["gate_up"].apply_grouped(
            gl, x_g, proj_out, self.gate_up_proj, expert_ids_g, offsets, pre=pre_gu
        )
    else:
        gate_up_delta = weight_pool.gate_up(hidden_states, gate_up_values, gate_up_indices)
        proj_out = proj_out + gate_up_delta[token_idx[perm]]
    if has_bias and gu_defer_bias:
        proj_out = proj_out + gu_bias

    # gating / activation
    if getattr(self, "has_gate", True):
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # down
    h_in = proj_out
    if down_on_h:
        down_values_g, down_indices_g = _wp_down_route_on_h(self, h_in)
    dn_bias = self.down_proj_bias[expert_ids_g] if has_bias else None
    dn_defer_bias = is_transposed or (wp is not None and wp["down"].ptype == "dora")
    base_bias = None if dn_defer_bias else dn_bias
    proj_out = gl(
        h_in, self.down_proj, offsets, bias=base_bias, is_transposed=is_transposed
    )
    if wp is not None:
        pre_dn = wp["down"].stage_delta(self.down_proj)
        proj_out = wp["down"].apply_grouped(
            gl, h_in, proj_out, self.down_proj, expert_ids_g, offsets, pre=pre_dn
        )
    else:
        proj_out = proj_out + weight_pool.down(h_in, down_values_g, down_indices_g)
    if has_bias and dn_defer_bias:
        proj_out = proj_out + dn_bias

    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
    weighted_out.masked_fill_(invalid_mask[perm].unsqueeze(-1), 0.0)
    weighted_out = weighted_out[inv_perm]
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def _find_experts_module(moe_block: nn.Module) -> Optional[nn.Module]:
    for m in moe_block.modules():
        if isinstance(getattr(m, "gate_up_proj", None), nn.Parameter) and isinstance(
            getattr(m, "down_proj", None), nn.Parameter
        ):
            return m
    return None


def inject_expert_weight_lora(
    model: "PreTrainedModel",
    moe_blocks: list[nn.Module],
    layer_ranks: list[int],
    base_rank: int,
    base_alpha: int,
) -> int:
    """Attach LoRA inside every fused expert weight for selected MoE layers.

    This is the FFN ``expert_weight`` branch of ``finetuning_type=moe_lora``. It deliberately
    attaches only the two expert projection stages; attention and base-router LoRA remain owned
    by ``moe_lora.py`` so the historical FFN-granularity arms retain their fixed LoRA branches.
    """
    if len(moe_blocks) != len(layer_ranks):
        raise ValueError(
            f"Expected one expert-weight rank per selected MoE layer, got {len(layer_ranks)} "
            f"ranks for {len(moe_blocks)} layers."
        )

    sample = next(model.parameters())
    device, dtype = sample.device, sample.dtype
    n_attached = 0
    for layer_idx, (moe_block, rank) in enumerate(zip(moe_blocks, layer_ranks)):
        experts = _find_experts_module(moe_block)
        if experts is None:
            raise ValueError(
                f"expert_weight FFN granularity could not find fused gate_up_proj/down_proj "
                f"parameters in selected MoE layer {layer_idx} ({type(moe_block).__name__})."
            )
        alpha = _scaled_alpha(base_rank, base_alpha, rank)
        is_transposed = getattr(experts, "is_transposed", False)
        adapters = nn.ModuleDict(
            {
                "gate_up": _StageAdapter(
                    experts.gate_up_proj, "lora", rank, alpha, block_size=1,
                    stage="gate_up", is_transposed=is_transposed,
                ),
                "down": _StageAdapter(
                    experts.down_proj, "lora", rank, alpha, block_size=1,
                    stage="down", is_transposed=is_transposed,
                ),
            }
        ).to(device=device, dtype=dtype)
        experts._wp_adapter = adapters
        experts._wp_orig_forward = experts.forward
        experts.forward = _wp_experts_forward.__get__(experts)
        n_attached += 2

    if n_attached:
        logger.info_rank0(
            "MoE-LoRA expert-weight compute path: "
            + (
                "grouped_mm (safe custom backward for adapter weights)"
                if _wp_grouped_enabled()
                else "per-expert loop (WEIGHT_PEFT_FORCE_LOOP or no grouped_mm kernel)"
            )
        )
    logger.info_rank0(
        "MoE-LoRA[ffn/expert_weight]: injected LoRA into "
        f"{n_attached} fused expert projection stages across {len(moe_blocks)} layers | "
        f"layer ranks={layer_ranks}"
    )
    return n_attached


def inject_weight_peft(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """Attach one weight-level method to FFN experts, attention, and router projections."""
    ptype = finetuning_args.weight_peft_type
    rank = finetuning_args.weight_peft_rank
    alpha = finetuning_args.weight_peft_alpha or (2 * rank)
    block = finetuning_args.weight_peft_oft_block_size
    fn = finetuning_args.weight_peft_fourier_n_frequencies
    fs = finetuning_args.weight_peft_fourier_scaling
    base_seed = finetuning_args.weight_peft_fourier_seed
    sample = next(model.parameters())
    device, dtype = sample.device, sample.dtype

    if rank < 1:
        raise ValueError(f"weight_peft_rank must be greater than 0, got {rank}")
    if ptype in ("oft", "oft_out") and block <= 1:
        raise ValueError(f"weight_peft_oft_block_size must be greater than 1, got {block}")


    # Per-adapter seed counter so every adapted weight gets a distinct but reproducible
    # FourierFT spectral support (the inject order is deterministic, so load replays it).
    _seed_ctr = [0]
    def _next_seed():
        _seed_ctr[0] += 1
        return base_seed + _seed_ctr[0]

    def _mk_stage(weight, stage, is_transposed=False):
        return _StageAdapter(weight, ptype, rank, alpha, block, stage=stage,
                             fourier_n_freq=fn, fourier_scaling=fs, fourier_seed=_next_seed(),
                             is_transposed=is_transposed)

    n_experts_attached = 0
    n_attn_attached = 0
    n_router_attached = 0

    # ---- experts (FFN): inject adapters, base grouped_mm stays on leaf weights ----
    if finetuning_args.weight_peft_attach_ffn:
        for blk in _find_moe_blocks(model):
            experts = _find_experts_module(blk)
            if experts is None:
                logger.warning_rank0("weight_peft: MoE block has no fused experts; skipping FFN.")
                continue
            is_transposed = getattr(experts, "is_transposed", False)
            adapters = nn.ModuleDict({
                "gate_up": _mk_stage(experts.gate_up_proj, "gate_up", is_transposed),
                "down": _mk_stage(experts.down_proj, "down", is_transposed),
            }).to(device=device, dtype=dtype)
            experts._wp_adapter = adapters
            experts._wp_orig_forward = experts.forward
            experts.forward = _wp_experts_forward.__get__(experts)
            n_experts_attached += 2
        if n_experts_attached:
            logger.info_rank0(
                "weight_peft expert compute path: "
                + ("grouped_mm (safe custom backward for adapter weights)"
                   if _wp_grouped_enabled()
                   else "per-expert loop (WEIGHT_PEFT_FORCE_LOOP or no grouped_mm kernel)")
            )

    # ---- attention: parametrize q/k/v/o .weight ----
    if finetuning_args.weight_peft_attach_attn:
        layers_container = _locate_decoder_layers(model)
        if layers_container is None:
            raise ValueError("weight_peft: cannot locate decoder layers for attention.")
        targets_cfg = getattr(finetuning_args, "weight_peft_attn_targets", "auto")
        for layer in layers_container.layers:
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                continue
            targets = _resolve_attn_targets(attn, targets_cfg)
            if ptype == "ia3":
                # Canonical IA3 rescales keys & values only.
                targets = [t for t in targets if ("k_proj" in t or "v_proj" in t)]
            for pname in targets:
                proj = getattr(attn, pname)
                adapter = _make_projection_adapter(
                    ptype, proj.weight, finetuning_args, "attn", seed=_next_seed()
                )
                adapter = adapter.to(device=device, dtype=dtype)
                _register_projection_adapter(proj, adapter, ptype)
                n_attn_attached += 1

    # ---- router: same method on the 2-D expert-logit projection ----
    if getattr(finetuning_args, "weight_peft_attach_router", False):
        for block_index, blk in enumerate(_find_moe_blocks(model)):
            router = _get_router_linear(blk)
            if router is None or not isinstance(getattr(router, "weight", None), torch.Tensor):
                raise ValueError(
                    f"weight_peft: MoE block {block_index} has no supported 2-D router projection."
                )
            adapter = _make_projection_adapter(
                ptype, router.weight, finetuning_args, "router", seed=_next_seed()
            )
            adapter = adapter.to(device=device, dtype=dtype)
            _register_projection_adapter(router, adapter, ptype)
            n_router_attached += 1

    # ---- freeze base, keep only adapter params trainable ----
    def _is_adapter(name: str) -> bool:
        return ("._wp_adapter." in name) or ("parametrizations." in name and ".original" not in name)

    for name, p in model.named_parameters():
        p.requires_grad = _is_adapter(name)

    def _real_numel(p):
        return getattr(p, "ds_numel", None) or p.numel()

    trainable = sum(_real_numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(_real_numel(p) for p in model.parameters())
    logger.info_rank0(
        f"weight_peft[{ptype}] setup complete | experts adapted: {n_experts_attached}, "
        f"attn adapted: {n_attn_attached}, router adapted: {n_router_attached} | "
        f"trainable: {trainable / 1e6:.2f}M "
        f"({100 * trainable / max(total, 1):.3f}%) | total: {total / 1e6:.2f}M"
    )
    if total > 0 and not (1e-5 < trainable / total < 0.10):
        logger.warning(
            f"weight_peft trainable ratio {trainable / total:.5f} out of expected range "
            "[1e-5, 0.1]; under ZeRO-3 base params may not be materialised yet -- verify after step 1."
        )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def _is_adapter_name(name: str) -> bool:
    return ("._wp_adapter." in name) or is_common_lora_parameter(name)


def save_weight_peft_state(model: nn.Module, save_dir: str,
                           finetuning_args: "FinetuningArguments") -> None:
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
        logger.warning_rank0("No trainable params, skip saving weight_peft state.")
        return
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    safetensors.torch.save_file(state, os.path.join(save_dir, "weight_peft_state.safetensors"))
    config = {
        "weight_peft_type": finetuning_args.weight_peft_type,
        "weight_peft_rank": finetuning_args.weight_peft_rank,
        "weight_peft_alpha": finetuning_args.weight_peft_alpha,
        "weight_peft_oft_block_size": finetuning_args.weight_peft_oft_block_size,
        "weight_peft_attn_rank": getattr(finetuning_args, "weight_peft_attn_rank", 0),
        "weight_peft_attn_alpha": getattr(finetuning_args, "weight_peft_attn_alpha", 0),
        "weight_peft_router_rank": getattr(finetuning_args, "weight_peft_router_rank", 0),
        "weight_peft_router_alpha": getattr(finetuning_args, "weight_peft_router_alpha", 0),
        "weight_peft_attn_oft_block_size": getattr(
            finetuning_args, "weight_peft_attn_oft_block_size", 0
        ),
        "weight_peft_router_oft_block_size": getattr(
            finetuning_args, "weight_peft_router_oft_block_size", 0
        ),
        "weight_peft_attach_attn": finetuning_args.weight_peft_attach_attn,
        "weight_peft_attach_ffn": finetuning_args.weight_peft_attach_ffn,
        "weight_peft_attach_router": getattr(finetuning_args, "weight_peft_attach_router", False),
        "weight_peft_attn_targets": finetuning_args.weight_peft_attn_targets,
        # FourierFT: the spectral support is regenerated from these on load (never stored).
        "weight_peft_fourier_n_frequencies": finetuning_args.weight_peft_fourier_n_frequencies,
        "weight_peft_attn_fourier_n_frequencies": getattr(
            finetuning_args, "weight_peft_attn_fourier_n_frequencies", 0
        ),
        "weight_peft_router_fourier_n_frequencies": getattr(
            finetuning_args, "weight_peft_router_fourier_n_frequencies", 0
        ),
        "weight_peft_fourier_scaling": finetuning_args.weight_peft_fourier_scaling,
        "weight_peft_fourier_seed": finetuning_args.weight_peft_fourier_seed,
        **common_lora_config(finetuning_args),
        "_meta_base_model": getattr(model.config, "name_or_path", "unknown"),
        # OFT convention marker: "input" = HF-peft-aligned input-side rotation (current
        # "oft"). Checkpoints saved before this marker existed used the legacy
        # output-side rotation; the loader remaps their "oft" to "oft_out".
        "_meta_oft_convention": (
            "input" if finetuning_args.weight_peft_type == "oft" else
            ("output" if finetuning_args.weight_peft_type == "oft_out" else None)
        ),
    }
    with open(os.path.join(save_dir, "weight_peft_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info_rank0(f"Saved weight_peft state ({len(state)} tensors) to {save_dir}")


def load_weight_peft_state(model: "PreTrainedModel", load_dir: str) -> "PreTrainedModel":
    from ...hparams import FinetuningArguments

    config_path = os.path.join(load_dir, "weight_peft_config.json")
    state_path = os.path.join(load_dir, "weight_peft_state.safetensors")
    if not os.path.exists(config_path) or not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing weight_peft_config.json or weight_peft_state.safetensors in {load_dir}")

    with open(config_path) as f:
        config = json.load(f)
    meta = {k: config.pop(k) for k in list(config.keys()) if k.startswith("_meta_")}

    # Legacy OFT checkpoints (saved before the HF alignment) carry no convention marker
    # and were trained with the output-side rotation -- load them as "oft_out" so the
    # rebuilt adapters use the math the skew parameters were trained under.
    if config.get("weight_peft_type") == "oft" and meta.get("_meta_oft_convention") != "input":
        logger.warning_rank0(
            "weight_peft: legacy OFT checkpoint (no input-convention marker); "
            "loading as weight_peft_type=oft_out (output-side rotation)."
        )
        config["weight_peft_type"] = "oft_out"

    finetuning_args = FinetuningArguments(finetuning_type="weight_peft", **config)
    inject_weight_peft(model, finetuning_args)
    inject_common_lora(model, finetuning_args)

    state = safetensors.torch.load_file(state_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert len(unexpected) == 0, f"Unexpected weights in checkpoint: {unexpected[:5]}"
    unloaded = [n for n in missing if _is_adapter_name(n)]
    assert not unloaded, f"Failed to load weight_peft params: {unloaded[:5]}"

    base_meta = meta.get("_meta_base_model")
    actual_base = getattr(model.config, "name_or_path", "unknown")
    if base_meta and base_meta != actual_base:
        logger.warning_rank0(f"Base model mismatch: loaded {actual_base} but checkpoint trained on {base_meta}.")
    logger.info_rank0(f"Loaded weight_peft state from {load_dir} ({len(state)} tensors)")
    return model


class WeightPeftSaveCallback(TrainerCallback):
    def __init__(self, finetuning_args: "FinetuningArguments"):
        self.finetuning_args = finetuning_args

    def on_save(self, args, state, control, model: Optional[nn.Module] = None, **kwargs):
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_weight_peft_state(model, ckpt_dir, self.finetuning_args)
        import glob
        for pattern in ("model.safetensors", "model-*.safetensors", "model.safetensors.index.json"):
            for f in glob.glob(os.path.join(ckpt_dir, pattern)):
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
