"""vLLM static-merge overlay for weight-level PEFT / expert-weight / partial-last checkpoints.

这些臂的训练侧前向都是**与 token/路由无关的静态权重变换**（不像 lora_pool 的
增量依赖每个 token 走哪条路由），因此可以在 vLLM 加载权重流里逐专家/逐投影地把
增量折进底座权重，之后照常走 FusedMoE 的 grouped 内核——拿到 vLLM 的推理加速。

覆盖的结构族（vllm_moe_lora.parse_overlay 按存在的 config 文件分派到这里）：
  * weight_peft_state.safetensors (weight_dora / weight_oft / weight_ia3 / weight_lora)
      - experts._wp_adapter.{gate_up,down}.*  逐专家融合权重变换
      - mlp.gate.parametrizations.weight.0.*  router 2D 权重变换
      FourierFT 不在此列：谱支撑索引由训练注入顺序的 seed 计数器复现，逐字节
      耦合注入序实现细节，静默错数风险高 → 仍走 HF。
  * moe_lora_state.safetensors 且含 _wp_adapter（FFN 粒度 expert_weight_rX / aux 链）
      - experts._wp_adapter.{gate_up,down}.{A,B}  逐专家 LoRA
      （attn `_lora_adapter` / router parametrize 仍由 vllm_moe_lora 原有分支合并）
  * partial_ft_state.safetensors (partial_last, modules=expert,router)
      - experts.{gate_up_proj,down_proj} 融合全权重 → 逐专家全替换
      - 其余完整权重名（mlp.gate.weight 等）→ 直接替换

不支持（fail fast → HF）：
  * gptoss：底座专家权重是 mxfp4 量化块（gate_up_proj_blocks/scales），merge 需要
    反量化-再量化，有损 → 数值口径无法保证。
  * gemma4：vLLM transformers backend 本就强制 HF（run_upstream_eval.sh）。
  * fourier（见上）。

vLLM qwen3 底座权重流里专家是逐专家 2D 命名（experts.E.{gate,up,down}_proj.weight），
gate_up 在 vLLM 内部才融合；训练侧融合权重 [E, 2*inter, hidden] 的前 inter 行是
gate、后 inter 行是 up（transformers Qwen3MoE `.chunk(2, dim=-1)` 口径，与 vLLM
`(gate_up_proj, gate_proj, 0), (gate_up_proj, up_proj, 1)` 一致），据此把融合增量
按输出维切半分别合入。

dtype 口径（延续 2026-07-22 router 1 ulp 教训）：
  * router 变换逐算子复刻训练侧 parametrization forward 的 bf16 顺序（在 CUDA 上
    执行以复现 GPU 归约序），追求逐位一致——router 输出决定 128 选 8 的路由边界。
  * 专家权重变换用 fp32 合并后单次 cast——训练侧本就是激活级旁路，合并权重在
    激活级无法逐位对齐，fp32 是数学上最准的合并（同 attn 先例）。

CPU 自检（直接对照训练侧 weight_peft/moe_lora 实现的数值）::

    python eval_scripts/vllm_weight_peft_overlay.py --self_test
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, Tuple

import torch

Transform = Callable[[torch.Tensor], torch.Tensor]

_RE_WP_EXPERT = re.compile(
    r"^(?P<prefix>.+\.experts)\._wp_adapter\.(?P<stage>gate_up|down)\.(?P<param>A|B|magnitude|skew|l_ff|spectrum)$"
)
_RE_ROUTER_PARAM = re.compile(
    r"^(?P<prefix>.+)\.parametrizations\.weight\.0\.(?P<param>A|B|magnitude|skew|l|spectrum)$"
)
_RE_ATTN_LORA = re.compile(r"^(?P<prefix>.+)\._lora_adapter\.(?P<ab>A|B)\.weight$")


def _guard_base_model(cfg: dict, family: str) -> None:
    """gptoss（mxfp4 量化专家）/ gemma4 的权重级 merge 不支持，fail fast 走 HF。"""
    base = str(cfg.get("_meta_base_model", ""))
    low = base.lower()
    if "gpt-oss" in low or "gpt_oss" in low:
        raise NotImplementedError(
            f"{family}: gptoss 底座专家权重是 mxfp4 量化块，静态 merge 需反量化-再量化"
            "（有损），不支持 vLLM overlay，请走 HF。"
        )
    if "gemma" in low:
        raise NotImplementedError(f"{family}: gemma4 一律走 HF（vLLM backend 限制）。")


def _dev() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# router 2D 变换：逐算子复刻训练侧 parametrization forward（bf16 顺序，CUDA 上执行）
# ---------------------------------------------------------------------------


def _router_lora_tf(A: torch.Tensor, B: torch.Tensor, scaling: float) -> Transform:
    """WeightLoRA / LoRAWeightDelta: W + (s*(B@A)).to(W.dtype)，B@A 在 W.dtype 下算。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        dev = _dev()
        wd = w.to(dev)
        delta = (scaling * (B.to(dev, wd.dtype) @ A.to(dev, wd.dtype))).to(wd.dtype)
        return (wd + delta).to(w.device)

    return tf


def _router_dora_tf(A: torch.Tensor, B: torch.Tensor, magnitude: torch.Tensor, scaling: float) -> Transform:
    """WeightDoRA.forward 逐算子复刻:
    v = W + (s*(B@A)).to(W.dtype); n = norm(v.float(),dim=1,keepdim).to(W.dtype).detach();
    W_eff = (magnitude.unsqueeze(-1) / n) * v
    """

    def tf(w: torch.Tensor) -> torch.Tensor:
        dev = _dev()
        wd = w.to(dev)
        v = wd + (scaling * (B.to(dev, wd.dtype) @ A.to(dev, wd.dtype))).to(wd.dtype)
        n = torch.linalg.norm(v.float(), dim=1, keepdim=True).to(wd.dtype)
        return ((magnitude.to(dev, wd.dtype).unsqueeze(-1) / n) * v).to(w.device)

    return tf


def _router_oft_tf(skew: torch.Tensor) -> Transform:
    """WeightOFTOut.forward 复刻（legacy 输出侧）: R = Cayley(skew)，块对角 R@W。"""
    n_blocks, n_free = skew.shape
    b = _block_size_from_free(n_free)

    def tf(w: torch.Tensor) -> torch.Tensor:
        dev = _dev()
        wd = w.to(dev)
        R = _cayley(skew.to(dev), b).to(wd.dtype)          # [nb, b, b]
        out, in_ = wd.shape
        if n_blocks * b != out:
            raise RuntimeError(f"OFT router 块结构不匹配: {n_blocks}x{b} vs out={out}")
        wv = wd.view(n_blocks, b, in_)
        return torch.einsum("nij,njk->nik", R, wv).reshape(out, in_).to(w.device)

    return tf


def _router_oft_in_tf(skew: torch.Tensor) -> Transform:
    """WeightOFT.forward 复刻（HF 对齐输入侧）: W_eff = W @ blockdiag(R)^T，块沿 in 维。"""
    n_blocks, n_free = skew.shape
    b = _block_size_from_free(n_free)

    def tf(w: torch.Tensor) -> torch.Tensor:
        dev = _dev()
        wd = w.to(dev)
        R = _cayley(skew.to(dev), b).to(wd.dtype)          # [nb, b, b]
        out, in_ = wd.shape
        if n_blocks * b != in_:
            raise RuntimeError(f"OFT(in) router 块结构不匹配: {n_blocks}x{b} vs in={in_}")
        wv = wd.view(out, n_blocks, b)
        return torch.einsum("onk,nck->onc", wv, R).reshape(out, in_).to(w.device)

    return tf


def _router_ia3_tf(l: torch.Tensor) -> Transform:
    """WeightIA3.forward 复刻: l.to(W.dtype).unsqueeze(1) * W（行缩放）。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        return l.to(w.device, w.dtype).unsqueeze(1) * w

    return tf


def _block_size_from_free(n_free: int) -> int:
    """由 skew 自由参数数 n_free = b(b-1)/2 反解块大小 b（不信 config，以 shape 为准）。"""
    b = int((1 + (1 + 8 * n_free) ** 0.5) / 2 + 0.5)
    if b * (b - 1) // 2 != n_free:
        raise RuntimeError(f"OFT skew 自由参数数 {n_free} 不是任何 b(b-1)/2")
    return b


def _cayley(skew_flat: torch.Tensor, b: int) -> torch.Tensor:
    """与训练侧 _cayley_R 相同的精确 Cayley（fp32 solve）。skew_flat: [..., n_free]。"""
    triu = torch.triu_indices(b, b, offset=1, device=skew_flat.device)
    lead = skew_flat.shape[:-1]
    Q = torch.zeros(*lead, b, b, device=skew_flat.device, dtype=torch.float32)
    Q[..., triu[0], triu[1]] = skew_flat.float()
    Q = Q - Q.transpose(-1, -2)
    eye = torch.eye(b, device=skew_flat.device, dtype=torch.float32).expand_as(Q)
    return torch.linalg.solve(eye + Q, eye - Q)


# ---------------------------------------------------------------------------
# 逐专家权重变换（fp32 合并，单次 cast）
# ---------------------------------------------------------------------------


def _expert_lora_tf(A_e: torch.Tensor, B_rows: torch.Tensor, scaling: float) -> Transform:
    """W_eff = W + s * B_rows @ A_e（fp32）。B_rows 已按 gate/up/down 切好行。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        if w.shape != (B_rows.shape[0], A_e.shape[1]):
            raise RuntimeError(f"expert LoRA 形状不匹配: W{tuple(w.shape)} vs ({B_rows.shape[0]},{A_e.shape[1]})")
        return (w.float() + scaling * (B_rows.float() @ A_e.float())).to(w.dtype)

    return tf


def _expert_dora_tf(A_e: torch.Tensor, B_rows: torch.Tensor, mag_rows: torch.Tensor, scaling: float) -> Transform:
    """W_eff = (m / ||W + sBA||_row) ⊙ (W + sBA)（fp32，与训练 apply_grouped 同数学）。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        v = w.float() + scaling * (B_rows.float() @ A_e.float())
        norm = torch.linalg.norm(v, dim=1, keepdim=True)
        return ((mag_rows.float().unsqueeze(-1) / norm) * v).to(w.dtype)

    return tf


def _expert_oft_tf(skew_blocks: torch.Tensor, b: int) -> Transform:
    """legacy 输出侧: W_eff = blockdiag(R) @ W。skew_blocks 已按 gate/up 切好块。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        R = _cayley(skew_blocks, b)                        # [nb, b, b] fp32
        out, in_ = w.shape
        nb = skew_blocks.shape[0]
        if nb * b != out:
            raise RuntimeError(f"expert OFT 块结构不匹配: {nb}x{b} vs out={out}")
        wv = w.float().view(nb, b, in_)
        return torch.einsum("nij,njk->nik", R, wv).reshape(out, in_).to(w.dtype)

    return tf


def _expert_oft_in_tf(skew_blocks: torch.Tensor, b: int) -> Transform:
    """HF 对齐输入侧: W_eff = W @ blockdiag(R)^T，块沿 in 维（gate/up 共享同一 R）。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        R = _cayley(skew_blocks, b)                        # [nb, b, b] fp32
        out, in_ = w.shape
        nb = skew_blocks.shape[0]
        if nb * b != in_:
            raise RuntimeError(f"expert OFT(in) 块结构不匹配: {nb}x{b} vs in={in_}")
        wv = w.float().view(out, nb, b)
        return torch.einsum("onk,nck->onc", wv, R).reshape(out, in_).to(w.dtype)

    return tf


def _expert_ia3_down_tf(l_e: torch.Tensor) -> Transform:
    """IA3 down: y = W(l⊙x) = (W diag(l)) x → 列缩放。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        if w.shape[1] != l_e.shape[0]:
            raise RuntimeError(f"expert IA3 形状不匹配: W{tuple(w.shape)} vs l({l_e.shape[0]})")
        return (w.float() * l_e.float().unsqueeze(0)).to(w.dtype)

    return tf


def _replace_tf(new_w: torch.Tensor) -> Transform:
    """partial_ft: 直接替换（shape 校验，cast 到底座 dtype）。"""

    def tf(w: torch.Tensor) -> torch.Tensor:
        if w.shape != new_w.shape:
            raise RuntimeError(f"partial_ft 替换形状不匹配: base{tuple(w.shape)} vs ckpt{tuple(new_w.shape)}")
        return new_w.to(w.dtype)

    return tf


def _chain(t1: Transform, t2: Transform) -> Transform:
    def tf(w: torch.Tensor) -> torch.Tensor:
        return t2(t1(w))

    return tf


# ---------------------------------------------------------------------------
# 融合专家 stage → 逐专家目标名展开
# ---------------------------------------------------------------------------


def _expand_expert_stage(
    transforms: Dict[str, Transform],
    prefix: str,           # e.g. model.layers.0.mlp.experts
    stage: str,            # gate_up | down
    make_tf,               # (expert_idx, split) -> Transform；split ∈ {gate,up,down}
    n_experts: int,
) -> None:
    for e in range(n_experts):
        if stage == "gate_up":
            transforms[f"{prefix}.{e}.gate_proj.weight"] = make_tf(e, "gate")
            transforms[f"{prefix}.{e}.up_proj.weight"] = make_tf(e, "up")
        else:
            transforms[f"{prefix}.{e}.down_proj.weight"] = make_tf(e, "down")


# ---------------------------------------------------------------------------
# 三个家族的解析入口
# ---------------------------------------------------------------------------


def parse_weight_peft(adapter_path: str) -> Tuple[Dict[str, Transform], dict]:
    """weight_peft_state.safetensors → {底座权重名: 变换}。

    返回 (transforms, config)。不支持的结构 raise NotImplementedError（→ HF）。
    """
    import safetensors.torch

    config_path = os.path.join(adapter_path, "weight_peft_config.json")
    state_path = os.path.join(adapter_path, "weight_peft_state.safetensors")
    if not (os.path.exists(config_path) and os.path.exists(state_path)):
        raise FileNotFoundError(f"{adapter_path} 缺 weight_peft_config.json / weight_peft_state.safetensors")
    with open(config_path) as f:
        cfg = json.load(f)
    _guard_base_model(cfg, "weight_peft")

    ptype = cfg.get("weight_peft_type")
    if ptype == "fourier":
        raise NotImplementedError(
            "weight_peft[fourier]: 谱支撑由训练注入顺序的 seed 计数器复现，"
            "静默错数风险高，不做 vLLM merge，请走 HF。"
        )
    if ptype not in ("lora", "dora", "oft", "oft_out", "ia3"):
        raise NotImplementedError(f"weight_peft type 不支持 vLLM merge: {ptype!r}")
    # 旧 checkpoint（无 convention 标记）的 "oft" 是 legacy 输出侧旋转 → 按 oft_out 处理
    if ptype == "oft" and cfg.get("_meta_oft_convention") != "input":
        ptype = "oft_out"

    state = safetensors.torch.load_file(state_path)

    rank = cfg.get("weight_peft_rank") or 1
    alpha = cfg.get("weight_peft_alpha") or (2 * rank)
    ffn_scaling = float(alpha) / float(rank)
    r_rank = cfg.get("weight_peft_router_rank") or rank
    r_alpha = cfg.get("weight_peft_router_alpha") or alpha
    router_scaling = float(r_alpha) / float(r_rank)
    a_rank = cfg.get("weight_peft_attn_rank") or rank
    a_alpha = cfg.get("weight_peft_attn_alpha") or alpha
    attn_scaling = float(a_alpha) / float(a_rank)

    transforms: Dict[str, Transform] = {}
    consumed: set = set()

    # ---- 逐专家 stage（按 (prefix, stage) 聚参数）----
    stages: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    for name in state:
        m = _RE_WP_EXPERT.match(name)
        if m:
            stages.setdefault((m.group("prefix"), m.group("stage")), {})[m.group("param")] = state[name]
            consumed.add(name)

    for (prefix, stage), params in stages.items():
        _build_expert_stage(transforms, prefix, stage, ptype, params, ffn_scaling)

    # ---- router parametrization ----
    routers: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in state:
        if name in consumed:
            continue
        m = _RE_ROUTER_PARAM.match(name)
        if m:
            routers.setdefault(m.group("prefix"), {})[m.group("param")] = state[name]
            consumed.add(name)

    for prefix, params in routers.items():
        target = f"{prefix}.weight"
        if ptype == "lora":
            transforms[target] = _router_lora_tf(params.pop("A"), params.pop("B"), router_scaling)
        elif ptype == "dora":
            transforms[target] = _router_dora_tf(
                params.pop("A"), params.pop("B"), params.pop("magnitude"), router_scaling
            )
        elif ptype == "oft":
            transforms[target] = _router_oft_in_tf(params.pop("skew"))
        elif ptype == "oft_out":
            transforms[target] = _router_oft_tf(params.pop("skew"))
        else:  # ia3
            transforms[target] = _router_ia3_tf(params.pop("l"))
        if params:
            raise NotImplementedError(f"router parametrization 余留未识别参数 {list(params)} @ {prefix}")

    # ---- attn common LoRA（attach_attn=True 的链；noattn 白名单里为空集）----
    for name in state:
        if name in consumed:
            continue
        m = _RE_ATTN_LORA.match(name)
        if m and m.group("ab") == "A":
            b_name = f"{m.group('prefix')}._lora_adapter.B.weight"
            if b_name not in state:
                raise RuntimeError(f"attn LoRA 缺 B: {b_name}")
            # attn 是运行时旁路，fp32 合并（同 vllm_moe_lora attn 口径）
            transforms[f"{m.group('prefix')}.weight"] = _expert_lora_tf(
                state[name], state[b_name], attn_scaling
            )
            consumed.update((name, b_name))

    unknown = [n for n in state if n not in consumed]
    if unknown:
        raise NotImplementedError(f"weight_peft 无法识别的张量（前 5 个）: {unknown[:5]}")
    if not transforms:
        raise NotImplementedError(f"weight_peft: {adapter_path} 未产出任何变换")
    return transforms, cfg


def _build_expert_stage(
    transforms: Dict[str, Transform],
    prefix: str,
    stage: str,
    ptype: str,
    params: Dict[str, torch.Tensor],
    scaling: float,
) -> None:
    """把一个 (experts prefix, stage) 的融合 adapter 参数展开成逐专家变换。"""
    if ptype in ("lora", "dora"):
        A, B = params.pop("A"), params.pop("B")           # A:[E,r,in] B:[E,out,r]
        E, out = B.shape[0], B.shape[1]
        inter = out // 2 if stage == "gate_up" else None
        mag = params.pop("magnitude", None)               # dora: [E,out]
        if ptype == "dora" and mag is None:
            raise RuntimeError(f"dora 缺 magnitude @ {prefix}.{stage}")

        def make_tf(e: int, split: str) -> Transform:
            if split == "gate":
                rows = slice(0, inter)
            elif split == "up":
                rows = slice(inter, out)
            else:
                rows = slice(0, out)
            if ptype == "lora":
                return _expert_lora_tf(A[e], B[e][rows], scaling)
            return _expert_dora_tf(A[e], B[e][rows], mag[e][rows], scaling)

        _expand_expert_stage(transforms, prefix, stage, make_tf, E)

    elif ptype == "oft":
        # HF 对齐输入侧：R 作用于 stage 输入维，gate/up/down 都用整个 skew（不切块）。
        skew = params.pop("skew")                          # [E, nb_in, n_free]
        E, nb, n_free = skew.shape
        b = _block_size_from_free(n_free)

        def make_tf(e: int, split: str) -> Transform:
            return _expert_oft_in_tf(skew[e], b)

        _expand_expert_stage(transforms, prefix, stage, make_tf, E)

    elif ptype == "oft_out":
        skew = params.pop("skew")                          # [E, nb, n_free]
        E, nb, n_free = skew.shape
        b = _block_size_from_free(n_free)
        half = nb // 2                                     # gate_up: 前半块=gate 行

        def make_tf(e: int, split: str) -> Transform:
            if split == "gate":
                blocks = skew[e][:half]
            elif split == "up":
                blocks = skew[e][half:]
            else:
                blocks = skew[e]
            return _expert_oft_tf(blocks, b)

        _expand_expert_stage(transforms, prefix, stage, make_tf, E)

    elif ptype == "ia3":
        if stage != "down":
            raise NotImplementedError(f"ia3 不应有 {stage} stage 参数 @ {prefix}")
        l_ff = params.pop("l_ff")                          # [E, inter]
        E = l_ff.shape[0]

        def make_tf(e: int, split: str) -> Transform:
            return _expert_ia3_down_tf(l_ff[e])

        _expand_expert_stage(transforms, prefix, "down", make_tf, E)

    else:
        raise NotImplementedError(f"expert stage 不支持: {ptype!r}")

    if params:
        raise NotImplementedError(f"expert stage 余留未识别参数 {list(params)} @ {prefix}.{stage}")


def extract_expert_weight_transforms(
    state: Dict[str, torch.Tensor], cfg: dict, consumed: set
) -> Dict[str, Transform]:
    """moe_lora_state 里的 _wp_adapter（FFN 粒度 expert_weight_rX）→ 逐专家 LoRA 变换。

    只消费 `_wp_adapter` 键并加入 consumed；attn/router 键留给 vllm_moe_lora 原有分支。
    per-layer rank（moe_lora_layer_ranks）时 _scaled_alpha 恒保 alpha/rank 比例，
    scaling 用全局比例即可。
    """
    _guard_base_model(cfg, "moe_lora[expert_weight]")
    scaling = float(cfg.get("moe_lora_alpha", 16)) / float(cfg.get("moe_lora_rank", 8))

    stages: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
    for name in state:
        m = _RE_WP_EXPERT.match(name)
        if m:
            stages.setdefault((m.group("prefix"), m.group("stage")), {})[m.group("param")] = state[name]
            consumed.add(name)

    transforms: Dict[str, Transform] = {}
    for (prefix, stage), params in stages.items():
        _build_expert_stage(transforms, prefix, stage, "lora", params, scaling)
    return transforms


def parse_partial_ft(adapter_path: str) -> Tuple[Dict[str, Transform], dict]:
    """partial_ft_state.safetensors → 直接替换变换。

    融合专家名（experts.gate_up_proj / experts.down_proj，[E,...]）拆成逐专家目标；
    其余完整权重名直接替换（mlp.gate.weight 等）。
    """
    import safetensors.torch

    config_path = os.path.join(adapter_path, "partial_ft_config.json")
    state_path = os.path.join(adapter_path, "partial_ft_state.safetensors")
    if not (os.path.exists(config_path) and os.path.exists(state_path)):
        raise FileNotFoundError(f"{adapter_path} 缺 partial_ft_config.json / partial_ft_state.safetensors")
    with open(config_path) as f:
        cfg = json.load(f)
    _guard_base_model(cfg, "partial_ft")

    state = safetensors.torch.load_file(state_path)
    transforms: Dict[str, Transform] = {}

    for name, w in state.items():
        m = re.match(r"^(?P<prefix>.+\.experts)\.gate_up_proj$", name)
        if m:
            E, out2, _ = w.shape
            inter = out2 // 2
            for e in range(E):
                transforms[f"{m.group('prefix')}.{e}.gate_proj.weight"] = _replace_tf(w[e][:inter])
                transforms[f"{m.group('prefix')}.{e}.up_proj.weight"] = _replace_tf(w[e][inter:])
            continue
        m = re.match(r"^(?P<prefix>.+\.experts)\.down_proj$", name)
        if m:
            for e in range(w.shape[0]):
                transforms[f"{m.group('prefix')}.{e}.down_proj.weight"] = _replace_tf(w[e])
            continue
        if w.dim() == 3:
            raise NotImplementedError(f"partial_ft 未识别的 3D 融合权重: {name} {tuple(w.shape)}")
        # 2D/1D 完整权重（router gate.weight、可能的 attn/norm 等）→ 直接替换
        transforms[name] = _replace_tf(w)

    if not transforms:
        raise NotImplementedError(f"partial_ft: {adapter_path} 空 state")
    return transforms, cfg


# ---------------------------------------------------------------------------
# CPU 自检：直接对照训练侧实现
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    import torch.nn as nn
    import torch.nn.utils.parametrize as parametrize
    from llamafactory.model.model_utils.weight_peft import (  # noqa: E402
        WeightDoRA, WeightIA3, WeightLoRA, WeightOFT, WeightOFTOut, _StageAdapter,
    )

    torch.manual_seed(0)
    E, inter, hidden, r = 4, 16, 32, 2
    alpha, scaling = 4, 2.0
    x = torch.randn(6, hidden)
    h_inter = torch.randn(6, inter)

    def rand_stage(stage, ptype, block=4):
        out = 2 * inter if stage == "gate_up" else hidden
        in_ = hidden if stage == "gate_up" else inter
        w = torch.randn(E, out, in_)
        ad = _StageAdapter(w, ptype, r, alpha, block, stage=stage)
        with torch.no_grad():
            for p in ad.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        return w, ad

    # ---- 1) expert lora/dora/oft/oft_out/ia3: apply_chunk vs merged W_eff ----
    for ptype in ("lora", "dora", "oft", "oft_out", "ia3"):
        for stage, xin in (("gate_up", x), ("down", h_inter)):
            if ptype == "ia3" and stage != "down":
                continue
            w, ad = rand_stage(stage, ptype)
            params = {k.rsplit(".", 1)[-1]: v.detach() for k, v in ad.named_parameters()}
            transforms: Dict[str, Transform] = {}
            _build_expert_stage(transforms, "model.layers.0.mlp.experts", stage, ptype, dict(params), scaling)
            for e in range(E):
                base_c = torch.nn.functional.linear(xin, w[e])
                ref = ad.apply_chunk(xin, base_c, w, e)
                if stage == "gate_up":
                    wg = transforms[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"](w[e][:inter])
                    wu = transforms[f"model.layers.0.mlp.experts.{e}.up_proj.weight"](w[e][inter:])
                    got = torch.nn.functional.linear(xin, torch.cat([wg, wu], dim=0))
                else:
                    wd = transforms[f"model.layers.0.mlp.experts.{e}.down_proj.weight"](w[e])
                    got = torch.nn.functional.linear(xin, wd)
                assert torch.allclose(ref, got, atol=1e-4), (
                    f"expert {ptype}/{stage} e{e} mismatch: {(ref - got).abs().max()}"
                )
    print("self_test: expert lora/dora/oft/ia3 (gate_up 拆分含) 与训练 apply_chunk 一致")

    # ---- 2) router 变换: parametrize forward vs transform，bf16 逐位 ----
    n_out = 24
    for ptype, mk_ad, mk_tf in (
        ("lora", lambda w: WeightLoRA(w, r, alpha), lambda p: _router_lora_tf(p["A"], p["B"], scaling)),
        ("dora", lambda w: WeightDoRA(w, r, alpha),
         lambda p: _router_dora_tf(p["A"], p["B"], p["magnitude"], scaling)),
        ("oft", lambda w: WeightOFT(w, 4), lambda p: _router_oft_in_tf(p["skew"])),
        ("oft_out", lambda w: WeightOFTOut(w, 4), lambda p: _router_oft_tf(p["skew"])),
        ("ia3", lambda w: WeightIA3(w), lambda p: _router_ia3_tf(p["l"])),
    ):
        gate = nn.Linear(hidden, n_out, bias=False).to(torch.bfloat16)
        w0 = gate.weight.detach().clone()
        ad = mk_ad(gate.weight).to(torch.bfloat16)
        with torch.no_grad():
            for p in ad.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        parametrize.register_parametrization(gate, "weight", ad)
        ref_w = gate.weight.detach()
        params = {k: v.detach() for k, v in ad.named_parameters()}
        got_w = mk_tf(params)(w0)
        if ptype in ("lora", "ia3"):
            assert torch.equal(ref_w, got_w), (
                f"router {ptype} 不逐位一致: {(ref_w != got_w).float().mean():.4%}"
            )
        else:
            # dora/oft 含 fp32 norm/solve，CPU 参考下应逐位；宽松断言防归约序差异
            assert torch.allclose(ref_w.float(), got_w.float(), atol=1e-2), (
                f"router {ptype} mismatch: {(ref_w.float() - got_w.float()).abs().max()}"
            )
            eq = (ref_w == got_w).float().mean().item()
            print(f"  router {ptype}: bitwise equal ratio = {eq:.4%}")
    print("self_test: router lora/dora/oft/ia3 与训练 parametrize 一致")

    # ---- 3) expert_weight (moe_lora _wp_adapter) 提取 ----
    w, ad = rand_stage("gate_up", "lora")
    state = {
        "model.layers.0.mlp.experts._wp_adapter.gate_up.A": ad.A.detach(),
        "model.layers.0.mlp.experts._wp_adapter.gate_up.B": ad.B.detach(),
    }
    consumed: set = set()
    tfs = extract_expert_weight_transforms(state, {"moe_lora_rank": r, "moe_lora_alpha": alpha}, consumed)
    assert len(consumed) == 2 and len(tfs) == 2 * E
    base_c = torch.nn.functional.linear(x, w[0])
    ref = ad.apply_chunk(x, base_c, w, 0)
    wg = tfs["model.layers.0.mlp.experts.0.gate_proj.weight"](w[0][:inter])
    wu = tfs["model.layers.0.mlp.experts.0.up_proj.weight"](w[0][inter:])
    got = torch.nn.functional.linear(x, torch.cat([wg, wu], dim=0))
    assert torch.allclose(ref, got, atol=1e-4)
    print("self_test: expert_weight (moe_lora _wp_adapter) 提取一致")

    # ---- 4) partial_ft 替换 ----
    fused_gu = torch.randn(E, 2 * inter, hidden, dtype=torch.bfloat16)
    fused_dn = torch.randn(E, hidden, inter, dtype=torch.bfloat16)
    gate_w = torch.randn(E, hidden, dtype=torch.bfloat16)
    import tempfile
    import safetensors.torch as st

    with tempfile.TemporaryDirectory() as td:
        st.save_file(
            {
                "model.layers.47.mlp.experts.gate_up_proj": fused_gu,
                "model.layers.47.mlp.experts.down_proj": fused_dn,
                "model.layers.47.mlp.gate.weight": gate_w,
            },
            os.path.join(td, "partial_ft_state.safetensors"),
        )
        with open(os.path.join(td, "partial_ft_config.json"), "w") as f:
            json.dump({"partial_ft_target": "last", "partial_ft_modules": "expert,router"}, f)
        tfs2, _ = parse_partial_ft(td)
    assert len(tfs2) == 3 * E + 1
    got_g = tfs2["model.layers.47.mlp.experts.1.gate_proj.weight"](torch.zeros(inter, hidden, dtype=torch.bfloat16))
    assert torch.equal(got_g, fused_gu[1][:inter])
    got_u = tfs2["model.layers.47.mlp.experts.1.up_proj.weight"](torch.zeros(inter, hidden, dtype=torch.bfloat16))
    assert torch.equal(got_u, fused_gu[1][inter:])
    got_d = tfs2["model.layers.47.mlp.experts.2.down_proj.weight"](torch.zeros(hidden, inter, dtype=torch.bfloat16))
    assert torch.equal(got_d, fused_dn[2])
    assert torch.equal(tfs2["model.layers.47.mlp.gate.weight"](torch.zeros(E, hidden, dtype=torch.bfloat16)), gate_w)
    print("self_test: partial_ft 逐专家拆分替换一致")

    # ---- 5) gptoss/gemma guard ----
    for bad in ("/x/gpt-oss-20b/snap", "/x/gemma-4-26B-A4B/snap"):
        try:
            _guard_base_model({"_meta_base_model": bad}, "t")
            raise AssertionError(f"guard 未拦截 {bad}")
        except NotImplementedError:
            pass
    print("self_test: gptoss/gemma guard OK")
    print("vllm_weight_peft_overlay self_test: 全部通过")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--parse", default=None, help="解析一个 adapter 目录并打印变换数")
    args = p.parse_args()
    if args.self_test:
        _self_test()
    if args.parse:
        d = args.parse
        if os.path.exists(os.path.join(d, "weight_peft_config.json")):
            tfs, _ = parse_weight_peft(d)
        elif os.path.exists(os.path.join(d, "partial_ft_config.json")):
            tfs, _ = parse_partial_ft(d)
        else:
            raise SystemExit("不是 weight_peft / partial_ft 目录")
        print(f"parsed {d}: {len(tfs)} weight transforms")
