"""vLLM overlay loader for self-implemented moe_lora checkpoints (attn / dense-FFN / router family).

背景：本仓库的 moe_lora adapter（`moe_lora_state.safetensors` + `moe_lora_config.json`）
是训练侧 monkey-patch 实现，vLLM 无法直接加载。本模块把 CSV 实验臂
（budget_align_lr / router_rank_pilot：attn_rXX / dense_rXXX / attn+dense / *_router_r8）
用到的三类子结构映射成 vLLM 可用的形式：

  1. attention LoRA  (`*.<proj>._lora_adapter.{A,B}.weight`)
       旁路 out = Wx + s·B(A(x)) 等价于合并 W_eff = W + s·(B@A)
       -> 在权重加载流里直接改写 `<proj>.weight`（TP 切分发生在合并之后，数学精确）。
  2. router LoRA     (`*.parametrizations.weight.0.{A,B}`，torch parametrize 低秩增量)
       W_eff = W + s·(B@A) 本身就是权重增量 -> 同样在加载流里合并。
  3. dense FFN 旁路  (`*.lora_pool.experts.0.{A,B}.weight`)
       仅支持 routing_mode=independent + n_experts=1 + top_k=1：单类 softmax 恒等于 1，
       LoRAPool 退化为稠密旁路 out += s·B(A(h))，与路由无关。
       -> 运行时给对应 MoE block/experts 模块的 forward 挂 wrapper（不可合并）。
       （`independent_router.gate.weight` 是无害死参数，读取后仅用于校验，不参与计算。）

不支持（fail fast）：n_experts>1 的 pool、learned/follow_moe/residual 路由、
global/grouped/cross-layer pool、hidden bottleneck。这些臂请继续用 HF 路径评测。

用法（评测脚本内，创建 LLM 之前）::

    import vllm_moe_lora
    vllm_moe_lora.activate(adapter_path)   # 注册三个架构的自定义模型类
    llm = LLM(model=base_model, tensor_parallel_size=1, enforce_eager=True, ...)

约束：需要 in-process engine（activate 会设 VLLM_ENABLE_V1_MULTIPROCESSING=0）；
建议 tensor_parallel_size=1（H200 单卡放得下三个模型）+ enforce_eager=True
（dense 旁路是 instance-level forward wrapper，不进 torch.compile 图最稳）。

自检（CPU，无需 GPU，直接对照训练侧实现的数值）::

    python eval_scripts/vllm_moe_lora.py --self_test
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch


_FORBIDDEN_PATTERNS = (
    "routing_projection",       # learned 路由（RoutingProjection W_l）
    "residual_router",          # residual 路由
    "cross_layer_lora_pool",    # 跨层 global pool
    "global_lora_pool",         # pool_share=global
    "h_proj",                   # hidden bottleneck
)

_RE_ATTN_A = re.compile(r"^(?P<prefix>.+)\._lora_adapter\.A\.weight$")
_RE_ROUTER_A = re.compile(r"^(?P<prefix>.+)\.parametrizations\.weight\.(?P<pidx>\d+)\.A$")
_RE_POOL_A = re.compile(r"^(?P<prefix>.+)\.lora_pool\.experts\.(?P<eidx>\d+)\.A\.weight$")
_RE_INDEP_GATE = re.compile(r"^(?P<prefix>.+)\.independent_router\.gate\.weight$")


@dataclass
class Overlay:
    adapter_path: str
    config: dict
    # ckpt weight name -> (A, B, scaling, kind); 目标张量 W_eff = W + scaling * (B @ A)
    # kind ∈ {"attn","router"}：router 必须按训练侧 LoRAWeightDelta 的 bf16 算子顺序
    # 合并（见 merged_weights_stream），attn 用 fp32 合并。
    weight_deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor, float, str]] = field(default_factory=dict)
    # 模块路径(与 ckpt 权重名同一命名空间) -> (A, B, scaling); forward 输出 += scaling * B(A(h))
    bypass: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = field(default_factory=dict)
    # 底座权重名 -> 变换函数 W -> W_eff（weight_peft / expert_weight / partial_ft 族）
    weight_transforms: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Overlay({self.adapter_path}): "
            f"{len(self.weight_deltas)} merged weight deltas, {len(self.bypass)} dense bypass modules, "
            f"{len(self.weight_transforms)} weight transforms"
        )


def _scaling(alpha: Optional[int], rank: Optional[int], fallback_alpha: int, fallback_rank: int) -> float:
    a = alpha if alpha else fallback_alpha
    r = rank if rank else fallback_rank
    return float(a) / float(r)


def parse_overlay(adapter_path: str) -> Overlay:
    """读取 adapter 目录，产出可合并的权重增量 + 稠密旁路张量 + 通用权重变换。

    命名映射（与训练侧 save 的 named_parameters 名字一一对应）：
      `<p>._lora_adapter.A.weight`             -> 目标 `<p>.weight`
      `<p>.parametrizations.weight.0.A`        -> 目标 `<p>.weight`
      `<p>.lora_pool.experts.0.A.weight`       -> 旁路挂在模块 `<p>`
    三个模型的 `<p>` 前缀（model.layers.* / model.language_model.layers.*）与
    base checkpoint safetensors 权重名逐字节一致（已对 qwen3/gemma4/gptoss 核验）。

    新增分派（weight_peft / partial_ft / expert_weight _wp_adapter）：
      weight_peft_config.json  -> vllm_weight_peft_overlay.parse_weight_peft
      partial_ft_config.json   -> vllm_weight_peft_overlay.parse_partial_ft
      moe_lora_state 含 _wp_adapter -> vllm_weight_peft_overlay.extract_expert_weight_transforms
    """
    import safetensors.torch
    import vllm_weight_peft_overlay as _wp

    # ---- weight_peft 族（dora/oft/ia3/lora；fourier→HF）----
    if os.path.exists(os.path.join(adapter_path, "weight_peft_config.json")):
        transforms, cfg = _wp.parse_weight_peft(adapter_path)
        ov = Overlay(adapter_path=adapter_path, config=cfg, weight_transforms=transforms)
        print(f"[vllm_moe_lora] weight_peft overlay {adapter_path}: {len(transforms)} transforms")
        return ov

    # ---- partial_ft 族 ----
    if os.path.exists(os.path.join(adapter_path, "partial_ft_config.json")):
        transforms, cfg = _wp.parse_partial_ft(adapter_path)
        ov = Overlay(adapter_path=adapter_path, config=cfg, weight_transforms=transforms)
        print(f"[vllm_moe_lora] partial_ft overlay {adapter_path}: {len(transforms)} transforms")
        return ov

    config_path = os.path.join(adapter_path, "moe_lora_config.json")
    state_path = os.path.join(adapter_path, "moe_lora_state.safetensors")
    if not (os.path.exists(config_path) and os.path.exists(state_path)):
        raise FileNotFoundError(f"{adapter_path} 缺 moe_lora_config.json / moe_lora_state.safetensors")
    with open(config_path) as f:
        cfg = json.load(f)
    state = safetensors.torch.load_file(state_path)

    for name in state:
        for pat in _FORBIDDEN_PATTERNS:
            if pat in name:
                raise NotImplementedError(
                    f"Adapter 含 '{pat}' 张量（{name}），该结构族不支持 vLLM overlay，请走 HF 评测路径。"
                )

    ov = Overlay(adapter_path=adapter_path, config=cfg)

    attn_scaling = _scaling(
        cfg.get("moe_lora_attn_alpha"), cfg.get("moe_lora_attn_rank"),
        cfg.get("moe_lora_alpha", 16), cfg.get("moe_lora_rank", 8),
    )
    router_scaling = _scaling(
        cfg.get("moe_lora_router_alpha"), cfg.get("moe_lora_router_rank"),
        cfg.get("moe_lora_alpha", 16), cfg.get("moe_lora_rank", 8),
    )
    pool_scaling = _scaling(cfg.get("moe_lora_alpha"), cfg.get("moe_lora_rank"), 16, 8)

    consumed = set()
    n_attn = n_router = 0

    # ---- expert_weight _wp_adapter（FFN 粒度 expert_weight_rX / aux 链）----
    if any("_wp_adapter" in n for n in state):
        ov.weight_transforms.update(
            _wp.extract_expert_weight_transforms(state, cfg, consumed)
        )

    for name, A in state.items():
        m = _RE_ATTN_A.match(name)
        if m:
            b_name = f"{m.group('prefix')}._lora_adapter.B.weight"
            B = state.get(b_name)
            if B is None:
                raise RuntimeError(f"attn LoRA 缺 B: {b_name}")
            target = f"{m.group('prefix')}.weight"
            ov.weight_deltas[target] = (A, B, attn_scaling, "attn")
            consumed.update((name, b_name))
            n_attn += 1
            continue

        m = _RE_ROUTER_A.match(name)
        if m:
            if m.group("pidx") != "0":
                raise NotImplementedError(f"多重 parametrization 不支持: {name}")
            b_name = f"{m.group('prefix')}.parametrizations.weight.0.B"
            B = state.get(b_name)
            if B is None:
                raise RuntimeError(f"router LoRA 缺 B: {b_name}")
            target = f"{m.group('prefix')}.weight"
            ov.weight_deltas[target] = (A, B, router_scaling, "router")
            consumed.update((name, b_name))
            n_router += 1
            continue

        m = _RE_POOL_A.match(name)
        if m:
            if m.group("eidx") != "0":
                raise NotImplementedError(
                    f"LoRAPool n_experts>1（{name}）不支持 overlay（路由是 token 相关的）。"
                )
            # n1 + top1 + independent 才能把 pool 化简为常权重 1.0 的稠密旁路
            if not (
                cfg.get("moe_lora_n_experts") == 1
                and cfg.get("moe_lora_top_k") == 1
                and cfg.get("moe_lora_routing_mode") == "independent"
                and cfg.get("moe_lora_pool_share", "per_layer") == "per_layer"
                and cfg.get("moe_lora_n_groups", 1) == 1
                and cfg.get("moe_lora_n_global", 0) == 0
                and cfg.get("moe_lora_hidden_bottleneck_dim", 0) == 0
            ):
                raise NotImplementedError(
                    "lora_pool 仅支持 independent + n_experts=1 + top_k=1 + per_layer 族 "
                    f"(config: n={cfg.get('moe_lora_n_experts')}, k={cfg.get('moe_lora_top_k')}, "
                    f"mode={cfg.get('moe_lora_routing_mode')})"
                )
            b_name = f"{m.group('prefix')}.lora_pool.experts.0.B.weight"
            B = state.get(b_name)
            if B is None:
                raise RuntimeError(f"pool LoRA 缺 B: {b_name}")
            ov.bypass[m.group("prefix")] = (A, B, pool_scaling)
            consumed.update((name, b_name))
            continue

        m = _RE_INDEP_GATE.match(name)
        if m:
            # 单类 softmax 恒 1，gate 权重不参与前向；仅确认形状 [1, d]
            if state[name].shape[0] != 1:
                raise NotImplementedError(f"independent_router 输出维 !=1: {name} {tuple(state[name].shape)}")
            consumed.add(name)
            continue

    # B 张量在配对时一并 consumed；此处剩下的才是真未知
    unknown = [n for n in state if n not in consumed]
    if unknown:
        raise NotImplementedError(f"无法识别的 adapter 张量（前 5 个）: {unknown[:5]}")

    print(
        f"[vllm_moe_lora] parsed {adapter_path}: attn={n_attn} router={n_router} "
        f"dense_bypass={len(ov.bypass)} | scaling attn={attn_scaling} router={router_scaling} pool={pool_scaling}"
    )
    return ov


def merged_weights_stream(
    weights: Iterable[Tuple[str, torch.Tensor]],
    overlay: Overlay,
    seen: set,
) -> Iterable[Tuple[str, torch.Tensor]]:
    """把 attn/router 低秩增量在权重流里合并进原始张量。

    在 vLLM 各模型自己的 load_weights 之前介入，此时名字仍是 checkpoint 原名、
    张量仍是全量（未 TP 切分），合并后再交给下游 mapper/sharder。

    dtype 口径（2026-07-22 教训）：
    - router 增量必须逐算子复刻训练侧 LoRAWeightDelta.forward 的 bf16 顺序
      `W + (s*(B@A)).to(W.dtype)`——之前用 fp32 合并一次舍入，与 HF 侧逐前向
      bf16 三次舍入差 ~1 ulp（实测 0.3-1% 元素不同位），128 选 8 的路由边界
      每 token-layer 翻 ~0.18%，1300-token prompt 全程期望 ~112 次专家集合
      翻转 → dense_r256_router_r8 臂 equiv 13/24 FAIL、首 token 解耦。
      rank≤8 时 B@A 的 fp32 累加在 CPU/GPU 上舍入结果一致，逐位可复现。
    - attn 增量保持 fp32 合并：训练侧 attn 是运行时旁路 Wx + s·B(A(x))，
      合并权重在激活级本就无法逐位对齐，fp32 是数学上最准的合并。
    """
    deltas = overlay.weight_deltas
    transforms = overlay.weight_transforms
    for name, w in weights:
        t = transforms.get(name)
        if t is not None:
            w = t(w)
            seen.add(name)
            yield name, w
            continue
        d = deltas.get(name)
        if d is not None:
            A, B, s, kind = d
            if w.shape != (B.shape[0], A.shape[1]):
                raise RuntimeError(
                    f"delta 形状不匹配 {name}: W{tuple(w.shape)} vs B@A "
                    f"({B.shape[0]},{A.shape[1]})"
                )
            if kind == "router":
                delta = (s * (B.to(w.dtype) @ A.to(w.dtype))).to(w.dtype)
                w = w + delta
            else:
                w = (w.float() + s * (B.float() @ A.float())).to(w.dtype)
            seen.add(name)
        yield name, w


def _resolve_module(model: torch.nn.Module, prefix: str) -> Tuple[str, torch.nn.Module]:
    """把 adapter 命名空间里的模块路径解析到 vLLM 模型实例里的模块。

    qwen3 native:  model.layers.N.mlp             （SparseMoeBlock，输入 = MoE block 入口 h）
    gptoss native: model.layers.N.mlp.experts     （FusedMoE，输入与 HF experts host 相同）
    gemma4 (transformers backend): model.language_model.layers.N.experts
    先精确匹配，再做唯一后缀匹配（容忍 wrapper 前缀差异）。
    """
    mods = dict(model.named_modules())
    if prefix in mods:
        return prefix, mods[prefix]
    tail = prefix[len("model."):] if prefix.startswith("model.") else prefix
    candidates = [n for n in mods if n.endswith(tail)]
    if len(candidates) == 1:
        return candidates[0], mods[candidates[0]]
    raise RuntimeError(
        f"找不到旁路目标模块 '{prefix}'（后缀候选 {candidates[:4]}）。"
        f"模型模块样例: {[n for n in list(mods)[:8]]}"
    )


def _install_bypass(model: torch.nn.Module, overlay: Overlay) -> None:
    """给每个目标模块 forward 挂 `out += s·(h @ A^T) @ B^T` 的 wrapper。

    h 取 forward 的第一个 positional 参数（gptoss FusedMoE 用 kwargs
    hidden_states= 调用，两种都兼容）。输出是 tuple 时改写第 0 项。
    仅推理用：A/B 存成普通属性张量（不进 named_parameters，绕开 loader 校验）。
    """
    try:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    except StopIteration:  # 理论不可能
        device, dtype = torch.device("cuda"), torch.bfloat16

    n = 0
    for prefix, (A, B, s) in overlay.bypass.items():
        resolved, mod = _resolve_module(model, prefix)
        if getattr(mod, "_moe_lora_bypass_installed", False):
            continue
        A_d = A.to(device=device, dtype=dtype)
        B_d = B.to(device=device, dtype=dtype)
        orig_forward = mod.forward

        def make_fwd(orig, A_t, B_t, scale):
            def fwd(*args, **kwargs):
                if args:
                    h = args[0]
                else:
                    h = kwargs.get("hidden_states")
                    if h is None:
                        h = kwargs.get("x")
                    if h is None:
                        raise RuntimeError(
                            f"moe_lora bypass 拿不到 hidden states（kwargs={list(kwargs)}）"
                        )
                # 必须在调用原 forward 之前算旁路增量：vLLM FusedMoE 前向
                # fused_experts(..., inplace=True)（custom op moe_forward 声明
                # mutates_args=["hidden_states"]）会把输出**原地写进输入缓冲**，
                # 先 orig 后读 h 拿到的是 MoE 输出而非输入 → delta 语义整个错掉
                #（2026-07-22 dense 臂 equiv 13/24 FAIL 的真根因）。
                delta = ((h.to(A_t.dtype) @ A_t.t()) @ B_t.t()) * scale
                out = orig(*args, **kwargs)
                if isinstance(out, tuple):
                    return (out[0] + delta.to(out[0].dtype),) + tuple(out[1:])
                return out + delta.to(out.dtype)
            return fwd

        mod.forward = make_fwd(orig_forward, A_d, B_d, s)
        mod._moe_lora_bypass_installed = True
        n += 1
    print(f"[vllm_moe_lora] installed {n} dense FFN bypass wrappers")


# ---------------------------------------------------------------------------
# vLLM 模型类注册
# ---------------------------------------------------------------------------

_ACTIVE_OVERLAY: Optional[Overlay] = None


def _wrap_load_weights(base_cls):
    """子类化 vLLM 模型类：load_weights 里合并增量、装旁路。"""

    class MoeLoraOverlayModel(base_cls):  # type: ignore[misc,valid-type]

        def load_weights(self, weights):
            overlay = _ACTIVE_OVERLAY
            seen: set = set()
            deltas = overlay.weight_deltas if overlay else {}
            transforms = overlay.weight_transforms if overlay else {}
            buffers = dict(self.named_buffers())
            loaded_buffers: list = []

            def stream():
                src = merged_weights_stream(weights, overlay, seen) if overlay else weights
                # AutoWeightsLoader 只认 parameter；checkpoint 里 buffer 型权重
                # （gemma4 的 layer_scalar / vision_tower.std_*）会让它 raise。
                # 拦截 buffer 权重手动灌入，不进下游 loader。base 臂同样需要。
                for name, w in src:
                    buf = buffers.get(name) if name not in deltas and name not in transforms else None
                    if buf is not None:
                        if buf.device.type == "meta":
                            mod_path, _, buf_name = name.rpartition(".")
                            mod = self.get_submodule(mod_path)
                            mod.register_buffer(
                                buf_name, w.detach().clone().to(dtype=buf.dtype),
                                persistent=True,
                            )
                        else:
                            buf.data.copy_(w.to(device=buf.device, dtype=buf.dtype))
                        loaded_buffers.append(name)
                        continue
                    yield name, w

            result = super().load_weights(stream())
            if overlay:
                all_targets = set(deltas) | set(overlay.weight_transforms)
                missing = all_targets - seen
                if missing:
                    raise RuntimeError(
                        f"权重流里没见到 {len(missing)} 个待合并目标（命名不匹配？）: "
                        f"{sorted(missing)[:5]}"
                    )
                _install_bypass(self, overlay)
            print(
                f"[vllm_moe_lora] merged {len(seen)} weight targets into base weights"
                + (f"; loaded {len(loaded_buffers)} buffer tensors directly" if loaded_buffers else "")
            )
            return result

    MoeLoraOverlayModel.__name__ = f"MoeLora{base_cls.__name__}"
    return MoeLoraOverlayModel


# ---------------------------------------------------------------------------
# vLLM 0.11 × transformers 5.6 兼容 shim（进程内 monkey-patch，不改安装目录）
# ---------------------------------------------------------------------------


def apply_vllm_transformers_compat() -> None:
    """修掉 vLLM 0.11 在 transformers 5.6 下的兼容缺口（崩溃两处 + 静默错值一处，
    2026-07-20/22 smoke 与全量评测实测）：

    1. `get_cached_tokenizer` 读 `tokenizer.all_special_tokens_extended`——
       transformers 5.6 已删除该属性（slow/fast/TokenizersBackend 全中）。
       补一个返回 all_special_tokens 的 property（vLLM 只拿它做缓存透传）。
    2. gemma4 的 rope 配置是 5.x 新式 per-layer `rope_parameters`
       （{"full_attention": {...}, "sliding_attention": {...}}，无顶层 rope_type，
       且以 rope_scaling 别名暴露），vLLM 的 `patch_rope_scaling_dict` 直接
       raise，`_get_and_verify_max_len` 里 `rope_scaling["rope_type"]` 会 KeyError。
       对这种嵌套形态：校验跳过、max-len 推导按无 rope scaling 处理
       （gemma4 max_position_embeddings=262144 本身已是 scaled 值，评测
       max_model_len=4096 远小于上限，无风险）。

    幂等；必须在构造 LLM(...) 之前调用。base 臂（无 adapter）也需要。
    """
    # --- 1) tokenizer 属性 shim ---
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    # --- 1b) gpt-oss config：transformers 5.6 把 rope_theta 挪进了 rope_parameters
    # dict，不再是顶层属性；vLLM gpt_oss.py 的 OAIAttention 读 config.rope_theta 崩。
    # 给 config 类补 property（实例 __dict__ 里有真值时不会走到 property）。
    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

        if not hasattr(GptOssConfig, "rope_theta"):
            GptOssConfig.rope_theta = property(
                lambda self: (getattr(self, "rope_parameters", None) or {}).get("rope_theta")
            )
    except ImportError:
        pass

    def _is_nested_rope(d) -> bool:
        return (
            isinstance(d, dict) and d
            and "rope_type" not in d and "type" not in d
            and all(isinstance(v, dict) for v in d.values())
        )

    # --- 2a) config 加载期校验 ---
    from vllm.transformers_utils import config as _vcfg

    if not getattr(_vcfg.patch_rope_scaling_dict, "_moe_lora_compat", False):
        _orig_patch = _vcfg.patch_rope_scaling_dict

        def _patched_rope_dict(rope_scaling):
            if _is_nested_rope(rope_scaling):
                return  # transformers 5.x per-layer rope_parameters；原样放行
            return _orig_patch(rope_scaling)

        _patched_rope_dict._moe_lora_compat = True
        _vcfg.patch_rope_scaling_dict = _patched_rope_dict

    # --- 2b) ModelConfig 的 max-len 推导 ---
    from vllm.config import model as _vmodel

    if not getattr(_vmodel._get_and_verify_max_len, "_moe_lora_compat", False):
        _orig_len = _vmodel._get_and_verify_max_len

        class _HideNestedRope:
            """getattr 代理：仅把嵌套式 rope_scaling 报告为 None，其余透传。"""

            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name):
                if name == "rope_scaling":
                    return None
                return getattr(object.__getattribute__(self, "_inner"), name)

        def _patched_len(hf_config, *a, **k):
            if _is_nested_rope(getattr(hf_config, "rope_scaling", None)):
                hf_config = _HideNestedRope(hf_config)
            return _orig_len(hf_config, *a, **k)

        _patched_len._moe_lora_compat = True
        _vmodel._get_and_verify_max_len = _patched_len

    # --- 3) 通用 rope_theta 回填 ---
    # transformers 5.6 把 rope_theta 从顶层属性挪进 rope_parameters dict
    # （Qwen3MoeConfig 等 hasattr(cfg,'rope_theta') 直接 False），而 vLLM 模型代码
    # 普遍写 `getattr(config, "rope_theta", 10000)`——不崩、静默用默认值。
    # qwen3 真值 1e6，错 100 倍 → MMLU-Pro base 43.9 vs HF 链 52.7、逐题一致率
    # 仅 63%（2026-07-22 全量 12032 题实测）。gpt_oss 是硬读所以当场崩（见 1b），
    # qwen3 这类 getattr 读法才是真正危险的形态。
    # 修法：在 vLLM get_config 出口把平铺 rope_parameters 里的 rope_theta 回填为
    # 实例属性。实例属性不与类 property 冲突（有 property 的类如 GptOssConfig
    # hasattr=True 会被跳过）；嵌套 per-layer 形态（gemma4）取不到标量，自动跳过。

    def _backfill_rope_theta(cfg) -> None:
        targets = [cfg]
        get_text = getattr(cfg, "get_text_config", None)
        if callable(get_text):
            try:
                targets.append(get_text())
            except Exception:
                pass
        seen: set[int] = set()
        for c in targets:
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            if hasattr(c, "rope_theta"):
                continue
            rp = getattr(c, "rope_parameters", None)
            theta = rp.get("rope_theta") if isinstance(rp, dict) else None
            if isinstance(theta, (int, float)):
                c.rope_theta = float(theta)

    def _hook_get_config(mod) -> None:
        fn = getattr(mod, "get_config", None)
        if fn is None or getattr(fn, "_moe_lora_rope_theta", False):
            return

        def _patched_get_config(*a, **k):
            cfg = fn(*a, **k)
            _backfill_rope_theta(cfg)
            return cfg

        _patched_get_config._moe_lora_rope_theta = True
        mod.get_config = _patched_get_config

    # vllm/config/model.py 顶部 `from ... import get_config` 是按名绑定，
    # 只 patch 定义模块不生效，两处都要挂。
    _hook_get_config(_vcfg)
    _hook_get_config(_vmodel)


def verify_config_fidelity(base_model: str) -> None:
    """启动前防呆：对照原始 config.json 与 vLLM 将读到的 config 视图，关键标量
    缺失或不等直接 raise。

    背景：transformers 升级会重排 config 属性（如 5.6 把 rope_theta 挪进
    rope_parameters），vLLM 里 getattr(config, key, default) 静默退回默认值，
    模型"能跑但变笨"，评测数字整体下沉且无任何报错。rope_theta 之外把同类
    标量一并盯住，宁可 fail fast 也不出静默错数。
    """
    import json as _json

    apply_vllm_transformers_compat()
    from vllm.transformers_utils import config as _vcfg

    with open(os.path.join(base_model, "config.json")) as f:
        raw = _json.load(f)
    cfg = _vcfg.get_config(base_model, trust_remote_code=True)

    scalar_keys = ("rope_theta", "max_position_embeddings", "head_dim",
                   "num_attention_heads", "num_key_value_heads", "sliding_window")
    bad: list[str] = []

    def _check(raw_side: dict, cfg_side, scope: str) -> None:
        for key in scalar_keys:
            raw_val = raw_side.get(key)
            if raw_val is None:
                continue
            seen = getattr(cfg_side, key, None)
            if seen is None or float(seen) != float(raw_val):
                bad.append(f"{scope}{key}: config.json={raw_val} vLLM视图={seen}")

    _check(raw, cfg, "")
    if isinstance(raw.get("text_config"), dict):
        _check(raw["text_config"], cfg.get_text_config(), "text_config.")
    if bad:
        raise RuntimeError(
            "[vllm_moe_lora] vLLM 读到的 config 与 config.json 不一致"
            "（transformers/vLLM 兼容层缺口，会静默错数）: " + "; ".join(bad)
        )
    print(f"[vllm_moe_lora] config fidelity OK: {base_model}")


def _fix_scaled_word_embedding(model: torch.nn.Module, hf_config) -> None:
    """gemma4 专用：把 vLLM transformers backend 丢掉的 embed_scale 补回来。

    HF 的 Gemma4TextScaledWordEmbedding 在 forward 里乘
    embed_scale = sqrt(hidden_size)（≈53.07，先 cast 到权重 dtype 再乘），
    而 vLLM backend 初始化时把该模块替换成裸 VocabParallelEmbedding，scale 丢失
    → 输入嵌入小 ~53 倍 → 输出乱码（2026-07-21 smoke 实测）。
    这里给替换后的 embedding 模块 forward 挂 wrapper 把 scale 乘回去。
    仅当模块自身没有 embed_scale（即确实被替换过）才补，防未来双重缩放。
    """
    text_cfg = getattr(hf_config, "text_config", hf_config)
    emb = model.model.get_input_embeddings()
    if hasattr(emb, "embed_scale") or getattr(emb, "_moe_lora_embed_scale_fixed", False):
        return
    scale = torch.tensor(float(text_cfg.hidden_size) ** 0.5)
    orig_forward = emb.forward

    def fwd(input_ids):
        out = orig_forward(input_ids)
        return out * scale.to(out.dtype)

    emb.forward = fwd
    emb._moe_lora_embed_scale_fixed = True
    print(f"[vllm_moe_lora] re-applied gemma4 embed_scale={float(scale):.4f} on input embeddings")


def register_models() -> None:
    """注册三个架构的 overlay 模型类 + 兼容 shim。

    base 臂（无 adapter）也必须调用：overlay=None 时 load_weights 退化为
    "原样加载 + buffer 权重拦截"（gemma4 的 layer_scalar 是 buffer，
    stock AutoWeightsLoader 会 raise）。幂等。
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    apply_vllm_transformers_compat()

    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
    from vllm.model_executor.models.gpt_oss import GptOssForCausalLM

    ModelRegistry.register_model("Qwen3MoeForCausalLM", _wrap_load_weights(Qwen3MoeForCausalLM))
    ModelRegistry.register_model("GptOssForCausalLM", _wrap_load_weights(GptOssForCausalLM))

    # gemma4 无原生 vLLM 实现，走 transformers backend；multimodal wrapper 兼容纯文本。
    impl = os.environ.get("MOE_LORA_GEMMA4_IMPL", "multimodal")
    if impl == "causal":
        from vllm.model_executor.models.transformers import TransformersForCausalLM as _G4Base
    else:
        from vllm.model_executor.models.transformers import TransformersForMultimodalLM as _G4Base

    _G4Overlay = _wrap_load_weights(_G4Base)

    class MoeLoraGemma4(_G4Overlay):  # type: ignore[misc,valid-type]
        def __init__(self, *, vllm_config, prefix: str = ""):
            # backend 的 create_attention_instances 只查顶层 config.layer_types，
            # 而 gemma4 的 sliding/full 交错配置在 text_config 上 → 不传播的话
            # 所有层都按 full attention 处理（>1024 token 偏离训练分布）。
            hf_cfg = vllm_config.model_config.hf_config
            text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
            if not hasattr(hf_cfg, "layer_types") and hasattr(text_cfg, "layer_types"):
                hf_cfg.layer_types = text_cfg.layer_types
                hf_cfg.sliding_window = text_cfg.sliding_window
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            _fix_scaled_word_embedding(self, hf_cfg)

    ModelRegistry.register_model("Gemma4ForConditionalGeneration", MoeLoraGemma4)


def activate(adapter_path: str) -> Overlay:
    """解析 adapter 并注册 overlay 模型类。必须在构造 LLM(...) 之前调用。"""
    global _ACTIVE_OVERLAY
    overlay = parse_overlay(adapter_path)
    _ACTIVE_OVERLAY = overlay
    register_models()
    return overlay


def is_vllm_supported(adapter_path: str) -> bool:
    """探测：该 adapter 是否属于 vLLM overlay 可映射的结构族（attn LoRA /
    router LoRA / n1top1-independent dense 旁路）。parse_overlay 对不支持的结构族
    fail-fast（NotImplementedError），据此判定，不加载模型、纯 CPU 读权重名。

    用于批量评测按权重结构自动选引擎：支持 → vLLM，否则 → HF（_common.load
    能吃全部结构族：weight_peft/partial_ft/das/moe_lora/PEFT）。"""
    try:
        parse_overlay(adapter_path)
        return True
    except (NotImplementedError, FileNotFoundError):
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU 自检：直接对照训练侧 llamafactory 实现的数值
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))
    from llamafactory.model.model_utils.moe_lora import (  # noqa: E402
        IndependentRouter, LinearLoRAAdapter, LoRAPool, LoRAWeightDelta,
    )
    import torch.nn.utils.parametrize as parametrize

    torch.manual_seed(0)
    d_in, d_out, rank, alpha = 64, 48, 8, 16
    x = torch.randn(5, d_in)

    # 1) attention LoRA: 旁路 vs 合并权重
    lin = torch.nn.Linear(d_in, d_out, bias=True)
    ad = LinearLoRAAdapter(d_in, d_out, rank, alpha)
    ref = lin(x) + ad(x)
    merged_w = lin.weight + (alpha / rank) * (ad.B.weight @ ad.A.weight)
    got = torch.nn.functional.linear(x, merged_w, lin.bias)
    assert torch.allclose(ref, got, atol=1e-5), f"attn merge mismatch: {(ref-got).abs().max()}"

    # 2) router LoRA: parametrize 增量 vs 合并权重
    gate = torch.nn.Linear(d_in, 128, bias=False)
    w0 = gate.weight.detach().clone()
    delta = LoRAWeightDelta(128, d_in, rank, alpha)
    parametrize.register_parametrization(gate, "weight", delta)
    ref = gate(x)
    merged_w = w0 + (alpha / rank) * (delta.B @ delta.A)
    got = torch.nn.functional.linear(x, merged_w)
    assert torch.allclose(ref, got, atol=1e-5), f"router merge mismatch: {(ref-got).abs().max()}"

    # 2b) router bf16 逐位对齐：合并流(merged_weights_stream router 分支)的
    # bf16 算子顺序必须与训练侧 parametrize 前向逐位一致——fp32 合并一次舍入
    # 曾造成 ~0.5% 权重元素差 1 ulp、路由边界翻转、equiv 13/24 FAIL（2026-07-22）。
    gate16 = torch.nn.Linear(d_in, 128, bias=False).to(torch.bfloat16)
    w0_16 = gate16.weight.detach().clone()
    delta16 = LoRAWeightDelta(128, d_in, rank, alpha).to(torch.bfloat16)
    parametrize.register_parametrization(gate16, "weight", delta16)
    ref_w16 = gate16.weight.detach()  # parametrize 计算的 W_eff (bf16)
    s16 = float(alpha) / float(rank)
    merged16 = w0_16 + (s16 * (delta16.B.to(w0_16.dtype) @ delta16.A.to(w0_16.dtype))).to(w0_16.dtype)
    assert torch.equal(ref_w16, merged16), (
        f"router bf16 merge 与训练 parametrize 不逐位一致: "
        f"{(ref_w16 != merged16).float().mean().item():.4%} 元素不同"
    )

    # 3) dense pool (independent n1 top1): LoRAPool 完整路径 vs 稠密旁路公式
    pool = LoRAPool(n_experts=1, d_model=d_in, rank=16, alpha=32, top_k=1)
    pool.eval()
    router = IndependentRouter(d_in, 1)
    p = router(x)
    assert torch.allclose(p, torch.ones_like(p)), "单类 softmax 应恒等于 1"
    ref = pool(x, p)
    A, B = pool.experts[0].A.weight, pool.experts[0].B.weight
    got = ((x @ A.t()) @ B.t()) * (32 / 16)
    assert torch.allclose(ref, got, atol=1e-5), f"pool bypass mismatch: {(ref-got).abs().max()}"

    print("vllm_moe_lora self_test: OK（attn merge / router merge / dense bypass 三路数值均与训练实现一致）")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--parse", default=None, help="解析一个 adapter 目录并打印 overlay 摘要")
    args = p.parse_args()
    if args.self_test:
        _self_test()
    if args.parse:
        parse_overlay(args.parse)
