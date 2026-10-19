#!/usr/bin/env python3
"""Check per-position trainable-parameter budgets for the PEFT comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ATTENTION_LORA_RANK = 32
ROUTER_LORA_RANK = 8
HIDDEN_BOTTLENECK = 128

MODEL_ROOT = Path(os.environ.get("CKPT_ROOT", Path(__file__).resolve().parents[3] / "models"))
DEFAULT_CONFIGS = {
    "qwen3": MODEL_ROOT
    / "Qwen3-30B-A3B-Base/config.json",
    "gemma4": MODEL_ROOT
    / "gemma-4-26B-A4B/config.json",
    "gptoss": MODEL_ROOT
    / "gpt-oss-20b/config.json",
}

SETTINGS = {
    "qwen3": {
        "ffn_oft_block": 4,
        "attn_oft_block": 128,
        "router_oft_block": 128,
        "ffn_fourier_n_freq": 4096,
        "attn_fourier_n_freq": 139264,
        "router_fourier_n_freq": 17408,
        "moe_lora_rank": 32,
        "moe2_rank": 24,
        "das_selected": 8,
        "das_rank": 32,
    },
    "gemma4": {
        "ffn_oft_block": 8,
        "attn_oft_block": 128,
        "router_oft_block": 128,
        "ffn_fourier_n_freq": 5632,
        "attn_fourier_n_freq": 199814,
        "router_fourier_n_freq": 23552,
        "moe_lora_rank": 32,
        "moe2_rank": 24,
        "das_selected": 8,
        "das_rank": 32,
    },
    "gptoss": {
        "ffn_oft_block": 6,
        "attn_oft_block": 64,
        "router_oft_block": 32,
        "ffn_fourier_n_freq": 5760,
        "attn_fourier_n_freq": 165888,
        "router_fourier_n_freq": 23296,
        "moe_lora_rank": 8,
        "moe2_rank": 1,
        "das_selected": 4,
        "das_rank": 16,
    },
}


class BudgetError(ValueError):
    pass


def config_value(config: dict, *keys: str):
    text_config = config.get("text_config") or {}
    for source in (config, text_config):
        for key in keys:
            if source.get(key) is not None:
                return source[key]
    return None


def require_int(config: dict, *keys: str) -> int:
    value = config_value(config, *keys)
    if not isinstance(value, int) or value <= 0:
        raise BudgetError(f"Expected a positive integer in one of {keys}, got {value!r}.")
    return value


def attention_layout(config: dict, hidden: int, layers: int) -> list[list[tuple[str, int, int]]]:
    """Return each layer's adapted attention matrices as (name, in, out)."""
    text_config = config.get("text_config") or config
    heads = require_int(config, "num_attention_heads")
    kv_heads = require_int(config, "num_key_value_heads")
    head_dim = require_int(config, "head_dim")
    layer_types = text_config.get("layer_types") or ["default"] * layers
    if len(layer_types) != layers:
        raise BudgetError(f"layer_types has {len(layer_types)} entries for {layers} layers.")

    global_head_dim = text_config.get("global_head_dim") or head_dim
    global_kv_heads = text_config.get("num_global_key_value_heads") or kv_heads
    attention_k_eq_v = bool(text_config.get("attention_k_eq_v", False))
    layout = []
    for layer_type in layer_types:
        sliding = layer_type == "sliding_attention"
        current_head_dim = head_dim if sliding else global_head_dim
        alternative_attention = attention_k_eq_v and not sliding
        current_kv_heads = global_kv_heads if alternative_attention else kv_heads
        q_width = heads * current_head_dim
        kv_width = current_kv_heads * current_head_dim
        matrices = [
            ("q_proj", hidden, q_width),
            ("k_proj", hidden, kv_width),
        ]
        if not alternative_attention:
            matrices.append(("v_proj", hidden, kv_width))
        matrices.append(("o_proj", q_width, hidden))
        layout.append(matrices)
    return layout


def lora_parameters(layout: list[list[tuple[str, int, int]]], rank: int) -> int:
    return rank * sum(in_features + out_features for layer in layout for _, in_features, out_features in layer)


def dora_parameters(layout: list[list[tuple[str, int, int]]], rank: int) -> int:
    magnitude = sum(out_features for layer in layout for _, _, out_features in layer)
    return lora_parameters(layout, rank) + magnitude


def oft_parameters(layout: list[list[tuple[str, int, int]]], block: int, position: str) -> int:
    # HF-peft convention: block-diagonal R over the INPUT features of each matrix.
    inputs = [in_features for layer in layout for _, in_features, _ in layer]
    invalid = sorted({width for width in inputs if width % block})
    if invalid:
        raise BudgetError(f"{position} OFT block {block} does not divide input widths {invalid}.")
    return sum(width * (block - 1) // 2 for width in inputs)


def fourier_parameters(layout: list[list[tuple[str, int, int]]], n_freq: int) -> int:
    return sum(min(n_freq, in_features * out_features) for layer in layout for _, in_features, out_features in layer)


def ia3_attention_parameters(layout: list[list[tuple[str, int, int]]]) -> int:
    return sum(
        out_features
        for layer in layout
        for name, _, out_features in layer
        if name in ("k_proj", "v_proj")
    )


def full_attention_parameters(layer: list[tuple[str, int, int]], has_bias: bool) -> int:
    return sum(in_features * out_features + (out_features if has_bias else 0)
               for _, in_features, out_features in layer)


def row(ffn: int, method_routing: int, attention: int, router: int, note: str) -> dict[str, int | str]:
    return {
        "ffn": ffn,
        "method_routing": method_routing,
        "attention": attention,
        "router": router,
        "total": ffn + method_routing + attention + router,
        "note": note,
    }


def compute_budgets(model: str, config: dict) -> tuple[dict[str, dict[str, int | str]], dict[str, int]]:
    hidden = require_int(config, "hidden_size")
    intermediate = require_int(config, "moe_intermediate_size", "intermediate_size")
    layers = require_int(config, "num_hidden_layers")
    experts = require_int(config, "num_experts", "num_local_experts")
    top_k = require_int(config, "num_experts_per_tok", "top_k_experts")
    settings = SETTINGS[model]
    attn_layout = attention_layout(config, hidden, layers)

    attention_lora = lora_parameters(attn_layout, ATTENTION_LORA_RANK)
    router_lora = layers * ROUTER_LORA_RANK * (hidden + experts)
    weight_lora_ffn_r1 = layers * experts * (2 * hidden + 3 * intermediate)
    expert_lora_ffn_r2 = layers * experts * 2 * hidden * 2

    budgets = {
        "weight_lora": row(
            weight_lora_ffn_r1, 0, attention_lora, router_lora,
            "LoRA: expert-weight r1 + attention r32 + router r8",
        ),
        "expert_lora": row(
            expert_lora_ffn_r2, 0, attention_lora, router_lora,
            "LoRA: expert-bypass r2 + attention r32 + router r8",
        ),
    }

    dora_ffn_magnitude = layers * experts * (hidden + 2 * intermediate)
    dora_attention = dora_parameters(attn_layout, ATTENTION_LORA_RANK)
    dora_router = router_lora + layers * experts
    budgets["weight_dora"] = row(
        weight_lora_ffn_r1 + dora_ffn_magnitude, 0, dora_attention, dora_router,
        "DoRA at all positions: FFN r1, attention r32, router r8",
    )

    ffn_block = settings["ffn_oft_block"]
    attn_block = settings["attn_oft_block"]
    router_block = settings["router_oft_block"]
    # HF-peft convention: R over the INPUT features (gate_up in=hidden, down in=inter;
    # router in=hidden).
    if hidden % ffn_block or intermediate % ffn_block:
        raise BudgetError(
            f"FFN OFT block {ffn_block} does not divide expert inputs {hidden}/{intermediate}."
        )
    if hidden % router_block:
        raise BudgetError(f"Router OFT block {router_block} does not divide hidden {hidden}.")
    oft_ffn = layers * experts * (hidden + intermediate) * (ffn_block - 1) // 2
    oft_attention = oft_parameters(attn_layout, attn_block, "Attention")
    oft_router = layers * hidden * (router_block - 1) // 2
    budgets["weight_oft"] = row(
        oft_ffn, 0, oft_attention, oft_router,
        f"OFT at all positions: blocks={ffn_block}/{attn_block}/{router_block}; discrete capacity",
    )

    ffn_n_freq = settings["ffn_fourier_n_freq"]
    attn_n_freq = settings["attn_fourier_n_freq"]
    router_n_freq = settings["router_fourier_n_freq"]
    fourier_ffn = layers * experts * (
        min(ffn_n_freq, 2 * intermediate * hidden)
        + min(ffn_n_freq, intermediate * hidden)
    )
    fourier_attention = fourier_parameters(attn_layout, attn_n_freq)
    fourier_router = layers * min(router_n_freq, experts * hidden)
    budgets["weight_fourier"] = row(
        fourier_ffn, 0, fourier_attention, fourier_router,
        f"FourierFT at all positions: n_freq={ffn_n_freq}/{attn_n_freq}/{router_n_freq}",
    )

    ia3_ffn = layers * experts * intermediate
    ia3_attention = ia3_attention_parameters(attn_layout)
    ia3_router = layers * experts
    budgets["weight_ia3"] = row(
        ia3_ffn, 0, ia3_attention, ia3_router,
        "EXEMPT: IA3 FFN down-input + Attention K/V output + Router logits output",
    )

    n_lora = 8
    moe_rank = settings["moe_lora_rank"]
    moe_pool = layers * n_lora * 2 * hidden * moe_rank
    independent_routing = layers * hidden * n_lora
    budgets["expert_moe_lora_n8"] = row(
        moe_pool, independent_routing, attention_lora, router_lora,
        f"LoRA: N=8 r={moe_rank}, independent top-k=2 + attention r32 + router r8",
    )

    moe2_rank = settings["moe2_rank"]
    moe2_pool = layers * n_lora * 2 * hidden * moe2_rank
    learned_routing = layers * (
        hidden * HIDDEN_BOTTLENECK + (experts + HIDDEN_BOTTLENECK) * n_lora
    )
    budgets["expert_moe2_lora"] = row(
        moe2_pool, learned_routing, attention_lora, router_lora,
        f"LoRA: N=8 r={moe2_rank}, learned top-k=2, hidden={HIDDEN_BOTTLENECK} + attention/router",
    )

    selected = settings["das_selected"]
    das_rank = settings["das_rank"]
    das_ffn = layers * selected * 2 * hidden * das_rank
    budgets["das_lora"] = row(
        das_ffn, 0, attention_lora, router_lora,
        f"LoRA: selected={selected} r={das_rank} + attention r32 + router r8",
    )

    expert_bias = experts * (2 * intermediate + hidden) if model == "gptoss" else 0
    partial_ffn = experts * 3 * hidden * intermediate + expert_bias
    attention_bias = bool(config_value(config, "attention_bias"))
    partial_attention = full_attention_parameters(attn_layout[-1], attention_bias)
    router_bias = experts if model == "gptoss" else 0
    partial_router = experts * hidden + router_bias
    budgets["partial_last"] = row(
        partial_ffn, 0, partial_attention, partial_router,
        "EXEMPT: full last-layer FFN + Attention projections + Router projection",
    )

    dimensions = {
        "hidden_size": hidden,
        "expert_intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_experts": experts,
        "top_k": top_k,
        "attention_lora_anchor": attention_lora,
        "router_lora_anchor": router_lora,
        "attention_matrices": sum(len(layer) for layer in attn_layout),
    }
    return budgets, dimensions


def percent_delta(value: int, reference: int) -> float:
    return 100.0 * (value - reference) / reference


def validate_alignment(budgets: dict[str, dict[str, int | str]]) -> None:
    expert = int(budgets["expert_lora"]["total"])
    if int(budgets["das_lora"]["total"]) != expert:
        raise BudgetError("das_lora is not exactly aligned to expert_lora.")
    # One Fourier coefficient count is shared by all matrices at a position, so
    # Gemma4's heterogeneous 115-matrix Attention layout has a tiny rounding residue.
    if abs(int(budgets["weight_fourier"]["total"]) - expert) > 100:
        raise BudgetError("weight_fourier is not aligned to expert_lora within rounding tolerance.")
    for name in ("expert_moe_lora_n8", "expert_moe2_lora"):
        delta = abs(percent_delta(int(budgets[name]["total"]), expert))
        if delta > 5.0:
            raise BudgetError(f"{name} differs from expert_lora by {delta:.2f}% (>5%).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DEFAULT_CONFIGS), required=True)
    parser.add_argument("--config", type=Path, help="Override the selected model config.json path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config or DEFAULT_CONFIGS[args.model]
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
        budgets, dimensions = compute_budgets(args.model, config)
        validate_alignment(budgets)
    except (OSError, json.JSONDecodeError, BudgetError) as exc:
        print(f"Budget check failed: {exc}", file=sys.stderr)
        return 2

    weight_reference = int(budgets["weight_lora"]["total"])
    expert_reference = int(budgets["expert_lora"]["total"])
    print(f"Model: {args.model} | config: {config_path}")
    print(
        "Dimensions: "
        f"d={dimensions['hidden_size']} d_expert={dimensions['expert_intermediate_size']} "
        f"layers={dimensions['num_hidden_layers']} experts={dimensions['num_experts']} "
        f"top_k={dimensions['top_k']} attention_matrices={dimensions['attention_matrices']}"
    )
    print(
        "LoRA capacity anchors (not shared adapters): "
        f"attention r32={dimensions['attention_lora_anchor']:,}, "
        f"router r8={dimensions['router_lora_anchor']:,}"
    )
    print()
    print(
        f"{'method':<24} {'FFN':>14} {'method route':>14} {'Attention':>14} {'Router':>12} "
        f"{'total':>14} {'vs weight':>10} {'vs expert':>10}  setting"
    )
    print("-" * 160)
    for name, values in budgets.items():
        total = int(values["total"])
        print(
            f"{name:<24} {int(values['ffn']):>14,} {int(values['method_routing']):>14,} "
            f"{int(values['attention']):>14,} {int(values['router']):>12,} {total:>14,} "
            f"{percent_delta(total, weight_reference):>+9.2f}% "
            f"{percent_delta(total, expert_reference):>+9.2f}%  {values['note']}"
        )
    print()
    print(
        "Budget check passed: every row accounts for its own FFN/Attention/Router method. "
        "FourierFT and DAS align to Expert LoRA (Fourier rounding <=100 params); routed LoRA "
        "methods are within 5%. DoRA/OFT use their nearest feasible capacity; IA3/Partial FT are exempt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
