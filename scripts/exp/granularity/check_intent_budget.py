#!/usr/bin/env python3
"""Report trainable-parameter budgets for the three-model Intent granularity study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ATTENTION_RANK = 32
ROUTER_RANK = 8
DEFAULT_LAYER_BYPASS_RANKS = (256,)
EXPERT_BYPASS_RANK = 2
EXPERT_WEIGHT_RANKS = (1, 2)

MODEL_ROOT = Path(os.environ.get("CKPT_ROOT", Path(__file__).resolve().parents[3] / "models"))
DEFAULT_CONFIGS = {
    "qwen3": MODEL_ROOT
    / "Qwen3-30B-A3B-Base/config.json",
    "gemma4": MODEL_ROOT
    / "gemma-4-26B-A4B/config.json",
    "gptoss": MODEL_ROOT
    / "gpt-oss-20b/config.json",
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


def parse_positive_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise BudgetError("--layer-bypass-ranks must be a comma-separated list of positive integers.") from exc
    if not ranks or any(rank <= 0 for rank in ranks):
        raise BudgetError("--layer-bypass-ranks must contain at least one positive integer.")
    return tuple(dict.fromkeys(ranks))


def compute_budgets(
    config: dict, layer_bypass_ranks: tuple[int, ...], attach_attention: bool
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    hidden = require_int(config, "hidden_size")
    intermediate = require_int(config, "moe_intermediate_size", "intermediate_size")
    layers = require_int(config, "num_hidden_layers")
    experts = require_int(config, "num_experts", "num_local_experts")
    top_k = require_int(config, "num_experts_per_tok", "top_k_experts")
    attention_heads = require_int(config, "num_attention_heads")
    kv_heads = require_int(config, "num_key_value_heads")
    head_dim = require_int(config, "head_dim")

    q_width = attention_heads * head_dim
    kv_width = kv_heads * head_dim
    attention_per_rank_per_layer = (
        (hidden + q_width)
        + (hidden + kv_width)
        + (hidden + kv_width)
        + (q_width + hidden)
    )
    attention = layers * ATTENTION_RANK * attention_per_rank_per_layer if attach_attention else 0
    router = layers * ROUTER_RANK * (hidden + experts)
    common = attention + router

    layer_bypass_routing = layers * hidden
    expert_bypass_ffn = layers * experts * EXPERT_BYPASS_RANK * 2 * hidden
    expert_weight_per_rank = layers * experts * (
        (hidden + 2 * intermediate) + (intermediate + hidden)
    )

    budgets = {}
    for rank in layer_bypass_ranks:
        ffn = layers * rank * 2 * hidden
        budgets[f"layer_bypass_r{rank}"] = {
            "ffn": ffn,
            "ffn_routing": layer_bypass_routing,
            "common": common,
            "total": common + ffn + layer_bypass_routing,
        }
    budgets["expert_bypass_r2"] = {
        "ffn": expert_bypass_ffn,
        "ffn_routing": 0,
        "common": common,
        "total": common + expert_bypass_ffn,
    }
    for rank in EXPERT_WEIGHT_RANKS:
        ffn = rank * expert_weight_per_rank
        budgets[f"expert_weight_r{rank}"] = {
            "ffn": ffn,
            "ffn_routing": 0,
            "common": common,
            "total": common + ffn,
        }

    dimensions = {
        "hidden_size": hidden,
        "expert_intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_experts": experts,
        "top_k": top_k,
        "attention": attention,
        "router": router,
    }
    return budgets, dimensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(DEFAULT_CONFIGS), required=True)
    parser.add_argument("--config", type=Path, help="Override the selected model's config.json path.")
    parser.add_argument(
        "--layer-bypass-ranks",
        default=",".join(map(str, DEFAULT_LAYER_BYPASS_RANKS)),
        help="Comma-separated layer-bypass ranks to report (default: 256).",
    )
    parser.add_argument(
        "--attach-attn",
        type=int,
        choices=(0, 1),
        default=1,
        help="Whether the common attention LoRA r32 branch is attached (default: 1).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config or DEFAULT_CONFIGS[args.model]
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
        layer_bypass_ranks = parse_positive_ranks(args.layer_bypass_ranks)
        budgets, dimensions = compute_budgets(
            config, layer_bypass_ranks, attach_attention=bool(args.attach_attn)
        )
    except (OSError, json.JSONDecodeError, BudgetError) as exc:
        print(f"Budget check failed: {exc}", file=sys.stderr)
        return 2

    print(f"Model: {args.model} | config: {config_path}")
    print(
        "Dimensions: "
        f"d={dimensions['hidden_size']} d_expert={dimensions['expert_intermediate_size']} "
        f"layers={dimensions['num_hidden_layers']} experts={dimensions['num_experts']} "
        f"top_k={dimensions['top_k']}"
    )
    print(f"Common adapters: attention={dimensions['attention']:,} router={dimensions['router']:,}")
    print()
    print(f"{'arm':<24} {'FFN LoRA':>14} {'FFN routing':>14} {'common':>14} {'total':>14}")
    print("-" * 86)
    for name, values in budgets.items():
        print(
            f"{name:<24} {values['ffn']:>14,} {values['ffn_routing']:>14,} "
            f"{values['common']:>14,} {values['total']:>14,}"
        )
    print()
    print(
        "Budget check passed. Uniform expert_weight r1/r2 are intentionally reported as "
        "lower/upper budgets; no mixed per-layer rank prior or 1% parity constraint is applied."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
