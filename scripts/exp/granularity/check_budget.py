#!/usr/bin/env python3
"""Check trainable-parameter parity for the Qwen3 FFN-granularity study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ATTENTION_RANK = 32
ROUTER_RANK = 8
LAYER_BYPASS_RANK = 256
EXPERT_BYPASS_RANK = 2
EXPERT_WEIGHT_HIGH_RANK_LAYERS = 13
MAX_RELATIVE_SPREAD = 0.01


class BudgetError(ValueError):
    pass


def canonical_layer_ranks(num_layers: int) -> list[int]:
    """Place 13 rank-2 layers as evenly as possible across model depth."""
    if num_layers < EXPERT_WEIGHT_HIGH_RANK_LAYERS:
        raise BudgetError(
            f"num_hidden_layers={num_layers} is smaller than the required "
            f"{EXPERT_WEIGHT_HIGH_RANK_LAYERS} rank-2 layers."
        )
    high_indices = [
        round((index + 0.5) * num_layers / EXPERT_WEIGHT_HIGH_RANK_LAYERS - 0.5)
        for index in range(EXPERT_WEIGHT_HIGH_RANK_LAYERS)
    ]
    if len(set(high_indices)) != len(high_indices):
        raise BudgetError(f"Could not distribute rank-2 layers uniquely: {high_indices}")
    return [2 if index in high_indices else 1 for index in range(num_layers)]


def parse_layer_ranks(value: str, num_layers: int) -> list[int]:
    try:
        ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise BudgetError("--layer-ranks must be a comma-separated integer list.") from exc
    if len(ranks) != num_layers:
        raise BudgetError(f"Expected {num_layers} layer ranks, got {len(ranks)}.")
    if any(rank <= 0 for rank in ranks):
        raise BudgetError("Layer ranks must all be positive.")
    return ranks


def require_int(config: dict, key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise BudgetError(f"config.json field {key!r} must be a positive integer, got {value!r}.")
    return value


def compute_budgets(config: dict, layer_ranks: list[int]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    hidden = require_int(config, "hidden_size")
    intermediate = require_int(config, "moe_intermediate_size")
    layers = require_int(config, "num_hidden_layers")
    experts = require_int(config, "num_experts")
    attention_heads = require_int(config, "num_attention_heads")
    kv_heads = require_int(config, "num_key_value_heads")
    head_dim = require_int(config, "head_dim")
    if len(layer_ranks) != layers:
        raise BudgetError(f"Expected {layers} layer ranks, got {len(layer_ranks)}.")

    q_width = attention_heads * head_dim
    kv_width = kv_heads * head_dim
    attention_per_rank_per_layer = (
        (hidden + q_width)
        + (hidden + kv_width)
        + (hidden + kv_width)
        + (q_width + hidden)
    )
    router_per_rank_per_layer = hidden + experts
    attention = layers * ATTENTION_RANK * attention_per_rank_per_layer
    router = layers * ROUTER_RANK * router_per_rank_per_layer
    common = attention + router

    layer_bypass_ffn = layers * LAYER_BYPASS_RANK * (2 * hidden)
    layer_bypass_routing = layers * hidden  # one-class IndependentRouter
    expert_bypass_ffn = layers * experts * EXPERT_BYPASS_RANK * (2 * hidden)
    expert_weight_per_rank = experts * ((hidden + 2 * intermediate) + (intermediate + hidden))
    expert_weight_ffn = sum(layer_ranks) * expert_weight_per_rank

    budgets = {
        "layer_bypass_r256": {
            "ffn": layer_bypass_ffn,
            "ffn_routing": layer_bypass_routing,
            "common": common,
            "total": common + layer_bypass_ffn + layer_bypass_routing,
        },
        "expert_bypass_r2": {
            "ffn": expert_bypass_ffn,
            "ffn_routing": 0,
            "common": common,
            "total": common + expert_bypass_ffn,
        },
        "expert_weight_mixed": {
            "ffn": expert_weight_ffn,
            "ffn_routing": 0,
            "common": common,
            "total": common + expert_weight_ffn,
        },
    }
    dimensions = {
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_experts": experts,
        "num_experts_per_tok": require_int(config, "num_experts_per_tok"),
        "attention": attention,
        "router": router,
    }
    return budgets, dimensions


def default_config_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    return (
        Path(os.environ.get("CKPT_ROOT", repo / "models"))
        / "Qwen3-30B-A3B-Base/config.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--layer-ranks",
        default="",
        help="Comma-separated expert-weight ranks. Empty uses the canonical evenly spaced 35x r1 + 13x r2 plan.",
    )
    parser.add_argument(
        "--print-layer-ranks",
        action="store_true",
        help="Print only the canonical comma-separated layer-rank list.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        with args.config.open() as config_file:
            config = json.load(config_file)
        num_layers = require_int(config, "num_hidden_layers")
        expected_ranks = canonical_layer_ranks(num_layers)
        if args.print_layer_ranks:
            print(",".join(str(rank) for rank in expected_ranks))
            return 0

        layer_ranks = parse_layer_ranks(args.layer_ranks, num_layers) if args.layer_ranks else expected_ranks
        if layer_ranks != expected_ranks:
            raise BudgetError(
                "The mixed-rank plan must use the canonical evenly spaced 13 rank-2 layers. "
                f"Expected {','.join(map(str, expected_ranks))}."
            )
        budgets, dimensions = compute_budgets(config, layer_ranks)
    except (OSError, json.JSONDecodeError, BudgetError) as exc:
        print(f"Budget check failed: {exc}", file=sys.stderr)
        return 2

    high_indices = [index for index, rank in enumerate(layer_ranks) if rank == 2]
    print(f"Model config: {args.config}")
    print(
        "Dimensions: "
        f"d={dimensions['hidden_size']} d_moe={dimensions['moe_intermediate_size']} "
        f"layers={dimensions['num_hidden_layers']} experts={dimensions['num_experts']} "
        f"top_k={dimensions['num_experts_per_tok']}"
    )
    print(f"Common adapters: attention={dimensions['attention']:,} router={dimensions['router']:,}")
    print(f"Mixed expert-weight ranks: sum={sum(layer_ranks)}, rank-2 zero-based layers={high_indices}")
    print()
    print(f"{'arm':<24} {'FFN LoRA':>14} {'FFN routing':>14} {'common':>14} {'total':>14}")
    print("-" * 86)
    for name, values in budgets.items():
        print(
            f"{name:<24} {values['ffn']:>14,} {values['ffn_routing']:>14,} "
            f"{values['common']:>14,} {values['total']:>14,}"
        )

    totals = [values["total"] for values in budgets.values()]
    spread = max(totals) - min(totals)
    relative_spread = spread / max(totals)
    print()
    print(f"Max total spread: {spread:,} ({relative_spread:.3%}; limit {MAX_RELATIVE_SPREAD:.1%})")
    if relative_spread > MAX_RELATIVE_SPREAD:
        print("Budget check failed: trainable-parameter spread exceeds 1%.", file=sys.stderr)
        return 1
    print("Budget check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
