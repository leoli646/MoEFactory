#!/usr/bin/env python3
"""汇总本次 MoE2 redesign 的下游与上游评测结果。

纯离线读取预测、指标与 summary.json；不需要 GPU 或模型权重。

用法:
  python3 scripts/exp/moe2/collect_moe2_redesign.py \
      --upstream-root results/upstream_moe2_redesign/qwen3 \
      --downstream-root results/moe2_redesign_p0 \
      --downstream-root results/moe2_redesign_sall \
      --out results/upstream_moe2_redesign/summary_qwen3.csv --model qwen3

缺失的新结果保留为空，不使用历史参考分数补齐。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


def fraction(value, path):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid fractional metric in {path}: {value!r}")
    return float(value)


def upstream_summaries(d: str, *, expect_base: bool = False):
    """Load task summaries and reject mixed checkpoints or base models."""
    summaries = {}
    for task in ("mmlu_pro", "ifeval"):
        path = Path(d) / task / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"upstream summary must be an object: {path}")
        base = summary.get("base_model")
        adapter = summary.get("adapter_path")
        if not isinstance(base, str) or not base.strip() or "adapter_path" not in summary:
            raise ValueError(f"missing upstream provenance: {path}")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError(f"invalid adapter_path: {path}")
        if expect_base and adapter:
            raise ValueError(f"base result contains an adapter: {path}")
        summaries[task] = summary
    if len(summaries) == 2:
        mmlu, ifeval = summaries["mmlu_pro"], summaries["ifeval"]
        if (mmlu["base_model"] != ifeval["base_model"]
                or (mmlu["adapter_path"] or None) != (ifeval["adapter_path"] or None)):
            raise ValueError(f"mismatched upstream adapter/base provenance: {d}")
    return summaries


def read_upstream(d: str, *, expect_base: bool = False):
    """返回 (mmlu_acc, ifeval_strict, empty_rate, trunc_rate) ，缺失为 None。"""
    mmlu = ifeval = empty = trunc = None
    summaries = upstream_summaries(d, expect_base=expect_base)
    if "mmlu_pro" in summaries:
        s = summaries["mmlu_pro"]
        score = fraction(s.get("accuracy"), Path(d) / "mmlu_pro/summary.json")
        if type(s.get("num_examples")) is int and s["num_examples"] == 12032:
            mmlu = score
    if "ifeval" in summaries:
        s = summaries["ifeval"]
        score = fraction((s.get("strict") or {}).get("prompt_level_accuracy"), Path(d) / "ifeval/summary.json")
        if type(s.get("num_examples")) is int and s["num_examples"] == 541:
            ifeval = score
        empty = s.get("empty_response_rate")
        trunc = s.get("truncated_rate")
    return mmlu, ifeval, empty, trunc


def adapter_of(upstream_dir: str) -> str | None:
    """从 summary.json 回读该结果对应的 adapter 目录。"""
    for summary in upstream_summaries(upstream_dir).values():
        if summary["adapter_path"]:
            return summary["adapter_path"]
    return None


def downstream_of(adapter: str | None, roots: list[Path]) -> float | None:
    """Match native pilot result tails to the adapter recorded by upstream eval."""
    if not adapter:
        return None
    tail = Path(*Path(adapter).parts[-4:])  # model/lr/task/moe2official_setting
    candidates = [root / tail / "intent_metrics.json" for root in roots]
    existing = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(existing) > 1:
        raise ValueError(f"ambiguous downstream results for {adapter}: {existing}")
    if not existing:
        return None
    metrics_path = existing[0]
    metrics = json.loads(metrics_path.read_text())
    average = fraction(metrics.get("average_score"), metrics_path)
    predictions_path = metrics_path.with_name("intent_predictions.jsonl")
    if not predictions_path.is_file():
        return None
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    ids = [row.get("idx") for row in predictions]
    scores = [row.get("score") for row in predictions]
    if (len(ids) != 500 or any(type(idx) is not int for idx in ids)
            or sorted(ids) != list(range(500)) or any(score not in (0, 1) for score in scores)
            or metrics.get("num_examples") != 500 or not math.isclose(sum(scores) / 500, average, abs_tol=1e-12)):
        raise ValueError(f"incomplete or inconsistent downstream results: {metrics_path}")
    return average * 100






def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--downstream-root", action="append", type=Path,
                    help="native pilot result root; repeat for P0, P1 and alpha/NTK runs")
    ap.add_argument("--base-upstream-dir", type=Path,
                    help="directory containing real base mmlu_pro/ifeval summaries; enables drift columns")
    args = ap.parse_args()

    if args.model not in ("qwen3", "gemma4", "gptoss"):
        print(f"未知 model={args.model}", file=sys.stderr)
        return 2
    bm = bi = None
    if args.base_upstream_dir:
        bm, bi, _, _ = read_upstream(str(args.base_upstream_dir), expect_base=True)
        if bm is None or bi is None:
            raise ValueError(f"missing base summaries: {args.base_upstream_dir}")
    downstream_roots = args.downstream_root or [Path(args.repo) / "results" / name for name in (
        "moe2_redesign_p0", "moe2_redesign_sall", "moe2_alpha_ntk_grid")]

    arms = []
    root = args.upstream_root
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                arms.append((name, d))

    if not arms:
        print(f"no upstream result directories: {root}", file=sys.stderr)
        return 1

    rows = []
    for label, d in arms:
        mmlu, ifeval, empty, trunc = read_upstream(d)
        if args.base_upstream_dir:
            base_summary = next(iter(upstream_summaries(str(args.base_upstream_dir), expect_base=True).values()))
            if any(s["base_model"] != base_summary["base_model"] for s in upstream_summaries(d).values()):
                raise ValueError(f"arm and base reference use different base models: {d}")
        adapter = adapter_of(d)
        down = downstream_of(adapter, downstream_roots)
        rows.append(
            {
                "arm": label,
                "downstream_intent_percent": down,
                "mmlu_pro_percent": None if mmlu is None else round(mmlu * 100, 2),
                "mmlu_pro_drift_pp": None if mmlu is None or bm is None else round((mmlu - bm) * 100, 2),
                "ifeval_strict_percent": None if ifeval is None else round(ifeval * 100, 2),
                "ifeval_drift_pp": None if ifeval is None or bi is None else round((ifeval - bi) * 100, 2),
                "ifeval_empty_rate_percent": None if empty is None else round(empty * 100, 1),
                "ifeval_truncated_rate_percent": None if trunc is None else round(trunc * 100, 1),
                "status": "done" if (mmlu is not None and ifeval is not None and down is not None) else "pending",
                "adapter_path": adapter or "",
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v, nd=2, w=8):
        return " " * (w - 2) + "--" if v is None else f"{v:>{w}.{nd}f}"

    if bm is not None and bi is not None:
        print(f"base ({args.model}): MMLU {bm*100:.2f}  IFEval {bi*100:.2f}")
    else:
        print("未提供本次 base 上游结果，漂移列留空；未使用历史分数。")
    print(f"{'arm':30}{'Intent':>8}{'dMMLU':>8}{'dIFEv':>8}{'empty%':>8}  status")
    for r in rows:
        print(f"{r['arm']:30}{fmt(r['downstream_intent_percent'],1)}{fmt(r['mmlu_pro_drift_pp'])}"
              f"{fmt(r['ifeval_drift_pp'])}{fmt(r['ifeval_empty_rate_percent'],1)}  {r['status']}")
    print(f"\n已写出 {args.out}")
    print("读表提醒：dIFEval 必须与 empty% 一起看 —— 生成塌陷(empty↑)与格式漂移是两回事。")
    return 0 if all(row["status"] == "done" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
