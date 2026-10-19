#!/usr/bin/env python3
"""Validate and export local ESFT Intent train/eval files without bundling data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


TASKS = ("intent",)
REGISTRY = Path(__file__).resolve().parents[2] / "data" / "dataset_info.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def validate_train_row(row: dict[str, Any], source: Path, index: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{source}:{index}: expected at least two messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"{source}:{index}: invalid message role")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{source}:{index}: message content must be text")


def validate_eval_row(row: dict[str, Any], source: Path, index: int) -> None:
    if not isinstance(row.get("prompt"), str):
        raise ValueError(f"{source}:{index}: prompt must be text")
    answers = row.get("raw_answers")
    if not isinstance(answers, list) or not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError(f"{source}:{index}: raw_answers must be a non-empty string list")


def export_dataset(source_root: Path, output_root: Path, tasks: tuple[str, ...] = TASKS) -> dict[str, int]:
    unsupported = set(tasks) - set(TASKS)
    if unsupported:
        raise ValueError(f"Unsupported ESFT tasks: {sorted(unsupported)}; only intent is released.")
    counts = {}
    for task in tasks:
        train_source = source_root / "datasets" / "train" / f"{task}.jsonl"
        eval_source = source_root / "datasets" / "eval" / f"{task}.jsonl"
        train_rows = read_jsonl(train_source)
        eval_rows = read_jsonl(eval_source)
        for index, row in enumerate(train_rows, 1):
            validate_train_row(row, train_source, index)
        task_ids: set[int] = set()
        for index, row in enumerate(eval_rows, 1):
            validate_eval_row(row, eval_source, index)
            idx = row.get("idx")
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(f"{eval_source}:{index}: idx must be an integer")
            if idx in task_ids:
                raise ValueError(f"{eval_source}:{index}: duplicate idx {idx}")
            task_ids.add(idx)

        train_target = output_root / "esft" / f"{task}_train.jsonl"
        eval_target = output_root / "esft_eval" / f"{task}.jsonl"
        train_target.parent.mkdir(parents=True, exist_ok=True)
        eval_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(train_source, train_target)
        shutil.copyfile(eval_source, eval_target)
        counts[f"{task}_train"] = len(train_rows)
        counts[f"{task}_eval"] = len(eval_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    registry_target = output_root / "dataset_info.json"
    if REGISTRY.resolve() != registry_target.resolve():
        shutil.copyfile(REGISTRY, registry_target)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Local checkout of the public ESFT dataset layout.")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = export_dataset(args.source_root, args.output_root, tuple(args.tasks))
    for name, count in counts.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
