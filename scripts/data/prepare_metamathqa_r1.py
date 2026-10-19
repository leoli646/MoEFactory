#!/usr/bin/env python3
"""Convert a local MetaMathQA-R1 export to LlamaFactory ShareGPT JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return text
    cleaned = THINK_PATTERN.sub("", text)
    if "<think>" in cleaned or "</think>" in cleaned:
        return cleaned
    return cleaned.strip()


def convert_sample(sample: dict[str, Any], keep_think: bool = False) -> dict[str, Any] | None:
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("complete") is False:
        return None

    prompt = sample.get("prompt") or ""
    response = sample.get("response") or ""
    if not keep_think:
        response = strip_think(response)
    if not response.strip():
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: expected an object")
                    yield row
        else:
            rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError(f"{path}: expected a JSON array")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: expected every item to be an object")
                yield row


def prepare(records: Iterable[dict[str, Any]], keep_think: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    output = []
    for input_index, record in enumerate(records):
        if limit is not None and input_index >= limit:
            break
        converted = convert_sample(record, keep_think=keep_think)
        if converted is not None:
            output.append(converted)
    return output


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local MetaMathQA-R1 JSON or JSONL export.")
    parser.add_argument("--output", type=Path, default=Path("data/metamathqa_r1_clean.json"))
    parser.add_argument("--keep-think", action="store_true")
    parser.add_argument("--limit", type=positive_int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare(read_records(args.input), keep_think=args.keep_think, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
