#!/usr/bin/env python3
"""Validate three-seed downstream results and select one LR by mean."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import sys

from protocol import FAMILY_SETTINGS, LRS, MODELS, SEEDS, supported_settings


def score_file(directory: Path, task: str, math_filename: str = "math500.jsonl") -> tuple[float, Path]:
    path = directory / ("intent_predictions.jsonl" if task == "intent" else math_filename)
    if not path.is_file():
        raise ValueError(f"missing result: {path}")
    ids, values = [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                idx = record["idx"]
                value = record["score" if task == "intent" else "correct"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid record {path}:{line_number}: {exc}") from exc
            if type(idx) is not int or not isinstance(value, (int, float, bool)) or value not in (0, 1):
                raise ValueError(f"invalid idx/score {path}:{line_number}")
            ids.append(idx)
            values.append(int(value))
    if len(ids) != 500 or sorted(ids) != list(range(500)):
        raise ValueError(f"expected exactly idx 0..499 in {path}, got {len(ids)} rows")
    accuracy = sum(values) / 500
    if task == "intent":
        metrics_path = directory / "intent_metrics.json"
        if not metrics_path.is_file():
            raise ValueError(f"missing result: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError(f"Intent metrics must be an object: {metrics_path}")
        if metrics.get("num_examples") != 500 or not math.isclose(metrics.get("average_score", -1), accuracy, abs_tol=1e-12):
            raise ValueError(f"Intent metrics disagree with predictions: {directory}")
    return accuracy, path


def adapter_complete(path: Path, family: str, setting: str, task: str) -> None:
    if family == "methods" and setting.startswith("weight_") and setting != "weight_lora":
        weight, config = "weight_peft_state.safetensors", "weight_peft_config.json"
    elif family == "methods" and setting == "das_lora":
        weight, config = "das_lora_state.safetensors", "das_lora_config.json"
    elif family == "methods" and setting == "partial_last":
        weight, config = "partial_ft_state.safetensors", "partial_ft_config.json"
    else:
        weight, config = "moe_lora_state.safetensors", "moe_lora_config.json"
    for filename in (weight, config, "trainer_state.json", "train_results.json"):
        target = path / filename
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"missing or empty training artifact: {target}")
    config_data = json.loads((path / config).read_text(encoding="utf-8"))
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "train_results.json").read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError(f"adapter config must be a JSON object: {path / config}")
    if family == "methods" and setting == "das_lora":
        selected_name = config_data.get("das_lora_selected_experts_path")
        if not isinstance(selected_name, str) or not selected_name:
            raise ValueError(f"DAS adapter config lacks selected-experts path: {path / config}")
        selected_path = Path(selected_name)
        if not selected_path.is_absolute():
            selected_path = path / selected_path
        if not selected_path.is_file() or selected_path.stat().st_size == 0:
            raise ValueError(f"missing or empty DAS selected-experts artifact: {selected_path}")
        selected_data = json.loads(selected_path.read_text(encoding="utf-8"))
        if not isinstance(selected_data, dict):
            raise ValueError(f"DAS selected-experts artifact must be a JSON object: {selected_path}")
    if not isinstance(state, dict) or not isinstance(metrics, dict):
        raise ValueError(f"training metadata must be JSON objects: {path}")
    target_epoch = 3 if task == "intent" else 1
    state_epoch, result_epoch = state.get("epoch"), metrics.get("epoch")
    global_step, max_steps = state.get("global_step"), state.get("max_steps")
    finite_epoch = lambda value: (isinstance(value, (int, float)) and not isinstance(value, bool)
                                  and math.isfinite(value) and target_epoch <= value <= 1000)
    valid_step = lambda value: (isinstance(value, int) and not isinstance(value, bool)
                                and 0 < value <= 1_000_000_000)
    if not (finite_epoch(state_epoch) and finite_epoch(result_epoch)
            and valid_step(global_step) and valid_step(max_steps) and global_step == max_steps):
        raise ValueError(f"training metadata is incomplete: {path}")


def relative_to_output(path: Path, output: Path) -> str:
    return os.path.relpath(path.resolve(), output.parent.resolve())


def cell_directory(root: Path, layout: str, family: str, setting: str,
                   task: str, model: str, seed: int, lr: str) -> Path:
    """Resolve original launcher outputs using the documented native roots."""
    if layout == "protocol" or family == "methods":
        return root / family / setting / task / model / f"seed{seed}" / lr
    epochs = 3 if task == "intent" else 1
    if family == "granularity":
        return root / family / task / model / f"seed{seed}" / f"{setting}_{lr}_ep{epochs}"
    if family == "loss":
        tag = f"expert_weight_r1_{lr}_{setting}_ep{epochs}"
        return root / family / setting / task / model / f"seed{seed}" / tag
    arm = setting.removesuffix("_router")
    trial = f"router_r8_{lr}" if setting.endswith("_router") else lr
    return root / family / task / model / f"seed{seed}" / f"{arm}_{trial}_ep{epochs}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=FAMILY_SETTINGS)
    parser.add_argument("--layout", choices=("protocol", "native"), default="protocol",
                        help="native reads original shell outputs under the roots in scripts/exp/README.md")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=("math", "intent"), default=["math", "intent"])
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--lrs", nargs="+", choices=LRS, default=list(LRS))
    args = parser.parse_args()
    settings = args.settings or FAMILY_SETTINGS[args.family]
    if args.layout == "native" and args.family == "methods":
        native_settings = tuple(s for s in FAMILY_SETTINGS["methods"] if not s.startswith("moe2_"))
        if args.settings is None:
            settings = tuple(s for s in native_settings if s != "expert_moe2_lora")
        elif any(s not in native_settings for s in settings):
            parser.error("MoE2 pilot/redesign outputs use separate roots; select comparison methods only with --layout native")
    if len(settings) != len(set(settings)):
        parser.error("--settings contains duplicates")
    if len(args.lrs) != len(set(args.lrs)):
        parser.error("--lrs contains duplicates")
    if len(args.models) != len(set(args.models)) or len(args.tasks) != len(set(args.tasks)):
        parser.error("--models and --tasks must not contain duplicates")
    unknown_settings = sorted(set(settings) - set(FAMILY_SETTINGS[args.family]))
    if unknown_settings:
        parser.error(f"unknown settings for {args.family}: {', '.join(unknown_settings)}")
    if set(args.lrs) != set(LRS) or len(args.lrs) != len(LRS):
        parser.error(f"the source selection protocol requires the full LR grid: {' '.join(LRS)}")
    supported_count = sum(len(supported_settings(model, args.family, settings)) for model in args.models)
    if args.settings is not None and supported_count == 0:
        parser.error("the requested models/settings contain no supported source-protocol combination")
    errors, selected_rows = [], []
    expected = 0
    for model in args.models:
        for task in args.tasks:
            model_settings = supported_settings(model, args.family, settings)
            expected += len(model_settings)
            for setting in model_settings:
                candidates = []
                for lr in args.lrs:
                    entries, scores = [], []
                    for seed in SEEDS:
                        result_dir = cell_directory(args.result_root, args.layout, args.family, setting, task, model, seed, lr)
                        adapter = cell_directory(args.output_root, args.layout, args.family, setting, task, model, seed, lr)
                        try:
                            adapter_complete(adapter, args.family, setting, task)
                            if setting == "das_lora" and args.layout == "protocol":
                                das_cache = result_dir / "das_selected.json"
                                if not das_cache.is_file() or das_cache.stat().st_size == 0:
                                    raise ValueError(f"missing or empty DAS selection: {das_cache}")
                            math_filename = "adapter_math500.jsonl" if args.layout == "native" and args.family == "position" else "math500.jsonl"
                            score, result_path = score_file(result_dir, task, math_filename)
                            scores.append(score)
                            entries.append({"seed": seed, "adapter": relative_to_output(adapter, args.output),
                                            "downstream_result": relative_to_output(result_path, args.output),
                                            "score": score})
                        except (OSError, ValueError, TypeError) as exc:
                            errors.append({"model": model, "domain": task, "setting": setting,
                                           "lr": lr, "seed": seed, "error": str(exc)})
                    if len(entries) == 3:
                        candidates.append({"lr": lr, "seed_scores": scores, "mean": mean(scores),
                                           "std": stdev(scores), "entries": entries})
                if len(candidates) != len(args.lrs):
                    continue
                # Source selection protocol: common LR with highest downstream three-seed mean; tie -> smaller LR.
                selected = max(candidates, key=lambda row: (row["mean"], -LRS.index(row["lr"])))
                selected_rows.append({"experiment": args.family, "model": model, "domain": task,
                                      "setting": setting, "lr": selected["lr"],
                                      "mean": selected["mean"], "std": selected["std"],
                                      "entries": selected["entries"], "candidates": candidates})
    payload = {"protocol": "MoEFactory source implementation protocol", "layout": args.layout, "seeds": list(SEEDS),
               "learning_rates": args.lrs, "selection": "highest common-LR downstream three-seed mean; tie uses smaller LR",
               "rows": selected_rows, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors or len(selected_rows) != expected:
        print(f"selection incomplete: rows={len(selected_rows)}/{expected}, errors={len(errors)}; see {args.output}", file=sys.stderr)
        return 1
    print(f"selection complete: rows={len(selected_rows)}, output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
