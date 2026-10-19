#!/usr/bin/env python3
"""Print repeatable commands for a source-protocol matrix; never runs them."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from protocol import FAMILY_SETTINGS, LRS, MODELS, SEEDS, supported_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILY_SETTINGS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=("math", "intent"), default=["math", "intent"])
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--lrs", nargs="+", choices=LRS, default=list(LRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--model-root", type=Path, required=True,
                        help="contains qwen3, gemma4 and gptoss model directories")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-upstream", action="store_true")
    parser.add_argument("--das-general", type=Path)
    args = parser.parse_args()
    settings = args.settings or FAMILY_SETTINGS[args.family]
    unknown = sorted(set(settings) - set(FAMILY_SETTINGS[args.family]))
    if unknown:
        parser.error(f"unknown settings for {args.family}: {', '.join(unknown)}")
    runner = Path(__file__).with_name("run_protocol.py")
    count = 0
    for model in args.models:
        for task in args.tasks:
            for setting in supported_settings(model, args.family, settings):
                if setting == "das_lora" and args.das_general is None:
                    parser.error("DAS-LoRA matrix generation requires --das-general")
                for lr in args.lrs:
                    for seed in args.seeds:
                        command = ["python3", str(runner), "--model", model, "--model-path",
                                   str(args.model_root / model), "--data-root", str(args.data_root),
                                   "--family", args.family, "--setting", setting, "--task", task,
                                   "--lr", lr, "--seed", str(seed)]
                        if args.run_upstream:
                            command.append("--run-upstream")
                        if setting == "das_lora":
                            command += ["--das-general", str(args.das_general)]
                        print(shlex.join(command))
                        count += 1
    if args.settings is not None and count == 0:
        parser.error("the requested models/settings/tasks contain no supported source-protocol combination")
    print(f"# commands={count}; all commands default to preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
