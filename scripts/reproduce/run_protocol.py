#!/usr/bin/env python3
"""Preview or explicitly execute one source-implementation experiment cell."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from protocol import FAMILY_SETTINGS, LRS, MODELS, RunSpec, commands, shell_join


def resolve_from_cwd(path: Path, cwd: Path) -> Path:
    return path.resolve() if path.is_absolute() else (cwd / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--family", required=True, choices=FAMILY_SETTINGS)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--task", required=True, choices=("math", "intent"))
    parser.add_argument("--lr", required=True, choices=LRS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, default=Path("saves/source_protocol"))
    parser.add_argument("--result-root", type=Path, default=Path("results/source_protocol"))
    parser.add_argument("--deepspeed", default="examples/deepspeed/ds_z2_config.json")
    parser.add_argument("--phase", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--das-general", type=Path)
    parser.add_argument("--run-upstream", action="store_true")
    parser.add_argument("--execute", action="store_true", help="run commands; default is preview only")
    parser.add_argument("--force", action="store_true", help="allow writing into an existing non-empty target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invocation_cwd = Path.cwd()
    repo = Path(__file__).resolve().parents[2]
    deepspeed = str(resolve_from_cwd(Path(args.deepspeed), invocation_cwd))
    spec = RunSpec(repo=repo, model=args.model,
                   model_path=resolve_from_cwd(args.model_path, invocation_cwd),
                   data_root=resolve_from_cwd(args.data_root, invocation_cwd),
                   family=args.family, setting=args.setting, task=args.task, lr=args.lr, seed=args.seed,
                   gpus=args.gpus, output_root=resolve_from_cwd(args.output_root, invocation_cwd),
                   result_root=resolve_from_cwd(args.result_root, invocation_cwd),
                   deepspeed=deepspeed, phase=args.phase,
                   das_general=(resolve_from_cwd(args.das_general, invocation_cwd)
                                if args.das_general is not None else None),
                   run_upstream=args.run_upstream)
    try:
        steps = commands(spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("mode=execute" if args.execute else "mode=preview (no command will run)")
    print(f"adapter={spec.adapter}")
    print(f"result={spec.result}")
    if args.execute and not args.force:
        if spec.phase in ("train", "full") and spec.adapter.exists() and any(spec.adapter.iterdir()):
            print(f"error: refusing to overwrite non-empty adapter target without --force: {spec.adapter}", file=sys.stderr)
            return 2
        downstream = spec.result / ("math500.jsonl" if spec.task == "math" else "intent_predictions.jsonl")
        if spec.phase in ("eval", "full") and downstream.exists():
            print(f"error: refusing to overwrite downstream result without --force: {downstream}", file=sys.stderr)
            return 2
    for kind, command in steps:
        print(f"[{kind}] {shell_join(command)}")
        if args.execute:
            spec.adapter.mkdir(parents=True, exist_ok=True)
            spec.result.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=repo, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
