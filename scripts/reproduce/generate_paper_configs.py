#!/usr/bin/env python3
"""Generate inspectable Weight-LoRA source-protocol YAML snapshots."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from protocol import MODELS, RunSpec, LR_VALUES, _base_config, _task_flags, method_flags


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "examples/paper"
HPARAM_FILES = ("model_args.py", "data_args.py", "training_args.py",
                "finetuning_args.py", "generating_args.py")
TRANSFORMERS_TRAINING_FIELDS = {
    "output_dir", "do_train", "per_device_train_batch_size",
    "gradient_accumulation_steps", "learning_rate", "num_train_epochs",
    "lr_scheduler_type", "warmup_ratio", "bf16", "gradient_checkpointing",
    "ddp_find_unused_parameters", "logging_steps", "save_steps",
    "save_only_model", "report_to", "run_name", "seed", "data_seed",
    "dataloader_num_workers", "deepspeed",
}
HEADER = ("# Generated source implementation protocol snapshot.\n"
          "# The generator and source runners are authoritative; this is not a paper-compliance claim.\n")


def scalar(text: str):
    return yaml.safe_load(text)


def accepted_argument_names() -> set[str]:
    names = set(TRANSFORMERS_TRAINING_FIELDS)
    hparams = REPO / "src/llamafactory/hparams"
    for filename in HPARAM_FILES:
        tree = ast.parse((hparams / filename).read_text(encoding="utf-8"), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for statement in node.body:
                    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                        names.add(statement.target.id)
    return names


def validate_snapshot(config: dict) -> None:
    unknown = sorted(set(config) - accepted_argument_names())
    if unknown:
        raise ValueError(f"snapshot has arguments unknown to the training parser: {', '.join(unknown)}")


def snapshot(model: str, task: str) -> dict:
    spec = RunSpec(repo=REPO, model=model, model_path=Path(f"models/{model}"), data_root=Path("data"),
                   family="methods", setting="weight_lora", task=task, lr="lr1e-5", seed=42,
                   gpus="0,1,2,3", output_root=Path("saves/source_protocol"),
                   result_root=Path("results/source_protocol"))
    config = yaml.safe_load((REPO / _base_config(spec)).read_text(encoding="utf-8"))
    overrides = _task_flags(spec) + method_flags(spec)
    for item in overrides:
        key, value = item.split("=", 1)
        config[key] = scalar(value)
    config.update({
        "model_name_or_path": str(spec.model_path), "dataset_dir": str(spec.data_root),
        "learning_rate": float(LR_VALUES[spec.lr]), "seed": 42, "data_seed": 42,
        "deepspeed": "examples/deepspeed/ds_z2_config.json", "report_to": "none",
        "overwrite_output_dir": False, "output_dir": str(spec.adapter),
        "run_name": f"{model}_methods_weight_lora_{task}_lr1e-5_seed42",
    })
    validate_snapshot(config)
    return config


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for task in ("math", "intent"):
            path = OUTPUT / f"weight_lora_{model}_{task}_lr1e-5_seed42.yaml"
            body = yaml.safe_dump(snapshot(model, task), sort_keys=False, allow_unicode=True)
            path.write_text(HEADER + body, encoding="utf-8")
            print(path.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
