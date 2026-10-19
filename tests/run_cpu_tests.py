"""Run the bounded public CPU checks without requiring pytest."""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SRC = (ROOT / "src").resolve()
sys.path.insert(0, str(EXPECTED_SRC))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from llamafactory.model.model_utils import moe_lora

    imported_src = Path(moe_lora.__file__).resolve().parents[3]
    if imported_src != EXPECTED_SRC:
        raise RuntimeError(f"Imported {imported_src}, expected {EXPECTED_SRC}")

    files = [
        ROOT / "tests" / "test_adapter_runtime.py",
        ROOT / "tests" / "test_weight_peft_variants.py",
    ]
    tests = []
    for path in files:
        module = _load(path)
        tests.extend(
            (f"{path.name}::{name}", value)
            for name, value in sorted(vars(module).items())
            if name.startswith("test_") and callable(value)
        )

    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - report all standalone failures
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print(f"{len(tests) - failures}/{len(tests)} passed; source={imported_src}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
