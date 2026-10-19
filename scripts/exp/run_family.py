#!/usr/bin/env python3
"""Public convenience wrapper for scripts/reproduce/print_matrix.py."""

from pathlib import Path
import runpy
import sys

reproduce = Path(__file__).resolve().parents[1] / "reproduce"
sys.path.insert(0, str(reproduce))
runpy.run_path(str(reproduce / "print_matrix.py"), run_name="__main__")
