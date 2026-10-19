"""Validate aggregation of native shell outputs using generated tiny artifacts."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/reproduce"))
from select_best_lr import cell_directory


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def predictions(directory, task, correct=300, filename=None):
    filename = filename or ("intent_predictions.jsonl" if task == "intent" else "math500.jsonl")
    directory.mkdir(parents=True, exist_ok=True)
    score_key = "score" if task == "intent" else "correct"
    (directory / filename).write_text("".join(
        json.dumps({"idx": idx, score_key: int(idx < correct)}) + "\n" for idx in range(500)))
    if task == "intent":
        write_json(directory / "intent_metrics.json", {"average_score": correct / 500, "num_examples": 500})


class NativeResultTests(unittest.TestCase):
    def test_native_selector_reads_all_four_original_directory_layouts(self):
        cases = [("methods", "weight_lora", "math"), ("granularity", "expert_weight_r1", "intent"),
                 ("position", "attn_r64_router", "math"), ("loss", "aux1e-3", "intent")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family, setting, task in cases:
                for lr, correct in [("lr1e-5", 300), ("lr1e-4", 310), ("lr1e-3", 200)]:
                    for seed in (42, 0, 1):
                        adapter = cell_directory(root / "saves", "native", family, setting, task, "qwen3", seed, lr)
                        result = cell_directory(root / "results", "native", family, setting, task, "qwen3", seed, lr)
                        write_json(adapter / "moe_lora_config.json", {})
                        (adapter / "moe_lora_state.safetensors").write_bytes(b"fixture")
                        epoch = 3 if task == "intent" else 1
                        write_json(adapter / "trainer_state.json", {"epoch": epoch, "global_step": 10, "max_steps": 10})
                        write_json(adapter / "train_results.json", {"epoch": epoch})
                        predictions(result, task, correct, "adapter_math500.jsonl" if family == "position" else None)
                output = root / f"{family}.json"
                command = [sys.executable, "-B", str(REPO / "scripts/reproduce/select_best_lr.py"),
                           "--layout", "native", "--family", family, "--settings", setting,
                           "--models", "qwen3", "--tasks", task, "--result-root", str(root / "results"),
                           "--output-root", str(root / "saves"), "--output", str(output)]
                with self.subTest(family=family):
                    proc = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    payload = json.loads(output.read_text())
                    self.assertEqual(payload["layout"], "native")
                    self.assertEqual(payload["rows"][0]["lr"], "lr1e-4")
                    self.assertEqual([v["seed"] for v in payload["rows"][0]["entries"]], [42, 0, 1])
                    last = cell_directory(root / "results", "native", family, setting, task, "qwen3", 1, "lr1e-3")
                    next(last.glob("*.jsonl")).unlink()
                    failed = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertEqual(json.loads(output.read_text())["rows"], [])

    def test_native_selector_does_not_accept_pilot_as_a_comparison_arm(self):
        proc = subprocess.run([sys.executable, "-B", str(REPO / "scripts/reproduce/select_best_lr.py"),
                               "--layout", "native", "--family", "methods", "--settings", "moe2_pilot_default",
                               "--result-root", "/unused", "--output-root", "/unused", "--output", "/unused/out"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("separate roots", proc.stderr)

    def test_moe2_collector_uses_fresh_downstream_and_optional_base_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tail = Path("qwen3/lr1e-3/intent/moe2official_n128_r32_wp_a11")
            adapter = str(root / "absent_models" / tail)
            for directory, mmlu, ifeval in [(root / "up/wp-a11", .6, .4), (root / "base", .5, .3)]:
                adapter_path = None if directory == root / "base" else adapter
                write_json(directory / "mmlu_pro/summary.json", {"accuracy": mmlu, "adapter_path": adapter_path,
                                                                "base_model": "/models/qwen3", "num_examples": 12032})
                write_json(directory / "ifeval/summary.json", {"strict": {"prompt_level_accuracy": ifeval},
                                                              "adapter_path": adapter_path,
                                                              "base_model": "/models/qwen3", "num_examples": 541})
            output = root / "summary.csv"
            command = [sys.executable, "-B", str(REPO / "scripts/exp/moe2/collect_moe2_redesign.py"),
                       "--upstream-root", str(root / "up"), "--downstream-root", str(root / "down"),
                       "--out", str(output)]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            row = next(csv.DictReader(io.StringIO(output.read_text())))
            self.assertEqual(row["downstream_intent_percent"], "")
            self.assertEqual(row["status"], "pending")
            predictions(root / "down" / tail, "intent", 321)
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            row = next(csv.DictReader(io.StringIO(output.read_text())))
            self.assertAlmostEqual(float(row["downstream_intent_percent"]), 64.2)
            self.assertEqual(row["mmlu_pro_drift_pp"], "")
            proc = subprocess.run([*command, "--base-upstream-dir", str(root / "base")],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            row = next(csv.DictReader(io.StringIO(output.read_text())))
            self.assertEqual(float(row["mmlu_pro_drift_pp"]), 10)
            self.assertEqual(float(row["ifeval_drift_pp"]), 10)
            for task in ("mmlu_pro", "ifeval"):
                path = root / "base" / task / "summary.json"
                summary = json.loads(path.read_text())
                summary["base_model"] = "/models/other"
                write_json(path, summary)
            mismatch = subprocess.run([*command, "--base-upstream-dir", str(root / "base")],
                                      capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("different base models", mismatch.stderr)

    def test_moe2_collector_rejects_mixed_seed_base_and_adapter_base_reference(self):
        spec = importlib.util.spec_from_file_location("moe2_provenance", REPO / "scripts/exp/moe2/collect_moe2_redesign.py")
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mmlu = {"accuracy": .6, "num_examples": 12032, "base_model": "/models/qwen3", "adapter_path": "/seed42/adapter"}
            ifeval = {"strict": {"prompt_level_accuracy": .4}, "num_examples": 541,
                      "base_model": "/models/qwen3", "adapter_path": "/seed42/adapter"}
            write_json(root / "mmlu_pro/summary.json", mmlu)
            for replacement in ({"adapter_path": "/seed0/adapter"}, {"base_model": "/models/other"}):
                write_json(root / "ifeval/summary.json", dict(ifeval, **replacement))
                for reader in (collector.read_upstream, collector.adapter_of):
                    with self.subTest(reader=reader.__name__, replacement=replacement):
                        with self.assertRaisesRegex(ValueError, "mismatched"):
                            reader(str(root))
            write_json(root / "ifeval/summary.json", ifeval)
            with self.assertRaisesRegex(ValueError, "base result contains an adapter"):
                collector.read_upstream(str(root), expect_base=True)

    def test_moe2_collector_marks_smoke_counts_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tail = Path("qwen3/lr1e-3/intent/moe2official_n128_r32_wp_a11")
            adapter = str(root / "absent" / tail)
            predictions(root / "down" / tail, "intent")
            command = [sys.executable, "-B", str(REPO / "scripts/exp/moe2/collect_moe2_redesign.py"),
                       "--upstream-root", str(root / "up"), "--downstream-root", str(root / "down"),
                       "--out", str(root / "out.csv")]
            for mmlu_count, ifeval_count in ((16, 541), (12032, 16), (None, 541)):
                write_json(root / "up/arm/mmlu_pro/summary.json", {"accuracy": .6, "num_examples": mmlu_count,
                           "base_model": "/models/qwen3", "adapter_path": adapter})
                write_json(root / "up/arm/ifeval/summary.json", {"strict": {"prompt_level_accuracy": .4},
                           "num_examples": ifeval_count, "base_model": "/models/qwen3", "adapter_path": adapter})
                with self.subTest(mmlu=mmlu_count, ifeval=ifeval_count):
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    row = next(csv.DictReader(io.StringIO((root / "out.csv").read_text())))
                    self.assertEqual(row["status"], "pending")
                    self.assertNotIn("delta_norm_mean", row)
                    self.assertNotIn("rel_delta_mean_percent", row)

    def test_moe2_collector_rejects_empty_inconsistent_and_ambiguous_results(self):
        spec = importlib.util.spec_from_file_location("moe2_collector", REPO / "scripts/exp/moe2/collect_moe2_redesign.py")
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tail = Path("qwen3/lr1e-3/intent/moe2official_n128_r32_wp_ntk")
            predictions(root / "first" / tail, "intent")
            write_json(root / "first" / tail / "intent_metrics.json", {"num_examples": 500, "average_score": .9})
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                collector.downstream_of(str(tail), [root / "first"])
            predictions(root / "second" / tail, "intent")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                collector.downstream_of(str(tail), [root / "first", root / "second"])
            proc = subprocess.run([sys.executable, "-B", str(REPO / "scripts/exp/moe2/collect_moe2_redesign.py"),
                                   "--upstream-root", str(root / "missing"), "--out", str(root / "out.csv")],
                                  capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse((root / "out.csv").exists())


if __name__ == "__main__":
    unittest.main()
