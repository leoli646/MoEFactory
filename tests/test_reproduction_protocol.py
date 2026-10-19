import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml


REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "scripts/reproduce"
sys.path.insert(0, str(REPRO))
import protocol


class ReproductionProtocolTests(unittest.TestCase):
    def spec(self, **updates):
        values = dict(repo=REPO, model="qwen3", model_path=Path("/models/qwen3"),
                      data_root=Path("/data"), family="methods", setting="weight_lora",
                      task="math", lr="lr1e-4", seed=42, gpus="0,1,2,3",
                      output_root=Path("saves/source_protocol"), result_root=Path("results/source_protocol"))
        values.update(updates)
        return protocol.RunSpec(**values)

    def test_preview_is_non_executing_and_preserves_method_overrides(self):
        command = [sys.executable, str(REPRO / "run_protocol.py"), "--model", "qwen3",
                   "--model-path", "/models/qwen3", "--data-root", "/data",
                   "--family", "methods", "--setting", "weight_lora", "--task", "math",
                   "--lr", "lr1e-4", "--seed", "42"]
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=True)
        self.assertIn("mode=preview", result.stdout)
        self.assertIn("moe_lora_ffn_granularity=expert_weight", result.stdout)
        self.assertIn("moe_lora_attach_attn=false", result.stdout)
        self.assertIn("moe_lora_attach_router=true", result.stdout)
        self.assertIn("moe_lora_router_rank=8", result.stdout)
        self.assertIn("MOEFACTORY_DATA_ROOT=/data", result.stdout)
        self.assertIn("moe_lora_b_init=normal", result.stdout)

    def test_preview_resolves_cli_paths_from_invocation_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp).resolve()
            command = [sys.executable, str(REPRO / "run_protocol.py"), "--model", "qwen3",
                       "--model-path", "models/qwen3", "--data-root", "datasets",
                       "--output-root", "outputs", "--result-root", "results",
                       "--deepspeed", "configs/ds.json", "--family", "methods",
                       "--setting", "weight_lora", "--task", "math",
                       "--lr", "lr1e-4", "--seed", "42"]
            result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
            self.assertIn(f"adapter={cwd / 'outputs/methods/weight_lora/math/qwen3/seed42/lr1e-4'}", result.stdout)
            self.assertIn(f"model_name_or_path={cwd / 'models/qwen3'}", result.stdout)
            self.assertIn(f"dataset_dir={cwd / 'datasets'}", result.stdout)
            self.assertIn(f"deepspeed={cwd / 'configs/ds.json'}", result.stdout)

    def test_das_and_moe2_semantics_are_explicit(self):
        das = self.spec(setting="das_lora", task="intent", das_general=Path("/data/general.json"))
        steps = protocol.commands(das)
        self.assertEqual(steps[0][0], "das")
        self.assertIn("eval_scripts/compute_das.py", steps[0][1])
        self.assertIn("CUDA_VISIBLE_DEVICES=0", steps[0][1])
        normalized = self.spec(setting="moe2_pilot_default_ntk")
        flags = protocol.method_flags(normalized)
        self.assertIn("moe_lora_pool_share=global", flags)
        self.assertIn("moe_lora_normalize_topk_prob=true", flags)
        default = self.spec(setting="moe2_pilot_default")
        default_flags = protocol.method_flags(default)
        self.assertNotIn("moe_lora_normalize_topk_prob=true", default_flags)
        self.assertIn("moe_lora_n_experts=280", default_flags)
        self.assertIn("moe_lora_attach_attn=false", default_flags)
        self.assertIn("moe_lora_balance_loss_coef=0.0", default_flags)
        self.assertIn("moe_lora_balance_loss_schedule=linear_decay", default_flags)
        final = self.spec(setting="moe2_sall_n128_r32_wp_up_ntk_a11_bz")
        final_flags = protocol.method_flags(final)
        self.assertIn("moe_lora_ffn_granularity=weight_pool", final_flags)
        self.assertIn("moe_lora_alpha=11", final_flags)
        self.assertIn("moe_lora_wp_gate_up_stage=up", final_flags)

    def test_position_source_batch_and_rank_overrides(self):
        qwen_math = self.spec(family="position", setting="attn_r64", task="math")
        self.assertIn("per_device_train_batch_size=12", protocol.train_command(qwen_math))
        self.assertIn("gradient_accumulation_steps=3", protocol.train_command(qwen_math))
        qwen_intent_router = self.spec(family="position", setting="attn_r64_router", task="intent")
        self.assertIn("per_device_train_batch_size=12", protocol.train_command(qwen_intent_router))
        self.assertIn("gradient_accumulation_steps=3", protocol.train_command(qwen_intent_router))
        self.assertEqual(len(protocol.FAMILY_SETTINGS["position"]), 8)
        for model in protocol.MODELS:
            for task in ("math", "intent"):
                position = self.spec(model=model, family="position", setting="attn32_dense128", task=task)
                flags = protocol.method_flags(position)
                self.assertIn("moe_lora_rank=128", flags)
                self.assertIn("moe_lora_n_experts=1", flags)
                self.assertIn("moe_lora_routing_mode=independent", flags)

    def test_seed_and_lr_constants(self):
        self.assertEqual(protocol.SEEDS, (42, 0, 1))
        self.assertEqual(set(protocol.LRS), {"lr1e-5", "lr1e-4", "lr1e-3"})
        upstream = protocol.upstream_commands(self.spec(run_upstream=True))[0]
        self.assertIn("MODEL=qwen3", upstream)
        self.assertIn("ENGINE=auto", upstream)
        self.assertIn("MAX_LEN=4096", upstream)
        self.assertIn("IFEVAL_MAXNEW=1280", upstream)
        self.assertIn("MOEFACTORY_DATA_ROOT=/data", upstream)
        intent = protocol.downstream_command(self.spec(task="intent"))
        self.assertIn("--esft-dir", intent)
        self.assertIn("/data/esft_eval", intent)
        loss = self.spec(family="loss", setting="aux1e-3")
        loss_flags = protocol.method_flags(loss)
        self.assertIn("moe_lora_balance_loss_coef=1.0e-3", loss_flags)
        self.assertIn("moe_lora_balance_loss_schedule=constant", loss_flags)

    def test_adapter_completion_requires_config_and_completed_training(self):
        spec = importlib.util.spec_from_file_location("selector", REPRO / "select_best_lr.py")
        selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selector)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "das_lora_state.safetensors").write_bytes(b"x")
            with self.assertRaises(ValueError):
                selector.adapter_complete(path, "methods", "das_lora", "math")
            (path / "das_lora_config.json").write_text(json.dumps({"das_lora_selected_experts_path": "das_selected_experts.json"}))
            (path / "trainer_state.json").write_text(json.dumps({"epoch": 1, "global_step": 10, "max_steps": 10}))
            (path / "train_results.json").write_text(json.dumps({"epoch": 1}))
            with self.assertRaises(ValueError):
                selector.adapter_complete(path, "methods", "das_lora", "math")
            (path / "das_selected_experts.json").write_text(json.dumps({"selected_experts_per_layer": [[0]]}))
            selector.adapter_complete(path, "methods", "das_lora", "math")
            (path / "das_lora_config.json").write_text("[]")
            with self.assertRaises(ValueError):
                selector.adapter_complete(path, "methods", "das_lora", "math")
            (path / "das_lora_config.json").write_text(json.dumps({"das_lora_selected_experts_path": "das_selected_experts.json"}))
            (path / "trainer_state.json").write_text(json.dumps({"epoch": float("nan"), "global_step": 10, "max_steps": 10}))
            with self.assertRaises(ValueError):
                selector.adapter_complete(path, "methods", "das_lora", "math")
            (path / "trainer_state.json").write_text(json.dumps({"epoch": 1, "global_step": True, "max_steps": True}))
            with self.assertRaises(ValueError):
                selector.adapter_complete(path, "methods", "das_lora", "math")

    def test_selector_tie_breaks_to_smaller_lr_and_emits_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results, outputs = root / "results", root / "saves"
            for lr, score_count in (("lr1e-5", 300), ("lr1e-4", 300), ("lr1e-3", 200)):
                for seed in protocol.SEEDS:
                    result = results / "granularity/expert_weight_r1/math/qwen3" / f"seed{seed}/{lr}"
                    adapter = outputs / "granularity/expert_weight_r1/math/qwen3" / f"seed{seed}/{lr}"
                    result.mkdir(parents=True)
                    adapter.mkdir(parents=True)
                    (adapter / "moe_lora_state.safetensors").write_bytes(b"x")
                    (adapter / "moe_lora_config.json").write_text("{}")
                    (adapter / "trainer_state.json").write_text(json.dumps({"epoch": 1, "global_step": 10, "max_steps": 10}))
                    (adapter / "train_results.json").write_text(json.dumps({"epoch": 1}))
                    with (result / "math500.jsonl").open("w", encoding="utf-8") as handle:
                        for idx in range(500):
                            handle.write(json.dumps({"idx": idx, "correct": int(idx < score_count)}) + "\n")
            output = root / "selection/selected.json"
            command = [sys.executable, str(REPRO / "select_best_lr.py"), "--family", "granularity",
                       "--models", "qwen3", "--tasks", "math", "--settings", "expert_weight_r1",
                       "--result-root", str(results), "--output-root", str(outputs), "--output", str(output)]
            subprocess.run(command, cwd=REPO, check=True, text=True, capture_output=True)
            row = json.loads(output.read_text())["rows"][0]
            self.assertEqual(row["lr"], "lr1e-5")
            self.assertEqual(row["experiment"], "granularity")
            self.assertEqual([entry["seed"] for entry in row["entries"]], [42, 0, 1])
            self.assertFalse(Path(row["entries"][0]["adapter"]).is_absolute())

    def test_selector_fails_on_missing_seed_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "selected.json"
            command = [sys.executable, str(REPRO / "select_best_lr.py"), "--family", "granularity",
                       "--models", "qwen3", "--tasks", "math", "--settings", "expert_weight_r1",
                       "--result-root", str(root / "results"), "--output-root", str(root / "saves"),
                       "--output", str(output)]
            result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["rows"], [])
            self.assertGreater(len(payload["errors"]), 0)

    def test_selector_rejects_partial_lr_grid_and_unknown_settings(self):
        base = [sys.executable, str(REPRO / "select_best_lr.py"), "--family", "granularity",
                "--models", "qwen3", "--tasks", "math", "--result-root", "/tmp/results",
                "--output-root", "/tmp/saves", "--output", "/tmp/selected.json"]
        partial = subprocess.run(base + ["--settings", "expert_weight_r1", "--lrs", "lr1e-4", "lr1e-3"],
                                 cwd=REPO, text=True, capture_output=True)
        self.assertEqual(partial.returncode, 2)
        unknown = subprocess.run(base + ["--settings", "not_a_setting"], cwd=REPO, text=True, capture_output=True)
        self.assertEqual(unknown.returncode, 2)

    def test_matrix_and_selector_share_model_specific_setting_filter(self):
        matrix = [sys.executable, str(REPRO / "print_matrix.py"), "--family", "methods",
                  "--models", "gemma4", "--tasks", "math",
                  "--settings", "moe2_pilot_n280_r32",
                  "--model-root", "/models", "--data-root", "/data"]
        result = subprocess.run(matrix, cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("no supported source-protocol combination", result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "selected.json"
            selector = [sys.executable, str(REPRO / "select_best_lr.py"), "--family", "methods",
                        "--models", "gemma4", "--tasks", "math",
                        "--settings", "moe2_pilot_n280_r32", "--result-root", str(root / "results"),
                        "--output-root", str(root / "saves"), "--output", str(output)]
            completed = subprocess.run(selector, cwd=REPO, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("no supported source-protocol combination", completed.stderr)
            self.assertFalse(output.exists())

    def test_paper_yaml_snapshots_match_generator(self):
        spec = importlib.util.spec_from_file_location("generator", REPRO / "generate_paper_configs.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        for model in protocol.MODELS:
            for task in ("math", "intent"):
                path = REPO / "examples/paper" / f"weight_lora_{model}_{task}_lr1e-5_seed42.yaml"
                self.assertTrue(path.read_text().startswith(generator.HEADER))
                self.assertEqual(yaml.safe_load(path.read_text()), generator.snapshot(model, task))
                generator.validate_snapshot(yaml.safe_load(path.read_text()))
        with self.assertRaises(ValueError):
            generator.validate_snapshot({"source_protocol_snapshot": True})

    def test_upstream_wrapper_preserves_safe_adapter_argv_and_source_environment(self):
        text = (REPRO / "run_upstream_eval.sh").read_text()
        self.assertIn("sys.argv[1]", text)
        self.assertIn('"$ADAPTER" 2>/dev/null', text)
        self.assertIn('[[ "$MODEL" == gemma4 ]]', text)
        self.assertIn('export NLTK_DATA="${NLTK_DATA:-$DATA_ROOT/nltk_data}"', text)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            bindir = root / "bin"
            bindir.mkdir()
            log = root / "python-args"
            fake = bindir / "python3"
            fake.write_text(f'#!/usr/bin/env bash\nprintf \'%s\n\' "$@" > "{log}"\n')
            fake.chmod(0o755)
            adapter = root / "adapter with ' quote"
            env = os.environ.copy()
            env.update(PATH=f"{bindir}:{os.environ['PATH']}", REPO=str(REPO),
                       BASE_MODEL=str(model), MODEL="qwen3", ADAPTER=str(adapter),
                       OUT_DIR=str(root / "out"), TASKS=" ")
            subprocess.run(["bash", str(REPRO / "run_upstream_eval.sh")], env=env, check=True)
            self.assertEqual(log.read_text().splitlines()[-1], str(adapter))


if __name__ == "__main__":
    unittest.main()
