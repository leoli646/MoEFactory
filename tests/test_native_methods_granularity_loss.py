"""Exercise the migrated native shell runners without loading models or using GPUs."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "methods": "methods/run_method_comparison.sh",
    "math": "granularity/run_math_ffn_granularity.sh",
    "intent": "granularity/run_intent_ffn_granularity.sh",
    "loss": "loss/run_auxiliary_loss.sh",
}
CONFIGS = {
    "qwen3": dict(hidden_size=2048, moe_intermediate_size=768, num_hidden_layers=48,
                  num_experts=128, num_experts_per_tok=8, num_attention_heads=32,
                  num_key_value_heads=4, head_dim=128),
    "gemma4": dict(hidden_size=2816, moe_intermediate_size=704, num_hidden_layers=30,
                   num_experts=128, top_k_experts=8, num_attention_heads=16,
                   num_key_value_heads=8, head_dim=256, global_head_dim=512,
                   num_global_key_value_heads=2, attention_k_eq_v=True,
                   layer_types=["sliding_attention"] * 5 + ["full_attention"]),
    "gptoss": dict(hidden_size=2880, intermediate_size=2880, num_hidden_layers=24,
                   num_local_experts=32, num_experts_per_tok=4, num_attention_heads=64,
                   num_key_value_heads=8, head_dim=64),
}


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native runners ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = {}
        for model, config in CONFIGS.items():
            config = dict(config)
            if model == "gemma4":
                config["layer_types"] = config["layer_types"] * 5
                config = {"text_config": config}
            directory = self.root / model
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(config))
            self.models[model] = directory
        self.bin = self.root / "bin"
        self.bin.mkdir()
        fake = f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
with open(os.environ["CALL_LOG"], "a") as handle:
    handle.write(json.dumps({"tool": pathlib.Path(sys.argv[0]).name, "args": sys.argv[1:],
                             "data": os.environ.get("MOEFACTORY_DATA_ROOT"),
                             "pythonpath": os.environ.get("PYTHONPATH")}) + "\\n")
raise SystemExit(int(os.environ.get("FAKE_EXIT", "0")))
'''
        for name in ("llamafactory-cli", "torchrun"):
            path = self.bin / name
            path.write_text(fake)
            path.chmod(0o755)

    def run_script(self, family, model="qwen3", dry=True, **overrides):
        env = os.environ.copy()
        env.update(REPO=str(REPO), MODEL=model, MODEL_PATH=str(self.models[model]),
                   MOEFACTORY_DATA_ROOT=str(self.root / "data with spaces"),
                   OUT_ROOT=str(self.root / "saves"), RES_ROOT=str(self.root / "results"),
                   DAS_JSON=str(self.root / "das.json"), PYTHON_BIN=sys.executable,
                   RUNNER_DRY_RUN=str(int(dry)), PHASE="train", TASK="math", TASKS="math",
                   LR_TRIALS="lr1e-5", METHODS="weight_lora", ARMS="expert_weight_r1",
                   COEF_TRIALS="aux1e-3", CALL_LOG=str(self.root / "calls.jsonl"),
                   PATH=str(self.bin) + os.pathsep + env.get("PATH", ""))
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(["bash", str(REPO / "scripts/exp" / SCRIPTS[family])],
                              env=env, cwd=self.root, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)

    def commands(self, result):
        assert result.returncode == 0, result.stdout
        return [shlex.split(line.lstrip()[1:]) for line in result.stdout.splitlines()
                if line.lstrip().startswith("+")]

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_all_model_task_batches_and_native_flags(self):
        for family in SCRIPTS:
            for model in CONFIGS:
                tasks = ("math", "intent") if family in ("methods", "loss") else (family,)
                for task in tasks:
                    with self.subTest(family=family, model=model, task=task):
                        command = self.commands(self.run_script(family, model, TASK=task, TASKS=task))[0]
                        assert "model_name_or_path=" + str(self.models[model]) in command
                        assert "dataset_dir=" + str(self.root / "data with spaces") in command
                        assert "moe_lora_attach_attn=false" in command
                        assert "moe_lora_attach_router=true" in command
                        assert "moe_lora_router_rank=8" in command
                        assert "moe_lora_rank=1" in command
                        assert "per_device_train_batch_size=" + ("4" if task == "intent" and model == "gemma4" else "6") in command
                        assert "gradient_accumulation_steps=" + ("9" if task == "intent" and model == "gemma4" else "6") in command
                        assert "dataset=" + ("esft_intent" if task == "intent" else "metamathqa_r1") in command
        assert not (self.root / "saves").exists()
        assert not (self.root / "results").exists()
        assert not self.calls()

    def test_method_variants_and_das_general_data(self):
        result = self.run_script("methods", METHODS="weight_dora partial_last expert_moe2_lora das_lora")
        commands = self.commands(result)
        assert len(commands) == 5  # four trainings plus DAS scoring
        assert "weight_peft_rank=1" in commands[0]
        assert "weight_peft_attach_attn=false" in commands[0]
        assert "partial_ft_modules=expert,router" in commands[1]
        assert "moe_lora_routing_mode=learned" in commands[2]
        das = commands[3]
        assert str(self.root / "data with spaces/general/commonsense_test/merged.json") in das
        assert "--general_split" in das
        assert "--max_samples" in das
        assert "das_lora_selected_experts_path=" + str(self.root / "das.json") in commands[4]
        (self.root / "das.json").write_text("{}")
        cached = self.commands(self.run_script("methods", METHODS="das_lora"))
        assert len(cached) == 1

    def test_native_arm_lr_loops_and_loss_schedule(self):
        for family in ("math", "intent"):
            commands = self.commands(self.run_script(family, ARMS="layer_bypass_r64 expert_bypass_r2 expert_weight_r1 expert_weight_r2", LR_TRIALS="lr1e-5 lr1e-4"))
            assert len(commands) == 8
            assert "moe_lora_rank=64" in commands[0]
            assert "moe_lora_routing_mode=independent" in commands[0]
            assert "moe_lora_rank=2" in commands[1]
            assert "learning_rate=1.0e-4" in commands[4]
        commands = self.commands(self.run_script("loss", COEF_TRIALS="aux0 aux1e-3", LR_TRIALS="lr1e-5 lr1e-4"))
        assert len(commands) == 4
        assert "moe_lora_balance_loss_coef=0.0" in commands[0]
        assert "moe_lora_balance_loss_schedule=constant" in commands[0]
        assert "moe_lora_balance_loss_coef=1.0e-3" in commands[1]

    def test_fake_execution_and_failure_exit(self):
        for family in SCRIPTS:
            with self.subTest(family=family):
                result = self.run_script(family, dry=False)
                assert result.returncode == 0, result.stdout
                call = self.calls()[-1]
                assert call["tool"] == "llamafactory-cli"
                assert call["data"] == str(self.root / "data with spaces")
                assert str(REPO / "src") in call["pythonpath"]
                result = self.run_script(family, dry=False, FAKE_EXIT=7)
                assert result.returncode != 0, result.stdout
                assert "[FAIL]" in result.stdout

    def test_completed_adapters_and_results_skip(self):
        for family in SCRIPTS:
            preview = self.commands(self.run_script(family, PHASE="full", SMOKE_EVAL_LIMIT=1))
            output = next(arg.split("=", 1)[1] for arg in preview[0] if arg.startswith("output_dir="))
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "moe_lora_state.safetensors").touch()
            (output / "trainer_state.json").write_text("{}")
            (output / "train_results.json").write_text("{}")
            evaluation = preview[1]
            if family == "intent":
                directory = Path(evaluation[evaluation.index("--output-dir") + 1])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "intent_predictions.jsonl").write_text("{}\n")
                (directory / "intent_metrics.json").write_text("{}")
            else:
                result_file = Path(evaluation[evaluation.index("--save_path") + 1])
                result_file.parent.mkdir(parents=True, exist_ok=True)
                result_file.write_text('{"idx": 0}\n')
            result = self.run_script(family, dry=False, PHASE="full", SMOKE_EVAL_LIMIT=1)
            assert result.returncode == 0, result.stdout
            assert "training skipped" in result.stdout
            assert "evaluation skipped" in result.stdout
        assert not self.calls()

    def test_eval_cli_failure_reaches_exit_status(self):
        for family in SCRIPTS:
            preview = self.commands(self.run_script(family))
            output = Path(next(arg.split("=", 1)[1] for arg in preview[0] if arg.startswith("output_dir=")))
            output.mkdir(parents=True, exist_ok=True)
            (output / "moe_lora_state.safetensors").touch()
            (output / "trainer_state.json").write_text("{}")
            (output / "train_results.json").write_text("{}")
            result = self.run_script(family, dry=False, PHASE="eval", SKIP_EXISTING=0, FAKE_EXIT=9)
            assert result.returncode != 0, result.stdout
            assert self.calls()[-1]["tool"] == "torchrun"

    def test_model_overrides_and_intent_eval_arguments(self):
        for model, variable in (("qwen3", "QWEN3_BASE"), ("gemma4", "GEMMA4_BASE"),
                                ("gptoss", "GPTOSS_BASE")):
            for family in SCRIPTS:
                command = self.commands(self.run_script(
                    family, model, MODEL_PATH="", **{variable: self.models[model]}))[0]
                assert "model_name_or_path=" + str(self.models[model]) in command
            for family in ("methods", "intent", "loss"):
                commands = self.commands(self.run_script(
                    family, model, TASK="intent", TASKS="intent", PHASE="full"))
                evaluation = commands[-1]
                assert "--prompt-protocol" in evaluation
                assert "--runtime-merge" in evaluation
                assert "--max-new-tokens" in evaluation
                assert "--hf-generate-batch-size" in evaluation
                if model == "gptoss":
                    assert "--flash-attn" in evaluation
                output = Path(next(arg.split("=", 1)[1] for arg in commands[0]
                                   if arg.startswith("output_dir=")))
                output.mkdir(parents=True, exist_ok=True)
                (output / "moe_lora_state.safetensors").touch()
                result = self.run_script(family, model, dry=False, TASK="intent",
                                         TASKS="intent", PHASE="eval", SKIP_EXISTING=0)
                assert result.returncode == 0, result.stdout
                assert self.calls()[-1]["tool"] == "torchrun"

    def test_aux_zero_reuse_requires_matching_seed_and_adapter(self):
        repo = self.root / "checkout"
        repo.mkdir()
        for name in ("scripts", "examples", "src"):
            (repo / name).symlink_to(REPO / name, target_is_directory=True)
        tag = "expert_weight_r1_lr1e-5_ep1"
        historic = repo / "results/ffn_granularity_math_noattn/qwen3" / tag
        historic.mkdir(parents=True)
        (historic / "math500.jsonl").write_text('{"idx": 0}\n')
        options = dict(REPO=repo, COEF_TRIALS="aux0", SMOKE_EVAL_LIMIT=1, UNIFIED_LAYOUT=1)
        # A result without its corresponding adapter cannot suppress training.
        assert len(self.commands(self.run_script("loss", **options))) == 1
        adapter = repo / "saves/ffn_granularity_math_noattn/qwen3" / tag
        adapter.mkdir(parents=True)
        (adapter / "moe_lora_state.safetensors").touch()
        for seed, data_seed in ((0, 0), (1, 1), (42, 0)):
            commands = self.commands(self.run_script(
                "loss", SEED=seed, DATA_SEED=data_seed, **options))
            assert len(commands) == 1
        result = self.run_script("loss", dry=False, **options)
        assert result.returncode == 0, result.stdout
        assert "reused noattn granularity baseline" in result.stdout
        assert not self.calls()
        assert any(path.is_symlink() for path in (self.root / "saves").rglob("*"))

    def test_failed_training_never_starts_evaluation(self):
        for family in SCRIPTS:
            before = len(self.calls())
            result = self.run_script(family, dry=False, PHASE="full", FAKE_EXIT=7)
            assert result.returncode != 0, result.stdout
            calls = self.calls()[before:]
            assert len(calls) == 1
            assert calls[0]["tool"] == "llamafactory-cli"


if __name__ == "__main__":
    unittest.main()
