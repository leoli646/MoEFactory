"""Offline shell contract tests: fake tools only, no model/data/GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "run_moe2_official_pilot.sh", "run_moe2_redesign_local4.sh",
    "run_moe2_alpha_ntk_local4.sh", "local_cuda_env.sh",
)


class NativeMoE2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native-moe2-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        scripts = self.repo / "scripts/exp/moe2"
        scripts.mkdir(parents=True)
        for name in SCRIPTS:
            shutil.copyfile(REPO / "scripts/exp/moe2" / name, scripts / name)
        upstream = self.repo / "scripts/reproduce"
        upstream.mkdir(parents=True)
        shutil.copyfile(REPO / "scripts/reproduce/run_upstream_eval.sh", upstream / "run_upstream_eval.sh")
        for family in ("peft_compare", "peft_compare_gemma4", "peft_compare_gptoss"):
            cfg = self.repo / "examples" / family / "expert_moe2_lora.yaml"
            cfg.parent.mkdir(parents=True)
            cfg.write_text("{}")
        model = self.repo / "models/Qwen3-30B-A3B-Base"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (self.repo / "data").mkdir()
        (self.repo / "data/dataset_info.json").write_text("{}")
        self.bin = self.repo / "bin"
        self.bin.mkdir()
        self.log = self.repo / "commands.jsonl"
        fake = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['FAKE_LOG'], 'a') as f:
    f.write(json.dumps([name] + args) + '\\n')
if name == 'fake-python' and args[:1] == ['-c']:
    print('12.8')
if name == 'llamafactory-cli':
    if os.environ.get('FAIL_TRAIN') == '1': sys.exit(7)
    output = pathlib.Path(next(x.split('=', 1)[1] for x in args if x.startswith('output_dir=')))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'moe_lora_state.safetensors').write_bytes(b'fake')
    (output / 'moe_lora_config.json').write_text('{}')
if os.environ.get('FAIL_UPSTREAM') == '1' and any('eval_vllm' in x for x in args):
    sys.exit(9)
'''
        for tool in ("fake-python", "llamafactory-cli", "torchrun"):
            path = self.bin / tool
            path.write_text(fake)
            path.chmod(0o755)
        nvcc = self.bin / "nvcc"
        nvcc.write_text('#!/bin/sh\necho "Cuda compilation tools, release 12.8"\n')
        nvcc.chmod(0o755)
        self.env = dict(os.environ, REPO=str(self.repo), PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        PYTHON_BIN=str(self.bin / "fake-python"), FAKE_LOG=str(self.log),
                        CUDA_HOME=str(self.repo), SETTINGS="n128_r32_wp_up_ntk_a11_bz", LR_TRIALS="lr1e-3")
        for name in ("MODEL_PATH", "BASE_MODEL", "QWEN3_BASE", "MOEFACTORY_DATA_ROOT", "DRY_RUN",
                     "RUNNER_DRY_RUN", "SKIP_TRAIN", "SKIP_UPSTREAM", "SMOKE", "PHASE", "PHASES"):
            self.env.pop(name, None)

    def run_script(self, script, **env):
        folder = "scripts/reproduce" if script == "run_upstream_eval.sh" else "scripts/exp/moe2"
        return subprocess.run(["bash", str(self.repo / folder / script)],
                              env=dict(self.env, **env), text=True, capture_output=True, timeout=30)

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_previews_need_no_python_models_cli_or_gpu(self):
        shutil.rmtree(self.repo / "models")
        for script, overrides in (
            ("run_moe2_official_pilot.sh", {"RUNNER_DRY_RUN": "1"}),
            ("run_moe2_redesign_local4.sh", {"DRY_RUN": "1"}),
            ("run_moe2_alpha_ntk_local4.sh", {"DRY_RUN": "1"}),
        ):
            result = self.run_script(script, PYTHON_BIN="/does/not/exist", **overrides)
            self.assert_success(result)
            self.assertIn("llamafactory-cli train", result.stdout)
            self.assertIn("eval_esft.py", result.stdout)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.repo / "saves").exists())
        self.assertFalse((self.repo / "results").exists())

    def test_setting_suffixes_and_model_overrides(self):
        for model, override, value in (("qwen3", "QWEN3_BASE", "/public/qwen"),
                                       ("gemma4", "GEMMA4_BASE", "/public/gemma"),
                                       ("gptoss", "GPTOSS_BASE", "/public/gpt")):
            result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="1", MODEL=model,
                                     ATTN_RANK="8", **{override: value})
            self.assert_success(result)
            for flag in ("moe_lora_ffn_granularity=weight_pool", "moe_lora_wp_b_init=zero",
                         "moe_lora_normalize_topk_prob=true", "moe_lora_alpha=11",
                         "moe_lora_wp_gate_up_stage=up", "moe_lora_attach_attn=true",
                         "model_name_or_path=" + value):
                self.assertIn(flag, result.stdout)
        for setting, expected in (("n128_r32_rt8_wp", "moe_lora_router_rank=8"),
                                   ("n128_r32_hb64_wla_tk4_wp", "moe_lora_wl_init=anchor"),
                                   ("n128_r32_g4_wp", "moe_lora_n_groups=4")):
            result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="1", SETTINGS=setting)
            self.assert_success(result)
            self.assertIn(expected, result.stdout)

    def test_official_train_eval_skip_and_missing_adapter(self):
        result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="0")
        self.assert_success(result)
        self.assertTrue(any(row[0] == "llamafactory-cli" for row in self.commands()))
        self.assertTrue(any(row[0] == "torchrun" for row in self.commands()))
        output = self.repo / "results/moe2_official_lr/qwen3/lr1e-3/intent/moe2official_n128_r32_wp_up_ntk_a11_bz"
        output.mkdir(parents=True, exist_ok=True)
        (output / "intent_predictions.jsonl").write_text('{}\n' * 500)
        (output / "intent_metrics.json").write_text('{}')
        self.log.unlink()
        result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="0")
        self.assert_success(result)
        self.assertFalse(any(row[0] in ("llamafactory-cli", "torchrun") for row in self.commands()))
        shutil.rmtree(self.repo / "saves")
        result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="0", PHASE="eval")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("adapter is missing", result.stdout + result.stderr)

    def test_p0_trains_new_single_factors_and_p1_combination(self):
        result = self.run_script("run_moe2_redesign_local4.sh", DRY_RUN="0", PHASES="P0 P1")
        self.assert_success(result)
        rows = self.commands()
        train = [row for row in rows if row[0] == "llamafactory-cli"]
        self.assertEqual(len(train), 7)
        self.assertTrue(all("dataset=esft_intent" in row for row in train))
        self.assertEqual(sum(any("eval_esft.py" in x for x in row) for row in rows), 7)
        self.assertEqual(sum("--merge_shards" in row for row in rows), 14)
        self.assertTrue(any("moe_lora_wp_b_init=zero" in row and "moe_lora_wp_gate_up_stage=up" in row for row in train))

    def test_training_failure_does_not_evaluate_and_is_nonzero(self):
        result = self.run_script("run_moe2_redesign_local4.sh", DRY_RUN="0", PHASES="P0",
                                 P0_ARMS="wp-a11|n128_r32_wp_a11", FAIL_TRAIN="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == "torchrun" or any("eval_vllm" in x for x in row) for row in self.commands()))

    def test_alpha_native_arms_all_models_and_smoke_isolation(self):
        for model, arm in (("qwen3", "n128_r32_wp_ntk_a96"),
                           ("gemma4", "n128_r32_wp_ntk_a128"),
                           ("gptoss", "n128_r32_wp_ntk_a128")):
            result = self.run_script("run_moe2_alpha_ntk_local4.sh", DRY_RUN="1", MODEL=model,
                                     SMOKE="1", PYTHON_BIN="/does/not/exist")
            self.assert_success(result)
            self.assertIn(arm, result.stdout)
            self.assertIn("moe2_alpha_ntk_grid_smoke", result.stdout)
            self.assertIn("max_steps=3", result.stdout)
            self.assertIn("--num_shards 4", result.stdout)
        self.assertFalse(self.log.exists())

    def test_upstream_completed_summary_skips_tools_and_unknown_task_fails(self):
        out = self.repo / "upstream/mmlu_pro"
        out.mkdir(parents=True)
        (out / "summary.json").write_text('{"accuracy": 0.5}')
        result = self.run_script("run_upstream_eval.sh", MODEL="qwen3", ENGINE="hf",
                                 OUT_DIR=str(out.parent), TASKS="mmlu_pro")
        self.assert_success(result)
        self.assertFalse(self.log.exists())
        result = self.run_script("run_upstream_eval.sh", DRY_RUN="1", MODEL="qwen3", ENGINE="hf",
                                 OUT_DIR=str(out.parent), TASKS="invalid")
        self.assertNotEqual(result.returncode, 0)

    def test_upstream_shard_failure_prevents_merge(self):
        result = self.run_script("run_upstream_eval.sh", DRY_RUN="0", MODEL="qwen3", ENGINE="hf",
                                 OUT_DIR=str(self.repo / "upstream"), TASKS="mmlu_pro", FAIL_UPSTREAM="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("--merge_shards" in row for row in self.commands()))

    def test_missing_checkpoint_and_unknown_inputs_fail(self):
        result = self.run_script("run_moe2_alpha_ntk_local4.sh", DRY_RUN="0", SKIP_TRAIN="1")
        self.assertNotEqual(result.returncode, 0)
        for extra in ({"SETTINGS": "n127_r32_g4_wp"}, {"SETTINGS": "n128_r32_up"}, {"MODEL": "unknown"}):
            result = self.run_script("run_moe2_official_pilot.sh", RUNNER_DRY_RUN="1", **extra)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
