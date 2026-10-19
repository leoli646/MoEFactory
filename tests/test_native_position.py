"""Check native launchers without loading models, GPUs, or datasets."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts/exp/position'
ARMS = 'attn_r64 dense_r256 attn64_dense256 attn32_dense128'


class NativePositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native position ')
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.capture = self.tmp / 'argv.jsonl'
        self.bin = self.tmp / 'bin'
        self.bin.mkdir()
        fake = '#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
with open(os.environ['CAPTURE'], 'a') as f:
    f.write(json.dumps(sys.argv) + '\\n')
if os.environ.get('FAKE_FAIL') == '1':
    raise SystemExit(7)
for arg in sys.argv[1:]:
    if arg.startswith('output_dir='):
        out = Path(arg.split('=', 1)[1]); out.mkdir(parents=True, exist_ok=True)
        for name in ('moe_lora_state.safetensors', 'trainer_state.json', 'train_results.json'):
            (out / name).touch()
'''
        for name in ('python', 'torchrun', 'llamafactory-cli'):
            path = self.bin / name
            path.write_text(fake)
            path.chmod(0o755)
        self.model = self.tmp / "model weights 'quoted'"
        self.model.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        CAPTURE=str(self.capture), QWEN_BASE_OUT_ROOT=str(self.tmp / 'qwen base adapters'),
                        QWEN_BASE_RES_ROOT=str(self.tmp / 'qwen base results'),
                        OUT_ROOT=str(self.tmp / 'saved adapters'),
                        RES_ROOT=str(self.tmp / 'eval results'), MODEL_PATH=str(self.model),
                        MOEFACTORY_DATA_ROOT=str(self.tmp / "data 'quoted'"), GPUS='0',
                        SKIP_EXISTING='0', ARMS=ARMS)
        for name in ('RUNNER_DRY_RUN', 'DRY_RUN', 'PYTHON_BIN', 'REPO'):
            self.env.pop(name, None)

    def run_script(self, script, *args, **env):
        result = subprocess.run(['bash', str(SCRIPTS / script), *args],
                                env=dict(self.env, **env), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def commands(self, output):
        return [shlex.split(line.lstrip()[1:]) for line in output.splitlines()
                if line.lstrip().startswith('+ ')]

    def test_shell_syntax_and_portability(self):
        files = list(SCRIPTS.glob('*.sh'))
        self.assertEqual(len(files), 6)
        for path in files:
            subprocess.run(['bash', '-n', str(path)], check=True)
            text = path.read_text()
            for forbidden in ('/mnt/', 'conda activate', 'DISABLE_VERSION_CHECK', '/snapshots/', 'chown -R'):
                self.assertNotIn(forbidden, text, path.name)

    def test_all_entry_dry_runs_execute_no_tools(self):
        for script in SCRIPTS.glob('run*.sh'):
            self.run_script(script.name, RUNNER_DRY_RUN='1', RUN_QWEN3_BASE='1',
                            RUN_QWEN3_ROUTER='1', RUN_NEW_MODELS='1')
        self.assertFalse(self.capture.exists())

    def test_qwen_native_batches_and_quoted_train_eval_paths(self):
        output = self.run_script('run_budget_align.sh', RUNNER_DRY_RUN='1',
                                 ATTACH_ROUTER='true', EVALS='math500')
        trains = [c for c in self.commands(output) if 'train' in c]
        self.assertEqual(len(trains), 4)
        self.assertFalse(any(x.startswith('per_device_train_batch_size=') for x in trains[0]))
        baseline = (ROOT / 'examples/train_moe_lora_qwen3/baseline1_qwen3_math_base_run.yaml').read_text()
        self.assertIn('per_device_train_batch_size: 12', baseline)
        self.assertIn('gradient_accumulation_steps: 3', baseline)
        for cmd in trains[1:]:
            self.assertIn('per_device_train_batch_size=6', cmd)
            self.assertIn('gradient_accumulation_steps=6', cmd)
            self.assertIn('moe_lora_attach_router=true', cmd)
        # Real shell dispatch to fake tools catches quoting errors hidden by previews.
        self.run_script('run_budget_align.sh', DOMAIN='intent', ARMS='attn32_dense128',
                        ATTACH_ROUTER='true', PYTHON_BIN=str(self.bin / 'python'))
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        train = next(c for c in calls if 'train' in c)
        evaluation = next(c for c in calls if '--task' in c)
        self.assertIn('model_name_or_path=' + str(self.model), train)
        self.assertIn('dataset_dir=' + self.env['MOEFACTORY_DATA_ROOT'], train)
        self.assertEqual(evaluation[evaluation.index('--model-name-or-path') + 1], str(self.model))
        self.assertEqual(evaluation[evaluation.index('--esft-dir') + 1], self.env['MOEFACTORY_DATA_ROOT'] + '/esft_eval')

    def test_newmodel_module_branch_ranks_and_intent_batches(self):
        math = self.commands(self.run_script('run_budget_align_position_nm_math.sh',
                                            RUNNER_DRY_RUN='1', ARMS='attn32_dense128'))
        intent = self.commands(self.run_script('run_budget_align_nm_intent.sh',
                                              RUNNER_DRY_RUN='1', ARMS='attn32_dense128'))
        math_trains = [c for c in math if 'train' in c]
        intent_trains = [c for c in intent if 'train' in c]
        self.assertEqual(len(math_trains), 4)  # two models, base and router
        self.assertEqual(len(intent_trains), 2)
        for cmd, rank in zip(math_trains, (128, 128, 128, 128)):
            self.assertIn('moe_lora_rank=' + str(rank), cmd)
            self.assertIn('model_name_or_path=' + str(self.model), cmd)
        for cmd, rank, pbs, gas in zip(intent_trains, (128, 128), (4, 6), (9, 6)):
            self.assertIn('moe_lora_rank=' + str(rank), cmd)
            self.assertIn('per_device_train_batch_size=' + str(pbs), cmd)
            self.assertIn('gradient_accumulation_steps=' + str(gas), cmd)
        for cmd in intent:
            if '--task' in cmd:
                self.assertIn('--esft-dir', cmd)
        self.assertFalse(self.capture.exists())

    def test_newmodel_real_dispatch_uses_only_fake_tools(self):
        for script in ('run_budget_align_nm_intent.sh', 'run_budget_align_position_nm_math.sh'):
            self.run_script(script, ARMS='attn32_dense128', MODELS='gemma4', VARIANTS='router',
                            RUN_ZERO_SHOT='0', ATTACH_ROUTER='true')
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        trains = [c for c in calls if 'train' in c]
        self.assertEqual(len(trains), 2)
        for train in trains:
            self.assertIn('model_name_or_path=' + str(self.model), train)
            self.assertIn('dataset_dir=' + self.env['MOEFACTORY_DATA_ROOT'], train)
            self.assertIn('moe_lora_attach_router=true', train)
        self.assertIn('moe_lora_rank=128', trains[0])
        self.assertIn('moe_lora_rank=128', trains[1])

    def test_qwen_sweep_seed_layout_and_success_status(self):
        self.run_script('run_budget_align_lr_sweep.sh', ARMS='attn_r64',
                        MODEL_SUBDIR='qwen3/seed42', LR_TRIALS='lr1e-5', DOMAIN='intent')
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        train = next(c for c in calls if 'train' in c)
        self.assertIn('output_dir=' + self.env['OUT_ROOT'] + '/qwen3/seed42/attn_r64_lr1e-5_ep3', train)
        self.assertIn('learning_rate=1.0e-5', train)
        self.run_script('run_budget_align_position_math.sh', ARMS='attn_r64',
                        LR_TRIALS='lr1e-4', RUN_QWEN3_BASE='1')

    def test_position_entries_propagate_tool_failure(self):
        for script in SCRIPTS.glob('run*.sh'):
            result = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                                    env=dict(self.env, ARMS='attn_r64', MODELS='gemma4',
                                             LR_TRIALS='lr1e-5', FAKE_FAIL='1'))
            self.assertNotEqual(result.returncode, 0, script.name + result.stdout + result.stderr)

    def test_common_helpers_quoted_paths_and_dry_run(self):
        command = 'source "$1"; model_meta gemma4; task_meta intent; do_eval_intent "$2" "$3"'
        result = subprocess.run(['bash', '-c', command, 'bash', str(SCRIPTS / 'common_nm.sh'),
                                 str(self.tmp / "adapter 'one'"), str(self.tmp / 'common results')],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call = json.loads(self.capture.read_text().splitlines()[0])
        self.assertIn(str(self.tmp / "adapter 'one'"), call)
        self.assertIn(self.env['MOEFACTORY_DATA_ROOT'] + '/esft_eval', call)


if __name__ == '__main__':
    unittest.main()
