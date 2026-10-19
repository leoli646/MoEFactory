"""Check release configuration and document links without model downloads."""
import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_training_examples_have_registered_data_and_deepspeed_configs(self):
        registry = json.loads((ROOT / "data/dataset_info.json").read_text())
        training_count = 0
        for path in (ROOT / "examples").rglob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict):
                continue
            with self.subTest(config=str(path.relative_to(ROOT))):
                datasets = config.get("dataset")
                if datasets:
                    training_count += 1
                    for name in datasets.split(",") if isinstance(datasets, str) else datasets:
                        self.assertIn(name.strip(), registry)
                deepspeed = config.get("deepspeed")
                if isinstance(deepspeed, str):
                    self.assertTrue((ROOT / deepspeed).is_file(), deepspeed)
                    json.loads((ROOT / deepspeed).read_text())
        self.assertGreaterEqual(training_count, 34)

    def test_release_keeps_experiments_and_loss_plotting_only(self):
        for relative in ("scripts/analysis", "scripts/plot", "docs/geometry.md",
                         "src/llamafactory/train/moe_probe_stats.py",
                         "src/llamafactory/train/router_stats.py",
                         "src/llamafactory/model/model_utils/mola.py",
                         "eval_scripts/eval_gsm8k.py", "eval_scripts/eval_aime.py"):
            self.assertFalse((ROOT / relative).exists(), relative)
        for family in ("position", "granularity", "methods", "loss", "moe2"):
            self.assertTrue((ROOT / "scripts/exp" / family).is_dir(), family)
        self.assertTrue((ROOT / "src/llamafactory/extras/ploting.py").is_file())
        workflow = (ROOT / "src/llamafactory/train/sft/workflow.py").read_text()
        self.assertIn("finetuning_args.plot_loss", workflow)
        self.assertIn("plot_loss(training_args.output_dir", workflow)
        methods = (ROOT / "scripts/exp/methods/run_method_comparison.sh").read_text()
        default = re.search(r"DEFAULT_METHODS=\((.*?)\)", methods, re.S).group(1).split()
        self.assertEqual(set(default), {"weight_lora", "weight_dora", "weight_oft",
                         "weight_fourier", "weight_ia3", "expert_lora", "expert_moe_lora_n8",
                         "layer_lora", "das_lora", "partial_last"})
        for variant in ("nora", "nora_init", "oft_v2", "oft_cayley_neumann"):
            self.assertFalse(any((ROOT / "examples").rglob(f"weight_{variant}.yaml")))

    def test_registry_contains_only_portable_relative_data_locations(self):
        registry = json.loads((ROOT / "data/dataset_info.json").read_text())
        for name, config in registry.items():
            with self.subTest(dataset=name):
                path = Path(config["file_name"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertEqual(config["formatting"], "sharegpt")
        self.assertIn("!data/dataset_info.json", (ROOT / ".gitignore").read_text().splitlines())

    def test_readme_local_links_resolve(self):
        for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
