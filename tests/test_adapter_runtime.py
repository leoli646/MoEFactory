"""Bounded CPU checks for the exported adapter implementations.

These tests call the real source modules with tiny tensors and intentionally
avoid model downloads, GPU kernels, distributed training, and simulated E2E claims.
"""

from __future__ import annotations

import json
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import safetensors.torch
import torch
import torch.nn as nn

from llamafactory.model.model_utils.das_lora import DASLoRAPool
from llamafactory.model.model_utils.moe_lora import LoRAPool
from llamafactory.model.model_utils.partial_ft import (
    _select_layer_ids,
    load_partial_ft_state,
    save_partial_ft_state,
)
from llamafactory.model.model_utils.weight_peft import WeightLoRAPool


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _set_scalar_lora(expert, value: float = 1.0) -> None:
    with torch.no_grad():
        expert.A.weight.fill_(value)
        expert.B.weight.fill_(value)


def test_moe_lora_selected_topk_normalization_is_opt_in():
    """Regression adapted from lora_moe-main test_lora_pool_fast_grouped.py."""
    legacy = LoRAPool(3, 1, 1, 1, 2)
    normalized = LoRAPool(3, 1, 1, 1, 2, normalize_topk_prob=True)
    for pool in (legacy, normalized):
        pool._use_grouped_mm = False
        pool.eval()
        for expert in pool.experts:
            _set_scalar_lora(expert)

    h = torch.ones(1, 1)
    probabilities = torch.tensor([[0.6, 0.3, 0.1]])
    torch.testing.assert_close(legacy(h, probabilities), torch.tensor([[0.9]]))
    torch.testing.assert_close(normalized(h, probabilities), torch.ones(1, 1))


def test_weight_lora_pool_selected_topk_normalization():
    gate_up = torch.zeros(3, 4, 2)
    down = torch.zeros(3, 2, 2)
    pool = WeightLoRAPool(
        gate_up, down, n_experts=3, rank=1, alpha=1, top_k=2, normalize_topk_prob=True
    )
    values, indices = pool.select(torch.tensor([[0.6, 0.3, 0.1]]), torch.float32)
    torch.testing.assert_close(values, torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]))
    assert indices.tolist() == [[0, 1]]
    assert pool.activation_count.tolist() == [1, 1, 0]
    assert pool.total_tokens.item() == 1


def test_das_lora_parallel_branch_uses_only_selected_routed_experts():
    pool = DASLoRAPool(selected_expert_ids=[1, 3], d_model=1, rank=1, alpha=1)
    pool.eval()
    for expert in pool.experts:
        _set_scalar_lora(expert)

    hidden = torch.tensor([[2.0], [5.0], [7.0]])
    routed_ids = torch.tensor([[1, 0], [2, 3], [0, 2]])
    routed_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.5, 0.5]])
    output = pool(hidden, routed_ids, routed_weights)

    # The source implements a parallel branch: selected expert 1 contributes
    # 0.75 * h, selected expert 3 contributes 0.4 * h, unmatched routes add 0.
    torch.testing.assert_close(output, torch.tensor([[1.5], [2.0], [0.0]]))


def test_partial_ft_layer_selection_boundaries():
    assert _select_layer_ids(48, "first", "") == [0]
    assert _select_layer_ids(48, "middle", "") == [24]
    assert _select_layer_ids(48, "last", "") == [47]
    assert _select_layer_ids(48, "first_third", "") == list(range(16))
    assert _select_layer_ids(48, "middle_third", "") == list(range(16, 32))
    assert _select_layer_ids(48, "last_third", "") == list(range(32, 48))
    assert _select_layer_ids(8, "custom", "7, 0, 7") == [0, 7]


class _TinyPartialModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.selected = nn.Linear(2, 2, bias=False, dtype=dtype)
        self.frozen = nn.Linear(2, 2, bias=False, dtype=dtype)
        self.config = SimpleNamespace(name_or_path="public-test-model")


def test_partial_ft_checkpoint_contains_only_trainable_and_round_trips():
    source = _TinyPartialModel()
    source.selected.weight.requires_grad = True
    source.frozen.weight.requires_grad = False
    with torch.no_grad():
        source.selected.weight.copy_(torch.tensor([[1.25, -2.5], [3.75, 4.5]]))
        source.frozen.weight.fill_(99.0)

    args = SimpleNamespace(
        partial_ft_target="custom", partial_ft_layers="0", partial_ft_modules="expert"
    )
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        save_partial_ft_state(source, checkpoint_dir, args)
        state_path = os.path.join(checkpoint_dir, "partial_ft_state.safetensors")
        config_path = os.path.join(checkpoint_dir, "partial_ft_config.json")
        state = safetensors.torch.load_file(state_path)
        assert set(state) == {"selected.weight"}
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        assert config["partial_ft_target"] == "custom"
        assert config["_meta_base_model"] == "public-test-model"

        # The config is optional during load. Removing it tests the checkpoint
        # tensor boundary without rebuilding model-specific adapter structure.
        os.remove(config_path)
        restored = _TinyPartialModel(dtype=torch.float64)
        frozen_before = restored.frozen.weight.detach().clone()
        load_partial_ft_state(restored, checkpoint_dir)
        torch.testing.assert_close(
            restored.selected.weight, source.selected.weight.to(torch.float64)
        )
        torch.testing.assert_close(restored.frozen.weight, frozen_before)
        assert restored.selected.weight.dtype == torch.float64


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Tiny supported-name block used to exercise the real DAS setup path."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2, 4, bias=False)
        self.base = nn.Linear(2, 2, bias=False)

    def forward(self, hidden_states):
        return self.base(hidden_states)


class _TinyDASLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Qwen3MoeSparseMoeBlock()


class _TinyDASModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyDASLayer()])
        self.config = SimpleNamespace(
            hidden_size=2,
            num_experts_per_tok=2,
            name_or_path="public-das-test-model",
        )


def test_das_setup_router_trainability_and_checkpoint_round_trip():
    """Exercise adapter._setup_das_lora, not the lower-level injectors alone."""
    from llamafactory.hparams import FinetuningArguments
    from llamafactory.model.adapter import _setup_das_lora
    from llamafactory.model.model_utils.das_lora import (
        load_das_lora_state,
        save_das_lora_state,
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        selected_path = os.path.join(checkpoint_dir, "selected.json")
        with open(selected_path, "w", encoding="utf-8") as handle:
            json.dump({"selected_experts_per_layer": [[1, 3]]}, handle)
        args = FinetuningArguments(
            finetuning_type="das_lora",
            das_lora_selected_experts_path=selected_path,
            das_lora_rank=1,
            das_lora_alpha=1,
            moe_lora_attach_router=True,
            moe_lora_router_rank=1,
            moe_lora_router_alpha=1,
        )

        source = _setup_das_lora(
            _TinyDASModel(), args, is_trainable=True, cast_trainable_params_to_fp32=False
        )
        block = source.model.layers[0].mlp
        assert block.das_lora_pool.selected_ids == [1, 3]
        trainable_names = {name for name, param in source.named_parameters() if param.requires_grad}
        assert any("das_lora_pool" in name for name in trainable_names)
        assert any("gate.parametrizations.weight.0" in name for name in trainable_names)
        assert not block.base.weight.requires_grad

        with torch.no_grad():
            for index, (name, param) in enumerate(source.named_parameters(), start=1):
                if param.requires_grad:
                    param.fill_(index / 10.0)
        save_das_lora_state(source, checkpoint_dir, args)
        saved = safetensors.torch.load_file(
            os.path.join(checkpoint_dir, "das_lora_state.safetensors")
        )
        assert any("das_lora_pool" in name for name in saved)
        assert any("gate.parametrizations.weight.0" in name for name in saved)

        restored = load_das_lora_state(_TinyDASModel(), checkpoint_dir)
        restored_block = restored.model.layers[0].mlp
        assert restored_block.das_lora_pool.selected_ids == [1, 3]
        restored_state = restored.state_dict()
        for name, value in saved.items():
            torch.testing.assert_close(restored_state[name], value)


def test_public_dataset_registrations_resolve():
    from llamafactory.data.parser import get_dataset_list

    attrs = get_dataset_list(["metamathqa_r1", "esft_intent"], str(REPO_ROOT / "data"))
    assert [(item.dataset_name, item.load_from, item.formatting, item.messages) for item in attrs] == [
        ("metamathqa_r1_clean.json", "file", "sharegpt", "messages"),
        ("esft/intent_train.jsonl", "file", "sharegpt", "messages"),
    ]


def test_eval_common_loader_imports_from_public_export():
    common_path = REPO_ROOT / "eval_scripts" / "_common.py"
    module = _load_module("public_eval_common", common_path)
    assert Path(module.__file__).resolve() == common_path.resolve()
    assert callable(module.common_arg_parser)
    assert callable(module.load)


def _cpu_parse_overrides():
    from omegaconf import OmegaConf

    # Preserve and inspect the source values before adding these parse-only
    # overrides. They bypass hardware/distributed validation without training.
    return OmegaConf.from_cli(
        [
            "bf16=false",
            "use_cpu=true",
            "deepspeed=null",
            "do_train=false",
            "warmup_ratio=0.0",
        ]
    )


def test_paper_yamls_parse_through_real_hparams_dataclasses():
    from omegaconf import OmegaConf

    from llamafactory.hparams.parser import _parse_train_args

    paths = sorted((REPO_ROOT / "examples" / "paper").glob("weight_lora_*.yaml"))
    assert len(paths) == 6
    for path in paths:
        raw = OmegaConf.load(path)
        config = OmegaConf.to_container(raw, resolve=True)
        assert isinstance(config["learning_rate"], float) and config["learning_rate"] == 1e-5
        assert isinstance(config["seed"], int) and config["seed"] == 42
        assert isinstance(config["data_seed"], int) and config["data_seed"] == 42
        assert config["finetuning_type"] == "moe_lora"
        assert config["overwrite_output_dir"] is False

        merged = OmegaConf.to_container(
            OmegaConf.merge(raw, _cpu_parse_overrides()), resolve=True
        )
        model_args, data_args, training_args, finetuning_args, _ = _parse_train_args(merged)
        assert model_args.model_name_or_path == config["model_name_or_path"]
        assert data_args.dataset == [config["dataset"]]
        assert training_args.learning_rate == 1e-5
        assert training_args.seed == training_args.data_seed == 42
        assert finetuning_args.finetuning_type == "moe_lora"

    unknown = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(paths[0]), _cpu_parse_overrides()), resolve=True
    )
    unknown["source_protocol_snapshot"] = True
    try:
        _parse_train_args(unknown)
    except ValueError as error:
        assert "source_protocol_snapshot" in str(error)
    else:
        raise AssertionError("HfArgumentParser accepted unknown snapshot metadata")


def test_protocol_matrix_omegaconf_merge_and_real_dataclass_parse():
    from omegaconf import OmegaConf

    from llamafactory.hparams.parser import _parse_train_args

    protocol = _load_module(
        "public_reproduction_protocol", REPO_ROOT / "scripts" / "reproduce" / "protocol.py"
    )
    parsed_count = 0
    saw_save_strategy_no = False
    for model in protocol.MODELS:
        for family in protocol.FAMILY_SETTINGS:
            for setting in protocol.supported_settings(model, family):
                for task in ("math", "intent"):
                    run = protocol.RunSpec(
                        repo=REPO_ROOT,
                        model=model,
                        model_path=Path("models") / model,
                        data_root=Path("data"),
                        family=family,
                        setting=setting,
                        task=task,
                        lr="lr1e-5",
                        seed=42,
                        gpus="0,1,2,3",
                        output_root=Path("saves/source_protocol"),
                        result_root=Path("results/source_protocol"),
                        das_general=Path("data/general.jsonl"),
                    )
                    command = protocol.train_command(run)
                    train_index = command.index("train")
                    base_path = REPO_ROOT / command[train_index + 1]
                    overrides = command[train_index + 2 :]
                    raw = OmegaConf.merge(OmegaConf.load(base_path), OmegaConf.from_cli(overrides))
                    raw_config = OmegaConf.to_container(raw, resolve=True)
                    assert isinstance(raw_config["learning_rate"], float)
                    assert raw_config["learning_rate"] == 1e-5
                    assert raw_config["seed"] == raw_config["data_seed"] == 42
                    assert isinstance(raw_config["seed"], int)
                    assert raw_config["dataset"] in ("metamathqa_r1", "esft_intent")
                    assert raw_config["finetuning_type"] in (
                        "moe_lora", "weight_peft", "das_lora", "partial_ft"
                    )

                    merged = OmegaConf.to_container(
                        OmegaConf.merge(raw, _cpu_parse_overrides()), resolve=True
                    )
                    _, data_args, training_args, finetuning_args, _ = _parse_train_args(merged)
                    assert data_args.dataset == [raw_config["dataset"]]
                    assert training_args.learning_rate == 1e-5
                    assert training_args.seed == training_args.data_seed == 42
                    assert finetuning_args.finetuning_type == raw_config["finetuning_type"]
                    if raw_config.get("save_strategy") == "no":
                        saw_save_strategy_no = True
                        assert training_args.save_strategy.value == "no"
                    parsed_count += 1

    assert parsed_count == 194
    assert saw_save_strategy_no
