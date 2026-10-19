#!/usr/bin/env python3
"""Portable command construction for the MoEFactory source protocol.

This module deliberately mirrors the checked-in source runners.  It is not a
claim that every source setting matches a paper appendix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

SEEDS = (42, 0, 1)
LRS = ("lr1e-5", "lr1e-4", "lr1e-3")
LR_VALUES = {"lr1e-5": "1.0e-5", "lr1e-4": "1.0e-4", "lr1e-3": "1.0e-3"}

MODELS = {
    "qwen3": {"experts": 128, "top_k": 8, "template": "qwen3", "layers": 48},
    "gemma4": {"experts": 128, "top_k": 8, "template": "default", "layers": 30},
    "gptoss": {"experts": 32, "top_k": 4, "template": "default", "layers": 24},
}

FAMILY_SETTINGS = {
    "granularity": ("layer_bypass_r256", "expert_bypass_r2", "expert_weight_r1", "expert_weight_r2"),
    "position": ("attn_r64", "dense_r256",
                 "attn32_dense128", "attn64_dense256", "attn_r64_router",
                 "dense_r256_router",
                 "attn32_dense128_router", "attn64_dense256_router"),
    "methods": ("weight_lora", "expert_lora", "weight_dora", "weight_oft",
                "weight_fourier", "weight_ia3", "expert_moe_lora_n8", "layer_lora",
                "das_lora", "partial_last", "expert_moe2_lora",
                "moe2_pilot_default", "moe2_pilot_default_ntk",
                "moe2_pilot_n280_r32", "moe2_pilot_n512_r17", "moe2_pilot_n512_r16",
                "moe2_pilot_n256_r22", "moe2_pilot_n256_r1", "moe2_pilot_n10_r32",
                "moe2_sall_n128_r32_wp_up_ntk_a11_bz"),
    "loss": ("aux0", "aux1e-4", "aux1e-3", "aux1e-2", "aux1e-1"),
}

MODEL_SPECIFIC_SETTINGS = {
    "moe2_pilot_n280_r32": "qwen3", "moe2_pilot_n512_r17": "qwen3",
    "moe2_pilot_n512_r16": "qwen3", "moe2_pilot_n256_r22": "gemma4",
    "moe2_pilot_n256_r1": "gptoss", "moe2_pilot_n10_r32": "gptoss",
    "moe2_sall_n128_r32_wp_up_ntk_a11_bz": "qwen3",
}


def setting_supported(model: str, family: str, setting: str) -> bool:
    return (family in FAMILY_SETTINGS and setting in FAMILY_SETTINGS[family]
            and MODEL_SPECIFIC_SETTINGS.get(setting, model) == model)


def supported_settings(model: str, family: str, settings: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    requested = FAMILY_SETTINGS[family] if settings is None else settings
    return tuple(setting for setting in requested if setting_supported(model, family, setting))


@dataclass(frozen=True)
class RunSpec:
    repo: Path
    model: str
    model_path: Path
    data_root: Path
    family: str
    setting: str
    task: str
    lr: str
    seed: int
    gpus: str
    output_root: Path
    result_root: Path
    deepspeed: str = "examples/deepspeed/ds_z2_config.json"
    phase: str = "full"
    das_general: Path | None = None
    run_upstream: bool = False

    def validate(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model: {self.model}")
        if not setting_supported(self.model, self.family, self.setting):
            raise ValueError(f"unknown {self.family} setting: {self.setting}")
        if self.task not in ("math", "intent"):
            raise ValueError(f"unknown task: {self.task}")
        if self.lr not in LR_VALUES:
            raise ValueError(f"unknown learning-rate tag: {self.lr}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.phase not in ("train", "eval", "full"):
            raise ValueError(f"unknown phase: {self.phase}")
        if len(self.gpus.split(",")) != 4:
            raise ValueError("the source implementation protocol uses exactly four GPUs")

    @property
    def adapter(self) -> Path:
        return self.output_root / self.family / self.setting / self.task / self.model / f"seed{self.seed}" / self.lr

    @property
    def result(self) -> Path:
        return self.result_root / self.family / self.setting / self.task / self.model / f"seed{self.seed}" / self.lr


def shell_join(command: list[str]) -> str:
    return shlex.join(command)


def _base_config(spec: RunSpec) -> str:
    if spec.setting.startswith("moe2_") and spec.setting != "expert_moe2_lora":
        suffix = {"qwen3": "", "gemma4": "_gemma4", "gptoss": "_gptoss"}[spec.model]
        directory = "peft_compare" if not suffix else f"peft_compare{suffix}"
        return f"examples/{directory}/expert_moe2_lora.yaml"
    if spec.family == "methods" and spec.setting not in ("weight_lora", "expert_lora", "layer_lora"):
        directory = {"qwen3": "peft_compare", "gemma4": "peft_compare_gemma4",
                     "gptoss": "peft_compare_gptoss"}[spec.model]
        candidate = spec.repo / "examples" / directory / f"{spec.setting}.yaml"
        if candidate.exists():
            return str(candidate.relative_to(spec.repo))
    return {
        "qwen3": "examples/ffn_granularity/qwen3_math.yaml",
        "gemma4": "examples/train_moe_lora_newmodels/baseline1_gemma4_math.yaml",
        "gptoss": "examples/train_moe_lora_newmodels/baseline1_gptoss_math.yaml",
    }[spec.model]


def _task_flags(spec: RunSpec) -> list[str]:
    if spec.task == "math":
        flags = ["dataset=metamathqa_r1", f"template={MODELS[spec.model]['template']}", "cutoff_len=2048",
                 "max_samples=80000", "num_train_epochs=1", "per_device_train_batch_size=6",
                 "gradient_accumulation_steps=6"]
        if spec.model == "qwen3" and spec.family == "position" and spec.setting.startswith("attn_r64"):
            flags[-2:] = ["per_device_train_batch_size=12", "gradient_accumulation_steps=3"]
        return flags
    batch = ("4", "9") if spec.model == "gemma4" else ("6", "6")
    flags = ["dataset=esft_intent", "template=esft_raw", "cutoff_len=4096", "packing=false",
             "max_samples=80000", "num_train_epochs=3", f"per_device_train_batch_size={batch[0]}",
             f"gradient_accumulation_steps={batch[1]}"]
    if spec.model == "gptoss":
        flags.append("flash_attn=disabled")
    if spec.model == "qwen3" and spec.family == "position" and spec.setting.startswith("attn_r64"):
        batch_start = flags.index(next(item for item in flags if item.startswith("per_device_train_batch_size=")))
        flags[batch_start:batch_start + 2] = ["per_device_train_batch_size=12",
                                               "gradient_accumulation_steps=3"]
    return flags


def _attachment_flags(attention: bool, router: bool = True) -> list[str]:
    return ["moe_lora_attach_ffn=true", f"moe_lora_attach_attn={str(attention).lower()}",
            f"moe_lora_attach_router={str(router).lower()}", "moe_lora_attn_targets=auto",
            "moe_lora_attn_rank=32", "moe_lora_attn_alpha=64", "moe_lora_router_rank=8",
            "moe_lora_router_alpha=16", "moe_lora_target_layers=all"]


def _granularity_flags(spec: RunSpec) -> list[str]:
    meta = MODELS[spec.model]
    common = _attachment_flags(False) + ["moe_lora_pool_share=per_layer", "moe_lora_w_share=per_layer",
                                         "moe_lora_b_init=normal", "moe_lora_wp_b_init=normal",
                                         "moe_lora_wl_init=default", "moe_lora_balance_loss_coef=0.0"]
    if spec.setting.startswith("layer_bypass_r"):
        rank = int(spec.setting.rsplit("r", 1)[1])
        return ["finetuning_type=moe_lora", "moe_lora_ffn_granularity=bypass",
                "moe_lora_routing_mode=independent", "moe_lora_n_experts=1", "moe_lora_top_k=1",
                f"moe_lora_rank={rank}", f"moe_lora_alpha={2 * rank}"] + common
    granularity, rank = {"expert_bypass_r2": ("bypass", 2), "expert_weight_r1": ("expert_weight", 1),
                         "expert_weight_r2": ("expert_weight", 2)}[spec.setting]
    return ["finetuning_type=moe_lora", f"moe_lora_ffn_granularity={granularity}",
            "moe_lora_routing_mode=follow_moe", f"moe_lora_n_experts={meta['experts']}",
            f"moe_lora_top_k={meta['top_k']}", f"moe_lora_rank={rank}",
            f"moe_lora_alpha={2 * rank}"] + common


def _position_flags(spec: RunSpec) -> list[str]:
    setting = spec.setting
    router = setting.endswith("_router")
    base = setting.removesuffix("_router")
    flags = ["finetuning_type=moe_lora", f"moe_lora_attach_router={str(router).lower()}"]
    if router:
        flags += ["moe_lora_router_rank=8", "moe_lora_router_alpha=16"]
    mapping = {
        "attn_r64": ["moe_lora_attach_ffn=false", "moe_lora_attach_attn=true",
                       "moe_lora_attn_targets=auto", "moe_lora_attn_rank=64", "moe_lora_attn_alpha=128"],
        "dense_r256": ["moe_lora_attach_ffn=true", "moe_lora_attach_attn=false",
                         "moe_lora_routing_mode=independent", "moe_lora_n_experts=1", "moe_lora_top_k=1",
                         "moe_lora_rank=256", "moe_lora_alpha=512"],
        "attn64_dense256": ["moe_lora_attach_ffn=true", "moe_lora_attach_attn=true",
                              "moe_lora_attn_rank=64", "moe_lora_attn_alpha=128",
                              "moe_lora_routing_mode=independent", "moe_lora_n_experts=1",
                              "moe_lora_top_k=1", "moe_lora_rank=256", "moe_lora_alpha=512"],
        "attn32_dense128": ["moe_lora_attach_ffn=true", "moe_lora_attach_attn=true",
                              "moe_lora_attn_rank=32", "moe_lora_attn_alpha=64",
                              "moe_lora_routing_mode=independent", "moe_lora_n_experts=1",
                              "moe_lora_top_k=1", "moe_lora_rank=128", "moe_lora_alpha=256"],
    }
    explicit = ["moe_lora_pool_share=per_layer", "moe_lora_w_share=per_layer",
                "moe_lora_balance_loss_coef=0.0"]
    if base in ("dense_r256", "attn64_dense256", "attn32_dense128"):
        explicit.append("moe_lora_ffn_granularity=bypass")
    return flags + ["moe_lora_attn_targets=auto"] + explicit + mapping[base]


def _method_flags(spec: RunSpec) -> list[str]:
    meta = MODELS[spec.model]
    setting = spec.setting
    if setting == "weight_lora":
        clone = RunSpec(**{**spec.__dict__, "family": "granularity", "setting": "expert_weight_r1"})
        return _granularity_flags(clone)
    if setting == "expert_lora":
        clone = RunSpec(**{**spec.__dict__, "family": "granularity", "setting": "expert_bypass_r2"})
        return _granularity_flags(clone)
    if setting == "layer_lora":
        rank = 64 if spec.model == "gptoss" else 256
        clone = RunSpec(**{**spec.__dict__, "family": "granularity", "setting": f"layer_bypass_r{rank}"})
        return _granularity_flags(clone)
    if setting.startswith("moe2_"):
        required_model = MODEL_SPECIFIC_SETTINGS.get(setting)
        if required_model and spec.model != required_model:
            raise ValueError(f"{setting} is a {required_model} source setting, not {spec.model}")
        defaults = {"qwen3": (280, 32), "gemma4": (256, 22), "gptoss": (256, 1)}
        n_experts, rank = defaults[spec.model]
        alpha = 2 * rank
        granularity = "bypass"
        extra: list[str] = []
        if setting.startswith("moe2_pilot_n"):
            parts = setting.removeprefix("moe2_pilot_n").split("_r")
            n_experts, rank = int(parts[0]), int(parts[1])
            alpha = 2 * rank
        elif setting == "moe2_sall_n128_r32_wp_up_ntk_a11_bz":
            n_experts, rank, alpha = 128, 32, 11
            granularity = "weight_pool"
            extra = ["moe_lora_wp_gate_up_stage=up", "moe_lora_normalize_topk_prob=true",
                     "moe_lora_wp_b_init=zero"]
        elif setting == "moe2_pilot_default_ntk":
            extra = ["moe_lora_normalize_topk_prob=true"]
        flags = ["finetuning_type=moe_lora", f"moe_lora_ffn_granularity={granularity}",
                 "moe_lora_pool_share=global", "moe_lora_w_share=per_layer",
                 "moe_lora_routing_mode=learned", "moe_lora_hidden_bottleneck_dim=128",
                 "moe_lora_n_groups=2", f"moe_lora_n_experts={n_experts}",
                 f"moe_lora_rank={rank}", f"moe_lora_alpha={alpha}",
                 f"moe_lora_top_k={meta['top_k']}", "moe_lora_attach_ffn=true",
                 "moe_lora_attach_attn=false", "moe_lora_attach_router=false",
                 "moe_lora_balance_loss_coef=0.0",
                 "moe_lora_balance_loss_schedule=linear_decay"]
        return flags + extra
    common_weight = ["finetuning_type=weight_peft", "weight_peft_attach_attn=false",
                     "weight_peft_attach_ffn=true", "weight_peft_attach_router=true"]
    if setting == "weight_dora":
        return common_weight + ["weight_peft_type=dora", "weight_peft_rank=1", "weight_peft_alpha=2",
                                "weight_peft_attn_rank=32", "weight_peft_attn_alpha=64",
                                "weight_peft_router_rank=8", "weight_peft_router_alpha=16", 'save_strategy="no"']
    if setting == "weight_oft":
        blocks = {"qwen3": (4, 128, 128), "gemma4": (8, 128, 128), "gptoss": (6, 64, 32)}[spec.model]
        return common_weight + ["weight_peft_type=oft", f"weight_peft_oft_block_size={blocks[0]}",
                                f"weight_peft_attn_oft_block_size={blocks[1]}",
                                f"weight_peft_router_oft_block_size={blocks[2]}",
                                'save_strategy="no"']
    if setting == "weight_fourier":
        frequencies = {"qwen3": (4096, 139264, 17408), "gemma4": (5632, 199814, 23552),
                       "gptoss": (5760, 165888, 23296)}[spec.model]
        return common_weight + ["weight_peft_type=fourier", f"weight_peft_fourier_n_frequencies={frequencies[0]}",
                                f"weight_peft_attn_fourier_n_frequencies={frequencies[1]}",
                                f"weight_peft_router_fourier_n_frequencies={frequencies[2]}", 'save_strategy="no"']
    if setting == "weight_ia3":
        return common_weight + ["weight_peft_type=ia3", 'save_strategy="no"']
    if setting == "expert_moe_lora_n8":
        rank = 8 if spec.model == "gptoss" else 32
        return ["finetuning_type=moe_lora", "moe_lora_n_experts=8", "moe_lora_top_k=2",
                "moe_lora_routing_mode=independent", f"moe_lora_rank={rank}", f"moe_lora_alpha={2 * rank}",
                "moe_lora_pool_share=per_layer", "moe_lora_w_share=per_layer",
                "moe_lora_balance_loss_coef=0.0"] + _attachment_flags(False)
    if setting == "expert_moe2_lora":
        rank = 1 if spec.model == "gptoss" else 24
        return ["finetuning_type=moe_lora", "moe_lora_n_experts=8", "moe_lora_n_groups=1",
                "moe_lora_top_k=2", "moe_lora_routing_mode=learned",
                "moe_lora_hidden_bottleneck_dim=128", f"moe_lora_rank={rank}",
                f"moe_lora_alpha={2 * rank}", "moe_lora_pool_share=per_layer",
                "moe_lora_w_share=per_layer"] + _attachment_flags(False)
    if setting == "das_lora":
        rank = 16 if spec.model == "gptoss" else 32
        return ["finetuning_type=das_lora", f"das_lora_selected_experts_path={spec.result / 'das_selected.json'}",
                f"das_lora_rank={rank}", f"das_lora_alpha={2 * rank}"] + _attachment_flags(False)
    if setting == "partial_last":
        return ["finetuning_type=partial_ft", "partial_ft_target=last", "partial_ft_modules=expert,router"]
    raise ValueError(f"unsupported method setting: {setting}")


def _loss_flags(spec: RunSpec) -> list[str]:
    coef = {"aux0": "0.0", "aux1e-4": "1.0e-4", "aux1e-3": "1.0e-3",
            "aux1e-2": "1.0e-2", "aux1e-1": "1.0e-1"}[spec.setting]
    clone = RunSpec(**{**spec.__dict__, "family": "granularity", "setting": "expert_weight_r1"})
    return _granularity_flags(clone) + [f"moe_lora_balance_loss_coef={coef}",
                                        "moe_lora_balance_loss_schedule=constant"]


def method_flags(spec: RunSpec) -> list[str]:
    return {"granularity": _granularity_flags, "position": _position_flags,
            "methods": _method_flags, "loss": _loss_flags}[spec.family](spec)


def das_command(spec: RunSpec) -> list[str] | None:
    if spec.setting != "das_lora":
        return None
    if spec.das_general is None:
        raise ValueError("DAS-LoRA requires --das-general")
    domain = spec.data_root / ("metamathqa_r1_clean.json" if spec.task == "math" else "esft/intent_train.jsonl")
    return ["env", f"CUDA_VISIBLE_DEVICES={spec.gpus.split(',')[0]}", "python3",
            "eval_scripts/compute_das.py", "--base_model", str(spec.model_path),
            "--domain_dataset", str(domain), "--general_dataset", str(spec.das_general),
            "--general_split", "test", "--top_k", str(MODELS[spec.model]["top_k"]),
            "--max_samples", "200", "--output", str(spec.result / "das_selected.json")]


def train_command(spec: RunSpec) -> list[str]:
    command = ["env", f"CUDA_VISIBLE_DEVICES={spec.gpus}", f"MOEFACTORY_DATA_ROOT={spec.data_root}",
               "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
               "FORCE_TORCHRUN=1", "llamafactory-cli", "train", _base_config(spec),
               f"model_name_or_path={spec.model_path}", f"deepspeed={spec.deepspeed}",
               f"dataset_dir={spec.data_root}", f"learning_rate={LR_VALUES[spec.lr]}",
               f"seed={spec.seed}", f"data_seed={spec.seed}"]
    command += _task_flags(spec) + method_flags(spec)
    command += ["report_to=none", "overwrite_output_dir=true",
                f"run_name={spec.model}_{spec.family}_{spec.setting}_{spec.task}_{spec.lr}_seed{spec.seed}",
                f"output_dir={spec.adapter}"]
    return command


def downstream_command(spec: RunSpec) -> list[str]:
    if spec.task == "math":
        return ["env", f"CUDA_VISIBLE_DEVICES={spec.gpus}", f"MOEFACTORY_DATA_ROOT={spec.data_root}",
                "HF_HUB_OFFLINE=1",
                "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "torchrun",
                "--nproc_per_node=4", "eval_scripts/eval_math500.py", "--base_model", str(spec.model_path),
                "--adapter_path", str(spec.adapter), "--batch_size", "32", "--max_new_tokens", "512",
                "--save_path", str(spec.result / "math500.jsonl")]
    command = ["env", f"CUDA_VISIBLE_DEVICES={spec.gpus}", f"MOEFACTORY_DATA_ROOT={spec.data_root}",
               "HF_HUB_OFFLINE=1",
               "HF_DATASETS_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
               f"PYTHONPATH={spec.repo / 'eval_scripts/esft'}:{spec.repo}", "torchrun",
               "--nproc_per_node=4", "eval_scripts/esft/eval_esft.py", "--task", "intent",
               "--mode", "all", "--esft-dir", str(spec.data_root / "esft_eval"),
               "--template", "esft_raw", "--prompt-protocol", "auto",
               "--runtime-merge", "--runtime-merge-device-map", "auto",
               "--hf-generate-batch-size", "8", "--tensor-parallel-size", "1",
               "--max-new-tokens", "512", "--model-name-or-path", str(spec.model_path),
               "--adapter-name-or-path", str(spec.adapter), "--output-dir", str(spec.result)]
    if spec.model == "gptoss":
        command += ["--flash-attn", "disabled"]
    return command


def upstream_commands(spec: RunSpec) -> list[list[str]]:
    return [["env", f"REPO={spec.repo}", f"MOEFACTORY_DATA_ROOT={spec.data_root}",
             f"MODEL={spec.model}", f"BASE_MODEL={spec.model_path}", f"ADAPTER={spec.adapter}",
             f"OUT_DIR={spec.result / 'upstream'}", f"GPUS={spec.gpus}", "TASKS=mmlu_pro ifeval",
             "ENGINE=auto", "MAX_LEN=4096", "IFEVAL_MAXNEW=1280", "bash",
             str(spec.repo / "scripts/reproduce/run_upstream_eval.sh")]]


def commands(spec: RunSpec) -> list[tuple[str, list[str]]]:
    spec.validate()
    result: list[tuple[str, list[str]]] = []
    if spec.phase in ("train", "full"):
        das = das_command(spec)
        if das:
            result.append(("das", das))
        result.append(("train", train_command(spec)))
    if spec.phase in ("eval", "full"):
        result.append(("downstream", downstream_command(spec)))
        if spec.run_upstream:
            result.extend(("upstream", command) for command in upstream_commands(spec))
    return result
