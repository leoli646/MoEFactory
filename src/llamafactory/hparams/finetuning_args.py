# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: int | None = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules. "
                "Use `none` to target no modules (only `lora_target_parameters`)."
            )
        },
    )
    lora_target_parameters: str = field(
        default="",
        metadata={
            "help": (
                "Comma-separated nn.Parameter names to apply LoRA via PEFT `target_parameters` "
                "(e.g. Qwen3-MoE fused experts: `experts.gate_up_proj`). Pair with `lora_target=none` "
                "to put plain LoRA only on the parameter. NOTE: PEFT cannot target two parameters on the "
                "same module, so pick one of gate_up_proj / down_proj per adapter."
            )
        },
    )
    loraplus_lr_ratio: float | None = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: str | None = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: str | None = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: int | None = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: float | None = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: int | None = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Literal["ascending", "descending", "random", "fixed"] | None = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: int | None = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: str | None = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: str | None = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: str | None = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: str | None = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: str | None = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: str | None = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal[
        "lora", "oft", "freeze", "full", "moe_lora", "das_lora", "weight_peft", "partial_ft"
    ] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    moe_lora_n_experts: int = field(
        default=8,
        metadata={"help": "MoE-LoRA: pool size N_L. V1=8, V2=512, V3=64, V4=4."},
    )
    moe_lora_rank: int = field(
        default=8,
        metadata={"help": "MoE-LoRA: rank of each LoRA expert. V1=8, V2=4, V3=32, V4=16."},
    )
    moe_lora_alpha: int = field(
        default=16,
        metadata={"help": "MoE-LoRA: scaling factor (alpha / rank). Standard alpha = 2 * rank."},
    )
    moe_lora_top_k: int = field(
        default=2,
        metadata={"help": "MoE-LoRA: top-k LoRA experts to activate per token."},
    )
    moe_lora_normalize_topk_prob: bool = field(
        default=False,
        metadata={
            "help": "MoE-LoRA: renormalize the selected top-k routing probabilities to sum to 1. "
            "Default False preserves historical checkpoints and training behavior."
        },
    )
    moe_lora_pool_share: Literal["per_layer", "global"] = field(
        default="per_layer",
        metadata={"help": "MoE-LoRA: LoRA pool sharing. V1/V4=per_layer, V2/V3=global."},
    )
    moe_lora_w_share: Literal["per_layer", "global"] = field(
        default="per_layer",
        metadata={
            "help": "MoE-LoRA: routing projection W sharing. Default per_layer keeps positional info; "
            "V2' ablation uses global."
        },
    )
    moe_lora_detach_p_e: bool = field(
        default=False,
        metadata={
            "help": "MoE-LoRA: detach router_logits before LoRA branch. "
            "No effect when MoE backbone is fully frozen; reserved for future stage-1 router tuning."
        },
    )
    moe_lora_routing_mode: Literal["learned", "follow_moe", "independent", "residual"] = field(
        default="learned",
        metadata={
            "help": "MoE-LoRA: routing source. 'learned' (default) uses RoutingProjection W to "
            "project router_logits to N_L LoRA experts. 'follow_moe' (control baseline) skips "
            "RoutingProjection and reuses original MoE router's top_k_index/top_k_weights "
            "directly; pool size and top_k must match original MoE's n_experts and top_k. "
            "'independent' (baseline2 / LoRAMoE-style) uses an independent router that takes "
            "hidden_states as input, completely decoupled from MoE routing. "
            "'residual' combines both signals: logits = W1 @ router_logits + W2 @ hidden_states."
        },
    )
    moe_lora_target_layers: str = field(
        default="all",
        metadata={
            "help": "MoE-LoRA: which MoE layers to inject. "
            "'all' / 'first_half' / 'last_half' / 'last_third' / explicit '0,5,10,15'."
        },
    )
    moe_lora_ffn_granularity: Literal["bypass", "expert_weight", "weight_pool"] = field(
        default="bypass",
        metadata={
            "help": "MoE-LoRA FFN granularity. 'bypass' keeps the historical d_model->rank->d_model "
            "LoRA branch (N_L=1 gives one layer-level branch; follow_moe with N_L=N_E gives one "
            "branch per base expert). 'expert_weight' applies one LoRA inside every native fused expert "
            "gate_up/down weight. 'weight_pool' keeps MoE^2-LoRA routing and shares a routed, aligned "
            "gate_up/down adapter pool across the native experts."
        },
    )
    moe_lora_b_init: Literal["normal", "zero"] = field(
        default="normal",
        metadata={
            "help": "MoE-LoRA: LoRA B initialization for the historical bypass adapter pool only. "
            "'normal' keeps the existing Normal(std=1e-4) initialization (default); 'zero' makes "
            "the adapter output exactly zero at step 0. No effect unless "
            "moe_lora_ffn_granularity='bypass'."
        },
    )
    moe_lora_wp_b_init: Literal["normal", "zero"] = field(
        default="normal",
        metadata={
            "help": "MoE-LoRA: LoRA B initialization for the weight_pool ('_wp') adapter pool only. "
            "'normal' keeps the historical MoE^2-LoRA normal(std=1e-4) init (default, preserves "
            "existing checkpoints and every other granularity). 'zero' makes the adapter an exact "
            "identity at step 0, which is the standard LoRA convention. No effect unless "
            "moe_lora_ffn_granularity='weight_pool'."
        },
    )
    moe_lora_layer_ranks: str = field(
        default="",
        metadata={
            "help": "Optional comma-separated FFN LoRA rank for each selected MoE layer. Empty uses "
            "moe_lora_rank uniformly. The alpha/rank ratio is preserved per layer. This is intended "
            "for parameter-budget matching and requires per-layer FFN adapters."
        },
    )
    moe_lora_wp_gate_up_stage: Literal["gate_up", "up"] = field(
        default="gate_up",
        metadata={
            "help": "MoE-LoRA weight_pool only: which half of the fused gate_up projection the adapter "
            "learns. 'gate_up' (default) adapts the full 2*intermediate output (historical behavior). "
            "'up' adapts only the up half (the linear branch of act_fn(gate)*up), leaving the gate half "
            "unadapted; this halves the gate_up B parameters and avoids perturbing the multiplicative "
            "gate input. No effect unless moe_lora_ffn_granularity='weight_pool'."
        },
    )
    moe_lora_balance_loss_coef: float = field(
        default=0.0,
        metadata={
            "help": "MoE-LoRA: load balancing auxiliary loss coefficient. "
            "0 = disabled. The aux loss is averaged over pools (per-layer mean, "
            "HF load_balancing_loss convention), so coef is comparable to common "
            "literature values (e.g. 1e-3 ~ 1e-2)."
        },
    )
    moe_lora_balance_loss_schedule: Literal["linear_decay", "constant"] = field(
        default="linear_decay",
        metadata={
            "help": "MoE-LoRA: balance loss coefficient schedule. "
            "'linear_decay' (default, legacy behavior) decays coef linearly to 0 over training; "
            "'constant' keeps coef fixed for the whole run (standard MoE aux-loss formulation, "
            "use this for aux-loss coef ablations)."
        },
    )
    moe_lora_hidden_bottleneck_dim: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: hidden state bottleneck dim added to routing input. "
            "0 = disabled (vanilla V2 with 64-d MoE logits only). "
            "When > 0, a Linear(d_model, dim) projects hidden state and concats with "
            "router_logits before W_l. Recommended: 16. Breaks the 64-d info bottleneck."
        },
    )
    moe_lora_wp_down_route_on_h: bool = field(
        default=False,
        metadata={
            "help": "MoE-LoRA weight_pool only: route the down-projection adapter from the post-gate "
            "activation h_in = act_fn(gate)*up instead of the block input hidden_states. Default False "
            "keeps the historical shared-input routing. When True, an independent down RoutingProjection "
            "consumes the intermediate activation (moe_intermediate_size) so the down adapter is selected "
            "in the space it actually acts on. No effect unless moe_lora_ffn_granularity='weight_pool'."
        },
    )
    moe_lora_hidden_bottleneck_gate_up_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE-LoRA weight_pool only: hidden bottleneck dim for an independent gate_up route. "
            "Set this together with moe_lora_hidden_bottleneck_down_dim to enable separate per-stage routing."
        },
    )
    moe_lora_hidden_bottleneck_down_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE-LoRA weight_pool only: hidden bottleneck dim for an independent down route. "
            "Set this together with moe_lora_hidden_bottleneck_gate_up_dim to enable separate per-stage routing."
        },
    )
    moe_lora_n_global: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: number of cross-layer global LoRA experts (routed, shared across all layers). "
            "0 = disabled. When > 0, per-layer W_l outputs (n_experts + n_global) logits and "
            "joint top-K competes across both pools. Requires pool_share='per_layer'. "
            "Designed to capture cross-layer recurring concepts via shared expert weights, "
            "while per-layer pool stays for layer-specific specialization. "
            "Joint mode disables balance loss to avoid per-pool boundary cases."
        },
    )
    moe_lora_global_rank: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: rank for cross-layer global LoRA experts. "
            "0 = use moe_lora_rank (same rank as per-layer pool). "
            "Set to a different value to give the global pool a different rank than the per-layer pool."
        },
    )
    moe_lora_global_alpha: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: alpha for cross-layer global LoRA experts. "
            "0 = use moe_lora_alpha (same alpha as per-layer pool)."
        },
    )
    moe_lora_top_k_local: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: split top-K mode — per-layer pool selects this many experts independently. "
            "0 (default) = use joint top-K over (per-layer + global) combined logits. "
            "When both top_k_local and top_k_global > 0, switch to split mode: per-layer and global pools select independently."
        },
    )
    moe_lora_top_k_global: int = field(
        default=0,
        metadata={
            "help": "MoE-LoRA: split top-K mode — cross-layer global pool selects this many experts independently. "
            "See moe_lora_top_k_local."
        },
    )
    moe_lora_das_init_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "MoE-LoRA: path to DAS json (e.g. eval_results/das_qwen3_math.json) "
            "for DAS-informed RoutingProjection initialization. When provided, W_l is "
            "initialized via hard-partition: each LoRA expert is assigned a group of "
            "base experts (sorted by per-layer DAS) with strong positive weight, "
            "preventing expert collapse on sparse-routing base models. "
            "Only effective for routing_mode='learned' and w_share='per_layer'."
        },
    )
    moe_lora_das_init_strength: float = field(
        default=1.0,
        metadata={
            "help": "MoE-LoRA: DAS init weight magnitude (gamma). Default 1.0; "
            "lower (e.g. 0.5) leaves more room for W_l to evolve during training."
        },
    )
    moe_lora_wl_init: str = field(
        default="default",
        metadata={
            "help": "MoE-LoRA: RoutingProjection W_l init mode. "
            "'default' = Kaiming default init. "
            "'anchor' = identity-like init that maps each LoRA expert i to a "
            "contiguous chunk of base experts on the z-channel, with the "
            "hidden-state channel zeroed. At step 0 the LoRA routing mirrors a "
            "quantized base routing (no routing collapse); training then "
            "gradually deviates W_l to learn task-specific routing. "
            "Ignored when moe_lora_das_init_path is also set (DAS takes precedence)."
        },
    )
    moe_lora_n_groups: int = field(
        default=1,
        metadata={
            "help": "MoE-LoRA: number of groups for Hierarchical Pool. "
            "1 = single global pool (default,适合浅层模型如 OLMoE 16 层). "
            ">1 = split layers into N groups,each group has its own LoRA pool. "
            "适合深层模型(如 Qwen3 48 层),减弱 cross-layer gradient conflict. "
            "需保证 n_lora 和 layer_count 都能被 n_groups 整除。"
            "示例: 48 层 + n_groups=3 → layer 0-15/16-31/32-47 各占一组。"
        },
    )
    # ------------------------------------------------------------------
    # MoE-LoRA position ablation: 控制 LoRA 注入的"位置"。
    #   - attach_ffn   : FFN expert pool 分支(=已有 baseline,默认开,保持向后兼容)
    #   - attach_attn  : Attention(Q/K/V/O 等 *_proj) 上的标准 LoRA
    #   - attach_router: MoE router/gate 权重上的低秩增量(weight 参数化)
    # 三个开关可任意组合,覆盖 {attn, ffn, router} 的 7 种非空子集。
    # ------------------------------------------------------------------
    moe_lora_attach_ffn: bool = field(
        default=True,
        metadata={
            "help": "MoE-LoRA position ablation: inject the FFN expert LoRA pool branch "
            "(the existing baseline). Default True keeps backward compatibility. "
            "Set False for attention-only / router-only ablations."
        },
    )
    moe_lora_attach_attn: bool = field(
        default=False,
        metadata={
            "help": "MoE-LoRA position ablation: inject standard LoRA on attention "
            "projections (see moe_lora_attn_targets). Default False."
        },
    )
    moe_lora_attach_router: bool = field(
        default=False,
        metadata={
            "help": "MoE-LoRA position ablation: inject a low-rank delta on the MoE "
            "router/gate weight (changes base expert selection). Default False."
        },
    )
    moe_lora_attn_targets: str = field(
        default="auto",
        metadata={
            "help": "MoE-LoRA: which attention Linear sub-modules get LoRA. "
            "'auto' = all '*_proj' linears in self_attn (q/k/v/o_proj for OLMoE/Qwen3, "
            "q_a/q_b/kv_a_proj_with_mqa/kv_b/o_proj for DeepSeek-V2 MLA). "
            "Or an explicit comma list, e.g. 'q_proj,k_proj,v_proj,o_proj'."
        },
    )
    moe_lora_attn_rank: int = field(
        default=0,
        metadata={"help": "MoE-LoRA: rank for attention LoRA. 0 = fall back to moe_lora_rank."},
    )
    moe_lora_attn_alpha: int = field(
        default=0,
        metadata={"help": "MoE-LoRA: alpha for attention LoRA. 0 = fall back to moe_lora_alpha."},
    )
    moe_lora_router_rank: int = field(
        default=0,
        metadata={"help": "MoE-LoRA: rank for router-gate LoRA delta. 0 = fall back to moe_lora_rank."},
    )
    moe_lora_router_alpha: int = field(
        default=0,
        metadata={"help": "MoE-LoRA: alpha for router-gate LoRA delta. 0 = fall back to moe_lora_alpha."},
    )
    das_lora_selected_experts_path: str = field(
        default="",
        metadata={
            "help": "DAS-LoRA: path to JSON file with DAS-selected experts per layer. "
            "Generate via eval_scripts/compute_das.py."
        },
    )
    das_lora_rank: int = field(
        default=32,
        metadata={"help": "DAS-LoRA: LoRA rank for each selected expert."},
    )
    das_lora_alpha: int = field(
        default=64,
        metadata={"help": "DAS-LoRA: LoRA alpha (scaling = alpha / rank). Convention: 2 * rank."},
    )
    weight_peft_type: Literal[
        "lora", "dora", "oft",
        "oft_out", "fourier", "ia3"
    ] = field(
        default="lora",
        metadata={
            "help": "weight_peft: which canonical weight-level adapter to attach to the fused "
            "MoE experts (gate_up_proj+down_proj), attention projections, and optional routers. "
            "lora / dora / oft (HF-peft exact input-side rotation) / "
            "oft_out (legacy "
            "output-side rotation, for old checkpoints) / fourier (FourierFT, DFT spectral "
            "delta) / ia3 (IA3 rescaling)."
        },
    )
    weight_peft_rank: int = field(
        default=2,
        metadata={"help": "weight_peft: LoRA/DoRA rank for each adapted weight (ignored by OFT)."},
    )
    weight_peft_alpha: int = field(
        default=0,
        metadata={"help": "weight_peft: LoRA/DoRA alpha (scaling = alpha / rank). 0 = use 2 * rank."},
    )
    weight_peft_oft_block_size: int = field(
        default=8,
        metadata={"help": "weight_peft (OFT): block size of the block-diagonal orthogonal matrix; "
                  "must divide in_features for OFT (out_features for legacy OFT-out). "
                  "Controls capacity."},
    )

    weight_peft_attn_rank: int = field(
        default=0,
        metadata={"help": "weight_peft: LoRA/DoRA rank for attention weights. "
                  "0 = fall back to weight_peft_rank."},
    )
    weight_peft_attn_alpha: int = field(
        default=0,
        metadata={"help": "weight_peft: LoRA/DoRA alpha for attention weights. "
                  "0 = fall back to weight_peft_alpha (or 2 * resolved rank when both are 0)."},
    )
    weight_peft_router_rank: int = field(
        default=0,
        metadata={"help": "weight_peft: LoRA/DoRA rank for router weights. "
                  "0 = fall back to weight_peft_rank."},
    )
    weight_peft_router_alpha: int = field(
        default=0,
        metadata={"help": "weight_peft: LoRA/DoRA alpha for router weights. "
                  "0 = fall back to weight_peft_alpha (or 2 * resolved rank when both are 0)."},
    )
    weight_peft_attn_oft_block_size: int = field(
        default=0,
        metadata={"help": "weight_peft (OFT): block size for attention weights. "
                  "0 = fall back to weight_peft_oft_block_size."},
    )
    weight_peft_router_oft_block_size: int = field(
        default=0,
        metadata={"help": "weight_peft (OFT): block size for router weights. "
                  "0 = fall back to weight_peft_oft_block_size."},
    )
    weight_peft_attach_attn: bool = field(
        default=True,
        metadata={"help": "weight_peft: attach adapter on attention q/k/v/o weights."},
    )
    weight_peft_attach_ffn: bool = field(
        default=True,
        metadata={"help": "weight_peft: attach adapter on all MoE expert weights (gate_up+down)."},
    )
    weight_peft_attach_router: bool = field(
        default=False,
        metadata={"help": "weight_peft: attach the selected weight adapter to each MoE router. "
                  "Default False preserves historical checkpoints and runs."},
    )
    weight_peft_attn_targets: str = field(
        default="auto",
        metadata={"help": "weight_peft: attention target linears. 'auto' = all q/k/v/o; "
                  "or comma list e.g. 'q_proj,v_proj'. For ia3 this is intersected with k/v "
                  "(canonical IA3 rescales keys & values only)."},
    )
    weight_peft_fourier_n_frequencies: int = field(
        default=2000,
        metadata={"help": "weight_peft (FourierFT): number of learnable spectral coefficients per "
                  "adapted weight (per expert for fused experts). Trainable params scale linearly "
                  "with this. Must be <= out_features * in_features of every adapted weight."},
    )
    weight_peft_attn_fourier_n_frequencies: int = field(
        default=0,
        metadata={"help": "weight_peft (FourierFT): spectral coefficients per attention weight. "
                  "0 = fall back to weight_peft_fourier_n_frequencies."},
    )
    weight_peft_router_fourier_n_frequencies: int = field(
        default=0,
        metadata={"help": "weight_peft (FourierFT): spectral coefficients per router weight. "
                  "0 = fall back to weight_peft_fourier_n_frequencies."},
    )
    weight_peft_fourier_scaling: float = field(
        default=150.0,
        metadata={"help": "weight_peft (FourierFT): scaling factor applied to the iDFT-reconstructed "
                  "weight delta (HF FourierFT default 150.0)."},
    )
    weight_peft_fourier_seed: int = field(
        default=2024,
        metadata={"help": "weight_peft (FourierFT): base seed for the (reproducible) random spectral "
                  "support. Saved in config and replayed on load so coefficients land at the same "
                  "frequency locations; never stored as weights."},
    )
    partial_ft_target: Literal[
        "first", "middle", "last", "first_third", "middle_third", "last_third", "custom"
    ] = field(
        default="last",
        metadata={"help": "partial_ft: which decoder layers to fine-tune. Single layer "
                  "(first/middle/last) or a contiguous third (first_third/middle_third/last_third), "
                  "or 'custom' with partial_ft_layers."},
    )
    partial_ft_layers: str = field(
        default="",
        metadata={"help": "partial_ft (custom): comma-separated decoder layer indices to fine-tune, "
                  "e.g. '0,1,2'. Only used when partial_ft_target=custom."},
    )
    partial_ft_modules: str = field(
        default="attn,expert",
        metadata={"help": "partial_ft: which weight groups inside each selected layer become "
                  "trainable. Comma list from {attn, expert, router, norm}. Default 'attn,expert' "
                  "mirrors the peft_compare injection site (attention q/k/v/o + all MoE experts)."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_hyper_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use HyperParallel distributed training backend (FSDP/TP). "
                "Only supported for the 'sft' stage with full fine-tuning."
            )
        },
    )
    hyper_parallel_args: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a JSON file containing HyperParallel strategy arguments "
                "(e.g., tp_size, param_dtype). Used when use_hyper_parallel=True."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_asft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the ASFT loss."},
    )
    asft_alpha: float = field(
        default=0.1,
        metadata={"help": "The alpha parameter for ASFT loss to control the power of adaptive weight."},
    )
    use_eaft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the EAFT loss."},
    )
    eaft_alpha: float = field(
        default=1.0,
        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    early_stopping_steps: int | None = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: list[str] | None = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        # `none`/`` clears it (mirrors `lora_target=none`), so a CLI override like
        # `lora_target_parameters=none` can cancel a yaml default (used by the
        # expert_lora position-ablation driver to keep attn/router-only runs FFN-free).
        self.lora_target_parameters: list[str] | None = (
            split_arg(self.lora_target_parameters)
            if self.lora_target_parameters and str(self.lora_target_parameters).strip().lower() != "none"
            else None
        )
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: list[str] | None = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        if self.weight_peft_rank < 1:
            raise ValueError("`weight_peft_rank` must be positive.")
        if self.weight_peft_oft_block_size <= 1:
            raise ValueError("`weight_peft_oft_block_size` must be greater than 1.")

        assert self.finetuning_type in ["lora", "oft", "freeze", "full", "moe_lora", "das_lora", "weight_peft", "partial_ft"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")


    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
