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

from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _set_attn(config, impl: str) -> None:
    setattr(config, "_attn_implementation", impl)
    setattr(config, "_attn_implementation_internal", impl)


def _configure_gpt_oss_attn(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    """gpt-oss 专用 attention 选择, 回退链: FA3 hub kernel → flex_attention → eager。

    - FA3 需要 transformers 对应 API(5.5+ 改名 load_and_register_attn_kernel) + `kernels` 包
      + 联网拉取 kernels-community/vllm-flash-attn3(离线容器不可用)。
    - flex_attention (torch>=2.5, 仅 CUDA): fused kernel, 原生传递 attention sinks(s_aux),
      不物化 O(L^2) 矩阵。eager 在 attn 挂 trainable 参数时(attn-LoRA / partial_ft)每层
      保留 PBS*heads*L^2 的 bf16 注意力矩阵用于反向 —— 4xH200 上 PBS6/cutoff2048 已实测 OOM
      (2026-07-12, gptoss math: 单矩阵 6*64*2048^2*2B = 3.0 GiB/层)。
    - FA2 2.8.3 无 s_aux/learnable_sink 参数, 会静默丢 attention sinks(数值错误), 永不使用。
    - 显式 flash_attn=disabled 跳过 FA3/flex 直接 eager(调试/CPU 逃生口;
      transformers 的 flex+sinks 路径在 CPU 上直接 raise)。
    """
    if model_args.flash_attn == AttentionFunction.DISABLED:
        _set_attn(config, "eager")
        logger.info_rank0("flash_attn=disabled: using eager attention for the gpt-oss model.")
        return

    try:
        try:
            # transformers >= 5.5
            from transformers.integrations.hub_kernels import (
                load_and_register_attn_kernel as _load_attn_kernel,
            )
        except ImportError:
            # transformers <= 4.5x
            from transformers.integrations.hub_kernels import (
                load_and_register_kernel as _load_attn_kernel,
            )
        from transformers.integrations.hub_kernels import is_kernels_available

        if not is_kernels_available():
            raise ImportError("`kernels` package is not installed")

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        _load_attn_kernel(flash_attn3_kernel)
        _set_attn(config, flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3
        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return
    except Exception as err:
        logger.warning_rank0(f"FlashAttention-3 hub kernel unavailable for gpt-oss ({err}).")

    try:
        import inspect as _inspect

        import torch

        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
        from transformers.integrations.flex_attention import flex_attention_forward

        if not torch.cuda.is_available():
            raise RuntimeError("flex attention with sinks requires CUDA")
        if "s_aux" not in str(_inspect.signature(flex_attention_forward)):
            raise ImportError("this transformers flex integration does not forward attention sinks (s_aux)")

        _set_attn(config, "flex_attention")
        model_args.flash_attn = AttentionFunction.DISABLED  # 防下游通用分支再覆盖
        logger.info_rank0("Using flex attention (native attention-sink support) for the gpt-oss model.")
    except Exception as err:
        logger.warning_rank0(
            f"flex attention unavailable for gpt-oss ({err}); falling back to eager attention "
            "(memory-heavy: expect OOM with attention adapters at long cutoff)."
        )
        _set_attn(config, "eager")
        model_args.flash_attn = AttentionFunction.DISABLED


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        _configure_gpt_oss_attn(config, model_args)
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            logger.warning_rank0("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")
