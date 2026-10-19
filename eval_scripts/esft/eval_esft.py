#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .tokenizer_compat import patch_qwen2_tokenizer_for_vllm
except ImportError:
    from tokenizer_compat import patch_qwen2_tokenizer_for_vllm

TASKS = ("intent",)
MODES = ("all", "generate", "score")
PROMPT_PROTOCOLS = ("auto", "raw", "chat")
DEFAULT_DATA_ROOT = Path(os.environ.get("MOEFACTORY_DATA_ROOT", ROOT / "data")).expanduser()
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_model_args(
    args: argparse.Namespace,
    *,
    infer_backend: str = "vllm",
) -> tuple[Any, Any]:
    from llamafactory.hparams import DataArguments, ModelArguments

    model_name_or_path = str(args.model_name_or_path)
    if args.model_name_or_path.exists():
        model_name_or_path = str(args.model_name_or_path.resolve())

    adapter_name_or_path = args.adapter_name_or_path
    if isinstance(adapter_name_or_path, str):
        adapter_path = Path(adapter_name_or_path).expanduser()
        if adapter_path.exists():
            adapter_name_or_path = str(adapter_path.resolve())

    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        trust_remote_code=True,
        infer_backend=infer_backend,
        infer_dtype="auto",
        vllm_config=args.vllm_config,
        # --flash-attn disabled: 强制 eager(逃生口)。gpt-oss 的 flex attention 在
        # 长序列(intent cutoff~4096)下会 illegal memory access(2026-07-14), zeroshot
        # base 评测走 load_model 会踩; custom adapter 路径(AutoModel 直加载)本就 eager。
        flash_attn=getattr(args, "flash_attn", "auto"),
    )
    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len, enable_thinking=False)
    return model_args, data_args


def load_tokenizer_and_template(model_args: Any, data_args: Any) -> tuple[Any, Any, Any]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    template.mm_plugin.expand_mm_tokens = False
    return tokenizer, processor, template


def render_chat_prompts(
    eval_rows: list[dict[str, Any]],
    model_args: Any,
    data_args: Any,
) -> tuple[list[list[int]], list[str], Any, Any]:
    tokenizer, processor, template = load_tokenizer_and_template(model_args, data_args)

    prompt_token_ids: list[list[int]] = []
    rendered_prompts: list[str] = []
    for row in eval_rows:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": ""},
        ]
        messages = template.mm_plugin.process_messages(messages, [], [], [], processor)
        input_ids, _ = template.encode_oneturn(tokenizer, messages, None, None)
        input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, [], [], [], tokenizer, processor)
        prompt_token_ids.append(input_ids)
        rendered_prompts.append(tokenizer.decode(input_ids, skip_special_tokens=False))

    return prompt_token_ids, rendered_prompts, tokenizer, template


def render_raw_prompts(
    eval_rows: list[dict[str, Any]],
    model_args: Any,
    data_args: Any,
) -> tuple[list[list[int]], list[str], Any, Any]:
    tokenizer, _, template = load_tokenizer_and_template(model_args, data_args)
    prefix_ids = template._convert_elements_to_ids(tokenizer, template.format_prefix.apply())

    prompt_token_ids: list[list[int]] = []
    rendered_prompts: list[str] = []
    for row in eval_rows:
        prompt_text = row["prompt"]
        input_ids = prefix_ids + tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_token_ids.append(input_ids)
        rendered_prompts.append(prompt_text)

    return prompt_token_ids, rendered_prompts, tokenizer, template


def resolve_prompt_protocol(args: argparse.Namespace) -> str:
    if args.prompt_protocol != "auto":
        return args.prompt_protocol

    if args.template == "esft_raw":
        return "raw"

    return "chat"


def get_stop_token_ids(tokenizer: Any, template: Any) -> list[int]:
    stop_token_ids = [int(token_id) for token_id in template.get_stop_token_ids(tokenizer) if token_id is not None]
    if not stop_token_ids and tokenizer.eos_token_id is not None:
        stop_token_ids = [int(tokenizer.eos_token_id)]
    return list(dict.fromkeys(stop_token_ids))


def _enforce_prompt_budget(
    prompt_token_ids: list[list[int]],
    rendered_prompts: list[str],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[list[list[int]], list[str]]:
    """超过 cutoff_len 的 prompt：默认仍 raise；--prompt-truncate-side 非 none 时截断。

    right: 保留开头(含 prefix/指令)，砍掉尾部 —— 用于保留前缀指令。
    left:  保留首 token(BOS) + 尾部。用于上下文比模型短的情形(如 OLMoE 4096)。
    """
    budget = args.cutoff_len
    side = getattr(args, "prompt_truncate_side", "none")
    over = [i for i, ids in enumerate(prompt_token_ids) if len(ids) > budget]
    if not over:
        return prompt_token_ids, rendered_prompts
    if side == "none":
        longest = max(len(prompt_token_ids[i]) for i in over)
        raise ValueError(
            f"Prompt length exceeds cutoff_len={budget}. Longest prompt is {longest} tokens."
        )
    new_ids, new_text = list(prompt_token_ids), list(rendered_prompts)
    for i in over:
        ids = prompt_token_ids[i]
        ids = ids[:budget] if side == "right" else ids[:1] + ids[-(budget - 1):]
        new_ids[i] = ids
        new_text[i] = tokenizer.decode(ids, skip_special_tokens=False)
    if _is_main():
        print(f"[eval] truncated {len(over)} prompt(s) to cutoff_len={budget} (side={side})")
    return new_ids, new_text


def generate_predictions_vllm(
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_args: Any,
    data_args: Any,
) -> tuple[list[str], list[str], list[int], str]:
    patch_qwen2_tokenizer_for_vllm()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompt_protocol = resolve_prompt_protocol(args)
    if prompt_protocol == "raw":
        prompt_token_ids, rendered_prompts, tokenizer, template = render_raw_prompts(eval_rows, model_args, data_args)
    else:
        prompt_token_ids, rendered_prompts, tokenizer, template = render_chat_prompts(
            eval_rows, model_args, data_args
        )

    prompt_token_ids, rendered_prompts = _enforce_prompt_budget(
        prompt_token_ids, rendered_prompts, tokenizer, args
    )
    prompt_lengths = [len(token_ids) for token_ids in prompt_token_ids]

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": args.cutoff_len + args.max_new_tokens,
        "disable_log_stats": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        stop_token_ids=get_stop_token_ids(tokenizer, template),
        skip_special_tokens=True,
        seed=args.seed,
    )

    lora_request = None
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])

    outputs = llm.generate(
        [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids],
        sampling,
        lora_request=lora_request,
    )
    predictions = [output.outputs[0].text for output in outputs]
    return predictions, rendered_prompts, prompt_lengths, prompt_protocol


def _dist_info() -> tuple[int, int]:
    """从 torchrun 环境读取 (world_size, local_rank)；非分布式时返回 (1, 0)。"""
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def generate_predictions_runtime_merge(
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[int], str]:
    import gc
    import os

    import torch

    from llamafactory.hparams import FinetuningArguments
    from llamafactory.model import load_model

    # adapter 可为 None：zero-shot 纯 base 评测也走这条 HF 路径，
    # 与 moe_lora adapter 评测同引擎（HF），保证 base vs adapter 可比。
    # 此时下面 is_moe_lora=False，走 load_model 仅加载 base。
    world_size, local_rank = _dist_info()
    distributed = world_size > 1
    model_args, data_args = build_model_args(args, infer_backend="huggingface")
    # 分布式(torchrun)：每个 rank 把整份模型放到自己的卡（数据并行），不能用 device_map="auto"
    # （会把 moe_lora 注入的共享模块拆到多卡导致跨设备调用）。
    model_args.device_map = {"": local_rank} if distributed else args.runtime_merge_device_map

    prompt_protocol = resolve_prompt_protocol(args)
    if prompt_protocol == "raw":
        prompt_token_ids, rendered_prompts, tokenizer, template = render_raw_prompts(eval_rows, model_args, data_args)
    else:
        prompt_token_ids, rendered_prompts, tokenizer, template = render_chat_prompts(
            eval_rows, model_args, data_args
        )

    prompt_token_ids, rendered_prompts = _enforce_prompt_budget(
        prompt_token_ids, rendered_prompts, tokenizer, args
    )
    prompt_lengths = [len(token_ids) for token_ids in prompt_token_ids]

    stop_token_ids = get_stop_token_ids(tokenizer, template)
    eos_token_id: int | list[int] | None = None
    if stop_token_ids:
        eos_token_id = stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids

    if args.hf_generate_batch_size < 1:
        raise ValueError("--hf-generate-batch-size must be >= 1.")

    finetuning_args = FinetuningArguments(stage="sft", finetuning_type="lora")
    model = None
    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token
    original_pad_token_id = tokenizer.pad_token_id
    predictions: list[str] = []

    # Detect custom lora_moe checkpoints (injected modules, not PEFT adapters).
    # PEFT/vLLM cannot load them; load the base in HF and re-inject + load the saved state.
    adapter_dir = model_args.adapter_name_or_path[0] if model_args.adapter_name_or_path else None
    def _has(fn):
        return adapter_dir is not None and os.path.exists(os.path.join(adapter_dir, fn))

    is_moe_lora = _has("moe_lora_state.safetensors")
    is_weight_peft = _has("weight_peft_state.safetensors")
    is_das_lora = _has("das_lora_state.safetensors")
    is_partial_ft = _has("partial_ft_state.safetensors")
    is_custom = is_moe_lora or is_weight_peft or is_das_lora or is_partial_ft

    try:
        if is_custom:
            from transformers import AutoModelForCausalLM

            # Custom adapters may share injected modules across layers; device_map="auto"
            # can split them across GPUs and break forward. Load the whole model onto
            # a single device for each data-parallel rank.
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": local_rank},
            )
            if is_moe_lora:
                from llamafactory.model.model_utils.moe_lora import load_moe_lora_state

                model = load_moe_lora_state(model, adapter_dir)
            elif is_weight_peft:
                from llamafactory.model.model_utils.weight_peft import load_weight_peft_state

                model = load_weight_peft_state(model, adapter_dir)
            elif is_das_lora:
                from llamafactory.model.model_utils.das_lora import load_das_lora_state

                model = load_das_lora_state(model, adapter_dir)
            else:
                from llamafactory.model.model_utils.partial_ft import load_partial_ft_state

                model = load_partial_ft_state(model, adapter_dir)
            model.eval()
        else:
            model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
        input_device = next(model.parameters()).device

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id for runtime merge eval.")

        tokenizer.padding_side = "left"
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        from tqdm import tqdm

        # 数据并行：每个 rank 只处理 [local_rank, local_rank+world_size, ...] 这一切片，
        # 各自单卡生成，最后 all_gather 合并回完整顺序。非分布式时 = 处理全部。
        my_indices = (
            list(range(local_rank, len(prompt_token_ids), world_size))
            if distributed
            else list(range(len(prompt_token_ids)))
        )
        local_pairs: list[tuple[int, str]] = []  # (global_idx, prediction_text)
        num_batches = (len(my_indices) + args.hf_generate_batch_size - 1) // args.hf_generate_batch_size
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(my_indices), args.hf_generate_batch_size),
                total=num_batches,
                desc=f"HF generate ({args.task}) rank{local_rank}",
                disable=not _is_main(),
            ):
                batch_idx = my_indices[start : start + args.hf_generate_batch_size]
                batch_token_ids = [prompt_token_ids[i] for i in batch_idx]
                batch = tokenizer.pad(
                    [{"input_ids": token_ids, "attention_mask": [1] * len(token_ids)} for token_ids in batch_token_ids],
                    padding=True,
                    return_tensors="pt",
                )
                prompt_width = batch["input_ids"].shape[1]
                batch = {name: tensor.to(input_device) for name, tensor in batch.items()}
                output_ids = model.generate(**batch, **generation_kwargs)
                generated_ids = output_ids[:, prompt_width:]
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                local_pairs.extend(zip(batch_idx, decoded))

        if distributed:
            import torch.distributed as dist

            torch.cuda.empty_cache()
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, local_pairs)
            merged: dict[int, str] = {}
            for sub in gathered:
                for gi, txt in sub:
                    merged[gi] = txt
            predictions = [merged[i] for i in range(len(prompt_token_ids))]
        else:
            predictions = [None] * len(prompt_token_ids)
            for gi, txt in local_pairs:
                predictions[gi] = txt
    finally:
        tokenizer.padding_side = original_padding_side
        if original_pad_token is not None:
            tokenizer.pad_token = original_pad_token
        tokenizer.pad_token_id = original_pad_token_id
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return predictions, rendered_prompts, prompt_lengths, prompt_protocol


def generate_predictions(
    eval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_args: Any,
    data_args: Any,
) -> tuple[list[str], list[str], list[int], str]:
    if args.runtime_merge:
        return generate_predictions_runtime_merge(eval_rows, args)

    return generate_predictions_vllm(eval_rows, args, model_args, data_args)


# Intent parsing/scoring is dependency-light and tested independently.
try:
    from .intent_metrics import post_process_intent, score_intent
except ImportError:
    from intent_metrics import post_process_intent, score_intent


def get_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    nested_path = args.esft_dir / "datasets" / "eval" / f"{args.task}.jsonl"
    flat_path = args.esft_dir / f"{args.task}.jsonl"
    eval_path = nested_path if nested_path.is_file() else flat_path
    if not eval_path.is_file():
        raise FileNotFoundError(
            f"ESFT evaluation data not found for {args.task}: expected {flat_path} "
            f"or {nested_path}. Run scripts/data/prepare_esft.py first."
        )
    rows = read_jsonl(eval_path)
    if args.debug:
        rows = rows[: args.debug_samples]
    return rows


def get_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions_file is not None:
        return args.predictions_file
    return args.output_dir / f"{args.task}_predictions.jsonl"


def get_metrics_path(args: argparse.Namespace) -> Path:
    if args.metrics_file is not None:
        return args.metrics_file
    return args.output_dir / f"{args.task}_metrics.json"


def build_prediction_rows(
    eval_rows: list[dict[str, Any]],
    raw_predictions: list[str],
    rendered_prompts: list[str],
    prompt_lengths: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "idx": row["idx"],
            "prompt": row["prompt"],
            "rendered_prompt": rendered_prompt,
            "prompt_length": prompt_length,
            "raw_prediction": raw_prediction,
            "raw_answers": row["raw_answers"],
        }
        for row, raw_prediction, rendered_prompt, prompt_length in zip(
            eval_rows, raw_predictions, rendered_prompts, prompt_lengths
        )
    ]


def score_prediction_row(task: str, row: dict[str, Any]) -> tuple[dict[str, Any], float]:
    if task not in TASKS:
        raise ValueError(f"Unsupported ESFT task: {task}; only intent is released.")
    prediction = post_process_intent(row["raw_prediction"])
    score = score_intent(prediction, row["raw_answers"][0])
    scored_row = dict(row)
    scored_row["prediction"] = prediction
    scored_row["score"] = score
    return scored_row, score


def generate_task(args: argparse.Namespace) -> dict[str, Any]:
    eval_rows = get_eval_rows(args)
    model_args, data_args = build_model_args(args)
    raw_predictions, rendered_prompts, prompt_lengths, prompt_protocol = generate_predictions(
        eval_rows, args, model_args, data_args
    )
    prediction_rows = build_prediction_rows(eval_rows, raw_predictions, rendered_prompts, prompt_lengths)

    predictions_path = get_predictions_path(args)
    # 数据并行下所有 rank 都已 all_gather 到完整结果，但只让 rank0 落盘，避免多进程写同一文件。
    if _is_main():
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(predictions_path, prediction_rows)

    return {
        "task": args.task,
        "stage": "generate",
        "num_examples": len(prediction_rows),
        "predictions_file": str(predictions_path),
        "config": {
            "model_name_or_path": str(args.model_name_or_path),
            "adapter_name_or_path": args.adapter_name_or_path,
            "template": args.template,
            "prompt_protocol": prompt_protocol,
            "runtime_merge": args.runtime_merge,
            "runtime_merge_device_map": args.runtime_merge_device_map if args.runtime_merge else None,
            "hf_generate_batch_size": args.hf_generate_batch_size if args.runtime_merge else None,
            "max_new_tokens": args.max_new_tokens,
            "cutoff_len": args.cutoff_len,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
    }


def score_task(args: argparse.Namespace) -> dict[str, Any]:
    # 评分只在 rank0 做（它持有 generate 阶段写出的预测文件）；其余 rank 直接跳过。
    if not _is_main():
        return {}
    predictions_path = get_predictions_path(args)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}")

    prediction_rows = read_jsonl(predictions_path)

    scored_rows = []
    scores = []
    for row in prediction_rows:
        scored_row, score = score_prediction_row(args.task, row)
        scored_rows.append(scored_row)
        scores.append(score)

    average = sum(scores) / len(scores) if scores else 0.0
    metrics_path = get_metrics_path(args)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "task": args.task,
        "stage": "score",
        "num_examples": len(scored_rows),
        "average_score": average,
        "predictions_file": str(predictions_path),
        "metrics_file": str(metrics_path),
    }

    write_jsonl(predictions_path, scored_rows)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "generate":
        return generate_task(args)

    if args.mode == "score":
        return score_task(args)

    generate_task(args)
    return score_task(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ESFT Intent with MoEFactory.")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=MODES, default="all")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Data root (default: MOEFACTORY_DATA_ROOT or repository data/); ESFT files are read from <root>/esft_eval.",
    )
    parser.add_argument(
        "--esft-dir",
        type=Path,
        default=None,
        help="Compatibility override containing <task>.jsonl, or an ESFT checkout containing datasets/eval/.",
    )
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--adapter-name-or-path", default=None)
    parser.add_argument(
        "--flash-attn",
        dest="flash_attn",
        choices=["auto", "disabled", "sdpa", "fa2", "fa3"],
        default="auto",
        help="attention 实现(透传 ModelArguments.flash_attn)。'disabled'=eager, "
        "用于 gpt-oss flex attention 长序列 illegal memory access 的逃生口。",
    )
    parser.add_argument("--template", default="esft_raw")
    parser.add_argument("--prompt-protocol", choices=PROMPT_PROTOCOLS, default="auto")
    parser.add_argument("--runtime-merge", action="store_true")
    parser.add_argument("--runtime-merge-device-map", default="auto")
    parser.add_argument("--hf-generate-batch-size", type=int, default=1)
    parser.add_argument("--cutoff-len", type=int, default=3584)
    parser.add_argument("--prompt-truncate-side", choices=["none", "left", "right"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path, default=None)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=8)
    parser.add_argument("--vllm-config", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.esft_dir = (
        args.esft_dir.expanduser() if args.esft_dir is not None else args.data_root / "esft_eval"
    ).resolve()
    if args.model_name_or_path.exists():
        args.model_name_or_path = args.model_name_or_path.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.predictions_file is not None:
        args.predictions_file = args.predictions_file.resolve()
    if args.metrics_file is not None:
        args.metrics_file = args.metrics_file.resolve()

    world_size, local_rank = _dist_info()
    distributed = world_size > 1
    if distributed:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    result = run_task(args)

    if distributed:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    if _is_main():
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
