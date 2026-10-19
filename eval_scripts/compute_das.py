"""Compute Domain Advantage Score (DAS) for MoE experts.

Simplified from "Exploring Expert Concentration" (ICLR 2026 submission):
- Skips Stage 1 (attention + router fine-tuning)
- Computes DAS on base model's pre-trained routing
- DAS(D_d, D_g) = mean(routing_score_d) - mean(routing_score_g)
- Selects top-k experts per layer by DAS

Usage:
    python eval_scripts/compute_das.py \\
        --base_model allenai/OLMoE-1B-7B-0924 \\
        --domain_dataset oumi-ai/MetaMathQA-R1 \\
        --general_dataset cais/mmlu \\
        --top_k 4 \\
        --max_samples 200 \\
        --output eval_results/das_math.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def _load_local_records(path, split=None):
    """Load a local JSON array or JSONL file.

    Some repo files use a .json suffix while storing one JSON object per line.
    Fall back to JSONL parsing when json.load sees extra top-level objects.
    """
    import json as _json

    def read_jsonl():
        rows = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(_json.loads(line))
                except _json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
        return rows

    if path.endswith(".jsonl"):
        return read_jsonl()

    with open(path, encoding="utf-8") as f:
        try:
            data = _json.load(f)
        except _json.JSONDecodeError as exc:
            if exc.msg != "Extra data":
                raise
            data = read_jsonl()

    if isinstance(data, dict):
        if split and split in data and isinstance(data[split], list):
            data = data[split]
        elif "data" in data and isinstance(data["data"], list):
            data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Local dataset must be a list or JSONL records: {path}")
    return data


def _locate_decoder_layers(model):
    """定位 decoder 层列表,兼容纯文本 LM 和多模态 LM。
    与 model_utils/moe_lora.py 中的同名函数保持一致的 fallback 顺序。
    """
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers
    if base is not None:
        for sub_name in ("text_model", "language_model"):
            sub = getattr(base, sub_name, None)
            if sub is not None and hasattr(sub, "layers"):
                return sub.layers
            if sub is not None:
                inner = getattr(sub, "model", None)
                if inner is not None and hasattr(inner, "layers"):
                    return inner.layers
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm.layers
        inner = getattr(lm, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    raise RuntimeError("Cannot locate decoder layers in model.")


def _get_router(block):
    return getattr(block, "gate", None) or getattr(block, "router", None)


def _router_projection(router):
    if isinstance(getattr(router, "weight", None), torch.Tensor):
        return router
    projection = getattr(router, "proj", None)
    if isinstance(getattr(projection, "weight", None), torch.Tensor):
        return projection
    return None


def _router_logits(router, hidden_states):
    """Compute dense router logits, including Gemma 4's router preprocessing."""
    projection = getattr(router, "proj", None)
    if projection is not None and isinstance(getattr(projection, "weight", None), torch.Tensor):
        router_input = hidden_states
        norm = getattr(router, "norm", None)
        if norm is not None:
            router_input = norm(router_input)
        scale = getattr(router, "scale", None)
        if scale is not None:
            router_input = router_input * scale
        scalar_root_size = getattr(router, "scalar_root_size", None)
        if scalar_root_size is not None:
            router_input = router_input * scalar_root_size
        return projection(router_input)
    projection = _router_projection(router)
    if projection is None:
        raise RuntimeError(f"Unsupported router module: {type(router).__name__}")
    return F.linear(hidden_states, projection.weight, getattr(projection, "bias", None))


def _find_moe_layers(model):
    """Return (decoder index, block, router, hook target) for each routed layer."""
    result = []
    for layer_idx, layer in enumerate(_locate_decoder_layers(model)):
        # Gemma 4 owns its experts/router directly on the decoder layer.
        if hasattr(layer, "experts") and hasattr(layer, "router"):
            block = layer
        else:
            block = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        if block is None:
            continue
        router = _get_router(block)
        if router is None or _router_projection(router) is None:
            continue
        # Legacy DeepSeek implementations may bypass gate.forward(), so retain the
        # block pre-hook there. New `.router` models call the router module directly,
        # which provides the exact post-attention Gemma hidden states.
        hook_target = router if hasattr(block, "router") else block
        result.append((layer_idx, block, router, hook_target))
    return result


@torch.no_grad()
def collect_routing_scores(model, tokenizer, samples, max_seq_len=512, desc=""):
    """Run forward on samples; for each MoE block, accumulate per-expert
    mean routing score. Returns tensor [n_layers, n_experts]."""
    moe_layers = _find_moe_layers(model)

    if not moe_layers:
        raise RuntimeError("No MoE layers with a supported .gate/.router projection found.")

    # n_experts from the router projection weight shape
    n_experts = _router_projection(moe_layers[0][2]).weight.shape[0]
    n_layers = len(moe_layers)
    device = next(model.parameters()).device

    score_sum = torch.zeros(n_layers, n_experts, device=device, dtype=torch.float32)
    token_count = torch.zeros(n_layers, device=device, dtype=torch.float32)

    # Forward pre-hook: legacy `.gate` models hook the MoE block because some
    # DeepSeek implementations read gate.weight directly. `.router` models hook
    # the router itself so Gemma 4 uses the exact post-attention router input.
    # 这种方式兼容多种 MoE 实现:
    #   - OLMoE / Qwen3-MoE: 它们的 MoE block forward 里也会先调 gate.linear
    #   - DeepSeek-V2 (transformers 5.x native): forward 直接用 F.linear(h, gate.weight)
    #     不走 gate.forward(),所以不能挂在 mlp.gate 上(hook 永远不触发)
    # 改挂在 mlp 上,统一处理。
    handles = []
    for li, (_orig_idx, _block, router, hook_target) in enumerate(moe_layers):
        def make_hook(layer_idx, router_module):
            def hook(module, input):
                h = input[0]
                if h.dim() == 3:
                    h = h.reshape(-1, h.shape[-1])
                logits = _router_logits(router_module, h).float()
                scores = F.softmax(logits, dim=-1)
                # device_map="auto" 下,该层可能在 cuda:N 而 score_sum 在 cuda:0,需 to(device)
                score_sum[layer_idx] += scores.sum(dim=0).to(score_sum.device)
                token_count[layer_idx] += scores.shape[0]
            return hook
        h = hook_target.register_forward_pre_hook(make_hook(li, router))
        handles.append(h)

    try:
        for text in tqdm(samples, desc=desc):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq_len
            ).to(device)
            _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Normalize: mean score per expert across all tokens
    mean_scores = score_sum / token_count.unsqueeze(-1).clamp(min=1)
    return mean_scores.cpu()


def extract_text(sample, dataset_name):
    """Extract a representative text field from various dataset formats."""
    if "messages" in sample:
        # ShareGPT / chat format (role/content)
        msgs = sample["messages"]
        if isinstance(msgs, list) and len(msgs) > 0:
            return " ".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    if "conversations" in sample:
        # LLaVA-style sharegpt (from/value),去掉 <image> 占位符
        msgs = sample["conversations"]
        if isinstance(msgs, list) and len(msgs) > 0:
            return " ".join(
                str(m.get("value", "")) for m in msgs if isinstance(m, dict)
            ).replace("<image>", "").strip()
    # MMLU-style (question + choices,先检查 choices,因为 answer 可能是 int label)
    if "question" in sample and "choices" in sample:
        choices = sample["choices"]
        q = str(sample["question"])
        if isinstance(choices, list):
            return q + " " + " ".join(str(c) for c in choices)
        return q
    # Generic question + answer(string answer 才行)
    if "question" in sample and "answer" in sample:
        ans = sample["answer"]
        if isinstance(ans, str):
            return str(sample["question"]) + " " + ans
        return str(sample["question"])
    if "prompt" in sample and "response" in sample:
        return str(sample["prompt"]) + " " + str(sample["response"])
    if "input" in sample and "output" in sample:  # commonsense_test merged.json 等 {input,output}
        return str(sample["input"]) + " " + str(sample["output"])
    if "instruction" in sample:
        return str(sample["instruction"]) + " " + str(sample.get("output", ""))
    if "text" in sample:
        return str(sample["text"])
    if "problem" in sample:
        return str(sample["problem"])
    # Fallback: stringify whole sample
    return str(sample)[:1000]


def load_samples(dataset_name, split, n, dataset_config=None):
    """Load N text samples from a dataset.

    Supports:
    - Local JSON file path (.json / .jsonl) — sharegpt 格式 {"messages": [...]} 或 alpaca
    - HuggingFace dataset name (e.g. "cais/mmlu")
    """
    # Local file path
    if os.path.exists(dataset_name) and dataset_name.endswith((".json", ".jsonl")):
        print(f"  [local file] {dataset_name}")
        data = _load_local_records(dataset_name, split)
        # Shuffle and select
        import random
        random.seed(42)
        if len(data) > n:
            data = random.sample(data, n)
        return [extract_text(s, dataset_name) for s in data]

    # HuggingFace dataset
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
    return [extract_text(s, dataset_name) for s in ds]


def load_model_for_das(base_model):
    """Load text or image-text MoE models with train-time quantization semantics."""
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    load_class = AutoModelForImageTextToText if config.model_type == "gemma4" else AutoModelForCausalLM
    init_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    quantization_config = getattr(config, "quantization_config", None) or {}
    if quantization_config.get("quant_method") == "mxfp4":
        from transformers import Mxfp4Config

        init_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        init_kwargs["ignore_mismatched_sizes"] = True
    return load_class.from_pretrained(base_model, **init_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--domain_dataset", required=True,
                        help="HF dataset name for domain data, e.g. oumi-ai/MetaMathQA-R1")
    parser.add_argument("--domain_config", default=None,
                        help="Optional dataset config name")
    parser.add_argument("--domain_split", default="train")
    parser.add_argument("--general_dataset", default="cais/mmlu")
    parser.add_argument("--general_config", default="all")
    parser.add_argument("--general_split", default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=4,
                        help="Top-K experts to select per layer (DAS-LoRA hyperparameter)")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = load_model_for_das(args.base_model)
    model.eval()

    print(f"Loading domain dataset: {args.domain_dataset} (split={args.domain_split})")
    domain_samples = load_samples(args.domain_dataset, args.domain_split,
                                   args.max_samples, args.domain_config)
    print(f"Loaded {len(domain_samples)} domain samples")

    print(f"Loading general dataset: {args.general_dataset} (split={args.general_split})")
    general_samples = load_samples(args.general_dataset, args.general_split,
                                    args.max_samples, args.general_config)
    print(f"Loaded {len(general_samples)} general samples")

    print("\n=== Computing routing scores on DOMAIN data ===")
    domain_scores = collect_routing_scores(
        model, tokenizer, domain_samples, args.max_seq_len, desc="domain"
    )

    print("\n=== Computing routing scores on GENERAL data ===")
    general_scores = collect_routing_scores(
        model, tokenizer, general_samples, args.max_seq_len, desc="general"
    )

    # DAS = domain - general
    das = domain_scores - general_scores  # [n_layers, n_experts]

    # Per-layer top-k selection
    top_k_experts = []
    for layer_idx in range(das.shape[0]):
        top_indices = das[layer_idx].argsort(descending=True)[: args.top_k].tolist()
        top_k_experts.append(top_indices)

    # Cumulative DAS@k (a sanity / informativeness metric from the paper)
    pos_das = das.clamp(min=0)
    cumulative_das_at_k = []
    for layer_idx in range(das.shape[0]):
        layer_pos = pos_das[layer_idx]
        total = layer_pos.sum().item()
        if total > 0:
            top_pos = layer_pos.sort(descending=True).values[: args.top_k].sum().item()
            cumulative_das_at_k.append(top_pos / total)
        else:
            cumulative_das_at_k.append(0.0)

    output = {
        "base_model": args.base_model,
        "domain_dataset": args.domain_dataset,
        "general_dataset": args.general_dataset,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "selected_experts_per_layer": top_k_experts,
        "das_per_layer": das.tolist(),
        "cumulative_das_at_k": cumulative_das_at_k,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== DAS computation complete ===")
    print(f"Saved to: {args.output}")
    print(f"Selected experts per layer (top-{args.top_k}):")
    for li, experts in enumerate(top_k_experts):
        print(f"  Layer {li:2d}: {experts}")
    avg_cdas = sum(cumulative_das_at_k) / len(cumulative_das_at_k)
    print(f"\nAverage C-DAS@{args.top_k}: {avg_cdas:.3f}")
    print(f"  (higher = more concentrated; ~0.5+ is typical for fine-tunable signal)")


if __name__ == "__main__":
    main()
