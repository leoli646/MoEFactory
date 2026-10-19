"""vLLM-based generative evaluation on MMLU-Pro and IFEval for base / moe_lora weights.

设计目标：
  - MMLU-Pro 与 `eval_mmlu_pro.py --eval_mode generation --gen_style direct` 协议逐字节一致
    （同 prompt 构造 / 同 4 级抽取器 / 0-shot / greedy / max_new_tokens=32），
    数字可与既有 HF generation 链（router_interventions_gen）直接对比。
  - IFEval 用官方 google-research 协议：裸 prompt、greedy，strict + loose 双口径，
    checker 为 vendored `eval_scripts/ifeval/`（本地 nltk punkt，离线可跑）。
  - adapter 通过 `vllm_moe_lora.activate()` 注入（attn/router 增量合并 + dense FFN 旁路）。
  - 单卡 vLLM 实例（H200 放得下三个模型），多卡用数据分片并行：
    每张卡起一个进程 `--shard_idx i --num_shards N`，之后 `--merge_shards` 汇总打分。

示例::

    # 单卡跑一个分片
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_vllm_general.py \
        --task mmlu_pro --base_model $QWEN3 --adapter_path saves/... \
        --out_dir results/vllm_general/intent/qwen3/arm/mmlu_pro --shard_idx 0 --num_shards 4

    # 全部分片完成后汇总（CPU，不加载模型）
    python eval_scripts/eval_vllm_general.py --task mmlu_pro --merge_shards \
        --base_model $QWEN3 --out_dir ... --num_shards 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_DATA_ROOT = os.environ.get("MOEFACTORY_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))
sys.path.insert(0, _THIS_DIR)

# 必须在 import vllm 之前：overlay 注册/monkey-patch 只在本进程有效
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_datasets"))
os.environ.setdefault("NLTK_DATA", os.path.join(_DATA_ROOT, "nltk_data"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval_mmlu_pro import (  # noqa: E402
    LETTERS, build_shot_bank, extract_choice, format_prompt_generation, load_split,
    normalize_answer, summarize,
)


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------


def build_mmlu_pro(args) -> list[dict[str, Any]]:
    rows = load_split(args.data_dir, "test")
    if args.limit:
        rows = rows[: args.limit]
    shot_bank = None
    if args.n_shot > 0:
        shot_bank = build_shot_bank(load_split(args.data_dir, "validation"))
    items = []
    for idx, row in enumerate(rows):
        items.append({
            "idx": idx,
            "row": row,
            "prompt": format_prompt_generation(row, shot_bank, args.n_shot, args.gen_style),
        })
    return items


def build_ifeval(args) -> list[dict[str, Any]]:
    items = []
    with open(args.ifeval_data) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            items.append({
                "idx": idx,
                "key": ex["key"],
                "prompt": ex["prompt"],
                "instruction_id_list": ex["instruction_id_list"],
                "kwargs": ex["kwargs"],
            })
    if args.limit:
        items = items[: args.limit]
    return items


# ---------------------------------------------------------------------------
# 生成（worker 模式）
# ---------------------------------------------------------------------------


def _extra_llm_kwargs(base_model: str) -> dict:
    """gemma4 是多模态壳（Gemma4ForConditionalGeneration），但本链只做纯文本评测。
    vLLM 会对 vision encoder 跑 profiling，而 transformers backend 的
    get_multimodal_embeddings 返回的不是纯 embedding，profile_run 的 sanity check
    直接 assert 失败。skip_mm_profiling=True 跳过 encoder profiling（无图输入，
    KV cache 预算按纯文本估即可）。"""
    import json as _json
    cfg_path = os.path.join(base_model, "config.json")
    try:
        with open(cfg_path) as f:
            arch = (_json.load(f).get("architectures") or [""])[0]
    except Exception:
        arch = ""
    if "Gemma4" in arch:
        return {"skip_mm_profiling": True}
    return {}


def collect_stop_ids(base_model: str) -> list[int]:
    """镜像 eval_scripts/_common.py::_collect_eos_ids 的口径（tokenizer eos +
    generation_config eos + <|im_end|>），防 Qwen3 生成 <|im_end|> 后不停。"""
    from transformers import AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    try:
        gc = GenerationConfig.from_pretrained(base_model)
        eos = gc.eos_token_id
        if eos is not None:
            ids.update(int(i) for i in (eos if isinstance(eos, (list, tuple)) else [eos]))
    except Exception:
        pass
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0 and im_end != tok.unk_token_id:
        ids.add(int(im_end))
    return sorted(ids)


def run_worker(args, items: list[dict[str, Any]]) -> None:
    shard = items[args.shard_idx :: args.num_shards]
    print(
        f"[worker] task={args.task} engine={args.engine} shard {args.shard_idx}/{args.num_shards}: "
        f"{len(shard)}/{len(items)} prompts | adapter={args.adapter_path or 'BASE'}"
    )
    if not shard:
        _write_shard(args, [])
        return

    if args.engine == "hf":
        completions, truncated_flags = _generate_hf(args, [it["prompt"] for it in shard])
    else:
        completions, truncated_flags = _generate_vllm(args, [it["prompt"] for it in shard])

    records = []
    for it, completion, truncated in zip(shard, completions, truncated_flags):
        if args.task == "mmlu_pro":
            row = it["row"]
            labels = LETTERS[: len(row["options"])]
            prediction, rule = extract_choice(completion, labels)
            gold = normalize_answer(row)
            records.append({
                "idx": it["idx"],
                "question_id": row.get("question_id"),
                "category": str(row.get("category", "unknown")),
                "src": str(row.get("src", "unknown")),
                "split": "test",
                "question": row["question"],
                "options": row["options"],
                "answer": gold,
                "answer_index": row.get("answer_index"),
                "prediction": prediction,
                "correct": prediction == gold,
                "completion": completion,
                "extract_rule": rule,
                "truncated": truncated,
                "score_type": f"generation_{args.gen_style}",
                "n_shot": args.n_shot,
                "engine": args.engine,
                "hf_batch_size": args.hf_batch_size if args.engine == "hf" else None,
                "prompt": None,
            })
        else:  # ifeval
            records.append({
                "idx": it["idx"],
                "key": it["key"],
                "prompt": it["prompt"],
                "response": completion,
                "truncated": truncated,
                "instruction_id_list": it["instruction_id_list"],
                "kwargs": it["kwargs"],
                "engine": args.engine,
                "hf_batch_size": args.hf_batch_size if args.engine == "hf" else None,
            })
    _write_shard(args, records)


def _generate_vllm(args, prompts: list[str]) -> tuple[list[str], list[bool]]:
    import vllm_moe_lora
    if args.adapter_path:
        vllm_moe_lora.activate(args.adapter_path)
    else:
        vllm_moe_lora.register_models()  # base 臂也要：compat shim + buffer 权重拦截
    # transformers 5.6 重排过 config 属性（rope_theta 等），vLLM getattr 默认值
    # 会静默错数（qwen3 曾因此 -8.8pt），启动前对照 config.json 逐标量断言。
    vllm_moe_lora.verify_config_fidelity(args.base_model)

    from vllm import LLM, SamplingParams

    max_tokens = args.gen_max_new_tokens
    stop_ids = collect_stop_ids(args.base_model)
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=not args.no_enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=0,
        disable_log_stats=True,
        **_extra_llm_kwargs(args.base_model),
    )
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=stop_ids,
        # 左截断保住尾部 anchor（与 HF 路径 truncation_side=left 一致）
        truncate_prompt_tokens=max(args.max_model_len - max_tokens, 512),
    )
    outputs = llm.generate(prompts, sp)
    completions = [o.outputs[0].text for o in outputs]
    truncated = [o.outputs[0].finish_reason == "length" for o in outputs]
    return completions, truncated


def _generate_hf(args, prompts: list[str]) -> tuple[list[str], list[bool]]:
    """HF fallback（gemma4 用）：vLLM transformers backend 撑不住 gemma4 的
    异构 KV 头（sliding 8 头 / full_attention 2 头 K=V）+ scaled embedding +
    per-layer rope（2026-07-21 smoke 三轮实测），退回训练同款 HF 加载路径。
    协议（greedy / 左截断 / eos 并集）与 vLLM 路径一致，engine 字段留痕。"""
    from _common import load  # noqa: PLC0415  (需要 DISABLE_VERSION_CHECK=1)
    from eval_mmlu_pro import generate_rows

    class _A:
        base_model = args.base_model
        adapter_path = args.adapter_path
        dtype = "bfloat16"
        trust_remote_code = True

    tokenizer, model = load(_A)
    max_input_len = max(args.max_model_len - args.gen_max_new_tokens, 512)
    completions: list[str] = []
    truncated: list[bool] = []
    bsz = args.hf_batch_size
    for start in range(0, len(prompts), bsz):
        batch = prompts[start : start + bsz]
        comps, truncs = generate_rows(model, tokenizer, batch, args.gen_max_new_tokens, max_input_len)
        completions.extend(comps)
        truncated.extend(truncs)
        done = min(start + bsz, len(prompts))
        print(f"[worker-hf] {done}/{len(prompts)}", flush=True)
    return completions, truncated


def _shard_path(out_dir: str, i: int, n: int) -> Path:
    return Path(out_dir) / f"result.shard{i}of{n}.json"


def _write_shard(args, records: list[dict[str, Any]]) -> None:
    out = _shard_path(args.out_dir, args.shard_idx, args.num_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[worker] wrote {len(records)} records -> {out}")


# ---------------------------------------------------------------------------
# 汇总打分（merge 模式，CPU）
# ---------------------------------------------------------------------------


def score_ifeval(records: list[dict[str, Any]]) -> dict[str, Any]:
    from ifeval import evaluation_lib

    strict_out, loose_out = [], []
    for r in records:
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in r["kwargs"]]
        inp = evaluation_lib.InputExample(
            key=r["key"], instruction_id_list=r["instruction_id_list"],
            prompt=r["prompt"], kwargs=kwargs,
        )
        prompt_to_response = {r["prompt"]: r["response"]}
        strict_out.append(evaluation_lib.test_instruction_following_strict(inp, prompt_to_response))
        loose_out.append(evaluation_lib.test_instruction_following_loose(inp, prompt_to_response))

    def agg(outputs):
        n_prompt = len(outputs)
        n_prompt_ok = sum(int(o.follow_all_instructions) for o in outputs)
        n_inst = sum(len(o.instruction_id_list) for o in outputs)
        n_inst_ok = sum(sum(o.follow_instruction_list) for o in outputs)
        return {
            "prompt_level_accuracy": n_prompt_ok / n_prompt if n_prompt else 0.0,
            "instruction_level_accuracy": n_inst_ok / n_inst if n_inst else 0.0,
            "num_prompts": n_prompt,
            "num_instructions": n_inst,
        }

    for r, s, l in zip(records, strict_out, loose_out):
        r["follow_instruction_list_strict"] = s.follow_instruction_list
        r["follow_all_strict"] = s.follow_all_instructions
        r["follow_instruction_list_loose"] = l.follow_instruction_list
        r["follow_all_loose"] = l.follow_all_instructions

    n = len(records)
    return {
        "task": "ifeval",
        "num_examples": n,
        "strict": agg(strict_out),
        "loose": agg(loose_out),
        "truncated_rate": (sum(int(bool(r.get("truncated"))) for r in records) / n) if n else 0.0,
        "empty_response_rate": (sum(int(not str(r.get("response", "")).strip()) for r in records) / n) if n else 0.0,
    }


def run_merge(args) -> None:
    out_dir = Path(args.out_dir)
    shards = []
    for i in range(args.num_shards):
        p = _shard_path(args.out_dir, i, args.num_shards)
        if not p.exists():
            raise FileNotFoundError(f"缺分片 {p}（对应 worker 失败或未跑完）")
        with p.open() as f:
            shards.extend(json.loads(line) for line in f if line.strip())
    shards.sort(key=lambda r: r["idx"])
    idxs = [r["idx"] for r in shards]
    if len(set(idxs)) != len(idxs):
        raise RuntimeError("分片间存在重复 idx，检查 num_shards 是否与 worker 一致")
    # engine 以记录为准（merge 进程未必带 --engine；worker 写入的才是真值）
    rec_engines = {str(r.get("engine")) for r in shards if r.get("engine")}
    engine = sorted(rec_engines)[0] if len(rec_engines) == 1 else (args.engine or "vllm")
    # hf batch 影响 batch 内动态 padding 组成，臂间/分片间必须一致才可比
    rec_bsz = {r["hf_batch_size"] for r in shards if r.get("hf_batch_size") is not None}
    if len(rec_bsz) > 1:
        raise RuntimeError(f"分片间 hf_batch_size 不一致：{sorted(rec_bsz)}，结果不可比，请统一后重跑")
    hf_batch_size = next(iter(rec_bsz)) if rec_bsz else None

    if args.task == "mmlu_pro":
        summary = summarize(shards)
        summary["engine"] = engine
        gen = summary.get("generation", {})
        print(
            f"\nMMLU-Pro [{engine} generation_direct]: acc={summary['accuracy']:.4f} "
            f"({sum(int(r['correct']) for r in shards)}/{len(shards)}) "
            f"parse_failure={gen.get('parse_failure_rate', 0):.4f} "
            f"truncated={gen.get('truncated_rate', 0):.4f}"
        )
    else:
        summary = score_ifeval(shards)
        summary["engine"] = engine
        print(
            f"\nIFEval [{engine}]: strict prompt={summary['strict']['prompt_level_accuracy']:.4f} "
            f"inst={summary['strict']['instruction_level_accuracy']:.4f} | "
            f"loose prompt={summary['loose']['prompt_level_accuracy']:.4f} "
            f"inst={summary['loose']['instruction_level_accuracy']:.4f} "
            f"(n={summary['num_examples']})"
        )

    summary["base_model"] = args.base_model
    summary["adapter_path"] = args.adapter_path
    summary["hf_batch_size"] = hf_batch_size
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        for r in shards:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_dir}/{{result.json,summary.json}}")
    if not args.keep_shards:
        for i in range(args.num_shards):
            _shard_path(args.out_dir, i, args.num_shards).unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["mmlu_pro", "ifeval"])
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None, help="moe_lora 目录；缺省评 base")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge_shards", action="store_true", help="仅汇总打分，不加载模型")
    p.add_argument("--keep_shards", action="store_true")
    p.add_argument("--data_dir", default=os.path.join(_DATA_ROOT, "MMLU-Pro"))
    p.add_argument("--ifeval_data", default=os.path.join(_DATA_ROOT, "general", "IFEval", "ifeval_input_data.jsonl"))
    p.add_argument("--gen_max_new_tokens", type=int, default=None,
                   help="默认 mmlu_pro direct=32（对齐 HF gen_direct 链）/ cot=1024，ifeval=1280")
    p.add_argument("--gen_style", default="direct", choices=["direct", "cot"],
                   help="mmlu_pro 专用。direct=只答字母（默认，与既有链可比）；"
                        "cot=think step by step + \"the answer is (X)\"（配合 --n_shot 5 "
                        "≈ 官方 MMLU-Pro 协议，用于对照官方报告数字）")
    p.add_argument("--n_shot", type=int, default=0,
                   help="mmlu_pro 专用：category 匹配的 validation few-shot 数（官方口径=5）")
    p.add_argument("--engine", default="vllm", choices=["vllm", "hf"],
                   help="hf = 训练同款 HF 加载路径（gemma4 用：vLLM transformers "
                        "backend 不支持其异构 KV 头/K=V full-attention 层）")
    p.add_argument("--hf_batch_size", type=int, default=8, help="engine=hf 的生成 batch")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument("--no_enforce_eager", action="store_true",
                   help="允许 torch.compile（默认 eager：dense 旁路 wrapper 最稳）")
    args = p.parse_args()

    if args.gen_max_new_tokens is None:
        if args.task == "mmlu_pro":
            args.gen_max_new_tokens = 32 if args.gen_style == "direct" else 1024
        else:
            args.gen_max_new_tokens = 1280
    if args.adapter_path and args.tensor_parallel_size != 1:
        raise SystemExit("adapter overlay 目前只验证过 tensor_parallel_size=1（单卡 + 数据分片并行）")
    if not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(f"非法分片 {args.shard_idx}/{args.num_shards}")

    if args.merge_shards:
        run_merge(args)
        return

    items = build_mmlu_pro(args) if args.task == "mmlu_pro" else build_ifeval(args)
    run_worker(args, items)


if __name__ == "__main__":
    main()
