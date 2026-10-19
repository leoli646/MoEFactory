"""vLLM-accelerated forced-choice **logprob** MMLU-Pro evaluation (遗忘探针主口径).

与 `eval_mmlu_pro.py --eval_mode logprob`（HF 路径）**逐 token 同口径**：
  - 同样的裸 prompt（`format_prompt`，不套 chat template，三模型统一）；
  - 对每个候选标签 A..J，打分 = 连续 token `" LABEL"` 的平均 log-prob；
  - argmax 取预测。三模型的 " A".." J" 都是单 token（已核验），所以每题
    10 个候选只在最后 1 个 token 不同 → vLLM prefix caching 让 prompt 前缀 KV
    只算一次，10 候选的增量成本≈10 个单 token（HF score_rows 是每候选重算整段
    prompt attention）。这正是 vLLM 相对 HF 的加速点。

数值对齐要点：
  - 直接把 `encode_continuation` 产出的 token_ids（prompt_ids + " LABEL" 的 ids，
    左截断保尾部）喂给 vLLM `TokensPrompt`，绕开字符串再 tokenize 的边界差异；
  - vLLM `prompt_logprobs[pos]` = 用 pos-1 预测 pos 处 token 的 log-prob，与 HF
    `log_softmax(logits[pos-1])[token]` 完全对应；两侧都是完整词表 log_softmax。
  - bf16 kernel 归约顺序差异会让近平手题的 argmax 偶尔翻（与 generation equiv 同源），
    用 `--equiv_vs_hf` 抽样对拍守门（argmax 一致率）。

gemma4：vLLM transformers backend 不支持其异构 KV 头（见 vllm_moe_lora 裁决），
`--engine hf` 退回 `eval_mmlu_pro.score_rows`（同口径，engine 字段留痕）。

用法::

    # 单卡一个分片（vLLM）
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_vllm_mmlu_logprob.py \
        --base_model $QWEN3 --adapter_path saves/... \
        --out_dir results/.../mmlu_pro_logprob --shard_idx 0 --num_shards 4

    # 全部分片完成后汇总（CPU）
    python eval_scripts/eval_vllm_mmlu_logprob.py --merge_shards \
        --base_model $QWEN3 --out_dir ... --num_shards 4

    # 对拍 HF（同批题 vLLM vs HF argmax 一致率，单卡两阶段子进程）
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/eval_vllm_mmlu_logprob.py \
        --base_model $QWEN3 --adapter_path saves/... --equiv_vs_hf --n_prompts 32
"""

from __future__ import annotations

import argparse
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval_mmlu_pro import (  # noqa: E402
    LETTERS, build_shot_bank, encode_continuation, format_prompt, load_split,
    normalize_answer, score_rows, summarize,
)


# ---------------------------------------------------------------------------
# 数据准备（与 eval_mmlu_pro logprob 口径一致：format_prompt 裸 prompt）
# ---------------------------------------------------------------------------


def build_items(args) -> list[dict[str, Any]]:
    rows = load_split(args.data_dir, args.split)
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
            "prompt": format_prompt(row, shot_bank, args.n_shot),
        })
    return items


def _extra_llm_kwargs(base_model: str) -> dict:
    """gemma4 多模态壳跳过 vision encoder profiling（纯文本评测）。"""
    cfg_path = os.path.join(base_model, "config.json")
    try:
        with open(cfg_path) as f:
            arch = (json.load(f).get("architectures") or [""])[0]
    except Exception:
        arch = ""
    return {"skip_mm_profiling": True} if "Gemma4" in arch else {}


# ---------------------------------------------------------------------------
# 打分：vLLM prompt_logprobs（每题每标签一个序列，共享 prompt 前缀）
# ---------------------------------------------------------------------------


def _score_vllm(args, shard: list[dict[str, Any]]) -> list[dict[str, float]]:
    import vllm_moe_lora
    if args.adapter_path:
        vllm_moe_lora.activate(args.adapter_path)
    else:
        vllm_moe_lora.register_models()
    vllm_moe_lora.verify_config_fidelity(args.base_model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 构造每 (题, 标签) 的 token_ids（与 HF encode_continuation 逐 token 一致）
    seqs: list[list[int]] = []
    meta: list[tuple[int, str, int, int, list[int]]] = []  # (shard_idx, label, plen, clen, ids)
    for si, it in enumerate(shard):
        labels = LETTERS[: len(it["row"]["options"])]
        for label in labels:
            ids, plen, clen = encode_continuation(tokenizer, it["prompt"], label, args.max_length)
            seqs.append(ids)
            meta.append((si, label, plen, clen, ids))

    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=not args.no_enforce_eager,
        max_model_len=args.max_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=not args.no_prefix_caching,   # 同题 10 候选共享 prompt 前缀
        trust_remote_code=True,
        seed=0,
        disable_log_stats=True,
        **_extra_llm_kwargs(args.base_model),
    )
    # prompt_logprobs=0：只要 prompt 里实际 token 的 logprob（vLLM 保证返回，
    # 即便不在 top-k）。max_tokens=1：vLLM 要求至少生成 1 token，我们只读 prompt 侧。
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], sp)

    scores: list[dict[str, float]] = [dict() for _ in shard]
    for out, (si, label, plen, clen, ids) in zip(outputs, meta):
        pl = out.prompt_logprobs  # list[Optional[dict[int, Logprob]]], 位置 0 为 None
        total = 0.0
        for j in range(clen):
            pos = plen + j
            token_id = ids[pos]
            entry = pl[pos]
            if entry is None or token_id not in entry:
                raise RuntimeError(
                    f"prompt_logprobs 缺 token {token_id} @pos {pos}（label={label}）；"
                    f"提高 prompt_logprobs 或检查 tokenization。"
                )
            total += entry[token_id].logprob
        scores[si][label] = total / clen
    return scores


def _score_hf(args, shard: list[dict[str, Any]]) -> list[dict[str, float]]:
    """gemma4 退路 / equiv 参照：训练同款 HF 加载 + eval_mmlu_pro.score_rows（同口径）。

    --hf_fp32：把 score_rows 内部的 log_softmax 提到 fp32（monkey-patch）。HF 现有
    链是 bf16 log_softmax（outputs.logits 未 upcast），对选项挤在一起的难题会因
    bf16 舍入把 argmax 判到相邻选项、甚至出现多个选项 logprob 完全相等靠字母序
    tie-break（2026-07-22 实测）。fp32 参照用于把 vLLM 的差异归因到精度而非实现 bug。
    """
    from _common import load

    if getattr(args, "hf_fp32", False):
        import eval_mmlu_pro as _emp
        if not getattr(_emp.log_softmax, "_fp32_wrapped", False):
            _orig = _emp.log_softmax

            def _fp32_log_softmax(x, *a, **k):
                return _orig(x.float(), *a, **k)

            _fp32_log_softmax._fp32_wrapped = True
            _emp.log_softmax = _fp32_log_softmax
            print("[worker-hf] log_softmax upcast to fp32")

    class _A:
        base_model = args.base_model
        adapter_path = args.adapter_path
        dtype = getattr(args, "hf_model_dtype", "bfloat16")
        trust_remote_code = True

    tokenizer, model = load(_A)
    rows = [it["row"] for it in shard]
    prompts = [it["prompt"] for it in shard]
    all_scores: list[dict[str, float]] = []
    bsz = args.hf_batch_size
    for start in range(0, len(shard), bsz):
        batch_rows = rows[start : start + bsz]
        batch_prompts = prompts[start : start + bsz]
        all_scores.extend(score_rows(model, tokenizer, batch_rows, batch_prompts, args.max_length))
        print(f"[worker-hf] {min(start + bsz, len(shard))}/{len(shard)}", flush=True)
    return all_scores


# ---------------------------------------------------------------------------
# worker：分片打分 → 写记录
# ---------------------------------------------------------------------------


def _records_from_scores(shard, scores, engine) -> list[dict[str, Any]]:
    records = []
    for it, sc in zip(shard, scores):
        row = it["row"]
        labels = LETTERS[: len(row["options"])]
        prediction = max(labels, key=lambda label: sc[label])
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
            "logprobs": {label: sc[label] for label in labels},
            "score_type": "label_logprob",
            "engine": engine,
            "prompt": None,
        })
    return records


def run_worker(args, items: list[dict[str, Any]]) -> None:
    shard = items[args.shard_idx :: args.num_shards]
    print(
        f"[worker] mmlu_pro logprob engine={args.engine} shard {args.shard_idx}/{args.num_shards}: "
        f"{len(shard)}/{len(items)} rows | adapter={args.adapter_path or 'BASE'}"
    )
    if not shard:
        _write_shard(args, [])
        return
    scores = _score_hf(args, shard) if args.engine == "hf" else _score_vllm(args, shard)
    _write_shard(args, _records_from_scores(shard, scores, args.engine))


def _shard_path(out_dir: str, i: int, n: int) -> Path:
    return Path(out_dir) / f"result.shard{i}of{n}.json"


def _write_shard(args, records: list[dict[str, Any]]) -> None:
    out = _shard_path(args.out_dir, args.shard_idx, args.num_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[worker] wrote {len(records)} records -> {out}")


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
    rec_engines = {str(r.get("engine")) for r in shards if r.get("engine")}
    engine = sorted(rec_engines)[0] if len(rec_engines) == 1 else (args.engine or "vllm")

    summary = summarize(shards)
    summary["engine"] = engine
    summary["base_model"] = args.base_model
    summary["adapter_path"] = args.adapter_path
    correct = sum(int(r["correct"]) for r in shards)
    print(f"\nMMLU-Pro [{engine} label_logprob]: acc={summary['accuracy']:.4f} ({correct}/{len(shards)})")

    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        for r in shards:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_dir}/{{result.json,summary.json}}")
    if not args.keep_shards:
        for i in range(args.num_shards):
            _shard_path(args.out_dir, i, args.num_shards).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# equiv 对拍：同批题 vLLM logprob vs HF logprob 的 argmax 一致率
# ---------------------------------------------------------------------------


def run_equiv(args) -> None:
    import subprocess
    import tempfile

    items = build_items(args)[: args.n_prompts]
    with tempfile.TemporaryDirectory() as td:
        results = {}
        for engine in ("vllm", "hf"):
            ids_out = os.path.join(td, f"{engine}.json")
            cmd = [
                sys.executable, os.path.abspath(__file__), "--stage_engine", engine,
                "--stage_out", ids_out, "--base_model", args.base_model,
                "--n_prompts", str(args.n_prompts), "--max_length", str(args.max_length),
                "--data_dir", args.data_dir, "--split", args.split, "--n_shot", str(args.n_shot),
            ]
            if args.adapter_path:
                cmd += ["--adapter_path", args.adapter_path]
            if args.no_prefix_caching and engine == "vllm":
                cmd += ["--no_prefix_caching"]
            if args.hf_fp32 and engine == "hf":
                cmd += ["--hf_fp32"]
            if getattr(args, "hf_model_dtype", "bfloat16") != "bfloat16" and engine == "hf":
                cmd += ["--hf_model_dtype", args.hf_model_dtype]
            print(f"[equiv] running {engine} stage ...", flush=True)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"[equiv] {engine} stage failed (exit {proc.returncode})", file=sys.stderr)
                sys.exit(proc.returncode or 1)
            with open(ids_out) as f:
                results[engine] = json.load(f)

    v, h = results["vllm"], results["hf"]
    n = len(v)
    agree = sum(int(a["prediction"] == b["prediction"]) for a, b in zip(v, h))
    ratio = agree / n if n else 1.0
    verdict = "PASS" if ratio >= args.min_agree else "FAIL"
    print(f"\n[equiv] argmax_agree={agree}/{n} ({ratio:.3f}) -> {verdict}")
    # mismatch 诊断：打印双方对各自 top 选项的 logprob margin，判近平手噪声 vs bug
    hmap = {b["idx"]: b for b in h}
    mism = []
    near_tie = 0
    for a in v:
        b = hmap[a["idx"]]
        if a["prediction"] == b["prediction"]:
            continue
        vlp, hlp = a.get("logprobs", {}), b.get("logprobs", {})
        # HF 在其选 hp 与 vLLM 选 vp 上的 logprob 差（HF 视角这俩多接近）
        hf_gap = abs(hlp.get(b["prediction"], 0.0) - hlp.get(a["prediction"], 0.0)) if hlp else None
        vl_gap = abs(vlp.get(a["prediction"], 0.0) - vlp.get(b["prediction"], 0.0)) if vlp else None
        tie = (hf_gap is not None and hf_gap < args.tie_eps) and (vl_gap is not None and vl_gap < args.tie_eps)
        near_tie += int(tie)
        mism.append({"idx": a["idx"], "vllm": a["prediction"], "hf": b["prediction"],
                     "hf_gap": hf_gap, "vllm_gap": vl_gap, "near_tie": tie,
                     "vllm_logprobs": vlp, "hf_logprobs": hlp})
    for m in mism[:12]:
        print(f"  idx{m['idx']}: vllm={m['vllm']} hf={m['hf']} "
              f"hf_gap={m['hf_gap']:.4f} vllm_gap={m['vllm_gap']:.4f} "
              f"{'近平手' if m['near_tie'] else '⚠非平手(疑bug)'}")
    print(f"[equiv] mismatch={len(mism)} 其中近平手(gap<{args.tie_eps})={near_tie} "
          f"→ {'全是近平手噪声,可接受' if near_tie==len(mism) else '存在非平手分歧,需查'}")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"argmax_agree": f"{agree}/{n}", "ratio": ratio, "verdict": verdict,
                       "near_tie": near_tie, "prefix_caching": not args.no_prefix_caching,
                       "mismatches": mism}, f, indent=2, ensure_ascii=False)
    sys.exit(0 if verdict == "PASS" else 1)


def _run_stage(args) -> None:
    """equiv 子进程：单引擎打分，把预测写出（独立子进程防显存残留）。"""
    items = build_items(args)[: args.n_prompts]
    scores = _score_hf(args, items) if args.stage_engine == "hf" else _score_vllm(args, items)
    recs = _records_from_scores(items, scores, args.stage_engine)
    slim = [{"idx": r["idx"], "prediction": r["prediction"], "answer": r["answer"],
             "logprobs": r["logprobs"]} for r in recs]
    with open(args.stage_out, "w") as f:
        json.dump(slim, f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None, help="moe_lora 目录；缺省评 base")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge_shards", action="store_true")
    p.add_argument("--keep_shards", action="store_true")
    p.add_argument("--data_dir", default=os.path.join(_DATA_ROOT, "MMLU-Pro"))
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--n_shot", type=int, default=0, help="category 匹配的 validation few-shot 数")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--engine", default="vllm", choices=["vllm", "hf"],
                   help="hf = eval_mmlu_pro.score_rows 同款（gemma4 用）")
    p.add_argument("--hf_batch_size", type=int, default=4)
    p.add_argument("--hf_fp32", action="store_true",
                   help="HF 打分的 log_softmax 提到 fp32（对齐 vLLM 精度；默认 bf16 同现有链）")
    p.add_argument("--hf_model_dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="HF 参照的模型加载精度。float32 = fp32 forward 真值参照"
                        "（判 vLLM/HF-bf16 谁更接近真值；30B fp32≈120G 单卡放得下）")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--no_enforce_eager", action="store_true")
    p.add_argument("--no_prefix_caching", action="store_true",
                   help="关闭 prefix caching（排查 prompt_logprobs×prefix caching 交互）")
    # equiv 对拍
    p.add_argument("--equiv_vs_hf", action="store_true", help="同批题 vLLM vs HF argmax 一致率")
    p.add_argument("--n_prompts", type=int, default=48)
    p.add_argument("--min_agree", type=float, default=0.83,
                   help="argmax 一致率阈值。vLLM(bf16) vs HF(bf16) 的固有一致率地板"
                        "≈0.85-0.90（两引擎 forward logits 差 ~0.12，难题 argmax 抖动，"
                        "2026-07-22 实测 qwen3 base 0.875；log_softmax/fp32 升精度不改变，"
                        "非实现 bug）。<0.7 才疑实现 bug（位置/tokenization 错）。")
    p.add_argument("--tie_eps", type=float, default=0.05,
                   help="mismatch 题双方 logprob gap 都 < 该值判近平手噪声（可接受）")
    p.add_argument("--out_json", default=None)
    # 内部：equiv 子进程
    p.add_argument("--stage_engine", choices=["vllm", "hf"], default=None)
    p.add_argument("--stage_out", default=None)
    args = p.parse_args()

    if args.stage_engine:
        _run_stage(args)
        return
    if args.equiv_vs_hf:
        run_equiv(args)
        return
    if args.merge_shards:
        if not args.out_dir:
            raise SystemExit("--merge_shards 需要 --out_dir")
        run_merge(args)
        return
    if not args.out_dir:
        raise SystemExit("worker 模式需要 --out_dir")
    if not (0 <= args.shard_idx < args.num_shards):
        raise SystemExit(f"非法分片 {args.shard_idx}/{args.num_shards}")
    if args.adapter_path and args.tensor_parallel_size != 1:
        raise SystemExit("adapter overlay 只验证过 tensor_parallel_size=1")
    run_worker(args, build_items(args))


if __name__ == "__main__":
    main()
