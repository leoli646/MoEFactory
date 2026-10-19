"""MMLU-Pro evaluation for local base/adapters (forced-choice logprob or generation).

Default mode (--eval_mode logprob) scores label log-probabilities over A..J and is
byte-identical to the historical protocol. Optional mode (--eval_mode generation)
lets the model free-form generate and parses the chosen letter; use it to separate
"calibration/format shift" from real capability loss (e.g. after intent SFT the
conditional probability of ' A'..' J' after "Answer:" can be distorted even when
the model still answers correctly when allowed to generate).

Generation styles:
  direct  prompt ends with "The answer is:" and the model emits (ideally) just the
          letter; cheap (default max_new_tokens=32), closest decision point to the
          logprob mode.
  cot     "Answer: Let's think step by step." then reasoning ending in
          "the answer is (X)" (official TIGER-Lab MMLU-Pro protocol shape); much
          more expensive (default max_new_tokens=1024).

Example:
    torchrun --nproc_per_node=4 eval_scripts/eval_mmlu_pro.py \
        --base_model /path/to/Qwen3-30B-A3B-Base \
        --adapter_path saves/expert_lora_sweep/intent/lr1e-4_bs144 \
        --data_dir data/MMLU-Pro \
        --batch_size 1 \
        --out_dir data/MMLU-Pro/test_result/expert_lora_sweep/intent/lr1e-4_bs144

    # generation mode (bigger batch is fine: one sequence per question)
    torchrun --nproc_per_node=4 eval_scripts/eval_mmlu_pro.py \
        --base_model /path/to/Qwen3-30B-A3B-Base \
        --eval_mode generation --gen_style direct --batch_size 16 \
        --data_dir data/MMLU-Pro --out_dir <isolated_out_dir>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_datasets"))

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.functional import log_softmax
from tqdm import tqdm


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = os.environ.get("MOEFACTORY_DATA_ROOT", str(_REPO_ROOT / "data"))

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from _common import common_arg_parser, load, _collect_eos_ids  # noqa: E402


LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_split(data_dir: str, split: str) -> list[dict[str, Any]]:
    data_root = Path(data_dir)
    parquet_glob = data_root / "data" / f"{split}-*.parquet"
    ds = load_dataset(
        "parquet",
        data_files={split: str(parquet_glob)},
        split=split,
    )
    return list(ds)


def normalize_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip().upper()
    answer_index = row.get("answer_index")
    if isinstance(answer_index, int) and 0 <= answer_index < len(LETTERS):
        return LETTERS[answer_index]
    raise ValueError(f"Cannot infer answer label for question_id={row.get('question_id')}")


def format_options(options: list[str]) -> list[str]:
    return [f"{LETTERS[i]}. {text}" for i, text in enumerate(options)]


def format_single_qa(row: dict[str, Any], include_answer: bool) -> str:
    lines = [f"Question: {row['question']}", "Options:"]
    lines.extend(format_options(row["options"]))
    if include_answer:
        lines.append(f"Answer: {normalize_answer(row)}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_shot_bank(validation_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validation_rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    return by_category


def format_prompt(row: dict[str, Any], shot_bank: dict[str, list[dict[str, Any]]] | None, n_shot: int) -> str:
    parts: list[str] = []
    if shot_bank and n_shot > 0:
        category = str(row.get("category", "unknown"))
        for shot in shot_bank.get(category, [])[:n_shot]:
            parts.append(format_single_qa(shot, include_answer=True))
            parts.append("")
    parts.append(format_single_qa(row, include_answer=False))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation mode (opt-in via --eval_mode generation; logprob path untouched)
# ---------------------------------------------------------------------------

GEN_HEADER_DIRECT = (
    "The following is a multiple choice question. Read the question and the options, "
    "then reply with only the letter of the correct option (one of {letters}). "
    "Do not explain."
)

GEN_HEADER_COT = (
    "The following is a multiple choice question. Think step by step and then finish "
    "your response with \"the answer is (X)\" where X is the letter of the correct option."
)


def format_question_block(row: dict[str, Any]) -> str:
    lines = [f"Question: {row['question']}", "Options:"]
    lines.extend(format_options(row["options"]))
    return "\n".join(lines)


def format_prompt_generation(
    row: dict[str, Any],
    shot_bank: dict[str, list[dict[str, Any]]] | None,
    n_shot: int,
    style: str,
) -> str:
    letters = LETTERS[: len(row["options"])]
    if style == "direct":
        header = GEN_HEADER_DIRECT.format(letters=", ".join(letters))
    else:
        header = GEN_HEADER_COT
    parts: list[str] = [header, ""]
    if shot_bank and n_shot > 0:
        category = str(row.get("category", "unknown"))
        for shot in shot_bank.get(category, [])[:n_shot]:
            parts.append(format_question_block(shot))
            gold = normalize_answer(shot)
            if style == "direct":
                parts.append(f"The answer is: {gold}")
            else:
                cot = str(shot.get("cot_content") or "").strip()
                if cot:
                    # dataset cot_content starts with "A: Let's think step by step."
                    cot = re.sub(r"^A:\s*", "Answer: ", cot)
                    parts.append(cot)
                else:
                    parts.append(f"Answer: The answer is ({gold}).")
            parts.append("")
    parts.append(format_question_block(row))
    if style == "direct":
        parts.append("The answer is:")
    else:
        parts.append("Answer: Let's think step by step.")
    return "\n".join(parts)


# ordered extraction rules; each returns the LAST match (final answer wins in CoT)
_RE_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*\**\(?([A-J])\)?", re.IGNORECASE)
_RE_ANSWER_COLON = re.compile(r"answer\s*:\s*\**\(?([A-J])\)?", re.IGNORECASE)
_RE_HEAD_LETTER = re.compile(r"^\s*\**\(?([A-J])\)?(?:[.):,\s]|$)")
_RE_STANDALONE = re.compile(r"\b([A-J])\b")


def extract_choice(completion: str, valid_letters: list[str]) -> tuple[str | None, str]:
    """Parse the chosen letter out of a free-form completion.

    Returns (letter or None, rule_name). Only letters that are valid for this
    question count; otherwise fall through to the next rule.
    """
    valid = set(valid_letters)

    matches = [m.group(1).upper() for m in _RE_ANSWER_IS.finditer(completion)]
    picks = [m for m in matches if m in valid]
    if picks:
        return picks[-1], "answer_is"

    matches = [m.group(1).upper() for m in _RE_ANSWER_COLON.finditer(completion)]
    picks = [m for m in matches if m in valid]
    if picks:
        return picks[-1], "answer_colon"

    m = _RE_HEAD_LETTER.match(completion)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper(), "head_letter"

    # last standalone uppercase letter (uppercase only: lowercase 'a' is an article)
    picks = [m.group(1) for m in _RE_STANDALONE.finditer(completion) if m.group(1) in valid]
    if picks:
        return picks[-1], "standalone_last"

    return None, "parse_failure"


@torch.inference_mode()
def generate_rows(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    max_input_len: int,
) -> tuple[list[str], list[bool]]:
    """Greedy batch generation. Left-truncates so the trailing answer anchor
    ("The answer is:") always survives; padding is already left-side (see load()).

    Returns (completions, truncated_flags). A row is truncated when it produced
    max_new_tokens tokens without ever emitting an EOS id.
    """
    old_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len
    ).to(model.device)
    tokenizer.truncation_side = old_side
    eos_ids = set(_collect_eos_ids(model, tokenizer))
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=sorted(eos_ids),
    )
    input_len = enc.input_ids.shape[1]
    completions: list[str] = []
    truncated: list[bool] = []
    for ids in out:
        gen_ids = ids[input_len:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        gen_list = gen_ids.tolist()
        truncated.append(len(gen_list) == max_new_tokens and not any(t in eos_ids for t in gen_list))
    return completions, truncated


def encode_continuation(
    tokenizer,
    prompt: str,
    label: str,
    max_length: int,
) -> tuple[list[int], int, int]:
    """Return (input_ids, prompt_len, continuation_len) for prompt + " LABEL"."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    cont_ids = tokenizer(" " + label, add_special_tokens=False).input_ids
    if not cont_ids:
        raise ValueError(f"Tokenizer produced no ids for label={label!r}")

    max_prompt_len = max_length - len(cont_ids)
    if max_prompt_len <= 0:
        raise ValueError(f"max_length={max_length} is too small for label={label!r}")
    if len(prompt_ids) > max_prompt_len:
        # Preserve the question/options/Answer tail where the scored continuation attaches.
        prompt_ids = prompt_ids[-max_prompt_len:]

    return prompt_ids + cont_ids, len(prompt_ids), len(cont_ids)


@torch.inference_mode()
def score_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    prompts: list[str],
    max_length: int,
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    flat_ids: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    groups: list[tuple[int, str]] = []

    for row_idx, (row, prompt) in enumerate(zip(rows, prompts)):
        labels = LETTERS[: len(row["options"])]
        for label in labels:
            ids, prompt_len, cont_len = encode_continuation(tokenizer, prompt, label, max_length)
            flat_ids.append(ids)
            spans.append((prompt_len, cont_len))
            groups.append((row_idx, label))

    pad_id = tokenizer.pad_token_id
    max_seq_len = max(len(ids) for ids in flat_ids)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    offsets: list[int] = []
    for ids in flat_ids:
        pad_len = max_seq_len - len(ids)
        padded.append([pad_id] * pad_len + ids)
        masks.append([0] * pad_len + [1] * len(ids))
        offsets.append(pad_len)

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = log_softmax(outputs.logits, dim=-1)

    scores: list[dict[str, float]] = [dict() for _ in rows]
    for seq_idx, ((row_idx, label), (prompt_len, cont_len), offset) in enumerate(zip(groups, spans, offsets)):
        total = 0.0
        ids = flat_ids[seq_idx]
        for j in range(cont_len):
            token_pos = offset + prompt_len + j
            token_id = ids[prompt_len + j]
            total += log_probs[seq_idx, token_pos - 1, token_id].item()
        scores[row_idx][label] = total / cont_len

    return scores


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def acc(items: list[dict[str, Any]]) -> float:
        return sum(int(r["correct"]) for r in items) / len(items) if items else 0.0

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        categories[record["category"]].append(record)
        sources[record["src"]].append(record)

    summary: dict[str, Any] = {
        "num_examples": len(records),
        "accuracy": acc(records),
        "categories": {
            key: {"num_examples": len(items), "accuracy": acc(items)}
            for key, items in sorted(categories.items())
        },
        "sources": {
            key: {"num_examples": len(items), "accuracy": acc(items)}
            for key, items in sorted(sources.items())
        },
    }

    score_types = {str(r.get("score_type")) for r in records}
    if score_types:
        summary["score_type"] = sorted(score_types)[0] if len(score_types) == 1 else sorted(score_types)
    if any(str(r.get("score_type", "")).startswith("generation") for r in records):
        n = len(records)
        failures = [r for r in records if r.get("extract_rule") == "parse_failure"]
        rule_counts: dict[str, int] = defaultdict(int)
        for r in records:
            rule_counts[str(r.get("extract_rule"))] += 1
        parsed = [r for r in records if r.get("extract_rule") != "parse_failure"]
        summary["generation"] = {
            "parse_failure_rate": len(failures) / n if n else 0.0,
            "num_parse_failures": len(failures),
            "extract_rules": dict(sorted(rule_counts.items())),
            "accuracy_on_parsed": acc(parsed),
            "num_parsed": len(parsed),
            "truncated_rate": sum(int(bool(r.get("truncated"))) for r in records) / n if n else 0.0,
        }
    return summary


def main() -> None:
    parser = common_arg_parser("MMLU-Pro evaluation (forced-choice logprob or generation).")
    parser.add_argument("--data_dir", default=os.path.join(_DATA_ROOT, "MMLU-Pro"))
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--out_dir", default=None, help="Write result.json and summary.json here.")
    parser.add_argument("--n_shot", type=int, default=0, help="Use category-matched validation shots.")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--categories", default=None, help="Comma-separated category filter.")
    parser.add_argument("--save_prompt", action="store_true", help="Store rendered prompt per record.")
    parser.add_argument(
        "--eval_mode",
        default="logprob",
        choices=["logprob", "generation"],
        help="logprob = historical forced-choice protocol (default, unchanged); "
        "generation = free-form generate then parse the chosen letter.",
    )
    parser.add_argument(
        "--gen_style",
        default="direct",
        choices=["direct", "cot"],
        help="generation only. direct = answer-letter-only prompt; cot = think step by step "
        "then 'the answer is (X)' (official MMLU-Pro shape).",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=None,
        help="generation only. Default: 32 for direct, 1024 for cot.",
    )
    args = parser.parse_args()

    if args.gen_max_new_tokens is None:
        args.gen_max_new_tokens = 32 if args.gen_style == "direct" else 1024

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    is_main = local_rank == 0

    if is_dist:
        torch.cuda.set_device(local_rank)
        # 24h timeout: generation (especially cot) makes per-rank runtimes uneven and the
        # default 10min NCCL watchdog kills the fast ranks at all_gather (same failure
        # mode used by the math evaluator).
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))

    rows = load_split(args.data_dir, args.split)
    if args.categories:
        wanted = {item.strip() for item in args.categories.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("category", "unknown")) in wanted]
    if args.limit:
        rows = rows[: args.limit]

    shot_bank = None
    if args.n_shot > 0:
        validation_rows = load_split(args.data_dir, "validation")
        shot_bank = build_shot_bank(validation_rows)

    gen_mode = args.eval_mode == "generation"
    score_type = f"generation_{args.gen_style}" if gen_mode else "label_logprob"

    if is_main:
        print(f"Loaded MMLU-Pro split={args.split}: {len(rows)} rows")
        if gen_mode:
            print(
                f"Scoring mode: generation ({args.gen_style}, greedy, "
                f"max_new_tokens={args.gen_max_new_tokens}), n_shot={args.n_shot}, "
                f"batch_size={args.batch_size}"
            )
        else:
            print(f"Scoring mode: label logprob over A..J, n_shot={args.n_shot}, batch_size={args.batch_size}")

    tokenizer, model = load(args)

    my_indices = list(range(local_rank, len(rows), world_size)) if is_dist else list(range(len(rows)))
    records: list[dict[str, Any]] = []
    starts = list(range(0, len(my_indices), args.batch_size))
    iterator = tqdm(starts, desc=f"mmlu-pro r{local_rank}") if is_main else starts

    for start in iterator:
        batch_indices = my_indices[start : start + args.batch_size]
        batch_rows = [rows[i] for i in batch_indices]

        if gen_mode:
            prompts = [
                format_prompt_generation(row, shot_bank, args.n_shot, args.gen_style)
                for row in batch_rows
            ]
            max_input_len = max(args.max_length - args.gen_max_new_tokens, 512)
            completions, truncated_flags = generate_rows(
                model, tokenizer, prompts, args.gen_max_new_tokens, max_input_len
            )
            for global_idx, row, prompt, completion, truncated in zip(
                batch_indices, batch_rows, prompts, completions, truncated_flags
            ):
                labels = LETTERS[: len(row["options"])]
                prediction, rule = extract_choice(completion, labels)
                gold = normalize_answer(row)
                records.append({
                    "idx": global_idx,
                    "question_id": row.get("question_id"),
                    "category": str(row.get("category", "unknown")),
                    "src": str(row.get("src", "unknown")),
                    "split": args.split,
                    "question": row["question"],
                    "options": row["options"],
                    "answer": gold,
                    "answer_index": row.get("answer_index"),
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "completion": completion,
                    "extract_rule": rule,
                    "truncated": truncated,
                    "score_type": score_type,
                    "prompt": prompt if args.save_prompt else None,
                })
        else:
            prompts = [format_prompt(row, shot_bank, args.n_shot) for row in batch_rows]
            batch_scores = score_rows(model, tokenizer, batch_rows, prompts, args.max_length)
            for global_idx, row, prompt, logprobs in zip(batch_indices, batch_rows, prompts, batch_scores):
                labels = LETTERS[: len(row["options"])]
                prediction = max(labels, key=lambda label: logprobs[label])
                gold = normalize_answer(row)
                record = {
                    "idx": global_idx,
                    "question_id": row.get("question_id"),
                    "category": str(row.get("category", "unknown")),
                    "src": str(row.get("src", "unknown")),
                    "split": args.split,
                    "question": row["question"],
                    "options": row["options"],
                    "answer": gold,
                    "answer_index": row.get("answer_index"),
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "logprobs": {label: logprobs[label] for label in labels},
                    "score_type": "label_logprob",
                    "prompt": prompt if args.save_prompt else None,
                }
                records.append(record)

    if is_dist:
        torch.cuda.empty_cache()
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [record for sub in gathered for record in sub]

    records.sort(key=lambda item: item["idx"])

    if is_main:
        summary = summarize(records)
        correct = sum(int(record["correct"]) for record in records)
        total = len(records)
        print(f"\nMMLU-Pro [{score_type}]: acc={summary['accuracy']:.4f} ({correct}/{total})")
        if "generation" in summary:
            gen = summary["generation"]
            print(
                f"  parse_failure_rate={gen['parse_failure_rate']:.4f} "
                f"({gen['num_parse_failures']}/{total}), "
                f"acc_on_parsed={gen['accuracy_on_parsed']:.4f} (n={gen['num_parsed']}), "
                f"truncated_rate={gen['truncated_rate']:.4f}"
            )
            print(f"  extract_rules={gen['extract_rules']}")

        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            result_path = out_dir / "result.json"
            summary_path = out_dir / "summary.json"
            with result_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Saved to {out_dir}/{{result.json,summary.json}}")

        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            with open(args.save_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary_path = args.save_path.replace(".jsonl", "_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
