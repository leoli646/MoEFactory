from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

metamath = load_module('prepare_metamathqa_r1', 'scripts/data/prepare_metamathqa_r1.py')
esft = load_module('prepare_esft', 'scripts/data/prepare_esft.py')
intent = load_module('intent_metrics', 'eval_scripts/esft/intent_metrics.py')
sys.path.insert(0, str(ROOT / 'eval_scripts/esft'))
evaluator = load_module('eval_esft', 'eval_scripts/esft/eval_esft.py')

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')

class PublicDataTests(unittest.TestCase):
    def test_dataset_registry_has_public_training_flows(self) -> None:
        registry = json.loads((ROOT / 'data/dataset_info.json').read_text(encoding='utf-8'))
        self.assertEqual(set(registry), {'metamathqa_r1', 'esft_intent'})
        for entry in registry.values():
            self.assertEqual(entry['formatting'], 'sharegpt')
            self.assertEqual(entry['columns']['messages'], 'messages')
            self.assertEqual(entry['tags'], {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'})

    def test_esft_exposes_only_intent_and_rejects_other_tasks(self) -> None:
        self.assertEqual(esft.TASKS, ('intent',))
        self.assertEqual(evaluator.TASKS, ('intent',))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "only intent"):
                esft.export_dataset(root / 'source', root / 'output', ('unsupported',))
            self.assertFalse((root / 'output').exists())
        with self.assertRaisesRegex(ValueError, "only intent"):
            evaluator.score_prediction_row('unsupported', {})
        self.assertFalse(hasattr(evaluator, 'LocalJudge'))

    def test_metamath_conversion_removes_think_and_skips_incomplete(self) -> None:
        records = [
            {'prompt': '2+2?', 'response': '<think>work</think>\\boxed{4}', 'metadata': {'complete': True}},
            {'prompt': 'skip', 'response': 'answer', 'metadata': {'complete': False}},
        ]
        self.assertEqual(metamath.prepare(records), [{'messages': [{'role': 'user', 'content': '2+2?'}, {'role': 'assistant', 'content': '\\boxed{4}'}]}])

    def test_metamath_preserves_source_semantics(self) -> None:
        self.assertEqual(metamath.strip_think("  <think>open"), "  <think>open")
        kept = metamath.convert_sample({"prompt": "  ", "response": " answer "}, keep_think=True)
        self.assertEqual(kept["messages"][0]["content"], "  ")
        self.assertEqual(kept["messages"][1]["content"], " answer ")
        records = [
            {"prompt": "skip", "response": "a", "metadata": {"complete": False}},
            {"prompt": "kept", "response": "b"},
        ]
        self.assertEqual(metamath.prepare(records, limit=1), [])

    def test_limit_parser_rejects_nonpositive_values(self) -> None:
        for value in ('0', '-1'):
            with self.assertRaises(Exception):
                metamath.positive_int(value)

    def test_esft_export_validates_preserves_and_copies_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; output = root / 'output'
            train_row = {'messages': [{'role': 'user', 'content': 'prompt'}, {'role': 'assistant', 'content': 'answer'}]}
            eval_row = {'idx': 0, 'prompt': 'User: prompt\n\nAssistant:', 'raw_answers': ['answer']}
            write_jsonl(source / 'datasets/train/intent.jsonl', [train_row])
            write_jsonl(source / 'datasets/eval/intent.jsonl', [eval_row])
            self.assertEqual(esft.export_dataset(source, output, ('intent',)), {'intent_train': 1, 'intent_eval': 1})
            self.assertEqual(esft.read_jsonl(output / 'esft/intent_train.jsonl'), [train_row])
            self.assertEqual(esft.read_jsonl(output / 'esft_eval/intent.jsonl'), [eval_row])
            self.assertTrue((output / 'dataset_info.json').is_file())

    def test_esft_export_rejects_invalid_or_duplicate_idx(self) -> None:
        cases = ([{'idx': '0', 'prompt': 'p', 'raw_answers': ['a']}], [{'idx': 0, 'prompt': 'p', 'raw_answers': ['a']}, {'idx': 0, 'prompt': 'q', 'raw_answers': ['b']}])
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'source'
                write_jsonl(source / 'datasets/train/intent.jsonl', [{'messages': [{'role': 'user', 'content': 'p'}, {'role': 'assistant', 'content': 'a'}]}])
                write_jsonl(source / 'datasets/eval/intent.jsonl', rows)
                with self.assertRaises(ValueError):
                    esft.export_dataset(source, root / 'output', ('intent',))

    def test_intent_metric_matches_original_normalization(self) -> None:
        raw = 'prefix {"intent":"Travel","slots":{"city":"Hong Kong","days":["Tue","Mon"]}} trailing {"x":1}'
        prediction = intent.post_process_intent(raw)
        ground_truth = '{"intent":"Travel","slots":{"city":"HongKong","days":["Mon","Tue"]}}'
        self.assertEqual(json.loads(prediction)['intent'], 'Travel')
        self.assertEqual(intent.score_intent(prediction, ground_truth), 1.0)
        self.assertEqual(intent.score_intent('{"intent":"Other","slots":{}}', ground_truth), 0.0)

    def test_intent_metric_returns_zero_for_malformed_model_values(self) -> None:
        ground_truth = '{"intent":"Travel","slots":{"city":"HongKong"}}'
        malformed = ('not json', '[]', '{"intent":"Travel","slots":[]}', '{"intent":"Travel","slots":{"city":{"nested":true}}}')
        for prediction in malformed:
            with self.subTest(prediction=prediction):
                self.assertEqual(intent.score_intent(prediction, ground_truth), 0.0)

    def test_prediction_and_scored_rows_retain_selector_fields(self) -> None:
        eval_rows = [{'idx': 7, 'prompt': 'p', 'raw_answers': ['{"intent":"A","slots":{}}']}]
        rows = evaluator.build_prediction_rows(eval_rows, ['{"intent":"A","slots":{}}'], ['p'], [1])
        self.assertEqual(rows[0]['idx'], 7)
        scored, score = evaluator.score_prediction_row('intent', rows[0])
        self.assertEqual(scored['idx'], 7)
        self.assertIsInstance(scored['score'], float)
        self.assertEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
