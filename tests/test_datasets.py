import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from looped_transformers.datasets import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "examples.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        self.path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    def test_manifest_pins_exact_bytes_and_ordered_selection(self):
        self.write([
            {"_meta": {"split": "validation", "purpose": "screening", "source": "authored"}},
            {"id": "a", "text": "A short example."},
            {"id": "b", "text": "Un esempio italiano."},
        ])
        all_rows = load_dataset(self.path)
        limited = load_dataset(self.path, limit=1)
        self.assertEqual(all_rows.fingerprint, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(all_rows.fingerprint, limited.fingerprint)
        self.assertNotEqual(all_rows.manifest()["selection_sha256"], limited.manifest()["selection_sha256"])
        self.assertEqual(limited.manifest()["total_examples"], 2)
        self.assertEqual(limited.manifest()["examples"], 1)
        self.assertEqual(limited.manifest()["split"], "validation")
        self.assertEqual(limited.rows[0]["id"], "a")
        json.dumps(limited.manifest(), allow_nan=False)

    def test_valid_qa_with_automatic_id(self):
        self.write([{"question": "What follows two?", "choices": ["Three", "One"], "answer": 0}])
        dataset = load_dataset(self.path, kind="qa")
        self.assertEqual(dataset.rows[0]["id"], "line-1")
        self.assertEqual(dataset.rows[0]["answer"], 0)

    def test_duplicate_ids_beyond_limit_are_still_rejected(self):
        self.write([{"id": "a", "text": "One."}, {"id": "a", "text": "Two."}])
        with self.assertRaisesRegex(ValueError, "duplicate example id"):
            load_dataset(self.path, limit=1)

    def test_duplicate_keys_and_nonstandard_numbers_are_rejected(self):
        for content in ['{"text":"a","text":"b"}\n', '{"text":"a","metadata":{"x":NaN}}\n', '{"text":"a","metadata":{"x":1e999}}\n']:
            with self.subTest(content=content):
                self.path.write_text(content)
                with self.assertRaises(ValueError):
                    load_dataset(self.path)

    def test_invalid_qa_records_are_rejected(self):
        for row in [
            {"question": "Q", "choices": ["A", "B"], "answer": True},
            {"question": "Q", "choices": ["A", "B"], "answer": 2},
            {"question": "Q", "choices": ["A", "A"], "answer": 0},
            {"question": "Q", "choices": ["A", " "], "answer": 0},
            {"question": "Q", "choices": ["A"], "answer": 0},
            {"question": "", "choices": ["A", "B"], "answer": 0},
        ]:
            with self.subTest(row=row):
                self.write([row])
                with self.assertRaises(ValueError):
                    load_dataset(self.path, kind="qa")

    def test_invalid_text_metadata_and_limits_are_rejected(self):
        for rows in [[], [{"text": " "}], [{"_meta": {}}], [{"text": "A"}, {"_meta": {}}], [{"text": "A", "typo": 1}]]:
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    load_dataset(self.path)
        self.write([{"text": "A"}])
        for limit in [0, -1, True, 1.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_dataset(self.path, limit=limit)

    def test_smoke_sets_are_labeled_and_answer_positions_balanced(self):
        root = Path(__file__).resolve().parents[1]
        texts = load_dataset(root / "data" / "smoke_text.jsonl")
        qa = load_dataset(root / "data" / "smoke_qa.jsonl", kind="qa")
        self.assertEqual(texts.manifest()["purpose"], "smoke")
        self.assertEqual(qa.manifest()["split"], "synthetic")
        answers = [row["answer"] for row in qa.rows]
        self.assertEqual([answers.count(i) for i in range(4)], [3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
