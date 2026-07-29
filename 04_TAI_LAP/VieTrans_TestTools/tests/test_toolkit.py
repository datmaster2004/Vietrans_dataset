from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from common import write_jsonl  # noqa: E402
from evaluate import evaluate_inpainting, evaluate_ocr, evaluate_translation  # noqa: E402
from prepare_benchmarks import parse_total_text_ground_truth  # noqa: E402


class ToolkitTests(unittest.TestCase):
    def test_total_text_multiline_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "poly_gt_img1.txt"
            path.write_text(
                "x: [[1 9 9 1]], y: [[1 1 5 5]], ornt: [u'h'], "
                "transcriptions: [u'OPEN']\n"
                "x: [[2 8 10\n 8 2]], y: [[10 8 12\n 16 14]], ornt: [u'c'], "
                'transcriptions: [u"Joe\'s"]\n',
                encoding="utf-8",
            )
            rows = parse_total_text_ground_truth(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["text"], "Joe's")

    def test_ocr_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            polygon = [[10, 10], [70, 10], [70, 30], [10, 30]]
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "case_id": "ocr_1",
                        "text": "OPEN",
                        "regions": [{"text": "OPEN", "polygon": polygon}],
                    }
                ],
            )
            write_jsonl(
                predictions,
                [
                    {
                        "case_id": "ocr_1",
                        "text": "OPEN",
                        "regions": [{"text": "OPEN", "polygon": polygon}],
                    }
                ],
            )
            summary, cases = evaluate_ocr(manifest, predictions)
            self.assertEqual(summary["coverage"], 1.0)
            self.assertEqual(cases[0]["detection_f1"], 100.0)
            self.assertEqual(cases[0]["text_spotting_f1"], 100.0)

    def test_inpainting_oracle(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean = np.full((20, 30, 3), 220, dtype=np.uint8)
            source = clean.copy()
            source[7:13, 8:22] = 10
            mask = np.zeros((20, 30), dtype=np.uint8)
            mask[7:13, 8:22] = 255
            Image.fromarray(clean).save(root / "clean.png")
            Image.fromarray(source).save(root / "input.png")
            Image.fromarray(mask).save(root / "mask.png")
            Image.fromarray(clean).save(root / "output.png")
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "case_id": "inp_1",
                        "input_image": "input.png",
                        "clean_image": "clean.png",
                        "mask_image": "mask.png",
                    }
                ],
            )
            write_jsonl(predictions, [{"case_id": "inp_1", "output_image": "output.png"}])
            summary, cases = evaluate_inpainting(manifest, predictions)
            self.assertEqual(summary["coverage"], 1.0)
            self.assertEqual(cases[0]["inside_mae"], 0.0)
            self.assertEqual(cases[0]["outside_mae"], 0.0)
            self.assertAlmostEqual(cases[0]["inside_ssim"], 1.0)

    def test_translation_oracle_and_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "case_id": "mt_1",
                        "source_text": "Keep your distance",
                        "references": ["Hãy giữ khoảng cách"],
                        "terminology": [{"accepted": ["khoảng cách"]}],
                    }
                ],
            )
            write_jsonl(
                predictions,
                [{"case_id": "mt_1", "translation": "Hãy giữ khoảng cách"}],
            )
            summary, cases = evaluate_translation(manifest, predictions)
            self.assertEqual(cases[0]["exact_match"], 100.0)
            self.assertEqual(cases[0]["terminology_accuracy"], 100.0)
            self.assertEqual(summary["corpus_metrics"]["corpus_chrf_plus_plus"], 100.0)
            self.assertIn("bleu", summary["corpus_metrics"]["metric_signatures"])

    def test_missing_prediction_reduces_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            write_jsonl(manifest, [{"case_id": "ocr_1", "text": "A"}])
            write_jsonl(predictions, [])
            summary, _ = evaluate_ocr(manifest, predictions)
            self.assertEqual(summary["coverage"], 0.0)
            self.assertEqual(summary["missing_prediction_count"], 1)


if __name__ == "__main__":
    unittest.main()

