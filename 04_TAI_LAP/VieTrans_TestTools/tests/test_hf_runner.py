from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_hf_server import (  # noqa: E402
    endpoint_names,
    inpainting_regions,
    ocr_prediction_from_payload,
)


class HuggingFaceRunnerTests(unittest.TestCase):
    def test_endpoint_names(self):
        payload = {"named_endpoints": {"/eval_ocr": {}, "/translate_no_qa": {}}}
        self.assertEqual(
            endpoint_names(payload),
            {"/eval_ocr", "/translate_no_qa"},
        )

    def test_ocr_payload_conversion(self):
        prediction = ocr_prediction_from_payload(
            "case1",
            {
                "regions": [
                    {
                        "text": "OPEN",
                        "box": [1, 2, 11, 12],
                        "confidence": 0.9,
                    }
                ]
            },
        )
        self.assertEqual(prediction["case_id"], "case1")
        self.assertEqual(prediction["text"], "OPEN")
        self.assertEqual(len(prediction["regions"][0]["polygon"]), 4)

    def test_otr_boxes_become_regions(self):
        payload = inpainting_regions(
            {
                "attributes": {
                    "words": ["hello"],
                    "word_bboxes": [[20, 30, 10, 5]],
                }
            }
        )
        self.assertEqual(payload["regions"][0]["box"], [10.0, 5.0, 20.0, 30.0])
        self.assertEqual(payload["regions"][0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()

