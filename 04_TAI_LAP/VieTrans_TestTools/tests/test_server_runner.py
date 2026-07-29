from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from run_component_server import (  # noqa: E402
    build_parser,
    endpoint_names,
    inpainting_regions,
    ocr_prediction_from_payload,
    server_probe,
)


class ServerRunnerTests(unittest.TestCase):
    def test_local_server_is_the_default(self):
        args = build_parser().parse_args(["probe"])
        self.assertEqual(args.server_url, "http://127.0.0.1:7860")
        self.assertEqual(args.server_id, "local-vietrans")

    def test_probe_records_generic_server_provenance(self):
        class FakeClient:
            def view_api(self, **_kwargs):
                return {
                    "named_endpoints": {
                        "/eval_ocr": {},
                        "/eval_inpaint": {},
                        "/eval_translate": {},
                        "/eval_info": {},
                    }
                }

            def predict(self, **_kwargs):
                return {"model_revision": "test"}

        with tempfile.TemporaryDirectory() as temp, patch(
            "run_component_server.get_client", return_value=FakeClient()
        ):
            probe = server_probe(
                "http://127.0.0.1:7860",
                "staging-vietrans",
                "build-123",
                None,
                None,
                Path(temp),
            )
        self.assertEqual(probe["server_id"], "staging-vietrans")
        self.assertEqual(probe["server_build_id"], "build-123")
        self.assertTrue(probe["component_endpoints_ready"])

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
