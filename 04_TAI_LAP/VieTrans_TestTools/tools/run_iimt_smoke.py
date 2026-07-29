from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TOOLS = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OCR, inpainting, and translation IIMT30K smoke tests against a local/private VieTrans server."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ["VIETRANS_DATASET_ROOT"]) if os.getenv("VIETRANS_DATASET_ROOT") else None,
        help="Directory containing IIMT30K; defaults to VIETRANS_DATASET_ROOT.",
    )
    parser.add_argument(
        "--server-url",
        default=os.getenv("VIETRANS_SERVER_URL", "http://127.0.0.1:7860"),
    )
    parser.add_argument(
        "--server-id",
        default=os.getenv("VIETRANS_SERVER_ID", "local-vietrans"),
    )
    parser.add_argument(
        "--build-id",
        default=os.getenv("VIETRANS_BUILD_ID", "unknown"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=1, help="Cases per suite; use 0 for every case.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.dataset_root is None:
        parser.error("--dataset-root or VIETRANS_DATASET_ROOT is required")
    root = args.dataset_root.resolve()
    iimt = root / "IIMT30K"
    if not iimt.is_dir():
        raise FileNotFoundError(f"IIMT30K directory not found: {iimt}")
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_dir or TOOLS.parent / "results" / f"iimt_smoke_{timestamp}").resolve()
    suites = (
        ("ocr", "ocr_iimt30k_en_arial"),
        ("inpainting", "inpainting_iimt30k_en_arial"),
        ("translation", "translation_iimt30k_en_vi"),
    )
    runner = TOOLS / "run_component_server.py"
    evaluator = TOOLS / "evaluate.py"
    for suite, manifest_name in suites:
        manifest = iimt / "manifests" / manifest_name / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        inference = output_root / suite / "inference"
        command = [
            sys.executable,
            str(runner),
            suite,
            "--server-url",
            args.server_url,
            "--server-id",
            args.server_id,
            "--build-id",
            args.build_id,
            "--manifest",
            str(manifest),
            "--output-dir",
            str(inference),
            "--workers",
            str(args.workers),
        ]
        if args.limit:
            command.extend(("--limit", str(args.limit)))
        run(command)
        run(
            [
                sys.executable,
                str(evaluator),
                suite,
                "--manifest",
                str(inference / "selected_manifest.jsonl"),
                "--predictions",
                str(inference / "predictions.jsonl"),
                "--output-dir",
                str(output_root / suite / "score"),
            ]
        )
    print(f"smoke_results={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
