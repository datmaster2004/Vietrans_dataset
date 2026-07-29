from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a JSONL prediction skeleton with exactly the manifest case IDs."
    )
    parser.add_argument("suite", choices=["ocr", "inpainting", "translation"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    output = []
    for row in rows:
        case_id = row["case_id"]
        if args.suite == "ocr":
            output.append(
                {
                    "case_id": case_id,
                    "image": row.get("image"),
                    "regions": [],
                    "text": "",
                }
            )
        elif args.suite == "inpainting":
            output.append(
                {
                    "case_id": case_id,
                    "input_image": row.get("input_image"),
                    "mask_image": row.get("mask_image"),
                    "output_image": "",
                }
            )
        else:
            output.append(
                {
                    "case_id": case_id,
                    "source_text": row.get("source_text"),
                    "translation": "",
                }
            )
    write_jsonl(args.output, output)
    print(f"cases={len(output)}")
    print(f"template={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

