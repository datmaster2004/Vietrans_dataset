from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


def resolve(base: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base.resolve().parent / path
    return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a tiny oracle subset only to self-test evaluator wiring."
    )
    parser.add_argument("suite", choices=["ocr", "inpainting", "translation"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    source_rows = read_jsonl(args.manifest)[: args.limit]
    rows = []
    predictions = []
    for source_row in source_rows:
        row = dict(source_row)
        if args.suite == "ocr":
            if row.get("image"):
                row["image"] = resolve(args.manifest, row["image"])
            predictions.append(
                {
                    "case_id": row["case_id"],
                    "regions": row.get("regions") or [],
                    "text": row.get("text") or "",
                }
            )
        elif args.suite == "inpainting":
            for field in ("input_image", "clean_image", "mask_image"):
                row[field] = resolve(args.manifest, row[field])
            predictions.append(
                {
                    "case_id": row["case_id"],
                    "output_image": row["clean_image"],
                }
            )
        else:
            references = row.get("references") or []
            predictions.append(
                {
                    "case_id": row["case_id"],
                    "translation": references[0],
                }
            )
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "manifest.jsonl", rows)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    print("WARNING: oracle files are wiring tests, never model results.")
    print(f"cases={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
