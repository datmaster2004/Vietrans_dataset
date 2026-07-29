from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import read_json, utc_now_iso, write_json


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate component summary.json files.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    raw = []
    for path in args.summaries:
        payload = read_json(path)
        raw.append(payload)
        corpus = payload.get("corpus_metrics") or {}
        metrics = payload.get("metrics") or {}
        selected = {
            **{key: value for key, value in corpus.items() if not isinstance(value, dict)},
            **metrics,
        }
        rows.append(
            {
                "summary": str(path.resolve()),
                "suite": payload.get("suite"),
                "coverage": payload.get("coverage"),
                "case_count": payload.get("case_count"),
                "error_count": payload.get("error_count"),
                "metrics": selected,
            }
        )
    output = args.output or args.root / "results" / "AGGREGATE_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VieTrans evaluation report",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "| Suite | Cases | Coverage | Errors | Metrics |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metric_text = "; ".join(
            f"{key}={format_value(value)}" for key, value in sorted(row["metrics"].items())
        )
        lines.append(
            f"| {row['suite']} | {row['case_count']} | {format_value(row['coverage'])} | "
            f"{row['error_count']} | {metric_text} |"
        )
    lines.extend(
        [
            "",
            "Do not average scores across unrelated datasets or components.",
            "A result is reportable only when coverage is 1.0, errors are 0, and provenance is frozen.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    write_json(output.with_suffix(".json"), {"generated_at": utc_now_iso(), "summaries": raw})
    print(f"report={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

