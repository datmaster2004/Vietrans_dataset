from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from common import read_jsonl, sha256_file, utc_now_iso, write_json


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_ARCHIVES = {
    "01_OCR/ToTalText/totaltext.zip": {
        "bytes": 432596071,
        "sha256": "bb2f555b5166deaca0ad04175e5e4b351f4255ab10a78aa47d3f035f7efd1e98",
    },
    "01_OCR/ToTalText/total-text-groundtruth-text.zip": {
        "bytes": 1451724,
        "sha256": "7f2c8007ddd200d1c11af5a7075594f04ab0c096fbf86782180c8cc9a34f62c2",
    },
    "03_Trans/FLORES200/flores200_dataset.tar.gz": {
        "bytes": 25585843,
        "sha256": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
    },
    "03_Trans/MASSIVE_1.0/amazon-massive-dataset-1.0.tar.gz": {
        "bytes": 39500415,
        "sha256": "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8",
    },
}
EXPECTED_MANIFESTS = {
    "manifests/ocr_totaltext/manifest.jsonl": 300,
    "manifests/inpainting_scut_enstext/manifest.jsonl": 813,
    "manifests/inpainting_otr_easy/manifest.jsonl": 5538,
    "manifests/translation_flores200_en_vi/manifest.jsonl": 1012,
    "manifests/translation_massive_en_vi/manifest.jsonl": 2974,
}


def check(condition: bool, name: str, detail: Any, severity: str = "error") -> dict[str, Any]:
    return {
        "name": name,
        "status": "ok" if condition else severity,
        "detail": detail,
    }


def validate_manifest(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    ids = [str(row.get("case_id") or "") for row in rows]
    results = [
        check(len(rows) == expected, f"{path.name}:case_count", {"actual": len(rows), "expected": expected}),
        check(all(ids), f"{path.name}:non_empty_ids", {"empty": sum(not value for value in ids)}),
        check(
            len(ids) == len(set(ids)),
            f"{path.name}:unique_ids",
            {"duplicate_count": len(ids) - len(set(ids))},
        ),
    ]
    return results


def validate_otr(root: Path, deep_hash: bool) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    data_dir = root / "02_Inpainting" / "OTR_easy" / "data"
    shards = sorted(data_dir.glob("OTR_easy-*.parquet"))
    results = [check(len(shards) == 12, "OTR_easy:shard_count", {"actual": len(shards), "expected": 12})]
    total_rows = 0
    required = {"id", "image", "gt_image", "class", "words", "word_bboxes"}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        total_rows += parquet.metadata.num_rows
        schema_names = set(parquet.schema_arrow.names)
        results.append(
            check(
                required.issubset(schema_names),
                f"OTR_easy:{shard.name}:schema",
                {"columns": sorted(schema_names)},
            )
        )
        metadata = (
            root
            / "02_Inpainting"
            / "OTR_easy"
            / ".cache"
            / "huggingface"
            / "download"
            / "data"
            / f"{shard.name}.metadata"
        )
        if deep_hash and metadata.is_file():
            lines = metadata.read_text(encoding="utf-8-sig").splitlines()
            expected_hash = lines[1].strip().lower() if len(lines) > 1 else ""
            actual_hash = sha256_file(shard)
            results.append(
                check(
                    actual_hash == expected_hash,
                    f"OTR_easy:{shard.name}:sha256",
                    {"actual": actual_hash, "expected": expected_hash},
                )
            )
    results.append(check(total_rows == 5538, "OTR_easy:row_count", {"actual": total_rows, "expected": 5538}))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local VieTrans evaluation data and manifests.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--deep-hash", action="store_true", help="Hash every OTR shard (slower).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results: list[dict[str, Any]] = []

    for relative, expected in KNOWN_ARCHIVES.items():
        path = root / relative
        exists = path.is_file()
        results.append(check(exists, f"{relative}:exists", str(path)))
        if not exists:
            continue
        results.append(
            check(
                path.stat().st_size == expected["bytes"],
                f"{relative}:bytes",
                {"actual": path.stat().st_size, "expected": expected["bytes"]},
            )
        )
        actual_hash = sha256_file(path)
        results.append(
            check(
                actual_hash == expected["sha256"],
                f"{relative}:sha256",
                {"actual": actual_hash, "expected": expected["sha256"]},
            )
        )

    totaltext_archive = root / "01_OCR" / "ToTalText" / "totaltext.zip"
    if totaltext_archive.is_file():
        with zipfile.ZipFile(totaltext_archive) as handle:
            images = [
                name
                for name in handle.namelist()
                if re_match_totaltext_image(name)
            ]
        results.append(check(len(images) == 300, "Total-Text:test_image_count", {"actual": len(images), "expected": 300}))

    scut = root / "02_Inpainting" / "SCUT_EnsText" / "test_set" / "test"
    scut_sets = {}
    for directory, extension in (("all_images", ".jpg"), ("all_labels", ".jpg"), ("all_gts", ".txt")):
        path = scut / directory
        stems = {item.stem for item in path.glob(f"*{extension}")} if path.is_dir() else set()
        scut_sets[directory] = stems
        results.append(check(len(stems) == 813, f"SCUT:{directory}:count", {"actual": len(stems), "expected": 813}))
    results.append(
        check(
            scut_sets["all_images"] == scut_sets["all_labels"] == scut_sets["all_gts"],
            "SCUT:aligned_ids",
            {
                "input_only": len(scut_sets["all_images"] - scut_sets["all_labels"]),
                "clean_only": len(scut_sets["all_labels"] - scut_sets["all_images"]),
            },
        )
    )
    results.extend(validate_otr(root, args.deep_hash))

    iimt = root / "02_Inpainting" / "IIMT30K"
    iimt_files = list(iimt.rglob("*")) if iimt.is_dir() else []
    results.append(
        check(
            not any(path.is_file() for path in iimt_files),
            "IIMT30K:not_used_as_final_test",
            "Directory is empty; this is expected because IIMT30k_Vi is training/development data.",
            severity="warning",
        )
    )

    for relative, expected in EXPECTED_MANIFESTS.items():
        path = root / relative
        if path.is_file():
            results.extend(validate_manifest(path, expected))
        else:
            results.append(
                check(
                    False,
                    f"{relative}:prepared",
                    "Not prepared yet; run tools/prepare_benchmarks.py.",
                    severity="warning",
                )
            )

    errors = [item for item in results if item["status"] == "error"]
    warnings = [item for item in results if item["status"] == "warning"]
    payload = {
        "generated_at": utc_now_iso(),
        "root": str(root),
        "deep_hash": args.deep_hash,
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": results,
    }
    output = args.output or root / "00_Doc" / "dataset_inventory.json"
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def re_match_totaltext_image(name: str) -> bool:
    parts = PurePath(name)
    return (
        len(parts) == 3
        and parts[0] == "Images"
        and parts[1] == "Test"
        and parts[2].lower().endswith(".jpg")
        and not parts[2].startswith("._")
    )


def PurePath(name: str) -> tuple[str, ...]:
    return tuple(part for part in name.replace("\\", "/").split("/") if part)


if __name__ == "__main__":
    raise SystemExit(main())

