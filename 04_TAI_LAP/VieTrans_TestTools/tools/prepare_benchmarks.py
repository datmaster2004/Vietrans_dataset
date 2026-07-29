from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from common import (
    natural_key,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    write_json,
    write_jsonl,
)


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
TOTAL_TEXT_LINE = re.compile(
    r"^x:\s*\[\[(?P<x>.*?)\]\],\s*"
    r"y:\s*\[\[(?P<y>.*?)\]\],\s*"
    r"ornt:\s*\[(?:(?:u|b)?(?P<orientation_quote>['\"])(?P<orientation>.*?)(?P=orientation_quote))?\],\s*"
    r"transcriptions:\s*\[(?:u|b)?(?P<text_quote>['\"])(?P<text>.*)(?P=text_quote)\]\s*$"
)
SLOT_PATTERN = re.compile(r"\[([^:\]]+)\s*:\s*([^\]]+)\]")


def relative_to_manifest(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest_dir.resolve())).as_posix()


def archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_output(base: Path, relative: PurePosixPath) -> Path:
    target = (base / Path(*relative.parts)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe archive member: {relative}")
    return target


def extract_zip_selected(
    archive: Path,
    destination: Path,
    select: Callable[[PurePosixPath], PurePosixPath | None],
) -> int:
    count = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            source_name = PurePosixPath(info.filename)
            relative = select(source_name)
            if relative is None or info.is_dir():
                continue
            target = _safe_output(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != info.file_size:
                with handle.open(info) as source, target.open("wb") as output:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
            count += 1
    return count


def extract_tar_member(archive: Path, suffix: str, output: Path) -> None:
    normalized = suffix.replace("\\", "/").lstrip("./")
    with tarfile.open(archive, "r:gz") as handle:
        matches = [
            member
            for member in handle.getmembers()
            if member.isfile() and member.name.replace("\\", "/").lstrip("./").endswith(normalized)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one tar member ending in {suffix!r}, found {len(matches)}")
        source = handle.extractfile(matches[0])
        if source is None:
            raise ValueError(f"Could not read {matches[0].name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)


def _numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value)]


def parse_total_text_ground_truth(path: Path) -> list[dict[str, Any]]:
    records: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("x: [[") and current:
                records.append(" ".join(current))
                current = []
            current.append(line)
    if current:
        records.append(" ".join(current))

    regions: list[dict[str, Any]] = []
    for record_number, line in enumerate(records, start=1):
        match = TOTAL_TEXT_LINE.match(line)
        if not match:
            raise ValueError(f"{path}:record {record_number}: unsupported Total-Text annotation")
        orientation = match.group("orientation") or "unknown"
        text = match.group("text").replace("\\'", "'").strip()
        if orientation == "#" or text in {"#", "###"}:
            continue
        xs = _numbers(match.group("x"))
        ys = _numbers(match.group("y"))
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError(f"{path}:record {record_number}: invalid polygon")
        regions.append(
            {
                "text": text,
                "polygon": [[x, y] for x, y in zip(xs, ys)],
                "orientation": orientation,
            }
        )
    return regions


def _ocr_category(regions: list[dict[str, Any]]) -> str:
    values = {str(region.get("orientation") or "unknown") for region in regions}
    if len(values) > 1:
        return "mixed_orientation"
    value = next(iter(values), "unknown")
    return {"c": "curved", "h": "horizontal", "m": "multi_oriented"}.get(value, value)


def prepare_ocr(root: Path) -> dict[str, Any]:
    source_dir = root / "01_OCR" / "ToTalText"
    image_archive = source_dir / "totaltext.zip"
    gt_archive = source_dir / "total-text-groundtruth-text.zip"
    for path in (image_archive, gt_archive):
        if not path.is_file():
            raise FileNotFoundError(f"Required Total-Text archive is missing: {path}")

    prepared = root / "prepared" / "ocr_totaltext"

    def select_image(name: PurePosixPath) -> PurePosixPath | None:
        parts = name.parts
        if len(parts) == 3 and parts[:2] == ("Images", "Test") and name.suffix.lower() == ".jpg":
            return name
        return None

    def select_gt(name: PurePosixPath) -> PurePosixPath | None:
        parts = name.parts
        if (
            len(parts) == 2
            and parts[0] == "Test"
            and name.name.startswith("poly_gt_")
            and name.suffix.lower() == ".txt"
        ):
            return PurePosixPath("Groundtruth", *parts)
        return None

    extract_zip_selected(image_archive, prepared, select_image)
    extract_zip_selected(gt_archive, prepared, select_gt)
    image_dir = prepared / "Images" / "Test"
    gt_dir = prepared / "Groundtruth" / "Test"

    output_dir = root / "manifests" / "ocr_totaltext"
    rows: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*.jpg"), key=lambda item: natural_key(item.stem)):
        annotation_path = gt_dir / f"poly_gt_{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing Total-Text annotation: {annotation_path}")
        regions = parse_total_text_ground_truth(annotation_path)
        rows.append(
            {
                "case_id": f"ocr_totaltext_{image_path.stem}",
                "image": relative_to_manifest(image_path, output_dir),
                "category": _ocr_category(regions),
                "language": "en",
                "text": " ".join(region["text"] for region in regions),
                "regions": regions,
                "attributes": {
                    "dataset": "Total-Text",
                    "split": "test",
                    "source_image_id": image_path.stem,
                    "region_count": len(regions),
                },
            }
        )
    if len(rows) != 300:
        raise ValueError(f"Expected 300 Total-Text test images, found {len(rows)}")
    write_jsonl(output_dir / "manifest.jsonl", rows)
    card = {
        "name": "Total-Text test",
        "created_at": utc_now_iso(),
        "case_count": len(rows),
        "region_count": sum(len(row["regions"]) for row in rows),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "role": "secondary external OCR benchmark",
        "license_note": "Official README limits the images to non-commercial research.",
        "archives": [archive_record(image_archive), archive_record(gt_archive)],
        "headline_metrics": ["detection_f1", "matched_region_cer", "text_spotting_f1"],
    }
    write_json(output_dir / "dataset_card.json", card)
    return {"id": "ocr_totaltext_test", **card}


def _polygon_mask(size: tuple[int, int], annotation_path: Path):
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    polygon_count = 0
    for line_number, raw_line in enumerate(
        annotation_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(),
        start=1,
    ):
        values = [value.strip() for value in raw_line.split(",") if value.strip()]
        try:
            coordinates = [float(value) for value in values]
        except ValueError as exc:
            raise ValueError(f"{annotation_path}:{line_number}: invalid coordinate") from exc
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"{annotation_path}:{line_number}: invalid polygon")
        points = list(zip(coordinates[0::2], coordinates[1::2]))
        draw.polygon(points, fill=255)
        polygon_count += 1
    return mask, polygon_count


def prepare_scut(root: Path) -> dict[str, Any]:
    from PIL import Image, ImageOps

    source = root / "02_Inpainting" / "SCUT_EnsText" / "test_set" / "test"
    input_dir = source / "all_images"
    clean_dir = source / "all_labels"
    gt_dir = source / "all_gts"
    for path in (input_dir, clean_dir, gt_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"SCUT-EnsText test directory is missing: {path}")

    output_dir = root / "manifests" / "inpainting_scut_enstext"
    mask_dir = root / "prepared" / "inpainting_scut_enstext" / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    input_by_id = {path.stem: path for path in input_dir.glob("*.jpg")}
    clean_by_id = {path.stem: path for path in clean_dir.glob("*.jpg")}
    gt_by_id = {path.stem: path for path in gt_dir.glob("*.txt")}
    ids = sorted(input_by_id.keys() & clean_by_id.keys() & gt_by_id.keys(), key=natural_key)
    if len(ids) != 813:
        raise ValueError(
            "Expected 813 aligned SCUT-EnsText cases, found "
            f"{len(ids)} (input={len(input_by_id)}, clean={len(clean_by_id)}, gt={len(gt_by_id)})"
        )

    rows = []
    for index, image_id in enumerate(ids, start=1):
        input_path = input_by_id[image_id]
        clean_path = clean_by_id[image_id]
        with Image.open(input_path) as handle:
            input_size = ImageOps.exif_transpose(handle).size
        with Image.open(clean_path) as handle:
            clean_size = ImageOps.exif_transpose(handle).size
        if input_size != clean_size:
            raise ValueError(f"SCUT image/label size mismatch for {image_id}")
        mask, polygon_count = _polygon_mask(input_size, gt_by_id[image_id])
        mask_path = mask_dir / f"{image_id}.png"
        mask.save(mask_path)
        rows.append(
            {
                "case_id": f"inp_scut_{int(image_id):04d}" if image_id.isdigit() else f"inp_scut_{image_id}",
                "category": "real_world",
                "input_image": relative_to_manifest(input_path, output_dir),
                "clean_image": relative_to_manifest(clean_path, output_dir),
                "mask_image": relative_to_manifest(mask_path, output_dir),
                "attributes": {
                    "dataset": "SCUT-EnsText",
                    "split": "official_test",
                    "source_image_id": image_id,
                    "polygon_count": polygon_count,
                },
            }
        )
        if index % 200 == 0:
            print(f"SCUT masks: {index}/{len(ids)}")
    write_jsonl(output_dir / "manifest.jsonl", rows)
    card = {
        "name": "SCUT-EnsText official test",
        "created_at": utc_now_iso(),
        "case_count": len(rows),
        "role": "primary external real-world paired inpainting benchmark",
        "mask_generation": "Union of official all_gts polygons; no model-derived mask.",
        "headline_metrics": [
            "inside_psnr",
            "inside_ssim",
            "inside_mae",
            "outside_mae",
            "boundary_mae",
        ],
        "restriction": "Valid as final evidence only if the official test split was not used for training or tuning.",
    }
    write_json(output_dir / "dataset_card.json", card)
    return {"id": "inpainting_scut_enstext_test", **card}


def _image_from_hf_struct(value: Any):
    from PIL import Image

    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("OTR image column does not contain embedded bytes")
    with Image.open(BytesIO(value["bytes"])) as handle:
        return handle.convert("RGB")


def _safe_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty OTR id")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def prepare_otr(root: Path, limit: int | None = None) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    source_dir = root / "02_Inpainting" / "OTR_easy" / "data"
    shards = sorted(source_dir.glob("OTR_easy-*.parquet"))
    if len(shards) != 12:
        raise ValueError(f"Expected 12 OTR_easy shards, found {len(shards)}")
    smoke = limit is not None
    suffix = "inpainting_otr_easy_smoke" if smoke else "inpainting_otr_easy"
    materialized = root / "prepared" / suffix
    output_dir = root / "manifests" / suffix
    input_dir = materialized / "inputs"
    clean_dir = materialized / "clean"
    mask_dir = materialized / "masks"
    for directory in (input_dir, clean_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    stop = False
    columns = ["id", "image", "gt_image", "class", "words", "word_bboxes"]
    for shard_index, shard in enumerate(shards):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for raw in batch.to_pylist():
                source_id = _safe_id(raw["id"])
                if source_id in seen_ids:
                    raise ValueError(f"Duplicate OTR id: {source_id}")
                seen_ids.add(source_id)
                case_token = f"{int(source_id):05d}" if source_id.isdigit() else source_id
                input_path = input_dir / f"{source_id}.png"
                clean_path = clean_dir / f"{source_id}.png"
                mask_path = mask_dir / f"{source_id}.png"
                input_image = _image_from_hf_struct(raw["image"])
                clean_image = _image_from_hf_struct(raw["gt_image"])
                if input_image.size != clean_image.size:
                    raise ValueError(f"OTR image/gt_image size mismatch for {source_id}")
                if not input_path.is_file():
                    input_image.save(input_path, format="PNG", optimize=False)
                if not clean_path.is_file():
                    clean_image.save(clean_path, format="PNG", optimize=False)

                mask = Image.new("L", input_image.size, 0)
                draw = ImageDraw.Draw(mask)
                valid_boxes = []
                for box in raw.get("word_bboxes") or []:
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    x1, y1, x2, y2 = (int(value) for value in box)
                    left, right = sorted((x1, x2))
                    top, bottom = sorted((y1, y2))
                    left = max(0, min(input_image.width - 1, left))
                    right = max(0, min(input_image.width - 1, right))
                    top = max(0, min(input_image.height - 1, top))
                    bottom = max(0, min(input_image.height - 1, bottom))
                    if right > left and bottom > top:
                        draw.rectangle((left, top, right, bottom), fill=255)
                        valid_boxes.append([left, top, right, bottom])
                mask_array = np.asarray(mask, dtype=np.uint8)
                if not np.any(mask_array):
                    raise ValueError(f"OTR mask is empty for {source_id}")
                mask.save(mask_path)
                rows.append(
                    {
                        "case_id": f"inp_otr_{case_token}",
                        "category": str(raw.get("class") or "unclassified"),
                        "input_image": relative_to_manifest(input_path, output_dir),
                        "clean_image": relative_to_manifest(clean_path, output_dir),
                        "mask_image": relative_to_manifest(mask_path, output_dir),
                        "attributes": {
                            "dataset": "OTR_easy",
                            "split": "official",
                            "source_id": source_id,
                            "source_shard": shard.name,
                            "words": raw.get("words") or [],
                            "word_bboxes": valid_boxes,
                        },
                    }
                )
                if len(rows) % 250 == 0:
                    print(f"OTR materialized: {len(rows)}")
                if limit is not None and len(rows) >= limit:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    expected = limit if limit is not None else 5538
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} OTR cases, found {len(rows)}")
    write_jsonl(output_dir / "manifest.jsonl", rows)
    card = {
        "name": "OTR_easy smoke" if smoke else "OTR_easy full benchmark",
        "created_at": utc_now_iso(),
        "case_count": len(rows),
        "role": "smoke test only" if smoke else "supplemental external paired inpainting benchmark",
        "source_shards": [archive_record(path) for path in shards],
        "mask_generation": "Union of official word_bboxes after coordinate normalization; no model-derived mask.",
        "headline_metrics": [
            "inside_psnr",
            "inside_ssim",
            "inside_mae",
            "outside_mae",
            "boundary_mae",
        ],
        "final_report_allowed": not smoke,
        "restriction": "Valid only if OTR_easy was not used for training, tuning or model selection.",
    }
    write_json(output_dir / "dataset_card.json", card)
    return {"id": suffix, **card}


def _slot_values(annotated: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for slot, value in SLOT_PATTERN.findall(annotated or ""):
        output[slot.strip()].append(value.strip())
    return output


def _judgment_summary(row: dict[str, Any]) -> dict[str, Any]:
    judgments = row.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        return {}

    def average(key: str) -> float | None:
        values = [item.get(key) for item in judgments if isinstance(item, dict)]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        return sum(numeric) / len(numeric) if numeric else None

    return {
        "judgment_count": len(judgments),
        "grammar_score_mean": average("grammar_score"),
        "spelling_score_mean": average("spelling_score"),
        "intent_score_mean": average("intent_score"),
    }


def prepare_translation(root: Path) -> list[dict[str, Any]]:
    flores_archive = root / "03_Trans" / "FLORES200" / "flores200_dataset.tar.gz"
    massive_archive = root / "03_Trans" / "MASSIVE_1.0" / "amazon-massive-dataset-1.0.tar.gz"
    for path in (flores_archive, massive_archive):
        if not path.is_file():
            raise FileNotFoundError(f"Required translation archive is missing: {path}")

    flores_source = root / "prepared" / "translation_flores200"
    for suffix, relative in (
        ("flores200_dataset/devtest/eng_Latn.devtest", "devtest/eng_Latn.devtest"),
        ("flores200_dataset/devtest/vie_Latn.devtest", "devtest/vie_Latn.devtest"),
        ("flores200_dataset/metadata_devtest.tsv", "metadata_devtest.tsv"),
    ):
        output = flores_source / relative
        if not output.is_file():
            extract_tar_member(flores_archive, suffix, output)
    english = (flores_source / "devtest" / "eng_Latn.devtest").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    vietnamese = (flores_source / "devtest" / "vie_Latn.devtest").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    with (flores_source / "metadata_devtest.tsv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    if not (len(english) == len(vietnamese) == len(metadata) == 1012):
        raise ValueError(
            f"FLORES alignment mismatch: en={len(english)}, vi={len(vietnamese)}, "
            f"metadata={len(metadata)}"
        )
    flores_rows = []
    for index, (source, target, attributes) in enumerate(
        zip(english, vietnamese, metadata), start=1
    ):
        flores_rows.append(
            {
                "case_id": f"mt_flores200_{index:04d}",
                "category": str(attributes.get("domain") or "unclassified"),
                "source_text": source.strip(),
                "references": [target.strip()],
                "terminology": [],
                "attributes": {
                    "dataset": "FLORES-200",
                    "partition": "devtest",
                    "aligned_row": index,
                    "source_locale": "eng_Latn",
                    "target_locale": "vie_Latn",
                    "url": attributes.get("URL"),
                    "topic": attributes.get("topic"),
                },
            }
        )
    flores_output = root / "manifests" / "translation_flores200_en_vi"
    write_jsonl(flores_output / "manifest.jsonl", flores_rows)
    flores_card = {
        "name": "FLORES-200 English-Vietnamese devtest",
        "created_at": utc_now_iso(),
        "case_count": len(flores_rows),
        "role": "primary external machine-translation benchmark",
        "license": "CC BY-SA 4.0",
        "archive": archive_record(flores_archive),
        "headline_metrics": ["corpus_chrf_plus_plus", "corpus_bleu"],
        "restriction": "Report-only: do not tune decoding or select checkpoints on devtest.",
    }
    write_json(flores_output / "dataset_card.json", flores_card)

    massive_source = root / "prepared" / "translation_massive"
    for locale in ("en-US", "vi-VN"):
        output = massive_source / f"{locale}.jsonl"
        if not output.is_file():
            extract_tar_member(massive_archive, f"1.0/data/{locale}.jsonl", output)
    english_rows = {
        str(row["id"]): row
        for row in read_jsonl(massive_source / "en-US.jsonl")
        if row.get("partition") == "test"
    }
    vietnamese_rows = {
        str(row["id"]): row
        for row in read_jsonl(massive_source / "vi-VN.jsonl")
        if row.get("partition") == "test"
    }
    ids = sorted(english_rows.keys() & vietnamese_rows.keys(), key=natural_key)
    massive_rows = []
    for case_id in ids:
        source = english_rows[case_id]
        target = vietnamese_rows[case_id]
        source_slots = _slot_values(str(source.get("annot_utt") or ""))
        target_slots = _slot_values(str(target.get("annot_utt") or ""))
        terminology = []
        for slot in sorted(source_slots.keys() & target_slots.keys()):
            for source_value, target_value in zip(source_slots[slot], target_slots[slot]):
                terminology.append(
                    {"source": source_value, "accepted": [target_value], "slot": slot}
                )
        massive_rows.append(
            {
                "case_id": (
                    f"mt_massive_{int(case_id):05d}"
                    if case_id.isdigit()
                    else f"mt_massive_{case_id}"
                ),
                "category": str(source.get("scenario") or "unclassified"),
                "source_text": str(source.get("utt") or "").strip(),
                "references": [str(target.get("utt") or "").strip()],
                "terminology": terminology,
                "attributes": {
                    "dataset": "MASSIVE 1.0",
                    "partition": "test",
                    "source_id": case_id,
                    "source_locale": "en-US",
                    "target_locale": "vi-VN",
                    "intent": source.get("intent"),
                    **_judgment_summary(target),
                },
            }
        )
    if len(massive_rows) != 2974:
        raise ValueError(f"Expected 2,974 MASSIVE test pairs, found {len(massive_rows)}")
    massive_output = root / "manifests" / "translation_massive_en_vi"
    write_jsonl(massive_output / "manifest.jsonl", massive_rows)
    massive_card = {
        "name": "MASSIVE English-Vietnamese test",
        "created_at": utc_now_iso(),
        "case_count": len(massive_rows),
        "role": "supplemental external short-utterance benchmark",
        "license": "CC BY 4.0",
        "archive": archive_record(massive_archive),
        "headline_metrics": [
            "corpus_chrf_plus_plus",
            "corpus_bleu",
            "terminology_accuracy",
        ],
        "restriction": "Report as assistant short utterances, not general document translation.",
    }
    write_json(massive_output / "dataset_card.json", massive_card)
    return [
        {"id": "translation_flores200_devtest_en_vi", **flores_card},
        {"id": "translation_massive_test_en_vi", **massive_card},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible manifests for the local VieTrans benchmark folders."
    )
    parser.add_argument(
        "--suite",
        choices=["all", "ocr", "scut", "otr", "translation"],
        default="all",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--limit-otr",
        type=int,
        help="Build a separate OTR smoke manifest. Omit for the official 5,538-case manifest.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summaries: list[dict[str, Any]] = []
    if args.suite in {"all", "ocr"}:
        summaries.append(prepare_ocr(root))
    if args.suite in {"all", "scut"}:
        summaries.append(prepare_scut(root))
    if args.suite in {"all", "otr"}:
        summaries.append(prepare_otr(root, args.limit_otr))
    if args.suite in {"all", "translation"}:
        summaries.extend(prepare_translation(root))
    write_json(root / "00_Doc" / "prepared_benchmarks.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

