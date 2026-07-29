from __future__ import annotations

import argparse
import math
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from common import (
    normalize_text,
    read_jsonl,
    sha256_file,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from quality_metrics import compute_ocr_ground_truth_metrics


def _identifier(row: dict[str, Any], label: str) -> str:
    value = str(row.get("case_id") or row.get("testcase_id") or "").strip()
    if not value:
        raise ValueError(f"{label} row is missing case_id")
    return value


def _index_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = _identifier(row, label)
        if case_id in indexed:
            raise ValueError(f"duplicate {label} case_id: {case_id}")
        indexed[case_id] = row
    return indexed


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_metrics(rows: list[dict[str, Any]], metric_names: Iterable[str]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name in metric_names:
        values = [_number(row.get(name)) for row in rows]
        usable = [value for value in values if value is not None]
        output[name] = mean(usable) if usable else None
    return output


def _build_summary(
    suite: str,
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    metric_names: list[str],
    corpus_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successful = [row for row in cases if row.get("status") == "success"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[str(row.get("category") or "unclassified")].append(row)
    categories = []
    for category, rows in sorted(grouped.items()):
        scored = [row for row in rows if row.get("status") == "success"]
        categories.append(
            {
                "category": category,
                "case_count": len(rows),
                "scored_count": len(scored),
                "metrics": _mean_metrics(scored, metric_names),
            }
        )
    manifest_ids = {_identifier(row, "manifest") for row in manifest_rows}
    prediction_ids = {_identifier(row, "prediction") for row in prediction_rows}
    return {
        "suite": suite,
        "generated_at": utc_now_iso(),
        "case_count": len(manifest_rows),
        "prediction_count": len(prediction_rows),
        "scored_count": len(successful),
        "coverage": len(successful) / len(manifest_rows) if manifest_rows else 0.0,
        "missing_prediction_count": len(manifest_ids - prediction_ids),
        "extra_prediction_count": len(prediction_ids - manifest_ids),
        "error_count": len(cases) - len(successful),
        "metrics": _mean_metrics(successful, metric_names),
        "corpus_metrics": corpus_metrics or {},
        "categories": categories,
    }


def _text_from_row(row: dict[str, Any], prediction: bool = False) -> str:
    keys = ("text", "ocr_text", "hypothesis_text") if prediction else ("text", "ocr_text")
    for key in keys:
        if row.get(key) is not None:
            return normalize_text(row[key])
    regions = row.get("regions")
    if isinstance(regions, list):
        return normalize_text(
            " ".join(str(region.get("text") or "") for region in regions if isinstance(region, dict))
        )
    return ""


def _as_box(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("bbox", "box", "polygon", "points"):
            if value.get(key) is not None:
                return _as_box(value[key])
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = (float(item) for item in value)
        return [x1, y1, x2, y2]
    points = [point for point in value if isinstance(point, (list, tuple)) and len(point) >= 2]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _boxes_from_row(row: dict[str, Any]) -> tuple[list[list[float]], bool]:
    if "regions" not in row:
        return [], False
    regions = row.get("regions")
    if not isinstance(regions, list):
        raise ValueError("regions must be a list")
    boxes = [_as_box(region) for region in regions]
    return [box for box in boxes if box is not None], True


def _as_polygon(value: Any) -> list[list[float]] | None:
    if isinstance(value, dict):
        for key in ("polygon", "points", "bbox", "box"):
            if value.get(key) is not None:
                return _as_polygon(value[key])
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = (float(item) for item in value)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points = [
        [float(point[0]), float(point[1])]
        for point in value
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    return points if len(points) >= 3 else None


def _regions_from_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    if "regions" not in row:
        return [], False
    raw_regions = row.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("regions must be a list")
    regions: list[dict[str, Any]] = []
    for raw in raw_regions:
        if not isinstance(raw, dict):
            continue
        polygon = _as_polygon(raw)
        if polygon is None:
            continue
        regions.append({"polygon": polygon, "text": normalize_text(raw.get("text") or "")})
    return regions, True


def _polygon_iou(left: list[list[float]], right: list[list[float]]) -> float:
    import numpy as np
    from PIL import Image, ImageDraw

    xs = [point[0] for point in left] + [point[0] for point in right]
    ys = [point[1] for point in left] + [point[1] for point in right]
    min_x, max_x = math.floor(min(xs)), math.ceil(max(xs))
    min_y, max_y = math.floor(min(ys)), math.ceil(max(ys))
    width, height = max_x - min_x + 3, max_y - min_y + 3
    if width <= 0 or height <= 0:
        return 0.0
    # Bound memory for malformed coordinates while preserving normal image-space polygons.
    scale = min(1.0, 4096.0 / max(width, height))
    size = (max(1, int(math.ceil(width * scale))), max(1, int(math.ceil(height * scale))))

    def rasterize(points: list[list[float]]):
        image = Image.new("1", size, 0)
        shifted = [
            ((point[0] - min_x + 1) * scale, (point[1] - min_y + 1) * scale)
            for point in points
        ]
        ImageDraw.Draw(image).polygon(shifted, fill=1)
        return np.asarray(image, dtype=bool)

    left_mask = rasterize(left)
    right_mask = rasterize(right)
    union = int(np.count_nonzero(left_mask | right_mask))
    return float(np.count_nonzero(left_mask & right_mask)) / union if union else 0.0


def _match_regions(
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    pairs = sorted(
        (
            (_polygon_iou(pred["polygon"], ref["polygon"]), pred_index, ref_index)
            for pred_index, pred in enumerate(predicted)
            for ref_index, ref in enumerate(reference)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, predicted_index, reference_index in pairs:
        if iou < iou_threshold:
            break
        if predicted_index in used_predictions or reference_index in used_references:
            continue
        used_predictions.add(predicted_index)
        used_references.add(reference_index)
        matches.append((predicted_index, reference_index, iou))
    return matches


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


OCR_METRICS = [
    "ocr_cer",
    "ocr_wer",
    "ocr_accuracy_score",
    "ocr_exact_match",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "matched_region_cer",
    "region_recognition_exact_rate",
    "text_spotting_precision",
    "text_spotting_recall",
    "text_spotting_f1",
]


def evaluate_ocr(
    manifest_path: Path,
    predictions_path: Path,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    manifest_rows = read_jsonl(manifest_path)
    prediction_rows = read_jsonl(predictions_path)
    predictions = _index_rows(prediction_rows, "prediction")
    _index_rows(manifest_rows, "manifest")
    cases: list[dict[str, Any]] = []
    for reference in manifest_rows:
        case_id = _identifier(reference, "manifest")
        base = {
            "case_id": case_id,
            "category": str(reference.get("category") or "unclassified"),
        }
        prediction = predictions.get(case_id)
        if prediction is None:
            cases.append({**base, "status": "missing_prediction", "error": "prediction not found"})
            continue
        try:
            reference_text = _text_from_row(reference)
            hypothesis_text = _text_from_row(prediction, prediction=True)
            if not reference_text and "text" not in reference and "ocr_text" not in reference:
                raise ValueError("manifest case needs text/ocr_text or regions with text")
            text_metrics = compute_ocr_ground_truth_metrics(hypothesis_text, reference_text)
            reference_boxes, boxes_are_labeled = _boxes_from_row(reference)
            predicted_boxes, _ = _boxes_from_row(prediction)
            reference_regions, regions_are_labeled = _regions_from_row(reference)
            predicted_regions, _ = _regions_from_row(prediction)
            detection_metrics: dict[str, Any] = {
                "detection_precision": None,
                "detection_recall": None,
                "detection_f1": None,
                "detection_matched_boxes": None,
                "matched_region_cer": None,
                "region_recognition_exact_rate": None,
                "text_spotting_precision": None,
                "text_spotting_recall": None,
                "text_spotting_f1": None,
            }
            if boxes_are_labeled and regions_are_labeled:
                matches = _match_regions(predicted_regions, reference_regions, iou_threshold)
                matched = len(matches)
                detection_precision = (
                    matched / len(predicted_regions)
                    if predicted_regions
                    else (1.0 if not reference_regions else 0.0)
                )
                detection_recall = (
                    matched / len(reference_regions)
                    if reference_regions
                    else (1.0 if not predicted_regions else 0.0)
                )
                exact_matches = 0
                region_cers: list[float] = []
                for predicted_index, reference_index, _ in matches:
                    predicted_text = predicted_regions[predicted_index]["text"]
                    reference_region_text = reference_regions[reference_index]["text"]
                    region_metrics = compute_ocr_ground_truth_metrics(
                        predicted_text, reference_region_text
                    )
                    region_cers.append(float(region_metrics["ocr_cer"]))
                    exact_matches += int(
                        normalize_text(predicted_text).casefold()
                        == normalize_text(reference_region_text).casefold()
                    )
                spotting_precision = (
                    exact_matches / len(predicted_regions) if predicted_regions else 0.0
                )
                spotting_recall = (
                    exact_matches / len(reference_regions) if reference_regions else 0.0
                )
                detection_metrics.update(
                    {
                        "detection_precision": detection_precision * 100.0,
                        "detection_recall": detection_recall * 100.0,
                        "detection_f1": _f1(detection_precision, detection_recall) * 100.0,
                        "detection_matched_boxes": matched,
                        "matched_region_cer": mean(region_cers) if region_cers else None,
                        "region_recognition_exact_rate": (
                            exact_matches / matched * 100.0 if matched else 0.0
                        ),
                        "text_spotting_precision": spotting_precision * 100.0,
                        "text_spotting_recall": spotting_recall * 100.0,
                        "text_spotting_f1": _f1(spotting_precision, spotting_recall) * 100.0,
                    }
                )
            normalized_reference = normalize_text(reference_text).casefold()
            normalized_hypothesis = normalize_text(hypothesis_text).casefold()
            cases.append(
                {
                    **base,
                    "status": "success",
                    "reference_text": reference_text,
                    "hypothesis_text": hypothesis_text,
                    **text_metrics,
                    "ocr_exact_match": float(normalized_hypothesis == normalized_reference) * 100.0,
                    **detection_metrics,
                    "reference_box_count": len(reference_boxes),
                    "predicted_box_count": len(predicted_boxes),
                }
            )
        except Exception as exc:
            cases.append({**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return (
        _build_summary("ocr", manifest_rows, prediction_rows, cases, OCR_METRICS),
        cases,
    )


def _resolve_path(base_file: Path, value: Any, field: str) -> Path:
    if not str(value or "").strip():
        raise ValueError(f"missing {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_file.resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path


def _masked_mae(left, right, mask) -> float | None:
    import numpy as np

    if not np.any(mask):
        return None
    return float(np.mean(np.abs(left[mask] - right[mask])))


def _masked_psnr(left, right, mask) -> float | None:
    import numpy as np

    if not np.any(mask):
        return None
    error = left[mask] - right[mask]
    mse = float(np.mean(error * error))
    return 100.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def _ssim_map(left, right):
    import numpy as np
    from skimage.metrics import structural_similarity

    _, similarity = structural_similarity(
        left,
        right,
        data_range=255.0,
        channel_axis=2,
        full=True,
    )
    if similarity.ndim == 3:
        similarity = np.mean(similarity, axis=2)
    return similarity


def _masked_ssim(similarity, mask) -> float | None:
    import numpy as np

    if not np.any(mask):
        return None
    return float(np.mean(similarity[mask]))


def _dilate(mask, radius: int = 2):
    import numpy as np

    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask)
    for y_offset in range(radius * 2 + 1):
        for x_offset in range(radius * 2 + 1):
            output |= padded[y_offset : y_offset + height, x_offset : x_offset + width]
    return output


INPAINTING_METRICS = [
    "full_psnr",
    "full_ssim",
    "inside_mae",
    "inside_psnr",
    "inside_ssim",
    "inside_similarity_score",
    "error_reduction_percent",
    "outside_mae",
    "outside_psnr",
    "outside_ssim",
    "outside_preservation_score",
    "boundary_mae",
    "boundary_ssim",
]


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    return image.size, np.asarray(image, dtype=np.float32)


def _load_mask(path: Path, expected_size: tuple[int, int]):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("L")
    if image.size != expected_size:
        raise ValueError(f"mask size {image.size} does not match image size {expected_size}")
    return np.asarray(image, dtype=np.uint8) >= 128


def evaluate_inpainting(
    manifest_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    manifest_rows = read_jsonl(manifest_path)
    prediction_rows = read_jsonl(predictions_path)
    predictions = _index_rows(prediction_rows, "prediction")
    _index_rows(manifest_rows, "manifest")
    cases: list[dict[str, Any]] = []
    for reference in manifest_rows:
        case_id = _identifier(reference, "manifest")
        base = {
            "case_id": case_id,
            "category": str(reference.get("category") or "unclassified"),
        }
        prediction = predictions.get(case_id)
        if prediction is None:
            cases.append({**base, "status": "missing_prediction", "error": "prediction not found"})
            continue
        try:
            input_path = _resolve_path(manifest_path, reference.get("input_image"), "input_image")
            clean_path = _resolve_path(manifest_path, reference.get("clean_image"), "clean_image")
            mask_path = _resolve_path(manifest_path, reference.get("mask_image"), "mask_image")
            output_path = _resolve_path(predictions_path, prediction.get("output_image"), "output_image")
            input_size, input_image = _load_rgb(input_path)
            clean_size, clean_image = _load_rgb(clean_path)
            output_size, output_image = _load_rgb(output_path)
            if clean_size != input_size or output_size != input_size:
                raise ValueError(
                    f"image sizes must match: input={input_size}, clean={clean_size}, output={output_size}"
                )
            mask = _load_mask(mask_path, input_size)
            if not np.any(mask):
                raise ValueError("mask_image contains no selected pixels")
            outside = ~mask
            boundary = _dilate(mask, 2) & outside
            full = np.ones(mask.shape, dtype=bool)
            similarity = _ssim_map(output_image, clean_image)
            inside_mae = _masked_mae(output_image, clean_image, mask)
            input_inside_mae = _masked_mae(input_image, clean_image, mask)
            error_reduction = None
            if input_inside_mae is not None and input_inside_mae > 1e-12 and inside_mae is not None:
                error_reduction = (1.0 - inside_mae / input_inside_mae) * 100.0
            outside_mae = _masked_mae(output_image, input_image, outside)
            cases.append(
                {
                    **base,
                    "status": "success",
                    "full_psnr": _masked_psnr(output_image, clean_image, full),
                    "full_ssim": _masked_ssim(similarity, full),
                    "inside_mae": inside_mae,
                    "inside_psnr": _masked_psnr(output_image, clean_image, mask),
                    "inside_ssim": _masked_ssim(similarity, mask),
                    "inside_similarity_score": (
                        max(0.0, min(100.0, (1.0 - inside_mae / 255.0) * 100.0))
                        if inside_mae is not None
                        else None
                    ),
                    "input_inside_mae": input_inside_mae,
                    "error_reduction_percent": error_reduction,
                    "outside_mae": outside_mae,
                    "outside_psnr": _masked_psnr(output_image, input_image, outside),
                    "outside_ssim": _masked_ssim(
                        _ssim_map(output_image, input_image),
                        outside,
                    ),
                    "outside_preservation_score": (
                        max(0.0, min(100.0, (1.0 - outside_mae / 255.0) * 100.0))
                        if outside_mae is not None
                        else None
                    ),
                    "boundary_mae": _masked_mae(output_image, clean_image, boundary),
                    "boundary_ssim": _masked_ssim(similarity, boundary),
                    "mask_pixel_count": int(np.count_nonzero(mask)),
                    "mask_ratio": float(np.mean(mask)),
                }
            )
        except Exception as exc:
            cases.append({**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return (
        _build_summary(
            "inpainting", manifest_rows, prediction_rows, cases, INPAINTING_METRICS
        ),
        cases,
    )


def _references(row: dict[str, Any]) -> list[str]:
    raw = row.get("references")
    if isinstance(raw, list):
        references = [normalize_text(value) for value in raw if normalize_text(value)]
    else:
        single = normalize_text(row.get("reference") or row.get("target_text") or "")
        references = [single] if single else []
    if not references:
        raise ValueError("translation case needs at least one non-empty reference")
    return references


def _hypothesis(row: dict[str, Any]) -> str:
    value = row.get("translation", row.get("hypothesis_text", row.get("text")))
    if value is None:
        raise ValueError("translation prediction needs translation or hypothesis_text")
    return normalize_text(value)


def _term_accuracy(hypothesis: str, requirements: Any) -> tuple[float | None, int, int]:
    if not isinstance(requirements, list) or not requirements:
        return None, 0, 0
    normalized_hypothesis = normalize_text(hypothesis).casefold()
    matched = 0
    total = 0
    for requirement in requirements:
        if isinstance(requirement, str):
            accepted = [requirement]
        elif isinstance(requirement, dict):
            raw = requirement.get("accepted") or requirement.get("targets") or []
            accepted = raw if isinstance(raw, list) else [raw]
        else:
            continue
        candidates = [normalize_text(value).casefold() for value in accepted if normalize_text(value)]
        if not candidates:
            continue
        total += 1
        matched += int(any(candidate in normalized_hypothesis for candidate in candidates))
    return (matched / total * 100.0 if total else None), matched, total


TRANSLATION_METRICS = [
    "sentence_bleu",
    "sentence_chrf",
    "sentence_chrf_plus_plus",
    "edit_similarity",
    "exact_match",
    "terminology_accuracy",
]


def evaluate_translation(
    manifest_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import sacrebleu

    manifest_rows = read_jsonl(manifest_path)
    prediction_rows = read_jsonl(predictions_path)
    predictions = _index_rows(prediction_rows, "prediction")
    _index_rows(manifest_rows, "manifest")
    cases: list[dict[str, Any]] = []
    corpus_rows: list[tuple[str, list[str]]] = []
    for reference in manifest_rows:
        case_id = _identifier(reference, "manifest")
        base = {
            "case_id": case_id,
            "category": str(reference.get("category") or "unclassified"),
        }
        prediction = predictions.get(case_id)
        if prediction is None:
            cases.append({**base, "status": "missing_prediction", "error": "prediction not found"})
            continue
        try:
            source_text = normalize_text(reference.get("source_text") or "")
            if not source_text:
                raise ValueError("translation case needs source_text")
            references = _references(reference)
            hypothesis = _hypothesis(prediction)
            term_score, term_matched, term_total = _term_accuracy(
                hypothesis, reference.get("terminology")
            )
            normalized_hypothesis = hypothesis.casefold()
            edit_similarity = max(
                SequenceMatcher(None, normalized_hypothesis, item.casefold()).ratio() * 100.0
                for item in references
            )
            exact_match = any(normalized_hypothesis == item.casefold() for item in references)
            cases.append(
                {
                    **base,
                    "status": "success",
                    "source_text": source_text,
                    "hypothesis_text": hypothesis,
                    "references": references,
                    "sentence_bleu": sacrebleu.sentence_bleu(
                        hypothesis, references, smooth_method="exp"
                    ).score,
                    "sentence_chrf": sacrebleu.sentence_chrf(hypothesis, references).score,
                    "sentence_chrf_plus_plus": sacrebleu.sentence_chrf(
                        hypothesis,
                        references,
                        word_order=2,
                    ).score,
                    "edit_similarity": edit_similarity,
                    "exact_match": float(exact_match) * 100.0,
                    "terminology_accuracy": term_score,
                    "terminology_matched": term_matched,
                    "terminology_total": term_total,
                }
            )
            corpus_rows.append((hypothesis, references))
        except Exception as exc:
            cases.append({**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    corpus_metrics: dict[str, Any] = {}
    if corpus_rows:
        hypotheses = [row[0] for row in corpus_rows]
        reference_count = max(len(row[1]) for row in corpus_rows)
        reference_streams = [
            [references[min(index, len(references) - 1)] for _, references in corpus_rows]
            for index in range(reference_count)
        ]
        bleu = sacrebleu.metrics.BLEU(effective_order=True)
        chrf = sacrebleu.metrics.CHRF(word_order=0)
        chrf_plus_plus = sacrebleu.metrics.CHRF(word_order=2)
        corpus_metrics = {
            "corpus_bleu": bleu.corpus_score(hypotheses, reference_streams).score,
            "corpus_chrf": chrf.corpus_score(hypotheses, reference_streams).score,
            "corpus_chrf_plus_plus": chrf_plus_plus.corpus_score(
                hypotheses,
                reference_streams,
            ).score,
            "metric_signatures": {
                "bleu": str(bleu.get_signature()),
                "chrf": str(chrf.get_signature()),
                "chrf_plus_plus": str(chrf_plus_plus.get_signature()),
            },
        }
    return (
        _build_summary(
            "translation",
            manifest_rows,
            prediction_rows,
            cases,
            TRANSLATION_METRICS,
            corpus_metrics,
        ),
        cases,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one isolated VieTrans benchmark set from JSONL ground truth and predictions."
    )
    subparsers = parser.add_subparsers(dest="suite", required=True)
    for suite in ("ocr", "inpainting", "translation"):
        child = subparsers.add_parser(suite)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--predictions", type=Path, required=True)
        child.add_argument("--output-dir", type=Path)
        if suite == "ocr":
            child.add_argument("--iou-threshold", type=float, default=0.5)
        child.add_argument(
            "--allow-partial",
            action="store_true",
            help="Return success even when coverage is incomplete. Never use for a final report.",
        )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")
    if not args.predictions.is_file():
        raise FileNotFoundError(f"predictions not found: {args.predictions}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or evaluation_dir / "results" / f"{args.suite}_{timestamp}"
    if args.suite == "ocr":
        summary, cases = evaluate_ocr(args.manifest, args.predictions, args.iou_threshold)
    elif args.suite == "inpainting":
        summary, cases = evaluate_inpainting(args.manifest, args.predictions)
    else:
        summary, cases = evaluate_translation(args.manifest, args.predictions)
    summary["provenance"] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_case.jsonl", cases)
    print(f"suite={args.suite} scored={summary['scored_count']}/{summary['case_count']}")
    print(f"summary={output_dir.resolve() / 'summary.json'}")
    incomplete = (
        summary["coverage"] < 1.0
        or summary["error_count"] > 0
        or summary["extra_prediction_count"] > 0
    )
    if incomplete and not args.allow_partial:
        print("status=FAILED_INCOMPLETE (use --allow-partial only for development)")
        return 2
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
