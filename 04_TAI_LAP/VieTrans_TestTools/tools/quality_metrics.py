from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def _levenshtein(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def compute_ocr_ground_truth_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    hypothesis_text = _normalize_text(hypothesis)
    reference_text = _normalize_text(reference)
    char_distance = _levenshtein(hypothesis_text, reference_text)
    reference_words = re.findall(r"\w+|[^\w\s]", reference_text, flags=re.UNICODE)
    hypothesis_words = re.findall(r"\w+|[^\w\s]", hypothesis_text, flags=re.UNICODE)
    cer = char_distance / max(1, len(reference_text))
    wer = _levenshtein(hypothesis_words, reference_words) / max(1, len(reference_words))
    return {
        "ocr_cer": cer,
        "ocr_wer": wer,
        "ocr_accuracy_score": max(0.0, min(100.0, (1.0 - cer) * 100.0)),
    }

