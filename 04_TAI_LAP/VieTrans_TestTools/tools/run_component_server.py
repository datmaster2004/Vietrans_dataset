from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

from common import read_jsonl, sha256_file, utc_now_iso, write_json, write_jsonl


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_URL = os.getenv("VIETRANS_SERVER_URL", "http://127.0.0.1:7860")
DEFAULT_SERVER_ID = os.getenv("VIETRANS_SERVER_ID", "local-vietrans")
DEFAULT_BUILD_ID = os.getenv("VIETRANS_BUILD_ID", "unknown")
COMPONENT_ENDPOINTS = {
    "ocr": "/eval_ocr",
    "inpainting": "/eval_inpaint",
    "translation": "/eval_translate",
}
DEPLOYED_PIPELINE_ENDPOINT = "/translate_no_qa"


def resolve_manifest_path(manifest: Path, value: Any, field: str) -> Path:
    if not str(value or "").strip():
        raise ValueError(f"Missing {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest.resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} not found: {path}")
    return path.resolve()


def absolute_manifest_rows(manifest: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest)
    if limit is not None:
        rows = rows[:limit]
    output = []
    for source in rows:
        row = dict(source)
        for field in ("image", "input_image", "clean_image", "mask_image"):
            if row.get(field):
                row[field] = str(resolve_manifest_path(manifest, row[field], field))
        output.append(row)
    return output


def endpoint_names(api_payload: Any) -> set[str]:
    if not isinstance(api_payload, dict):
        return set()
    endpoints = api_payload.get("named_endpoints")
    return set(endpoints) if isinstance(endpoints, dict) else set()


def ocr_prediction_from_payload(case_id: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("OCR endpoint did not return JSON")
    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("OCR payload.regions is not a list")
    regions = []
    for raw in raw_regions:
        if not isinstance(raw, dict):
            continue
        polygon = raw.get("polygon")
        box = raw.get("box") or raw.get("bbox")
        if polygon is None and isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = [float(value) for value in box]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            continue
        regions.append(
            {
                "polygon": polygon,
                "text": str(raw.get("text") or raw.get("detector_text") or ""),
                "confidence": raw.get("confidence"),
            }
        )
    text = str(payload.get("text") or "").strip()
    if not text:
        text = " ".join(region["text"] for region in regions if region["text"].strip())
    return {"case_id": case_id, "text": text, "regions": regions}


def inpainting_regions(row: dict[str, Any]) -> dict[str, Any]:
    raw_regions = row.get("regions")
    if isinstance(raw_regions, list):
        return {"regions": raw_regions}
    attributes = row.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    boxes = attributes.get("word_bboxes")
    words = attributes.get("words")
    if not isinstance(boxes, list):
        return {"regions": []}
    words = words if isinstance(words, list) else []
    regions = []
    for index, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        regions.append(
            {
                "box": [left, top, right, bottom],
                "polygon": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
                "text": str(words[index]) if index < len(words) else "",
            }
        )
    return {"regions": regions}


def _source_candidates(value: Any) -> list[str]:
    candidates: list[str] = []

    def add(raw: Any) -> None:
        if isinstance(raw, (str, os.PathLike)):
            text = str(raw).strip()
            if text and text not in candidates:
                candidates.append(text)

    if isinstance(value, dict):
        for key in ("url", "path", "name"):
            add(value.get(key))
        nested = value.get("data")
        if isinstance(nested, dict):
            for key in ("url", "path", "name"):
                add(nested.get(key))
    else:
        add(value)
        for attr in ("url", "path", "name"):
            add(getattr(value, attr, None))
    return candidates


def save_remote_image(
    value: Any,
    output: Path,
    space_url: str,
    token: str | None,
) -> None:
    import httpx
    from PIL import Image

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    errors = []
    for raw in _source_candidates(value):
        source = raw
        if raw.startswith(("/gradio_api/", "gradio_api/", "/file=", "file=")):
            source = urljoin(f"{space_url.rstrip('/')}/", raw.lstrip("/"))
        try:
            if source.startswith(("http://", "https://")):
                with httpx.Client(timeout=120.0, follow_redirects=True) as client:
                    response = client.get(source, headers=headers)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(Path(source)).convert("RGB")
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, "PNG")
            return
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not download Space image for {output.name}; candidates={_source_candidates(value)}; "
        f"errors={errors[:3]}"
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _space_sha(space_id: str, token: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).space_info(space_id)
        return str(info.sha or "") or None
    except Exception:
        return None


def get_client(space_url: str, token: str | None):
    from gradio_client import Client

    kwargs = {"token": token} if token else {}
    try:
        return Client(space_url, download_files=False, verbose=False, **kwargs)
    except TypeError:
        return Client(space_url, **kwargs)


def call_predict(
    client: Any,
    api_name: str,
    arguments: list[Any],
    retries: int,
    retry_delay: float,
) -> Any:
    error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return client.predict(*arguments, api_name=api_name)
        except Exception as exc:
            error = exc
            if attempt >= max(1, retries):
                break
            time.sleep(max(0.0, retry_delay) * attempt)
    raise RuntimeError(
        f"Space request {api_name} failed after {max(1, retries)} attempt(s): {error}"
    ) from error


def server_probe(
    server_url: str,
    server_id: str,
    build_id: str,
    hf_space_id: str | None,
    token: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    client = get_client(server_url, token)
    api = client.view_api(print_info=False, return_format="dict")
    names = sorted(endpoint_names(api))
    hf_space_sha = _space_sha(hf_space_id, token) if hf_space_id else None
    payload: dict[str, Any] = {
        "checked_at": utc_now_iso(),
        "server_url": server_url,
        "server_id": server_id,
        "server_build_id": build_id or hf_space_sha or "unknown",
        "hf_space_id": hf_space_id,
        "hf_space_sha": hf_space_sha,
        # Legacy fields keep prior reports and external tooling readable.
        "space_url": server_url,
        "space_id": hf_space_id or server_id,
        "space_sha": hf_space_sha,
        "endpoints": names,
        "component_endpoints_ready": all(
            endpoint in names for endpoint in COMPONENT_ENDPOINTS.values()
        ),
        "required_component_endpoints": COMPONENT_ENDPOINTS,
        "deployed_pipeline_endpoint_ready": DEPLOYED_PIPELINE_ENDPOINT in names,
    }
    if "/eval_info" in names:
        payload["server_component_provenance"] = client.predict(api_name="/eval_info")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "server_probe.json", payload)
    return payload


def _handle_file(path: Path):
    from gradio_client import handle_file

    return handle_file(str(path))


def _threaded_image_run(
    rows: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    resume: bool,
    worker: Callable[[dict[str, Any], Any], tuple[dict[str, Any], dict[str, Any]]],
    client_factory: Callable[[], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions_path = output_dir / "predictions.jsonl"
    raw_log_path = output_dir / "raw_log.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if resume and predictions_path.is_file():
        completed = {str(row["case_id"]): row for row in read_jsonl(predictions_path)}
    elif not resume:
        for path in (predictions_path, raw_log_path):
            if path.exists():
                path.unlink()
    selected = [row for row in rows if str(row["case_id"]) not in completed]
    thread_state = threading.local()

    def invoke(row: dict[str, Any]):
        client = getattr(thread_state, "client", None)
        if client is None:
            client = client_factory()
            thread_state.client = client
        started = time.perf_counter()
        try:
            prediction, details = worker(row, client)
            log = {
                "case_id": row["case_id"],
                "status": "success",
                "duration_seconds": time.perf_counter() - started,
                "details": details,
            }
            return prediction, log
        except Exception as exc:
            log = {
                "case_id": row["case_id"],
                "status": "error",
                "duration_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return None, log

    new_predictions: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vietrans-server") as executor:
        futures = {executor.submit(invoke, row): row for row in selected}
        for future in as_completed(futures):
            prediction, log = future.result()
            logs.append(log)
            _append_jsonl(raw_log_path, log)
            if prediction is not None:
                new_predictions.append(prediction)
                _append_jsonl(predictions_path, prediction)
            print(
                f"{log['case_id']} status={log['status']} "
                f"duration={log['duration_seconds']:.2f}s"
            )
    combined = list(completed.values()) + new_predictions
    order = {str(row["case_id"]): index for index, row in enumerate(rows)}
    combined.sort(key=lambda row: order.get(str(row["case_id"]), len(order)))
    write_jsonl(predictions_path, combined)
    return combined, logs


def run_ocr(args: argparse.Namespace, rows: list[dict[str, Any]], probe: dict[str, Any]) -> None:
    endpoint = (
        DEPLOYED_PIPELINE_ENDPOINT
        if args.mode == "deployed-ocr"
        else COMPONENT_ENDPOINTS["ocr"]
    )
    if endpoint not in probe["endpoints"]:
        raise RuntimeError(
            f"Server does not expose {endpoint}. Deploy the updated Space/app.py before isolated OCR evaluation."
        )

    def worker(row: dict[str, Any], client: Any):
        image_path = Path(row["image"])
        result = call_predict(
            client,
            endpoint,
            [_handle_file(image_path)],
            args.retries,
            args.retry_delay,
        )
        payload = (
            result[7]
            if args.mode == "deployed-ocr"
            and isinstance(result, (list, tuple))
            and len(result) >= 8
            else result
        )
        if args.mode == "deployed-ocr":
            payload = {
                "regions": (payload or {}).get("ocr_records") or [],
                "text": str(result[5] or ""),
            }
        prediction = ocr_prediction_from_payload(str(row["case_id"]), payload)
        return prediction, {"endpoint": endpoint, "image_sha256": sha256_file(image_path)}

    predictions, logs = _threaded_image_run(
        rows,
        args.output_dir,
        args.workers,
        args.resume,
        worker,
        lambda: get_client(args.space_url, args.hf_token),
    )
    _write_run_summary(args, rows, predictions, logs, probe, endpoint)


def run_inpainting(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    probe: dict[str, Any],
) -> None:
    endpoint = (
        DEPLOYED_PIPELINE_ENDPOINT
        if args.mode == "deployed-inpainting"
        else COMPONENT_ENDPOINTS["inpainting"]
    )
    if endpoint not in probe["endpoints"]:
        raise RuntimeError(
            f"Server does not expose {endpoint}. Deploy the updated Space/app.py before isolated inpainting evaluation."
        )

    def worker(row: dict[str, Any], client: Any):
        input_path = Path(row["input_image"])
        artifact_dir = args.output_dir / "artifacts" / str(row["case_id"])
        output_path = artifact_dir / "inpainted.png"
        if args.mode == "deployed-inpainting":
            result = call_predict(
                client,
                endpoint,
                [_handle_file(input_path)],
                args.retries,
                args.retry_delay,
            )
            if not isinstance(result, (list, tuple)) or len(result) < 8:
                raise ValueError("Unexpected /translate_no_qa response")
            save_remote_image(result[3], output_path, args.space_url, args.hf_token)
            debug = result[7] if isinstance(result[7], dict) else {}
            write_json(artifact_dir / "debug.json", debug)
            details = {
                "endpoint": endpoint,
                "mode": "deployed_pipeline_conditional_on_server_ocr_mask",
                "debug": str((artifact_dir / "debug.json").resolve()),
            }
        else:
            mask_path = Path(row["mask_image"])
            result = call_predict(
                client,
                endpoint,
                [
                    _handle_file(input_path),
                    _handle_file(mask_path),
                    inpainting_regions(row),
                ],
                args.retries,
                args.retry_delay,
            )
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise ValueError("Unexpected /eval_inpaint response")
            save_remote_image(result[0], output_path, args.space_url, args.hf_token)
            write_json(artifact_dir / "metadata.json", result[1])
            details = {
                "endpoint": endpoint,
                "mode": "isolated_production_inpainting_with_ground_truth_mask",
                "metadata": str((artifact_dir / "metadata.json").resolve()),
            }
        return (
            {"case_id": row["case_id"], "output_image": str(output_path.resolve())},
            details,
        )

    predictions, logs = _threaded_image_run(
        rows,
        args.output_dir,
        args.workers,
        args.resume,
        worker,
        lambda: get_client(args.space_url, args.hf_token),
    )
    _write_run_summary(args, rows, predictions, logs, probe, endpoint)


def run_translation(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    probe: dict[str, Any],
) -> None:
    endpoint = COMPONENT_ENDPOINTS["translation"]
    if endpoint not in probe["endpoints"]:
        raise RuntimeError(
            f"Server does not expose {endpoint}. Deploy the updated Space/app.py; "
            "the current image-only endpoint cannot isolate translation on FLORES/MASSIVE."
        )
    predictions_path = args.output_dir / "predictions.jsonl"
    raw_log_path = args.output_dir / "raw_log.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and predictions_path.is_file():
        completed = {str(row["case_id"]): row for row in read_jsonl(predictions_path)}
    elif not args.resume:
        for path in (predictions_path, raw_log_path):
            if path.exists():
                path.unlink()
    remaining = [row for row in rows if str(row["case_id"]) not in completed]
    client = get_client(args.space_url, args.hf_token)
    logs = []
    predictions = list(completed.values())
    for offset in range(0, len(remaining), args.batch_size):
        batch = remaining[offset : offset + args.batch_size]
        started = time.perf_counter()
        try:
            result = call_predict(
                client,
                endpoint,
                [{"texts": [str(row["source_text"]) for row in batch]}],
                args.retries,
                args.retry_delay,
            )
            if not isinstance(result, dict) or not isinstance(result.get("translations"), list):
                raise ValueError("Unexpected /eval_translate response")
            translations = result["translations"]
            if len(translations) != len(batch):
                raise ValueError(
                    f"Translation count mismatch: expected {len(batch)}, got {len(translations)}"
                )
            duration = time.perf_counter() - started
            for row, translated in zip(batch, translations):
                prediction = {
                    "case_id": row["case_id"],
                    "translation": str(translated),
                }
                predictions.append(prediction)
                _append_jsonl(predictions_path, prediction)
                log = {
                    "case_id": row["case_id"],
                    "status": "success",
                    "duration_seconds": duration / len(batch),
                    "endpoint": endpoint,
                }
                logs.append(log)
                _append_jsonl(raw_log_path, log)
            print(f"translated={min(offset + len(batch), len(remaining))}/{len(remaining)}")
        except Exception as exc:
            duration = time.perf_counter() - started
            for row in batch:
                log = {
                    "case_id": row["case_id"],
                    "status": "error",
                    "duration_seconds": duration / len(batch),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                logs.append(log)
                _append_jsonl(raw_log_path, log)
            print(f"batch_offset={offset} status=error error={exc}")
    order = {str(row["case_id"]): index for index, row in enumerate(rows)}
    predictions.sort(key=lambda row: order.get(str(row["case_id"]), len(order)))
    write_jsonl(predictions_path, predictions)
    _write_run_summary(args, rows, predictions, logs, probe, endpoint)


def _write_run_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    logs: list[dict[str, Any]],
    probe: dict[str, Any],
    endpoint: str,
) -> None:
    prediction_ids = {str(row["case_id"]) for row in predictions}
    selected_ids = {str(row["case_id"]) for row in rows}
    errors = [row for row in logs if row.get("status") == "error"]
    summary = {
        "generated_at": utc_now_iso(),
        "mode": args.mode,
        "endpoint": endpoint,
        "server_url": args.server_url,
        "server_id": args.server_id,
        "server_build_id": probe.get("server_build_id"),
        "space_url": args.server_url,
        "space_id": probe.get("space_id"),
        "space_sha": probe.get("space_sha"),
        "case_count": len(rows),
        "prediction_count": len(prediction_ids & selected_ids),
        "coverage": len(prediction_ids & selected_ids) / len(rows) if rows else 0.0,
        "error_count_this_invocation": len(errors),
        "server_probe": str((args.output_dir / "server_probe.json").resolve()),
        "predictions": str((args.output_dir / "predictions.jsonl").resolve()),
    }
    write_json(args.output_dir / "inference_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark predictions from a VieTrans server (local, private, or Hugging Face)."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in (
        "probe",
        "ocr",
        "inpainting",
        "translation",
        "deployed-ocr",
        "deployed-inpainting",
    ):
        child = subparsers.add_parser(mode)
        child.add_argument(
            "--server-url",
            "--space-url",
            dest="server_url",
            default=DEFAULT_SERVER_URL,
            help="Base URL of the running VieTrans server. Defaults to VIETRANS_SERVER_URL or localhost.",
        )
        child.add_argument(
            "--server-id",
            "--space-id",
            dest="server_id",
            default=DEFAULT_SERVER_ID,
            help="Deployment identifier written to provenance.",
        )
        child.add_argument(
            "--build-id",
            default=DEFAULT_BUILD_ID,
            help="Git commit, container tag, or deployment version written to provenance.",
        )
        child.add_argument(
            "--hf-space-id",
            default=os.getenv("HF_SPACE_ID"),
            help="Optional Hugging Face Space ID used only to retrieve its SHA.",
        )
        child.add_argument(
            "--auth-token",
            "--hf-token",
            dest="auth_token",
            default=os.getenv("VIETRANS_AUTH_TOKEN") or os.getenv("HF_TOKEN"),
            help="Optional bearer token for a protected Gradio server.",
        )
        child.add_argument("--output-dir", type=Path)
        if mode != "probe":
            child.add_argument("--manifest", type=Path, required=True)
            child.add_argument("--limit", type=int)
            child.add_argument("--workers", type=int, default=3)
            child.add_argument("--resume", action="store_true")
            child.add_argument("--retries", type=int, default=2)
            child.add_argument("--retry-delay", type=float, default=2.0)
            if mode == "translation":
                child.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Internal code historically used these names. Keep them as aliases while
    # the public CLI uses server terminology.
    args.space_url = args.server_url
    args.space_id = args.server_id
    args.hf_token = args.auth_token
    if args.hf_space_id is None and ".hf.space" in args.server_url:
        args.hf_space_id = args.server_id
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_ROOT / "results" / f"server_{args.mode}_{timestamp}"
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = server_probe(
        args.server_url,
        args.server_id,
        args.build_id,
        args.hf_space_id,
        args.auth_token,
        args.output_dir,
    )
    print(
        f"server_build_id={probe.get('server_build_id')} "
        f"endpoints={','.join(probe.get('endpoints') or [])}"
    )
    if args.mode == "probe":
        print(f"probe={args.output_dir / 'server_probe.json'}")
        return 0 if probe["deployed_pipeline_endpoint_ready"] else 2
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    if args.mode == "translation" and not 1 <= args.batch_size <= 32:
        raise ValueError("--batch-size must be between 1 and 32")
    rows = absolute_manifest_rows(args.manifest, args.limit)
    selected_manifest = args.output_dir / "selected_manifest.jsonl"
    write_jsonl(selected_manifest, rows)
    if args.mode in {"ocr", "deployed-ocr"}:
        run_ocr(args, rows, probe)
    elif args.mode in {"inpainting", "deployed-inpainting"}:
        run_inpainting(args, rows, probe)
    else:
        run_translation(args, rows, probe)
    summary = json.loads(
        (args.output_dir / "inference_summary.json").read_text(encoding="utf-8")
    )
    return 0 if summary["coverage"] == 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
