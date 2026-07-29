from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from common import sha256_file, utc_now_iso, write_json


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze model/code/config and benchmark hashes before opening final-test scores."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    manifests = {}
    for path in sorted((root / "manifests").glob("*/manifest.jsonl")):
        manifests[str(path.relative_to(root)).replace("\\", "/")] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except Exception as exc:
        pip_freeze = [f"unavailable: {type(exc).__name__}: {exc}"]

    payload = {
        "run_id": args.run_id,
        "frozen_at": utc_now_iso(),
        "model": {"id": args.model_id, "revision": args.model_revision},
        "code_commit": args.code_commit,
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "manifests": manifests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pip_freeze": pip_freeze,
        },
        "attestation": (
            "No final-test score may be used to change this model revision, code commit, "
            "configuration or decoding. If changed, create a new untouched holdout."
        ),
    }
    output = args.output or root / "runs" / args.run_id / "run_lock.json"
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

