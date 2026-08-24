"""Create content hashes that make research inputs and configurations traceable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.research.schemas.study import StudyConfig


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_manifest(config: StudyConfig) -> list[dict[str, Any]]:
    """Describe every configured source without reading its analytical content."""

    sources = [config.target, *config.exogenous]
    by_path: dict[Path, list[str]] = {}
    for source in sources:
        by_path.setdefault(source.path, []).append(source.name)
    return [
        {
            "series": sorted(series_names),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, series_names in sorted(by_path.items(), key=lambda item: str(item[0]))
    ]


def study_fingerprint(config: StudyConfig, inputs: list[dict[str, Any]]) -> str:
    """Create a stable short identifier from configuration and input hashes."""

    payload = {
        "config": config.model_dump(mode="json", exclude={"analysis": {"output_directory"}}),
        "inputs": inputs,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
