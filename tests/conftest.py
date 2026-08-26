"""Repository-wide isolation for writable desktop runtime paths."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRICE_RESEARCH_APP_DATA_DIRECTORY", str(tmp_path / "app-data"))
    monkeypatch.setenv("PRICE_RESEARCH_OUTPUT_DIRECTORY", str(tmp_path / "research-output"))
