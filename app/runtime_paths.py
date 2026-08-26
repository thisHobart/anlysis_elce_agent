"""Writable runtime locations that remain valid in source and packaged builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIRECTORY_NAME = "PriceResearchAgent"


def application_data_directory() -> Path:
    """Return a per-user directory for internal databases and audit records."""

    configured = os.environ.get("PRICE_RESEARCH_APP_DATA_DIRECTORY")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        configured = os.environ.get("XDG_DATA_HOME")
        root = Path(configured) if configured else Path.home() / ".local" / "share"
    return (root / APP_DIRECTORY_NAME).resolve()


def default_research_output_directory() -> Path:
    """Return a durable, user-visible default for generated research packages."""

    configured = os.environ.get("PRICE_RESEARCH_OUTPUT_DIRECTORY")
    if configured:
        return Path(configured).expanduser().resolve()
    documents = Path.home() / "Documents"
    return (documents / APP_DIRECTORY_NAME / "research").resolve()


def source_worktree() -> Path | None:
    """Locate a development Git worktree without treating an EXE bundle as one."""

    if getattr(sys, "frozen", False):
        return None
    candidate = Path(__file__).resolve().parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None
