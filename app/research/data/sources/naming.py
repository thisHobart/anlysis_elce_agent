"""Map stored column names onto the words a power-market analyst uses.

The mapping is a word list rather than a model translation on purpose: the same
column has to read the same way in every session, otherwise a research package
stops being reproducible.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from app.runtime_paths import source_worktree

if TYPE_CHECKING:  # A runtime import would make the two modules import each other.
    from app.research.data.sources.summary import VariableKind, VariableLabel

WORD_LIST_FILENAME = "variable_names.yaml"
WORD_LIST_ENVIRONMENT_VARIABLE = "PRICE_RESEARCH_VARIABLE_NAMES"


def word_list_path() -> Path | None:
    """Locate the word list in a source checkout, a bundle, or an operator override."""

    configured = os.environ.get(WORD_LIST_ENVIRONMENT_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.is_file() else None
    roots = [Path(__file__).resolve().parent]
    worktree = source_worktree()
    if worktree is not None:
        roots.append(worktree / "configs")
    for root in roots:
        candidate = root / WORD_LIST_FILENAME
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def _word_list() -> dict[str, str]:
    """Load the word list once, flattened to lookup keys.

    Keys are either ``表名.列名`` or a bare column name; both are stored exactly as
    written plus in a normalized form, so the caller only walks the key order.
    """

    path = word_list_path()
    if path is None:
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    entries: dict[str, str] = {}
    for section, value in raw.items():
        if not isinstance(value, dict):
            continue
        prefix = "" if str(section) == "columns" else f"{section}."
        for column, display in value.items():
            if not display:
                continue
            entries[f"{prefix}{column}"] = str(display)
    return entries


def normalize(name: str) -> str:
    """Fold case and separators so ``Wind_Power`` and ``windpower`` match."""

    return "".join(character for character in str(name).lower() if character.isalnum())


def label_variable(
    column: str,
    *,
    table: str | None = None,
    kind: VariableKind = "unknown",
) -> VariableLabel:
    """Resolve one column name, falling back to the column itself when unknown.

    Showing ``p001`` is deliberate: an analyst can ask an operator what it means,
    whereas a placeholder like 「变量 1」 carries no information at all.
    """

    from app.research.data.sources.summary import VariableLabel

    entries = _word_list()
    if table:
        exact = entries.get(f"{table}.{column}")
        if exact:
            return VariableLabel(display=exact, resolved=True, kind=kind)
    exact = entries.get(str(column))
    if exact:
        return VariableLabel(display=exact, resolved=True, kind=kind)
    wanted = normalize(column)
    for key, display in entries.items():
        if "." in key:
            continue
        if normalize(key) == wanted:
            return VariableLabel(display=display, resolved=True, kind=kind)
    return VariableLabel(display=str(column), resolved=False, kind=kind)


def reload_word_list() -> None:
    """Drop the cached word list; used by tests and by an operator hot-reload."""

    _word_list.cache_clear()
