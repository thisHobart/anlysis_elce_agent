"""Deterministic canonicalization of region and asset mentions.

Extraction must keep the source wording, because every field carries a verbatim evidence
span. Identity must not. A model that writes ``辽宁`` on one run and ``辽宁电网`` on the
next is naming one grid twice, but a raw-string event identity turns that into two
different events and silently defeats deduplication.

So the two jobs are split, the way EDC-style pipelines split them: the extractor records
what the source said, and this module maps that mention to a stable key used for identity,
merging and scoring. Mapping is pure and table-driven — never a model call — so the same
mention always folds to the same key.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

ENTITY_NORMALIZER_VERSION = "1.0.0"

# Words that name the operator or the administrative wrapper rather than the region.
# Stripping them is what makes 辽宁电网 and 辽宁 the same key; order matters because the
# longer forms must be removed before the shorter ones they contain.
_REGION_SUFFIXES: tuple[str, ...] = (
    "电网公司",
    "电力公司",
    "电力集团",
    "供电公司",
    "电网",
    "电力",
    "全网",
    "全省",
    "全市",
    "省份",
    "省",
    "市",
    "自治区",
    "地区",
    "区域",
    "电网区域",
)

# Known mentions folded to one market/region code. Unknown mentions still fold
# deterministically via suffix stripping, so this table improves precision but is
# never required for stability.
_REGION_CANONICAL: dict[str, str] = {
    "江苏": "CN-JIANGSU",
    "广东": "CN-GUANGDONG",
    "广西": "CN-GUANGXI",
    "浙江": "CN-ZHEJIANG",
    "辽宁": "CN-LIAONING",
    "吉林": "CN-JILIN",
    "黑龙江": "CN-HEILONGJIANG",
    "山东": "CN-SHANDONG",
    "山西": "CN-SHANXI",
    "天津": "CN-TIANJIN",
    "北京": "CN-BEIJING",
    "上海": "CN-SHANGHAI",
    "安徽": "CN-ANHUI",
    "福建": "CN-FUJIAN",
    "河南": "CN-HENAN",
    "河北": "CN-HEBEI",
    "湖南": "CN-HUNAN",
    "湖北": "CN-HUBEI",
    "四川": "CN-SICHUAN",
    "全国": "CN-NATIONAL",
    "中国": "CN-NATIONAL",
    "华东": "CN-EAST",
    "华北": "CN-NORTH",
    "华中": "CN-CENTRAL",
    "华南": "CN-SOUTH",
    "西北": "CN-NORTHWEST",
    "东北": "CN-NORTHEAST",
    "greatbritain": "GB",
    "britain": "GB",
    "unitedkingdom": "GB",
    "uk": "GB",
    "gb": "GB",
    "england": "GB",
    "texas": "ERCOT",
    "ercot": "ERCOT",
    "qld": "NEM-QLD",
    "queensland": "NEM-QLD",
    "nemqld": "NEM-QLD",
}

_KNOWN_REGION_CODES = frozenset(_REGION_CANONICAL.values())
_REGION_CODE_BY_FOLDED = {
    re.sub(r"[^\w]", "", code, flags=re.UNICODE).casefold(): code
    for code in _KNOWN_REGION_CODES
}

# A stem shorter than this is not a region name. Without the floor, 市 would be stripped from
# 沙市 and collapse it into 沙 — and any other single-character name that happens to end the
# same way would collapse with it.
_MIN_REGION_STEM = 2

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)


def _fold(value: str) -> str:
    """Reduce a mention to comparable characters: NFKC, no punctuation or spacing, casefolded."""

    normalized = unicodedata.normalize("NFKC", value)
    return _NON_WORD.sub("", normalized).casefold()


def _strip_region_suffixes(folded: str) -> str:
    """Remove operator/administrative tails repeatedly; keep the input if nothing survives."""

    current = folded
    changed = True
    while changed:
        changed = False
        for suffix in _REGION_SUFFIXES:
            if len(current) - len(suffix) >= _MIN_REGION_STEM and current.endswith(suffix):
                current = current[: -len(suffix)]
                changed = True
                break
    return current or folded


def canonical_region(value: str) -> str:
    """Fold one region mention to its stable key; unknown regions still fold deterministically."""

    folded = _fold(value)
    if not folded:
        return ""
    if folded in _REGION_CODE_BY_FOLDED:
        return _REGION_CODE_BY_FOLDED[folded]
    stripped = _strip_region_suffixes(folded)
    return _REGION_CANONICAL.get(stripped, stripped)


def canonical_asset(value: str) -> str:
    """Fold one asset mention. Assets keep their own words; only spacing and case are dropped."""

    return _fold(value)


def _keys(values: Iterable[str], folder) -> tuple[str, ...]:
    """Deduplicate and sort, so a reordered model answer is not a different identity."""

    return tuple(sorted({key for key in (folder(value) for value in values) if key}))


def canonical_regions(values: Iterable[str]) -> tuple[str, ...]:
    return _keys(values, canonical_region)


def canonical_assets(values: Iterable[str]) -> tuple[str, ...]:
    return _keys(values, canonical_asset)


def is_region_mention(value: str) -> bool:
    """True when a mention resolves to a region in the controlled vocabulary."""

    return canonical_region(value) in _KNOWN_REGION_CODES


def region_source_aliases(value: str) -> tuple[str, ...]:
    """Source spellings for the controlled keys; no substring-derived geography."""
    spaced = {"greatbritain": "Great Britain", "unitedkingdom": "United Kingdom"}
    key = canonical_region(value)
    return tuple(sorted((spaced.get(alias, alias) for alias in _REGION_CANONICAL
                         if canonical_region(alias) == key), key=len, reverse=True))


def split_entity_keys(
    regions: Iterable[str],
    assets: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Decide region-vs-asset placement from the vocabulary rather than from the model.

    Whether 天津电网 is "the region" or "the affected asset" is a question the source text
    does not answer and the model answers differently from one run to the next. Anything the
    region vocabulary recognises is treated as a region wherever the model wrote it, so the
    same event has one identity no matter which list it landed in. A name the vocabulary does
    not recognise — a plant, a unit, a line — stays an asset.
    """

    asset_values = list(assets)
    promoted = [value for value in asset_values if is_region_mention(value)]
    remaining = [value for value in asset_values if not is_region_mention(value)]
    return canonical_regions([*regions, *promoted]), canonical_assets(remaining)


def entity_identity(
    regions: Iterable[str],
    assets: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """The entity half of an event identity, in canonical form."""

    region_keys, asset_keys = split_entity_keys(regions, assets)
    return {"affected_assets": asset_keys, "affected_regions": region_keys}
