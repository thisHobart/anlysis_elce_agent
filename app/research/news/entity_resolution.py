"""Versioned, source-backed entity resolution. No model or external lookup runs here.

Legacy records keep their original keys. New records persist both the resolved keys and
the exact mentions that produced them, so a future vocabulary cannot rewrite a replay.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.research.news.entities import (
    canonical_asset,
    canonical_region,
    is_region_mention,
    region_source_aliases,
)

if TYPE_CHECKING:
    from app.research.news.contracts import EvidenceSpan, NewsDocument

ENTITY_RESOLUTION_VERSION = "2.0.0"


class MarketScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1)
    raw_value: str = Field(min_length=1)
    source_type: Literal["market_tag"] = "market_tag"
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    source_ref: str = Field(pattern=r"^market_tags\[\d+\]$")


class RawEntityMention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mention_id: str = Field(pattern=r"^mention_[a-f0-9]{24}$")
    document_version_id: str = Field(pattern=r"^newsv_[a-f0-9]{24}$")
    source_type: Literal["source_text"] = "source_text"
    text_field: Literal["title", "body"]
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_span(self):
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("entity mention length must match its source span")
        return self


class ResolvedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str = Field(min_length=1)
    key: str = Field(min_length=1)
    source_type: Literal["source_text", "rule_derived"]
    rule_id: Literal["verbatim", "region_alias", "asset_alias", "asset_group", "explicit_unit_list"]
    mention_ids: tuple[str, ...] = Field(min_length=1)


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["2.0.0"] = ENTITY_RESOLUTION_VERSION
    market_scope: tuple[MarketScope, ...] = ()
    mentioned_regions: tuple[ResolvedEntity, ...] = ()
    affected_assets: tuple[ResolvedEntity, ...] = ()
    asset_groups: tuple[ResolvedEntity, ...] = ()
    raw_entity_mentions: tuple[RawEntityMention, ...] = ()

    @model_validator(mode="after")
    def valid_links(self):
        mentions = {item.mention_id: item for item in self.raw_entity_mentions}
        ids = set(mentions)
        if len(ids) != len(self.raw_entity_mentions):
            raise ValueError("duplicate raw entity mention ids")
        for entity in (*self.mentioned_regions, *self.affected_assets, *self.asset_groups):
            if not set(entity.mention_ids).issubset(ids):
                raise ValueError("resolved entity has missing raw mention references")
            if (entity.source_type == "source_text") != (entity.rule_id == "verbatim"):
                raise ValueError("derived entity must declare rule_derived provenance")
        for entity in self.mentioned_regions:
            if entity.rule_id != "region_alias" or any(
                canonical_region(mentions[mid].quote) != entity.key for mid in entity.mention_ids
            ):
                raise ValueError("region key is not supported by its source mentions")
        for entity in self.asset_groups:
            if entity.rule_id != "asset_group" or any(
                (group_key(mentions[mid].quote) or canonical_asset(mentions[mid].quote)) != entity.key
                for mid in entity.mention_ids
            ):
                raise ValueError("group key is not supported by its source mentions")
        for entity in self.affected_assets:
            for mid in entity.mention_ids:
                raw = mentions[mid].quote
                supported = _asset_source(entity.value, raw)
                if (
                    group_key(raw)
                    or supported is None
                    or entity.key not in {canonical_asset(value) for value in supported[2]}
                    or entity.rule_id != supported[3]
                ):
                    raise ValueError("asset key is not supported by its source expression")
        return self

    @property
    def region_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.key for item in self.mentioned_regions}))

    @property
    def asset_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.key for item in self.affected_assets}))

    @property
    def group_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.key for item in self.asset_groups}))

    @property
    def market_keys(self) -> tuple[str, ...]:
        return tuple(sorted({item.value for item in self.market_scope}))

    def semantic_keys(self) -> dict:
        return {
            "regions": self.region_keys,
            "assets": self.asset_keys,
            "groups": self.group_keys,
            "markets": self.market_keys,
        }

    def matches_market(self, market: str) -> bool:
        """Tags route candidates. Ambiguous or conflicting event scope cannot price-match."""
        market = canonical_region(market)
        if market not in self.market_keys:
            return False
        known_regions = tuple(key for key in self.region_keys if is_region_mention(key))
        if known_regions:
            # No allocation of a multi-region total to each individual price market.
            return known_regions == (market,)
        return len(self.market_keys) == 1


class EntityResolutionError(ValueError):
    pass


def raw_mentions_from_spans(spans: list[EvidenceSpan]) -> tuple[RawEntityMention, ...]:
    """Retain already-validated source evidence even when entity resolution is quarantined."""
    mentions = {}
    for span in spans:
        token = f"{span.document_version_id}:{span.text_field}:{span.start_char}:{span.end_char}"
        mid = "mention_" + hashlib.sha256(token.encode()).hexdigest()[:24]
        mentions[mid] = RawEntityMention(
            mention_id=mid,
            document_version_id=span.document_version_id,
            text_field=span.text_field,
            start_char=span.start_char,
            end_char=span.end_char,
            quote=span.quote,
        )
    return tuple(mentions[key] for key in sorted(mentions))


# Deliberately closed vocabulary: unfamiliar collections are retained in quarantine,
# never promoted into an invented set of named plants.
_GROUPS = {
    "agrs": "AGR",
    "agr": "AGR",
    "advancedgascooledreactorstations": "AGR",
    "advancedgascooledreactornuclearpowerstations": "AGR",
    "advancedgascooledreactors": "AGR",
    "coalfleet": "coal_fleet",
    "coalplants": "coal_fleet",
    "windgeneration": "wind_generation",
    "windfarms": "wind_generation",
    "solargeneration": "solar_generation",
    "燃煤机组": "coal_fleet",
    "煤电机组": "coal_fleet",
    "风电": "wind_generation",
}
_EN_LIST = r"\d+(?:\s*(?:,\s*(?:and\s+)?|and\s+)\d+)+"
_CN_LIST = r"\d+(?:\s*[、,，及和]\s*\d+)+"
_EN_COMPOSITE = re.compile(
    rf"(?P<base>.+?(?:Power Station|Power Plant|Plant|Station))\s+"
    rf"(?:Generating\s+)?Units\s+(?P<ids>{_EN_LIST})",
    re.IGNORECASE,
)
_CN_COMPOSITE = re.compile(rf"(?P<base>.+?(?:电厂|电站))\s*(?P<ids>{_CN_LIST})号机组")
_EN_SINGLE = re.compile(
    r"(?P<base>.+?(?:Power Station|Power Plant|Plant|Station))\s+(?:Generating\s+)?Unit\s+(?P<id>\d+)", re.IGNORECASE
)
_CN_SINGLE = re.compile(r"(?P<base>.+?(?:电厂|电站))\s*(?P<id>\d+)号机组")


def group_key(value: str) -> str | None:
    return _GROUPS.get(canonical_asset(re.sub(r"^\s*\d+\s+", "", value)))


def _named_parent(base: str) -> bool:
    return canonical_asset(base) not in {
        "powerstation",
        "powerplant",
        "plant",
        "station",
        "thepowerstation",
        "theplant",
        "thepowerplant",
        "thisplant",
        "电厂",
        "电站",
        "该电厂",
        "某电厂",
        "本电厂",
        "该电站",
        "某电站",
    }


def split_explicit_units(value: str) -> tuple[str, ...] | None:
    """Only full named-parent enumerations; no ranges, alternatives or inferred counts."""
    for pattern, english in ((_EN_COMPOSITE, True), (_CN_COMPOSITE, False)):
        match = pattern.fullmatch(value.strip())
        if match:
            if not _named_parent(match["base"]):
                raise EntityResolutionError("组合资产缺少可识别的父电厂名称")
            numbers = re.findall(r"\d+", match["ids"])
            if len(set(numbers)) != len(numbers):
                raise EntityResolutionError("组合资产包含重复编号，保留原文待复核")
            return tuple(
                f"{match['base']} Unit {number}" if english else f"{match['base']}{number}号机组" for number in numbers
            )
    return None


def _exact(value: str, quote: str):
    # Latin word boundaries prevent Unit 1 matching Unit 10, or AGR matching AGRs.
    prefix = r"(?<![\w])" if value and value[0].isascii() and value[0].isalnum() else ""
    suffix = r"(?![\w])" if value and value[-1].isascii() and value[-1].isalnum() else ""
    return re.search(prefix + re.escape(value) + suffix, quote, re.IGNORECASE)


def _asset_source(value: str, quote: str) -> tuple[int, int, tuple[str, ...], str] | None:
    direct = _exact(value, quote)
    if direct:
        raw = direct[0]
        if re.fullmatch(r"(?:Unit\s+\d+|\d+号机组)", raw, re.IGNORECASE):
            raise EntityResolutionError(f"机组缺少可识别的父电厂名称：{raw}")
        split = split_explicit_units(raw)
        if split:
            return direct.start(), direct.end(), split, "explicit_unit_list"
        if re.search(r"\bunits\b|\d+\s*[-–/、和及或至到~]\s*\d+", raw, re.IGNORECASE):
            raise EntityResolutionError(f"无法安全拆分组合资产：{raw}")
        if re.search(r"\b(?:fleet|generation|plants|reactors|stations)\b|^\d+\s*(?:台|座)", raw, re.IGNORECASE):
            raise EntityResolutionError(f"集合表达不能作为具体资产：{raw}；应保留为 asset_groups")
        single = _EN_SINGLE.fullmatch(raw)
        parent = single or _CN_SINGLE.fullmatch(raw)
        if parent and not _named_parent(parent["base"]):
            raise EntityResolutionError(f"机组缺少可识别的父电厂名称：{raw}")
        normalized = f"{single['base']} Unit {single['id']}" if single else raw
        return direct.start(), direct.end(), (normalized,), "asset_alias" if single else "verbatim"
    # The model may already have expanded the list. Validate and expand the same exact
    # source expression, so either output shape yields the same assets and backlinks.
    for single_pattern, english in ((_EN_SINGLE, True), (_CN_SINGLE, False)):
        single = single_pattern.fullmatch(value)
        if not single:
            continue
        if english:
            singular = re.search(
                re.escape(single["base"]) + r"\s+(?:Generating\s+)?Unit\s+" + re.escape(single["id"]) + r"(?!\w)",
                quote,
                re.IGNORECASE,
            )
            if singular:
                normalized = f"{single['base']} Unit {single['id']}"
                return singular.start(), singular.end(), (normalized,), "asset_alias"
        tail = rf"\s+(?:Generating\s+)?Units\s+{_EN_LIST}" if english else rf"\s*{_CN_LIST}号机组"
        for found in re.finditer(re.escape(single["base"]) + tail, quote, re.IGNORECASE):
            # A partial match in a range/alternative is not a complete enumeration.
            if re.match(r"\s*(?:[-–/]\s*\d|or\b|或|至|到)", quote[found.end() :], re.IGNORECASE):
                continue
            expanded = split_explicit_units(found[0])
            if expanded and canonical_asset(value.replace("Generating ", "")) in {
                canonical_asset(item) for item in expanded
            }:
                selected = tuple(
                    item
                    for item in expanded
                    if canonical_asset(item) == canonical_asset(value.replace("Generating ", ""))
                )
                return found.start(), found.end(), selected, "explicit_unit_list"
    return None


def resolve_entities(
    document: NewsDocument,
    regions: tuple[str, ...],
    assets: tuple[str, ...],
    groups: tuple[str, ...],
    evidence: list[EvidenceSpan],
) -> EntityResolution:
    mentions: dict[str, RawEntityMention] = {}
    resolved: dict[str, dict[str, ResolvedEntity]] = {"regions": {}, "assets": {}, "groups": {}}
    scopes = tuple(
        MarketScope(
            value=canonical_region(tag),
            raw_value=tag,
            document_version_id=document.document_version_id,
            source_ref=f"market_tags[{i}]",
        )
        for i, tag in enumerate(document.market_tags)
    )

    def save(kind, value, key, span, start, end, rule):
        start += span.start_char
        end += span.start_char
        quote = getattr(document, span.text_field)[start:end]
        token = f"{document.document_version_id}:{span.text_field}:{start}:{end}"
        mid = "mention_" + hashlib.sha256(token.encode()).hexdigest()[:24]
        mentions[mid] = RawEntityMention(
            mention_id=mid,
            document_version_id=document.document_version_id,
            text_field=span.text_field,
            start_char=start,
            end_char=end,
            quote=quote,
        )
        previous = resolved[kind].get(key)
        ids = tuple(sorted({mid, *(previous.mention_ids if previous else ())}))
        resolved[kind][key] = ResolvedEntity(
            value=value,
            key=key,
            source_type=("rule_derived" if rule != "verbatim" else "source_text"),
            rule_id=rule,
            mention_ids=ids,
        )

    for field_name, values in (("affected_regions", regions), ("affected_assets", assets), ("asset_groups", groups)):
        spans = [item for item in evidence if item.field_name == field_name]
        for value in values:
            is_region = field_name == "affected_regions" or is_region_mention(value)
            gkey = group_key(value) if not is_region else None
            found = False
            for span in spans:
                if is_region:
                    match = _exact(value, span.quote)
                    # Region aliases must be supported by an actual named source region.
                    if not match:
                        for alias in region_source_aliases(value):
                            match = _exact(alias, span.quote)
                            if match:
                                break
                    if not match:
                        code = re.fullmatch(r"([A-Za-z0-9_]+)(?:电网|电力|区域)", value)
                        if code:
                            match = _exact(code[1], span.quote)
                    if match:
                        save(
                            "regions",
                            match[0],
                            canonical_region(value),
                            span,
                            match.start(),
                            match.end(),
                            "region_alias",
                        )
                        found = True
                        break
                elif gkey:
                    match = _exact(value, span.quote)
                    if match:
                        count = re.search(r"\b\d+\s+$", span.quote[: match.start()])
                        start = count.start() if count else match.start()
                        save("groups", span.quote[start : match.end()], gkey, span, start, match.end(), "asset_group")
                        found = True
                        break
                else:
                    specific = bool(
                        _EN_SINGLE.fullmatch(value)
                        or _CN_SINGLE.fullmatch(value)
                        or _EN_COMPOSITE.fullmatch(value)
                        or _CN_COMPOSITE.fullmatch(value)
                    )
                    if field_name == "asset_groups" and not specific:
                        # Unknown groups keep their explicit source wording as a stable key.
                        match = _exact(value, span.quote)
                        if match:
                            save(
                                "groups",
                                match[0],
                                canonical_asset(value),
                                span,
                                match.start(),
                                match.end(),
                                "asset_group",
                            )
                            found = True
                            break
                    match = _asset_source(value, span.quote)
                    if match:
                        start, end, expanded, rule = match
                        for item in expanded:
                            save("assets", item, canonical_asset(item), span, start, end, rule)
                        found = True
                        break
            if not found:
                if is_region and canonical_region(value) in {scope.value for scope in scopes}:
                    # A tag is metadata, never a substitute source-text evidence span.
                    continue
                raise EntityResolutionError(f"{field_name} 的值 {value!r} 未被对应原文证据支持")
    return EntityResolution(
        market_scope=scopes,
        mentioned_regions=tuple(resolved["regions"][k] for k in sorted(resolved["regions"])),
        affected_assets=tuple(resolved["assets"][k] for k in sorted(resolved["assets"])),
        asset_groups=tuple(resolved["groups"][k] for k in sorted(resolved["groups"])),
        raw_entity_mentions=tuple(mentions[k] for k in sorted(mentions)),
    )


def entity_sources_are_valid(resolution: EntityResolution, documents: dict[str, NewsDocument]) -> bool:
    """Check persisted provenance against immutable input documents, including tag indices."""
    try:
        EntityResolution.model_validate(resolution.model_dump())
    except ValueError:
        return False
    for scope in resolution.market_scope:
        document = documents.get(scope.document_version_id)
        index = int(scope.source_ref[len("market_tags[") : -1])
        if (
            document is None
            or index >= len(document.market_tags)
            or document.market_tags[index] != scope.raw_value
            or canonical_region(scope.raw_value) != scope.value
        ):
            return False
    for mention in resolution.raw_entity_mentions:
        document = documents.get(mention.document_version_id)
        if (
            document is None
            or getattr(document, mention.text_field)[mention.start_char : mention.end_char] != mention.quote
        ):
            return False
    return True
