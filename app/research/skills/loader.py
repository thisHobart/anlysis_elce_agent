"""Load Agent Skills without executing bundled scripts or assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml

from app.research.skills.contracts import SkillDefinition


class SkillLoadError(ValueError):
    """A SKILL.md bundle is invalid or unsafe to load."""


def _frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillLoadError(f"{path} 缺少 YAML frontmatter")
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillLoadError(f"{path} 的 YAML frontmatter 未闭合") from exc
    metadata = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(metadata, dict):
        raise SkillLoadError(f"{path} 的 frontmatter 必须是对象")
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise SkillLoadError(f"{path} 缺少 Skill 指令正文")
    return metadata, body


def load_skill(path: str | Path, *, source: Literal["builtin", "external"]) -> SkillDefinition:
    """Load only declarative Skill instructions; bundled scripts are never executed."""

    skill_path = Path(path).resolve()
    if skill_path.is_dir():
        skill_path = skill_path / "SKILL.md"
    if not skill_path.is_file():
        raise SkillLoadError(f"Skill 文件不存在：{skill_path}")
    frontmatter, instructions = _frontmatter(skill_path.read_text(encoding="utf-8"), skill_path)
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    raw_metadata = frontmatter.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        raise SkillLoadError(f"{skill_path} 的 metadata 必须是对象")
    raw_allowed = frontmatter.get("allowed-tools", "")
    if isinstance(raw_allowed, str):
        allowed_tools = raw_allowed.split()
    elif isinstance(raw_allowed, list) and all(isinstance(item, str) for item in raw_allowed):
        allowed_tools = raw_allowed
    else:
        raise SkillLoadError(f"{skill_path} 的 allowed-tools 必须是字符串或字符串列表")
    return SkillDefinition(
        name=name,
        description=description,
        version=str(raw_metadata.get("version", "unversioned")),
        domain=str(raw_metadata.get("domain", "general")),
        allowed_tools=list(dict.fromkeys(allowed_tools)),
        instructions=instructions,
        metadata=raw_metadata,
        root=skill_path.parent,
        source=source,
    )


def discover_skill_paths(root: str | Path) -> list[Path]:
    """Discover a direct Skill bundle or immediate child bundles."""

    resolved = Path(root).resolve()
    if (resolved / "SKILL.md").is_file():
        return [resolved]
    if not resolved.is_dir():
        return []
    return sorted(path for path in resolved.iterdir() if path.is_dir() and (path / "SKILL.md").is_file())
