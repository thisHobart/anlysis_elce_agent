"""Registry combining packaged and user-provided professional Skills."""

from __future__ import annotations

import os
from pathlib import Path

from app.config import Settings, get_settings, runtime_env_file
from app.research.skills.contracts import SkillDefinition
from app.research.skills.loader import SkillLoadError, discover_skill_paths, load_skill


class SkillRegistry:
    """Expose Skill metadata to the main Agent and load full instructions on activation."""

    def __init__(self, skills: list[SkillDefinition], *, load_errors: list[str] | None = None) -> None:
        self._skills: dict[str, SkillDefinition] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise SkillLoadError(f"Skill 名称重复：{skill.name}")
            self._skills[skill.name] = skill
        self.load_errors = load_errors or []

    @classmethod
    def default(cls, settings: Settings | None = None) -> SkillRegistry:
        configured = settings or get_settings()
        package_root = Path(__file__).resolve().parent
        builtin_paths = discover_skill_paths(package_root)
        skills = [load_skill(path, source="builtin") for path in builtin_paths]
        known = {skill.name for skill in skills}
        errors: list[str] = []

        external_roots = [runtime_env_file().parent / "skills"]
        external_roots.extend(
            Path(value).expanduser()
            for value in configured.skill_paths.split(os.pathsep)
            if value.strip()
        )
        for root in dict.fromkeys(path.resolve() for path in external_roots):
            for path in discover_skill_paths(root):
                try:
                    skill = load_skill(path, source="external")
                    if skill.name in known:
                        raise SkillLoadError(f"外部 Skill 不能覆盖已有 Skill：{skill.name}")
                    skills.append(skill)
                    known.add(skill.name)
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
        return cls(skills, load_errors=errors)

    def get(self, name: str) -> SkillDefinition:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillLoadError(f"未找到研究 Skill：{name}") from exc

    def metadata(self) -> list[dict[str, object]]:
        """Return lightweight discovery metadata without loading extra references."""

        return [
            {
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "domain": skill.domain,
                "allowed_functions": skill.allowed_functions,
                "research_protocol": (
                    {
                        "protocol_id": skill.research_protocol.protocol_id,
                        "version": skill.research_protocol.version,
                        "kind": skill.research_protocol.kind,
                    }
                    if skill.research_protocol is not None
                    else None
                ),
                "source": skill.source,
            }
            for skill in self._skills.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._skills
