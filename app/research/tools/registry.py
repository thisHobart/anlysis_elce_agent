"""Registry and discovery metadata for controlled research functions."""

from __future__ import annotations

from app.research.tools.contracts import ToolSpec


class ToolRegistryError(ValueError):
    """A function is missing, duplicated, or version-incompatible."""


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ToolRegistryError(f"工具名称重复：{spec.name}")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolRegistryError(f"工具未注册：{name}") from exc

    def function_schemas(self, allowed_functions: list[str] | None = None) -> list[dict]:
        names = set(allowed_functions) if allowed_functions is not None else set(self._specs)
        unknown = names.difference(self._specs)
        if unknown:
            raise ToolRegistryError(f"Skill 引用了未注册工具：{', '.join(sorted(unknown))}")
        return [spec.function_schema() for name, spec in self._specs.items() if name in names]

    @property
    def names(self) -> set[str]:
        return set(self._specs)
