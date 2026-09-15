"""Tool-like analysis recipes expanded into auditable atomic functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    version: str
    description: str
    when_to_use: str
    when_not_to_use: str
    functions: tuple[str, ...]

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"{self.description} 适用时机：{self.when_to_use} "
                    f"不适用：{self.when_not_to_use}。展开后的基础调用会分别记录。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }


class RecipeRegistry:
    def __init__(self, recipes: list[RecipeSpec]) -> None:
        self._recipes = {recipe.name: recipe for recipe in recipes}
        if len(self._recipes) != len(recipes):
            raise ValueError("分析配方名称不能重复")

    def get(self, name: str) -> RecipeSpec:
        try:
            return self._recipes[name]
        except KeyError as exc:
            raise ValueError(f"分析配方未注册：{name}") from exc

    def schemas_for(self, authorized_functions: set[str]) -> list[dict[str, Any]]:
        return [
            recipe.function_schema()
            for recipe in self._recipes.values()
            if set(recipe.functions).issubset(authorized_functions)
        ]

    @property
    def names(self) -> set[str]:
        return set(self._recipes)


def build_recipe_registry() -> RecipeRegistry:
    return RecipeRegistry(
        [
            RecipeSpec(
                name="price_quick_profile",
                version="1.0.0",
                description="快速建立电价水平、日历结构和滚动波动的整体画像。",
                when_to_use="用户首次要求整体了解一批电价数据",
                when_not_to_use="用户只要求单一统计量，或明确限定了其他方法",
                functions=(
                    "price_descriptive_distribution",
                    "price_calendar_group_profile",
                    "price_rolling_mean_std",
                ),
            )
        ]
    )
