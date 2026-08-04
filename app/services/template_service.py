"""回答模板渲染。

- ``render_data_answer``：把 data 工具的扁平结果拼成一句话（数据路由用，保持原样）。
- ``render_faq_template``：渲染 FAQ 模板，支持 ``{{var}}`` / ``{{if}}`` / ``{{#each}}``。
  这是 P2 的**确定性**渲染器，用于在接入 LLM（P3）之前预览回答骨架、并作为兜底。
  真正上线后 compose 交给 LLM 按输出铁律组织自然语言，本渲染器作为降级路径。
"""

from __future__ import annotations

import re
from typing import Any

# 缺失变量占位符：让回答骨架清晰可见，避免静默丢字段
MISSING = "〔{name}〕"

_EACH_RE = re.compile(r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}", re.DOTALL)
_IF_RE = re.compile(r"\{\{if\s+(\w+)\}\}(.*?)(?:\{\{else\}\}(.*?))?\{\{endif\}\}", re.DOTALL)
_VAR_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def format_value(key: str, value: Any) -> str:
    """data 路由用：按字段后缀补单位。"""
    if isinstance(value, float):
        value = f"{value:,.1f}"
    if key.endswith("_kw"):
        return f"{value} kW"
    if key.endswith("_percent"):
        return f"{value}%"
    return str(value)


def render_data_answer(data: dict[str, Any]) -> str:
    if not data:
        return "未查询到可展示的数据。"
    items = "；".join(f"{key}: {format_value(key, value)}" for key, value in data.items())
    return f"查询结果：{items}。"


def _fmt(value: Any) -> str:
    """FAQ 模板取值格式化：数值保留 1 位小数；模板已自带单位，故不再补单位。"""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _render_each_item(inner: str, item: Any, keep_missing: bool) -> str:
    """渲染 ``{{#each}}`` 单个元素（item 为参数而非闭包变量，规避循环变量捕获）。"""

    def sub_field(vm: re.Match[str]) -> str:
        field = vm.group(1)
        if isinstance(item, dict) and field in item and item[field] is not None:
            return _fmt(item[field])
        return MISSING.format(name=field) if keep_missing else ""

    return _VAR_RE.sub(sub_field, inner)


def render_faq_template(template: str, data: dict[str, Any], *, keep_missing: bool = True) -> str:
    """按 ``data`` 渲染 FAQ 模板。

    支持三种构造：
    - ``{{var}}``：简单替换；缺失时（keep_missing）保留占位符 ``〔var〕``
    - ``{{if cond}}A{{else}}B{{endif}}``：按 ``data[cond]`` 真值选分支
    - ``{{#each items}}...{{field}}...{{/each}}``：遍历 ``data[items]`` 列表（元素为 dict）

    静态 FAQ（模板无变量）原样返回，即得到完整回答。
    """

    def render_each(match: re.Match[str]) -> str:
        key, inner = match.group(1), match.group(2)
        items = data.get(key)
        if not isinstance(items, list):
            return MISSING.format(name=key) if keep_missing else ""
        return "".join(_render_each_item(inner, item, keep_missing) for item in items)

    def render_if(match: re.Match[str]) -> str:
        cond, truthy, falsy = match.group(1), match.group(2), match.group(3) or ""
        return truthy if data.get(cond) else falsy

    def render_var(vm: re.Match[str]) -> str:
        name = vm.group(1)
        if name in data and data[name] is not None:
            return _fmt(data[name])
        return MISSING.format(name=name) if keep_missing else ""

    # 顺序：先展开 each / if 块（其内部可能含 {{var}}），再替换剩余简单变量
    result = _EACH_RE.sub(render_each, template)
    result = _IF_RE.sub(render_if, result)
    result = _VAR_RE.sub(render_var, result)
    return result
