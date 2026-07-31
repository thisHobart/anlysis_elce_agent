from typing import Any


def format_value(key: str, value: Any) -> str:
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
