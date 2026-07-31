from collections.abc import Callable
from typing import Any


class VPPTools:
    """Adapter boundary for the three existing VPP Python tools.

    Replace the sample return values below with calls to your production SDK/API.
    The graph only interacts with this class, keeping tool integration isolated.
    """

    def get_station_status(self, station_id: str = "all", **_: Any) -> dict[str, Any]:
        return {"station_id": station_id, "status": "online", "available_kw": 1250.0}

    def get_aggregate_metrics(self, **_: Any) -> dict[str, Any]:
        return {"online_stations": 18, "available_kw": 12500.0, "soc_percent": 72.4}

    def get_device_detail(self, device_id: str, **_: Any) -> dict[str, Any]:
        return {"device_id": device_id, "status": "online", "power_kw": 320.0}

    def execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        tool: Callable[..., dict[str, Any]] | None = getattr(self, tool_name, None)
        if tool is None or not callable(tool):
            raise ValueError(f"Unknown VPP tool: {tool_name}")
        return tool(**params)
