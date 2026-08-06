"""VPP API 模拟服务（Mock）

提供模拟的 API 数据，用于开发和测试多步查询编排。
真实 API 准备好后，替换为 app/services/vpp_api.py。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VPPAPIError(Exception):
    """API 调用异常"""


# 模拟数据：企业信息
MOCK_COMPANIES = {
    "山东鲁能新能源": {"id": "COMP001", "name": "山东鲁能新能源", "capacity": 50.5},
    "青岛海尔工业园": {"id": "COMP002", "name": "青岛海尔工业园", "capacity": 35.2},
    "济南钢铁集团": {"id": "COMP003", "name": "济南钢铁集团", "capacity": 28.8},
    "潍坊光伏发电站": {"id": "COMP004", "name": "潍坊光伏发电站", "capacity": 42.3},
}

# 模拟数据：企业实时负荷（按企业ID）
MOCK_REALTIME_LOAD = {
    "COMP001": {"load_kw": 42300, "capacity_kw": 50500, "ratio": 83.8, "status": "正常"},
    "COMP002": {"load_kw": 28640, "capacity_kw": 35200, "ratio": 81.4, "status": "正常"},
    "COMP003": {"load_kw": 22100, "capacity_kw": 28800, "ratio": 76.7, "status": "正常"},
    "COMP004": {"load_kw": 38950, "capacity_kw": 42300, "ratio": 92.1, "status": "正常"},
}

# 模拟数据：企业可调能力（按企业ID）
MOCK_ADJUSTABLE_CAPACITY = {
    "COMP001": {"adjustable_mw": 15.2, "peak_shaving_mw": 10.5, "valley_filling_mw": 4.7, "level": "A"},
    "COMP002": {"adjustable_mw": 10.6, "peak_shaving_mw": 7.0, "valley_filling_mw": 3.6, "level": "B"},
    "COMP003": {"adjustable_mw": 8.6, "peak_shaving_mw": 5.5, "valley_filling_mw": 3.1, "level": "B"},
    "COMP004": {"adjustable_mw": 12.7, "peak_shaving_mw": 8.5, "valley_filling_mw": 4.2, "level": "A"},
}

# 模拟数据：总体实时数据
MOCK_OVERALL_REALTIME = {
    "total_power_mw": 45.6,
    "pv_power_mw": 32.3,
    "storage_discharge_mw": 8.5,
    "storage_charge_mw": 0.0,
    "load_power_mw": 41.2,
}

# 模拟数据：存储 SOC
MOCK_STORAGE_SOC = {
    "average_soc": 65.5,
    "stations": [
        {"name": "储能站点A", "soc": 72.3, "status": "放电"},
        {"name": "储能站点B", "soc": 58.7, "status": "充电"},
    ],
}


class MockVPPAPI:
    """VPP API 模拟服务

    模拟真实 API 的接口和数据结构，用于开发和测试。
    """

    def __init__(self):
        self.enabled = True

    def execute(self, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        执行 API 命令（模拟）

        Args:
            command: API 命令名称
            params: 命令参数

        Returns:
            {"code": 0, "data": {...}} 或 {"code": 1, "error": "..."}
        """
        params = params or {}

        try:
            if command == "companies":
                # 获取企业列表或查询企业信息
                return self._get_companies(params)

            elif command == "realtime-load":
                # 获取企业实时负荷
                return self._get_realtime_load(params)

            elif command == "adjustable-capacity":
                # 获取企业可调能力
                return self._get_adjustable_capacity(params)

            elif command == "realtime-overall":
                # 获取总体实时数据
                return self._get_realtime_overall(params)

            elif command == "storage-soc":
                # 获取储能 SOC
                return self._get_storage_soc(params)

            else:
                return {"code": 1, "error": f"未知命令: {command}"}

        except VPPAPIError as e:
            logger.error(f"Mock API 调用失败: {command} - {e}")
            return {"code": 1, "error": str(e)}

    def _get_companies(self, params: dict) -> dict:
        """获取企业信息（模拟）"""
        company_name = params.get("name")

        if company_name:
            # 按名称查询（支持模糊匹配）
            for name, info in MOCK_COMPANIES.items():
                if company_name in name or name in company_name:
                    return {
                        "code": 0,
                        "data": {
                            "company_id": info["id"],
                            "company_name": info["name"],
                            "capacity_mw": info["capacity"],
                        },
                    }
            return {"code": 1, "error": f"未找到企业: {company_name}"}

        # 返回所有企业
        return {
            "code": 0,
            "data": {
                "companies": [
                    {"id": info["id"], "name": info["name"], "capacity_mw": info["capacity"]}
                    for info in MOCK_COMPANIES.values()
                ],
                "total": len(MOCK_COMPANIES),
            },
        }

    def _get_realtime_load(self, params: dict) -> dict:
        """获取企业实时负荷（模拟）"""
        company_id = params.get("company_id")

        if not company_id:
            return {"code": 1, "error": "缺少 company_id 参数"}

        if company_id not in MOCK_REALTIME_LOAD:
            return {"code": 1, "error": f"未找到企业数据: {company_id}"}

        data = MOCK_REALTIME_LOAD[company_id]
        return {
            "code": 0,
            "data": {
                "company_id": company_id,
                "load_kw": data["load_kw"],
                "capacity_kw": data["capacity_kw"],
                "load_ratio": data["ratio"],
                "status": data["status"],
            },
        }

    def _get_adjustable_capacity(self, params: dict) -> dict:
        """获取企业可调能力（模拟）"""
        company_id = params.get("company_id")

        if not company_id:
            return {"code": 1, "error": "缺少 company_id 参数"}

        if company_id not in MOCK_ADJUSTABLE_CAPACITY:
            return {"code": 1, "error": f"未找到企业数据: {company_id}"}

        data = MOCK_ADJUSTABLE_CAPACITY[company_id]
        return {
            "code": 0,
            "data": {
                "company_id": company_id,
                "adjustable_capacity_mw": data["adjustable_mw"],
                "peak_shaving_mw": data["peak_shaving_mw"],
                "valley_filling_mw": data["valley_filling_mw"],
                "supply_level": data["level"],
            },
        }

    def _get_realtime_overall(self, params: dict) -> dict:
        """获取总体实时数据（模拟）"""
        return {"code": 0, "data": MOCK_OVERALL_REALTIME.copy()}

    def _get_storage_soc(self, params: dict) -> dict:
        """获取储能 SOC（模拟）"""
        return {"code": 0, "data": MOCK_STORAGE_SOC.copy()}


# 全局单例
_api_instance: MockVPPAPI | None = None


def get_mock_api() -> MockVPPAPI:
    """获取 Mock API 单例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = MockVPPAPI()
    return _api_instance
