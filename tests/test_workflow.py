from app.graph.workflow import build_workflow


def test_workflow_answers_faq():
    result = build_workflow().invoke({"question": "什么是虚拟电厂", "role": "viewer"})
    assert result["route"] == "faq"
    assert "虚拟电厂" in result["answer"]


def test_workflow_queries_station_and_emits_action():
    result = build_workflow().invoke(
        {"question": "查询电站状态", "role": "operator", "params": {"station_id": "station-01"}, "request_action": True}
    )
    assert result["route"] == "data"
    assert result["data"] == {}
    assert result["fallback"] is True
    assert result["fallback_reason"] == "data_query_not_connected"
    assert result["screen_action"] == {"type": "highlight_station", "target": "station-01"}


def test_workflow_blocks_unprivileged_device_query():
    result = build_workflow().invoke(
        {"question": "查询设备", "role": "viewer", "params": {"device_id": "device-01"}}
    )
    assert result["validation_errors"]
