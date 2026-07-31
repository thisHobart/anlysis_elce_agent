from app.security.policy import is_action_allowed, validate_sql, validate_tool_request


def test_viewer_cannot_access_device_detail():
    errors = validate_tool_request("get_device_detail", "viewer", {"device_id": "dev-01"})
    assert errors


def test_policy_rejects_unsafe_identifier_and_write_sql():
    assert validate_tool_request("get_station_status", "operator", {"station_id": "a;drop"})
    assert not validate_sql("DELETE FROM stations")
    assert validate_sql("SELECT * FROM stations")


def test_screen_actions_need_operator_role():
    assert not is_action_allowed("highlight_station", "viewer")
    assert is_action_allowed("highlight_station", "operator")
