from fastapi.testclient import TestClient

from app.main import app


def test_query_api_returns_p3_metadata_and_reusable_session():
    with TestClient(app) as client:
        first = client.post(
            "/v1/query",
            json={"question": "什么是虚拟电厂", "role": "viewer"},
        )
        assert first.status_code == 200
        first_payload = first.json()

        second = client.post(
            "/v1/query",
            json={
                "session_id": first_payload["session_id"],
                "question": "谢谢",
                "role": "viewer",
            },
        )
        assert second.status_code == 200
        second_payload = second.json()

    assert first_payload["route"] == "faq"
    assert first_payload["faq_id"] == "FAQ-01"
    assert first_payload["classification_source"] == "rules"
    assert first_payload["compose_source"] == "template"
    assert second_payload["session_id"] == first_payload["session_id"]
    assert second_payload["route"] == "chitchat"
