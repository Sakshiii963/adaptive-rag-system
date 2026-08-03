"""API foundation tests."""

from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_endpoint_returns_liveness_payload() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert response.headers["x-request-id"]


def test_unknown_route_uses_standard_error_contract() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/not-implemented")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
