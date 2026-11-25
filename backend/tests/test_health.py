"""Tests for the health check endpoint."""

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check() -> None:
    """Verify GET /health returns HTTP 200 and exact JSON payload."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "waferinspect-backend"}
