import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import app.services.job_store as store


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    import asyncio
    asyncio.get_event_loop().run_until_complete(store.init_db())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    import asyncio
    asyncio.get_event_loop().run_until_complete(store.init_db())
    from app.main import app
    return TestClient(app)


def test_analyze_returns_job_id(client):
    with patch("app.api.routes.analyze.run_pipeline", new_callable=AsyncMock):
        response = client.post("/analyze/438281")
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert len(data["job_id"]) > 10


def test_status_pending(client):
    with patch("app.api.routes.analyze.run_pipeline", new_callable=AsyncMock):
        job_resp = client.post("/analyze/438281")
    job_id = job_resp.json()["job_id"]

    status_resp = client.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ("PENDING", "RUNNING", "DONE")


def test_status_not_found(client):
    response = client.get("/status/nonexistent-job-id")
    assert response.status_code == 404


def test_health_endpoint(client):
    with (
        patch("app.api.routes.health.OllamaClient") as mock_ollama_cls,
        patch("app.api.routes.health.get_db_connection") as mock_db,
    ):
        mock_instance = MagicMock()
        mock_instance.health_check = AsyncMock(return_value=True)
        mock_instance.aclose = AsyncMock()
        mock_ollama_cls.return_value = mock_instance
        mock_db.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_categories_list(client, tmp_path, monkeypatch):
    import app.services.category_registry as reg
    cats = {"categories": [{"code": "URGENCY_HIGH", "group": "Urgentnost", "description": "test", "source": "db"}]}
    (tmp_path / "categories.json").write_text(json.dumps(cats))
    (tmp_path / "pending_categories.json").write_text("[]")
    monkeypatch.setattr(reg, "CATEGORIES_PATH", tmp_path / "categories.json")
    monkeypatch.setattr(reg, "PENDING_PATH", tmp_path / "pending_categories.json")

    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["code"] == "URGENCY_HIGH"
