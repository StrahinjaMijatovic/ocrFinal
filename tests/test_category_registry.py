import json
import pytest
from pathlib import Path
import app.services.category_registry as reg


@pytest.fixture(autouse=True)
def use_temp_data(tmp_path, monkeypatch):
    cats = [{"code": "URGENCY_HIGH", "group": "Urgentnost", "description": "Arrival 3-8 days", "source": "db"}]
    (tmp_path / "categories.json").write_text(json.dumps({"categories": cats}))
    (tmp_path / "pending_categories.json").write_text(json.dumps([]))
    monkeypatch.setattr(reg, "CATEGORIES_PATH", tmp_path / "categories.json")
    monkeypatch.setattr(reg, "PENDING_PATH", tmp_path / "pending_categories.json")


def test_load_categories_returns_list():
    cats = reg.load_categories()
    assert isinstance(cats, list)
    assert cats[0]["code"] == "URGENCY_HIGH"


def test_save_pending_category_persists():
    reg.save_pending_category({"code": "NEW_CAT", "reason": "test", "description": "A new one"})
    pending = reg.load_pending_categories()
    assert len(pending) == 1
    assert pending[0]["code"] == "NEW_CAT"


def test_approve_pending_moves_to_active(tmp_path):
    reg.save_pending_category({"code": "NEW_CAT", "reason": "test", "description": "A new one", "group": "Rizik"})
    reg.approve_pending_category("NEW_CAT")
    active = reg.load_categories()
    codes = [c["code"] for c in active]
    assert "NEW_CAT" in codes
    pending = reg.load_pending_categories()
    assert all(c["code"] != "NEW_CAT" for c in pending)
