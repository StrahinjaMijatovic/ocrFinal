import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.pipeline.categorizer import apply_db_rules, apply_ocr_rules, apply_llm_categories
from app.models.category import Category
from app.models.document import ExtractedDocument
import app.services.category_registry as reg


SAMPLE_DB = {
    "urgency_kategorija": "URGENCY_HIGH",
    "starosna_kategorija": "APPLICANT_MINOR",
    "pasos_status": "RISK_PASSPORT_EXPIRING",
    "duzina_boravka_dana": 6,
    "dana_do_dolaska": 5,
    "pasos_istice_za_dana": 90,
    "broj_prethodnih_zahteva": 1,
    "datum_podnosenja": "2026-06-01",
    "starost": 16,
}


def test_db_rules_urgency():
    cats = apply_db_rules(SAMPLE_DB)
    codes = [c.code for c in cats]
    assert "URGENCY_HIGH" in codes
    assert all(c.source == "db" for c in cats)


def test_db_rules_minor():
    cats = apply_db_rules(SAMPLE_DB)
    codes = [c.code for c in cats]
    assert "APPLICANT_MINOR" in codes


def test_db_rules_passport_expiring():
    cats = apply_db_rules(SAMPLE_DB)
    codes = [c.code for c in cats]
    assert "RISK_PASSPORT_EXPIRING" in codes


def test_db_rules_stay_duration():
    cats = apply_db_rules(SAMPLE_DB)
    codes = [c.code for c in cats]
    assert "STAY_4_7_DAYS" in codes


def test_db_rules_resubmitted():
    cats = apply_db_rules(SAMPLE_DB)
    codes = [c.code for c in cats]
    assert "ADMIN_RESUBMITTED" in codes


def test_ocr_rules_low_confidence():
    docs = [
        ExtractedDocument(file="2.pdf", document_type="INVITATION_LETTER", confidence=0.3)
    ]
    cats = apply_ocr_rules(docs)
    codes = [c.code for c in cats]
    assert "DOC_LOW_CONFIDENCE" in codes


def test_ocr_rules_no_passport_categories():
    # Passport/MRZ/completeness categories are gone — identity docs are skipped upstream.
    docs = [
        ExtractedDocument(file="2.pdf", document_type="INVITATION_LETTER", confidence=0.9)
    ]
    codes = [c.code for c in apply_ocr_rules(docs)]
    assert codes == []


@pytest.mark.asyncio
async def test_llm_categories(tmp_path, monkeypatch):
    cats_data = {"categories": [
        {"code": "PURPOSE_CULTURAL", "group": "Svrha", "description": "Cultural event", "source": "llm"}
    ]}
    (tmp_path / "categories.json").write_text(json.dumps(cats_data))
    (tmp_path / "pending_categories.json").write_text("[]")
    monkeypatch.setattr(reg, "CATEGORIES_PATH", tmp_path / "categories.json")
    monkeypatch.setattr(reg, "PENDING_PATH", tmp_path / "pending_categories.json")
    monkeypatch.setattr("app.pipeline.categorizer.settings.active_categorization_prompt", "v1_basic")

    mock_ollama = MagicMock()
    mock_ollama.generate = AsyncMock(return_value={
        "categories": [{"code": "PURPOSE_CULTURAL", "reason": "Invitation to violin competition"}],
        "new_categories": []
    })

    cats = await apply_llm_categories({}, [], [], mock_ollama)
    codes = [c.code for c in cats]
    assert "PURPOSE_CULTURAL" in codes
    assert cats[0].source == "llm"
