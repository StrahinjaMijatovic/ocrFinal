import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from app.pipeline.document_structurer import structure_document
from app.models.document import ExtractedDocument


@pytest.fixture
def mock_ollama():
    client = MagicMock()
    client.generate = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_structure_passport_classifies_without_mrz_parsing(mock_ollama, monkeypatch):
    # Passport handling is gone: structure_document only returns the LLM's output as-is,
    # it no longer enriches with parsed MRZ fields (the orchestrator skips these docs).
    mock_ollama.generate.return_value = {
        "document_type": "PASSPORT",
        "confidence": 0.95,
        "language": "English",
        "fields": {}
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    ocr_text = (
        "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "E759035120CHN0103015F2705136<<<<<<<<<<<<<<00"
    )
    doc = await structure_document(Path("1.pdf"), ocr_text, mock_ollama)

    assert isinstance(doc, ExtractedDocument)
    assert doc.document_type == "PASSPORT"
    assert "mrz_valid" not in doc.fields
    assert "nationality" not in doc.fields


@pytest.mark.asyncio
async def test_structure_invitation(mock_ollama, monkeypatch):
    mock_ollama.generate.return_value = {
        "document_type": "INVITATION_LETTER",
        "confidence": 0.88,
        "language": "English",
        "fields": {"inviting_org": "Muzički Atelje", "period_from": "2019-04-04"}
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    doc = await structure_document(Path("2.pdf"), "some invitation text", mock_ollama)

    assert doc.document_type == "INVITATION_LETTER"
    assert doc.fields["inviting_org"] == "Muzički Atelje"
    assert "mrz_valid" not in doc.fields


@pytest.mark.asyncio
async def test_structure_coalesces_explicit_nulls(mock_ollama, monkeypatch):
    # The LLM can emit keys with explicit null values; dict.get's default does NOT
    # cover that, so None must be coalesced to the fallbacks rather than reaching Pydantic.
    mock_ollama.generate.return_value = {
        "document_type": None, "confidence": None, "language": None, "fields": None
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    doc = await structure_document(Path("3.pdf"), "garbled text", mock_ollama)

    assert doc.document_type == "OTHER"
    assert doc.confidence == 0.0
    assert doc.language == "unknown"
    assert doc.fields == {}


@pytest.mark.asyncio
async def test_structure_uses_confidence_clamp(mock_ollama, monkeypatch):
    mock_ollama.generate.return_value = {
        "document_type": "OTHER", "confidence": 2.5, "language": "unknown", "fields": {}
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    doc = await structure_document(Path("9.pdf"), "random text", mock_ollama)
    assert doc.confidence <= 1.0
