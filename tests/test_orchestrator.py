import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import app.services.job_store as store
from app.pipeline.orchestrator import run_pipeline
from app.models.job import JobStatus


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")


@pytest.fixture
def mock_docs_folder(tmp_path):
    folder = tmp_path / "438281"
    folder.mkdir()
    (folder / "1.pdf").write_bytes(b"fake pdf")
    (folder / "2.pdf").write_bytes(b"fake pdf")
    (folder / "3.jpg").write_bytes(b"fake jpg")
    return tmp_path


MOCK_DB_DATA = {
    "zahtev_id": 438281, "urgency_kategorija": "URGENCY_HIGH",
    "starosna_kategorija": "APPLICANT_STANDARD", "pasos_status": "PASSPORT_OK",
    "duzina_boravka_dana": 6, "dana_do_dolaska": 5, "pasos_istice_za_dana": 400,
    "broj_prethodnih_zahteva": 0, "datum_podnosenja": "2026-06-01", "starost": 30,
}


@pytest.mark.asyncio
async def test_pipeline_skips_documents_by_filename(mock_docs_folder, monkeypatch):
    # Files "1" (passport) and "3" (unneeded image) must be dropped before OCR;
    # only "2.pdf" should ever reach the OCR/structuring stage.
    await store.init_db()
    job = await store.create_job(438281)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))

    ocr_mock = MagicMock(return_value=("sample ocr text", 1))
    structure_mock = AsyncMock(return_value=MagicMock(
        document_type="INVITATION_LETTER", file="2.pdf", confidence=0.85, language="English", fields={},
        model_dump=lambda: {"file": "2.pdf", "document_type": "INVITATION_LETTER",
                            "confidence": 0.85, "language": "English", "fields": {}},
    ))

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_pdf_text_layer", return_value=""),
        patch("app.pipeline.orchestrator.extract_text_from_file", ocr_mock),
        patch("app.pipeline.orchestrator.structure_document", structure_mock),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock, return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    # Only 2.pdf was OCR'd; 1.pdf and 3.jpg never reached the OCR call.
    ocr_paths = {call.args[0].name for call in ocr_mock.call_args_list}
    assert ocr_paths == {"2.pdf"}
    assert all(call.args[0].name == "2.pdf" for call in structure_mock.call_args_list)


@pytest.mark.asyncio
async def test_pipeline_runs_to_done(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)

    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_text_from_file", return_value=("sample ocr text", 1)),
        patch("app.pipeline.orchestrator.structure_document", new_callable=AsyncMock,
              return_value=MagicMock(
                  document_type="INVITATION_LETTER",
                  file="2.pdf",
                  confidence=0.85,
                  language="English",
                  fields={},
                  model_dump=lambda: {"file": "2.pdf", "document_type": "INVITATION_LETTER",
                                      "confidence": 0.85, "language": "English", "fields": {}}
              )),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock,
              return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    final = await store.get_job(job.job_id)
    assert final.status == JobStatus.DONE
    assert final.result is not None
    assert "categories" in final.result


@pytest.mark.asyncio
async def test_pipeline_sets_failed_on_error(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)

    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))

    with patch("app.pipeline.orchestrator.fetch_request_data", side_effect=Exception("DB down")):
        with pytest.raises(Exception, match="DB down"):
            await run_pipeline(job.job_id, 438281)

    final = await store.get_job(job.job_id)
    assert final.status == JobStatus.FAILED
    assert "DB down" in final.error


@pytest.mark.asyncio
async def test_pipeline_skips_identity_documents(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))

    mrz_text = (
        "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "E759035120CHN0103015F2705136<<<<<<<<<<<<<<00"
    )
    structure_mock = AsyncMock()

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_pdf_text_layer", return_value=""),
        patch("app.pipeline.orchestrator.extract_text_from_file", return_value=(mrz_text, 1)),
        patch("app.pipeline.orchestrator.structure_document", structure_mock),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock, return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    final = await store.get_job(job.job_id)
    assert final.status == JobStatus.DONE
    # All docs had an MRZ → all skipped before the LLM; none structured.
    assert final.result["extracted_documents"] == []
    structure_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_uses_digital_pdf_text_layer(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))

    # Disable filename skipping so this test exercises triage across all three files.
    monkeypatch.setattr("app.pipeline.orchestrator.settings.skip_document_filenames", "")

    digital_text = "INSURANCE POLICY\n" + "coverage details " * 10  # >50 chars, no MRZ
    ocr_mock = MagicMock(return_value=("ocr fallback text", 1))
    structure_mock = AsyncMock(return_value=MagicMock(
        document_type="INSURANCE", file="x", confidence=0.9, language="English", fields={},
        model_dump=lambda: {"file": "x", "document_type": "INSURANCE",
                            "confidence": 0.9, "language": "English", "fields": {}},
    ))

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_pdf_text_layer", return_value=digital_text),
        patch("app.pipeline.orchestrator.extract_text_from_file", ocr_mock),
        patch("app.pipeline.orchestrator.structure_document", structure_mock),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock, return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    # Two PDFs used the embedded text layer; OCR ran only for the .jpg.
    assert ocr_mock.call_count == 1
    # The digital text (not OCR) was passed to the structurer for the PDFs.
    assert any(call.args[1] == digital_text for call in structure_mock.call_args_list)


@pytest.mark.asyncio
async def test_pipeline_vision_fallback_on_low_confidence(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))
    monkeypatch.setattr("app.pipeline.orchestrator.settings.vision_fallback_enabled", True)

    vision_mock = AsyncMock(return_value="clean vision text")
    structure_mock = AsyncMock(return_value=MagicMock(
        document_type="INVITATION_LETTER", file="x", confidence=0.9, language="English", fields={},
        model_dump=lambda: {"file": "x", "document_type": "INVITATION_LETTER",
                            "confidence": 0.9, "language": "English", "fields": {}},
    ))

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_pdf_text_layer", return_value=""),
        patch("app.pipeline.orchestrator.extract_text_from_file", return_value=("grbld txt", 0.2)),
        patch("app.pipeline.orchestrator.vision_transcribe", vision_mock),
        patch("app.pipeline.orchestrator.structure_document", structure_mock),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock, return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    # Low OCR confidence → re-read with the vision model; its text reached the structurer.
    assert vision_mock.called
    assert all(call.args[1] == "clean vision text" for call in structure_mock.call_args_list)


@pytest.mark.asyncio
async def test_pipeline_no_vision_fallback_when_disabled(mock_docs_folder, monkeypatch):
    await store.init_db()
    job = await store.create_job(438281)
    monkeypatch.setattr("app.pipeline.orchestrator.settings.docs_base_path", str(mock_docs_folder))
    # Explicitly disabled — low confidence must NOT trigger vision.
    monkeypatch.setattr("app.pipeline.orchestrator.settings.vision_fallback_enabled", False)

    vision_mock = AsyncMock(return_value="clean vision text")
    structure_mock = AsyncMock(return_value=MagicMock(
        document_type="INVITATION_LETTER", file="x", confidence=0.9, language="English", fields={},
        model_dump=lambda: {"file": "x", "document_type": "INVITATION_LETTER",
                            "confidence": 0.9, "language": "English", "fields": {}},
    ))

    with (
        patch("app.pipeline.orchestrator.fetch_request_data", return_value=MOCK_DB_DATA),
        patch("app.pipeline.orchestrator.extract_pdf_text_layer", return_value=""),
        patch("app.pipeline.orchestrator.extract_text_from_file", return_value=("grbld txt", 0.2)),
        patch("app.pipeline.orchestrator.vision_transcribe", vision_mock),
        patch("app.pipeline.orchestrator.structure_document", structure_mock),
        patch("app.pipeline.orchestrator.apply_llm_categories", new_callable=AsyncMock, return_value=[]),
        patch("app.pipeline.orchestrator.OllamaClient"),
    ):
        await run_pipeline(job.job_id, 438281)

    assert not vision_mock.called
