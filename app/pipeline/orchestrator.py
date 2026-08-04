import asyncio
import inspect
import time
from pathlib import Path

from app.config import settings
from app.models.job import JobStatus
from app.pipeline.categorizer import apply_db_rules, apply_llm_categories, apply_ocr_rules
from app.pipeline.db_fetcher import fetch_request_data
from app.pipeline.document_structurer import structure_document
from app.pipeline.ocr_engine import (
    extract_pdf_text_layer,
    extract_text_from_file,
    is_identity_mrz,
    is_photo_only,
)
from app.pipeline.vision_ocr import vision_transcribe
from app.services.job_store import set_job_result, update_job_status
from app.services.ollama_client import OllamaClient
from app.services.result_saver import append_result

_SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png"}

# Identity documents are skipped entirely — all their data already comes from the DB.
_SKIP_DOCUMENT_TYPES = {"PASSPORT", "ID_CARD"}

# A PDF with at least this many embedded characters is treated as digital (no OCR needed).
_DIGITAL_TEXT_MIN = 50


async def _read_document_text(f: Path, ollama: OllamaClient) -> str:
    """Format triage: a digital PDF (with an embedded text layer) is read directly
    with no OCR; everything else (scanned PDFs, images) goes through OCR. When OCR
    confidence is low and the vision fallback is enabled, the document is re-read with
    the vision model. Blocking work runs off the event loop so the API stays responsive."""
    if f.suffix.lower() == ".pdf":
        embedded = await asyncio.to_thread(extract_pdf_text_layer, f)
        if len(embedded.strip()) >= _DIGITAL_TEXT_MIN:
            return embedded

    text, confidence = await asyncio.to_thread(extract_text_from_file, f)

    if (settings.vision_fallback_enabled
            and text.strip()  # empty → photo-only, nothing to re-read
            and confidence < settings.ocr_confidence_threshold):
        vision_text = await vision_transcribe(f, ollama)
        if vision_text.strip():
            return vision_text

    return text


async def run_pipeline(job_id: str, request_id: int) -> None:
    start = time.time()
    ollama = OllamaClient()

    try:
        await update_job_status(job_id, JobStatus.RUNNING)

        db_data = await asyncio.to_thread(fetch_request_data, request_id)

        docs_path = Path(settings.docs_base_path) / str(request_id)
        if not docs_path.exists():
            raise FileNotFoundError(f"Document folder not found: {docs_path}")

        # Skip known-irrelevant documents by filename before any OCR (e.g. "1" is
        # always the passport, "3" an unneeded image). Saves the full OCR + LLM cost.
        skip_names = {
            n.strip() for n in settings.skip_document_filenames.split(",") if n.strip()
        }
        files = sorted(
            f for f in docs_path.iterdir()
            if f.suffix.lower() in _SUPPORTED and f.stem not in skip_names
        )

        extracted_docs = []
        for f in files:
            text = await _read_document_text(f, ollama)

            if is_photo_only(text):
                continue  # no text → applicant photo or unreadable scan; skip

            # Cheap identity-document gate: skip passports/ID cards before the
            # expensive LLM call — their data already comes from the DB. Visas and
            # other MRZ documents pass through (they are not identity documents).
            if is_identity_mrz(text):
                continue

            doc = await structure_document(f, text, ollama)

            # Safety net: catch identity docs whose MRZ did not OCR cleanly.
            if doc.document_type in _SKIP_DOCUMENT_TYPES:
                continue

            extracted_docs.append(doc)

        rule_cats = apply_db_rules(db_data)
        ocr_cats = apply_ocr_rules(extracted_docs)
        existing_codes = [c.code for c in rule_cats + ocr_cats]

        llm_cats = await apply_llm_categories(
            db_data,
            [d.model_dump() for d in extracted_docs],
            existing_codes,
            ollama,
        )

        all_cats = rule_cats + ocr_cats + llm_cats

        result = {
            "job_id": job_id,
            "request_id": request_id,
            "status": "DONE",
            "processing_time_seconds": int(time.time() - start),
            "categories": [c.model_dump() for c in all_cats],
            "extracted_documents": [d.model_dump() for d in extracted_docs],
        }
        await set_job_result(job_id, result)
        await append_result(result, settings.results_file_path)

    except Exception as exc:
        await update_job_status(job_id, JobStatus.FAILED, error=str(exc))
        raise
    finally:
        result = ollama.aclose()
        if inspect.isawaitable(result):
            await result
