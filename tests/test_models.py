import pytest
from app.models.job import Job, JobStatus
from app.models.document import ExtractedDocument
from app.models.category import Category


def test_job_defaults():
    job = Job(job_id="abc", request_id=123, status=JobStatus.PENDING, created_at="2026-06-09T10:00:00")
    assert job.finished_at is None
    assert job.error is None
    assert job.result is None


def test_job_status_enum():
    assert JobStatus.PENDING.value == "PENDING"
    assert JobStatus.RUNNING.value == "RUNNING"
    assert JobStatus.DONE.value == "DONE"
    assert JobStatus.FAILED.value == "FAILED"


def test_extracted_document_defaults():
    doc = ExtractedDocument(file="1.pdf", document_type="PASSPORT", confidence=0.9)
    assert doc.language == "unknown"
    assert doc.fields == {}


def test_extracted_document_confidence_clamp():
    doc = ExtractedDocument(file="1.pdf", document_type="PASSPORT", confidence=1.5)
    assert doc.confidence <= 1.0


def test_category_source_values():
    c = Category(code="URGENCY_HIGH", source="db", reason="Arrival in 5 days")
    assert c.code == "URGENCY_HIGH"
    assert c.source == "db"
