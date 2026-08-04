# VISAOcr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline visa application processing pipeline that combines SQL Server data with OCR-extracted document data to produce categorized JSON results for consular officers.

**Architecture:** Monolithic FastAPI app with SQLite-backed async background tasks. `POST /analyze/{request_id}` creates a job and runs the pipeline in the background: DB fetch → OCR → LLM structuring → rule-based + LLM categorization. Results polled via `GET /status/{job_id}`.

**Tech Stack:** Python 3.10 venv (PaddleOCR requires ≤3.11), FastAPI, PaddleOCR v2 CPU, Qwen2.5:7B via Ollama, pyodbc (Windows Auth), aiosqlite, httpx, pyyaml, pymupdf, pydantic-settings, pytest

---

## File Map

```
visa-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app + lifespan
│   ├── config.py                     # pydantic-settings from .env
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── analyze.py            # POST /analyze/{request_id}
│   │       ├── status.py             # GET /status/{job_id}
│   │       ├── categories.py         # GET/POST /categories*
│   │       └── health.py             # GET /health
│   ├── models/
│   │   ├── __init__.py
│   │   ├── job.py                    # Job, JobStatus
│   │   ├── document.py               # ExtractedDocument
│   │   └── category.py               # Category
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # run_pipeline() coords all steps
│   │   ├── db_fetcher.py             # pyodbc → SQL Server
│   │   ├── ocr_engine.py             # PaddleOCR + MRZ parser
│   │   ├── document_structurer.py    # LLM: OCR text → ExtractedDocument
│   │   └── categorizer.py            # rule-based + LLM categories
│   ├── services/
│   │   ├── __init__.py
│   │   ├── job_store.py              # aiosqlite CRUD
│   │   ├── ollama_client.py          # httpx async Ollama wrapper
│   │   └── category_registry.py     # categories.json + pending CRUD
│   └── prompts/
│       ├── __init__.py
│       ├── registry.py               # YAML loader + template formatter
│       ├── structuring/
│       │   ├── v1_basic.yaml
│       │   └── v2_few_shot.yaml
│       └── categorization/
│           ├── v1_basic.yaml
│           └── v2_few_shot.yaml
├── eval/
│   ├── __init__.py
│   ├── metrics.py
│   ├── run_eval.py
│   └── test_cases/
│       ├── structuring/
│       └── categorization/
├── data/
│   ├── categories.json
│   └── pending_categories.json
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_job_store.py
│   ├── test_category_registry.py
│   ├── test_db_fetcher.py
│   ├── test_ocr_engine.py
│   ├── test_ollama_client.py
│   ├── test_prompt_registry.py
│   ├── test_document_structurer.py
│   ├── test_categorizer.py
│   └── test_routes.py
├── jobs.db                           # created at runtime
├── requirements.txt
├── .env
└── .env.example
```

---

## Task 1: Python Environment + Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create virtual environment with Python 3.10**

```powershell
cd C:\Users\strahinja\Desktop\VISAOcr
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python --version   # should print Python 3.10.x
```

- [ ] **Step 2: Create `requirements.txt`**

```
fastapi==0.115.5
uvicorn==0.32.1
paddlepaddle==2.6.2
paddleocr==2.9.1
pymupdf==1.24.14
pyodbc==5.2.0
httpx==0.28.0
pydantic-settings==2.6.1
python-dotenv==1.0.1
pyyaml==6.0.2
aiosqlite==0.20.0
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-mock==3.14.0
httpx==0.28.0
```

- [ ] **Step 3: Install dependencies**

```powershell
pip install -r requirements.txt
```

Expected: All packages install without error. PaddleOCR will download language models on first use (~500MB).

- [ ] **Step 4: Create `.env.example`**

```ini
DOCS_BASE_PATH=C:\Users\strahinja\Desktop\Test
DB_SERVER=localhost
DB_NAME=NoviVis
DB_DRIVER=ODBC Driver 17 for SQL Server
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
ACTIVE_STRUCTURING_PROMPT=v2_few_shot
ACTIVE_CATEGORIZATION_PROMPT=v1_basic
SLA_WARNING_DAYS=5
SLA_BREACH_DAYS=10
```

Copy to `.env` and adjust paths if needed.

- [ ] **Step 5: Create `app/__init__.py`** (empty)

- [ ] **Step 6: Create `app/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    docs_base_path: str = r"C:\Users\strahinja\Desktop\Test"
    db_server: str = "localhost"
    db_name: str = "NoviVis"
    db_driver: str = "ODBC Driver 17 for SQL Server"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    active_structuring_prompt: str = "v2_few_shot"
    active_categorization_prompt: str = "v1_basic"
    sla_warning_days: int = 5
    sla_breach_days: int = 10

    model_config = {"env_file": ".env"}


settings = Settings()
```

- [ ] **Step 7: Create `app/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.services.job_store import init_db
from app.api.routes import analyze, status, categories, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="VISAOcr", lifespan=lifespan)

app.include_router(analyze.router)
app.include_router(status.router)
app.include_router(categories.router)
app.include_router(health.router)
```

- [ ] **Step 8: Create `tests/__init__.py`** (empty)

- [ ] **Step 9: Create `tests/conftest.py`**

```python
import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

- [ ] **Step 10: Commit**

```bash
git init
git add requirements.txt .env.example app/ tests/
git commit -m "feat: project scaffold with config and FastAPI skeleton"
```

---

## Task 2: Pydantic Data Models

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/job.py`
- Create: `app/models/document.py`
- Create: `app/models/category.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.models.job'`

- [ ] **Step 3: Create `app/models/__init__.py`** (empty)

- [ ] **Step 4: Create `app/models/job.py`**

```python
from enum import Enum
from typing import Any
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class Job(BaseModel):
    job_id: str
    request_id: int
    status: JobStatus
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
```

- [ ] **Step 5: Create `app/models/document.py`**

```python
from pydantic import BaseModel, field_validator
from typing import Any


class ExtractedDocument(BaseModel):
    file: str
    document_type: str
    confidence: float
    language: str = "unknown"
    fields: dict[str, Any] = {}

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))
```

- [ ] **Step 6: Create `app/models/category.py`**

```python
from typing import Literal
from pydantic import BaseModel


class Category(BaseModel):
    code: str
    source: Literal["db", "ocr", "llm", "llm_new"]
    reason: str
```

- [ ] **Step 7: Run tests to verify pass**

```powershell
pytest tests/test_models.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add app/models/ tests/test_models.py
git commit -m "feat: Pydantic data models for Job, ExtractedDocument, Category"
```

---

## Task 3: SQLite Job Store

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/job_store.py`
- Create: `tests/test_job_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_job_store.py
import pytest
import pytest_asyncio
from pathlib import Path
import app.services.job_store as store
from app.models.job import JobStatus


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_jobs.db")


@pytest.mark.asyncio
async def test_create_and_get_job():
    await store.init_db()
    job = await store.create_job(request_id=12345)
    assert job.request_id == 12345
    assert job.status == JobStatus.PENDING

    fetched = await store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id


@pytest.mark.asyncio
async def test_get_nonexistent_job():
    await store.init_db()
    result = await store.get_job("nonexistent-uuid")
    assert result is None


@pytest.mark.asyncio
async def test_update_job_status():
    await store.init_db()
    job = await store.create_job(request_id=999)
    await store.update_job_status(job.job_id, JobStatus.RUNNING)
    updated = await store.get_job(job.job_id)
    assert updated.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_set_job_result():
    await store.init_db()
    job = await store.create_job(request_id=555)
    result = {"categories": [{"code": "URGENCY_HIGH", "source": "db", "reason": "test"}]}
    await store.set_job_result(job.job_id, result)
    done = await store.get_job(job.job_id)
    assert done.status == JobStatus.DONE
    assert done.result == result
    assert done.finished_at is not None
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_job_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.services.job_store'`

- [ ] **Step 3: Create `app/services/__init__.py`** (empty)

- [ ] **Step 4: Create `app/services/job_store.py`**

```python
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from app.models.job import Job, JobStatus

DB_PATH = Path("jobs.db")


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id      TEXT PRIMARY KEY,
                request_id  INTEGER NOT NULL,
                status      TEXT NOT NULL DEFAULT 'PENDING',
                created_at  TEXT NOT NULL,
                finished_at TEXT,
                error       TEXT,
                result      TEXT
            )
        """)
        await db.commit()


async def create_job(request_id: int) -> Job:
    job = Job(
        job_id=str(uuid.uuid4()),
        request_id=request_id,
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO jobs (job_id, request_id, status, created_at) VALUES (?, ?, ?, ?)",
            (job.job_id, job.request_id, job.status.value, job.created_at),
        )
        await db.commit()
    return job


async def get_job(job_id: str) -> Job | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return Job(
        job_id=row["job_id"],
        request_id=row["request_id"],
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        error=row["error"],
        result=json.loads(row["result"]) if row["result"] else None,
    )


async def update_job_status(job_id: str, status: JobStatus, error: str | None = None) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status = ?, error = ? WHERE job_id = ?",
            (status.value, error, job_id),
        )
        await db.commit()


async def set_job_result(job_id: str, result: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status = ?, result = ?, finished_at = ? WHERE job_id = ?",
            (JobStatus.DONE.value, json.dumps(result), now, job_id),
        )
        await db.commit()
```

- [ ] **Step 5: Run tests to verify pass**

```powershell
pytest tests/test_job_store.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/__init__.py app/services/job_store.py tests/test_job_store.py
git commit -m "feat: SQLite job store with CRUD operations"
```

---

## Task 4: Category Registry

**Files:**
- Create: `app/services/category_registry.py`
- Create: `data/categories.json`
- Create: `data/pending_categories.json`
- Create: `tests/test_category_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_category_registry.py
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
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_category_registry.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `data/categories.json`**

```json
{
  "version": "1.0",
  "last_updated": "2026-06-09",
  "categories": [
    {"code": "PURPOSE_HUMANITARIAN_DEATH",  "group": "Svrha putovanja", "description": "Humanitarian purpose: death of relative", "source": "llm"},
    {"code": "PURPOSE_DIPLOMATIC",          "group": "Svrha putovanja", "description": "Diplomatic mission", "source": "llm"},
    {"code": "PURPOSE_HUMANITARIAN_MED",    "group": "Svrha putovanja", "description": "Humanitarian: medical treatment", "source": "llm"},
    {"code": "PURPOSE_EXPO2027",            "group": "Svrha putovanja", "description": "Related to EXPO 2027 Belgrade", "source": "llm"},
    {"code": "PURPOSE_BUSINESS_CRITICAL",   "group": "Svrha putovanja", "description": "Critical business purpose", "source": "llm"},
    {"code": "PURPOSE_WORK",                "group": "Svrha putovanja", "description": "Work permit or employment", "source": "llm"},
    {"code": "PURPOSE_FAMILY_REUNION",      "group": "Svrha putovanja", "description": "Family reunification", "source": "llm"},
    {"code": "PURPOSE_STUDY",               "group": "Svrha putovanja", "description": "Study or education", "source": "llm"},
    {"code": "PURPOSE_TOURISM",             "group": "Svrha putovanja", "description": "Tourism", "source": "llm"},
    {"code": "PURPOSE_TRANSIT",             "group": "Svrha putovanja", "description": "Transit through Serbia", "source": "llm"},
    {"code": "PURPOSE_SPORTS",              "group": "Svrha putovanja", "description": "Sports competition or event", "source": "llm"},
    {"code": "PURPOSE_CULTURAL",            "group": "Svrha putovanja", "description": "Cultural event, concert, exhibition", "source": "llm"},
    {"code": "PURPOSE_RELIGIOUS",           "group": "Svrha putovanja", "description": "Religious pilgrimage or gathering", "source": "llm"},
    {"code": "PURPOSE_CONFERENCE",          "group": "Svrha putovanja", "description": "Conference or seminar", "source": "llm"},
    {"code": "APPLICANT_VIP",               "group": "Podnosilac",      "description": "VIP or notable applicant", "source": "llm"},
    {"code": "APPLICANT_MINOR",             "group": "Podnosilac",      "description": "Applicant is under 18", "source": "db"},
    {"code": "APPLICANT_DISABILITY",        "group": "Podnosilac",      "description": "Applicant has documented disability", "source": "llm"},
    {"code": "APPLICANT_ELDERLY",           "group": "Podnosilac",      "description": "Applicant is over 70", "source": "db"},
    {"code": "APPLICANT_RETURNING",         "group": "Podnosilac",      "description": "Applicant has visited Serbia before", "source": "llm"},
    {"code": "APPLICANT_REFUSED_BEFORE",    "group": "Podnosilac",      "description": "Previously refused a visa", "source": "llm"},
    {"code": "APPLICANT_UNACCOMPANIED_MINOR","group": "Podnosilac",     "description": "Minor traveling without parent/guardian", "source": "llm"},
    {"code": "APPLICANT_GROUP",             "group": "Podnosilac",      "description": "Group visa application", "source": "llm"},
    {"code": "PASSPORT_DIPLOMATIC",         "group": "Pasos",           "description": "Diplomatic passport", "source": "llm"},
    {"code": "PASSPORT_OFFICIAL",           "group": "Pasos",           "description": "Official/service passport", "source": "llm"},
    {"code": "PASSPORT_EXPIRED",            "group": "Pasos",           "description": "Passport is expired", "source": "db"},
    {"code": "RISK_PASSPORT_EXPIRING",      "group": "Pasos",           "description": "Passport expires within 180 days", "source": "db"},
    {"code": "DOC_COMPLETE",                "group": "Dokumenta",       "description": "All required documents present", "source": "ocr"},
    {"code": "DOC_MRZ_VALID",               "group": "Dokumenta",       "description": "Passport MRZ zone parsed successfully", "source": "ocr"},
    {"code": "DOC_MISSING_MINOR",           "group": "Dokumenta",       "description": "Minor supporting document missing", "source": "ocr"},
    {"code": "DOC_LOW_CONFIDENCE",          "group": "Dokumenta",       "description": "OCR confidence below 50% on at least one doc", "source": "ocr"},
    {"code": "DOC_MISSING_CRITICAL",        "group": "Dokumenta",       "description": "Critical document missing (e.g. passport)", "source": "ocr"},
    {"code": "DOC_TRANSLATED",              "group": "Dokumenta",       "description": "Documents in foreign language with translation", "source": "llm"},
    {"code": "DOC_MULTIPLE_PASSPORTS",      "group": "Dokumenta",       "description": "More than one travel document submitted", "source": "llm"},
    {"code": "STAY_1_3_DAYS",               "group": "Duzina boravka",  "description": "Stay 1-3 days", "source": "db"},
    {"code": "STAY_4_7_DAYS",               "group": "Duzina boravka",  "description": "Stay 4-7 days", "source": "db"},
    {"code": "STAY_8_14_DAYS",              "group": "Duzina boravka",  "description": "Stay 8-14 days", "source": "db"},
    {"code": "STAY_15_30_DAYS",             "group": "Duzina boravka",  "description": "Stay 15-30 days", "source": "db"},
    {"code": "STAY_OVER_30",                "group": "Duzina boravka",  "description": "Stay over 30 days", "source": "db"},
    {"code": "FINANCE_FULL_COVERAGE",       "group": "Finansije",       "description": "Full financial coverage confirmed", "source": "llm"},
    {"code": "FINANCE_SELF_SUFFICIENT",     "group": "Finansije",       "description": "Applicant is self-sufficient financially", "source": "llm"},
    {"code": "FINANCE_INSUFFICIENT",        "group": "Finansije",       "description": "Insufficient proof of funds", "source": "llm"},
    {"code": "FINANCE_SPONSOR",             "group": "Finansije",       "description": "Costs covered by third-party sponsor/organization", "source": "llm"},
    {"code": "ADMIN_PENDING_INFO",          "group": "Finansije",       "description": "Pending additional financial information", "source": "llm"},
    {"code": "ADMIN_SLA_BREACH",            "group": "Administrativno", "description": "SLA deadline breached", "source": "db"},
    {"code": "ADMIN_SLA_WARNING",           "group": "Administrativno", "description": "SLA deadline approaching", "source": "db"},
    {"code": "ADMIN_RESUBMITTED",           "group": "Administrativno", "description": "Applicant has submitted before", "source": "db"},
    {"code": "URGENCY_CRITICAL",            "group": "Urgentnost",      "description": "Arrival within 3 days", "source": "db"},
    {"code": "URGENCY_HIGH",                "group": "Urgentnost",      "description": "Arrival within 3-8 days", "source": "db"},
    {"code": "URGENCY_MEDIUM",              "group": "Urgentnost",      "description": "Arrival within 8-15 days", "source": "db"},
    {"code": "URGENCY_LOW",                 "group": "Urgentnost",      "description": "Arrival more than 15 days away", "source": "db"},
    {"code": "OVERSTAY_RISK",               "group": "Rizik",           "description": "Risk of overstay based on profile", "source": "llm"},
    {"code": "RISK_SUSPICIOUS_PATTERN",     "group": "Rizik",           "description": "Suspicious application pattern", "source": "llm"},
    {"code": "RISK_COUNTRY_MEDIUM",         "group": "Rizik",           "description": "Medium-risk nationality", "source": "llm"},
    {"code": "RISK_COUNTRY_HIGH",           "group": "Rizik",           "description": "High-risk nationality", "source": "llm"},
    {"code": "RISK_INCOMPLETE_DOCS",        "group": "Rizik",           "description": "Incomplete documentation pattern", "source": "llm"},
    {"code": "RISK_BLACKLIST_MATCH",        "group": "Rizik",           "description": "Potential blacklist match", "source": "llm"},
    {"code": "RISK_OVERSTAY_HISTORY",       "group": "Rizik",           "description": "History of overstaying visa", "source": "llm"},
    {"code": "RISK_FORGED_DOC_SUSPECTED",   "group": "Rizik",           "description": "LLM detected inconsistencies suggesting forgery", "source": "llm"}
  ]
}
```

- [ ] **Step 4: Create `data/pending_categories.json`**

```json
[]
```

- [ ] **Step 5: Create `app/services/category_registry.py`**

```python
import json
from pathlib import Path

CATEGORIES_PATH = Path("data/categories.json")
PENDING_PATH = Path("data/pending_categories.json")


def load_categories() -> list[dict]:
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)["categories"]


def load_pending_categories() -> list[dict]:
    with open(PENDING_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_pending_category(entry: dict) -> None:
    pending = load_pending_categories()
    if not any(p["code"] == entry["code"] for p in pending):
        pending.append(entry)
        with open(PENDING_PATH, "w", encoding="utf-8") as f:
            json.dump(pending, f, indent=2, ensure_ascii=False)


def approve_pending_category(code: str) -> bool:
    pending = load_pending_categories()
    entry = next((p for p in pending if p["code"] == code), None)
    if not entry:
        return False

    active_data = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    new_cat = {
        "code": entry["code"],
        "group": entry.get("group", "Other"),
        "description": entry.get("description", ""),
        "source": "llm",
    }
    active_data["categories"].append(new_cat)
    CATEGORIES_PATH.write_text(json.dumps(active_data, indent=2, ensure_ascii=False), encoding="utf-8")

    remaining = [p for p in pending if p["code"] != code]
    PENDING_PATH.write_text(json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8")
    return True
```

- [ ] **Step 6: Run tests to verify pass**

```powershell
pytest tests/test_category_registry.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add app/services/category_registry.py data/ tests/test_category_registry.py
git commit -m "feat: category registry with predefined categories and pending approval flow"
```

---

## Task 5: DB Fetcher

**Files:**
- Create: `app/pipeline/__init__.py`
- Create: `app/pipeline/db_fetcher.py`
- Create: `tests/test_db_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db_fetcher.py
import json
import pytest
from unittest.mock import MagicMock, patch
from app.pipeline.db_fetcher import fetch_request_data


SAMPLE_DB_JSON = {
    "zahtev_id": 438281,
    "zahtev_kod": "VR-2024-001",
    "datum_podnosenja": "2024-03-01",
    "je_maloletnik": 0,
    "urgency_kategorija": "URGENCY_HIGH",
    "starosna_kategorija": "APPLICANT_STANDARD",
    "pasos_status": "PASSPORT_OK",
    "duzina_boravka_dana": 6,
    "dana_do_dolaska": 5,
    "pasos_istice_za_dana": 400,
    "broj_prethodnih_zahteva": 0,
    "starost": 30,
    "ime": "Test",
    "prezime": "User",
}


def test_fetch_returns_dict(mocker):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (json.dumps(SAMPLE_DB_JSON),)
    mocker.patch("app.pipeline.db_fetcher.get_db_connection", return_value=mock_conn)

    result = fetch_request_data(438281)
    assert result["zahtev_id"] == 438281
    assert result["urgency_kategorija"] == "URGENCY_HIGH"


def test_fetch_raises_on_missing(mocker):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (None,)
    mocker.patch("app.pipeline.db_fetcher.get_db_connection", return_value=mock_conn)

    with pytest.raises(ValueError, match="No data found"):
        fetch_request_data(999999)
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_db_fetcher.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/pipeline/__init__.py`** (empty)

- [ ] **Step 4: Create `app/pipeline/db_fetcher.py`**

```python
import json
import pyodbc
from app.config import settings

SQL_QUERY = """
SELECT (
    SELECT
        vr.Id                                               AS zahtev_id,
        vr.Code                                             AS zahtev_kod,
        CONVERT(VARCHAR, vr.SubmissionDate, 23)             AS datum_podnosenja,
        CONVERT(VARCHAR, vr.CreatedDate, 23)                AS datum_kreiranja,
        vr.SubmittedBy                                      AS podneo,
        vr.IsMinor                                          AS je_maloletnik,
        vr.IsFromEXPO                                       AS expo_zahtev,
        vr.IsDigital                                        AS digitalni_zahtev,
        vr.Priority                                         AS trenutni_prioritet,
        vr.ConsulOpinion                                    AS misljenje_konzula,
        vs.Name                                             AS status,
        vs.NameEnglish                                      AS status_en,
        vc.Name                                             AS kategorija_vize,
        vc.NameEnglish                                      AS kategorija_vize_en,
        vc.Code                                             AS kategorija_kod,
        vd.NumberOfDays                                     AS broj_dana_boravka,
        CONVERT(VARCHAR, vd.ArrivalDate, 23)                AS datum_dolaska,
        CONVERT(VARCHAR, vd.DepartureDate, 23)              AS datum_odlaska,
        vd.TransportMeans                                   AS prevozno_sredstvo,
        vd.IsPersonInRS                                     AS vec_u_srbiji,
        vd.PreviousStaysInRs                                AS prethodni_boravci,
        vd.OtherVisas                                       AS druge_vize,
        tp.Name                                             AS svrha_putovanja,
        tp.NameEnglish                                      AS svrha_putovanja_en,
        stp.Name                                            AS podsvrha_putovanja,
        stp.NameEnglish                                     AS podsvrha_putovanja_en,
        noe.Name                                            AS broj_ulazaka,
        noe.NameEnglish                                     AS broj_ulazaka_en,
        bc.Name                                             AS granicni_prelaz,
        bc.NameEnglish                                      AS granicni_prelaz_en,
        vra.FirstName                                       AS ime,
        vra.LastName                                        AS prezime,
        vra.BirthLastName                                   AS rodjeno_prezime,
        CONVERT(VARCHAR, vra.BirthDate, 23)                 AS datum_rodjenja,
        vra.BirthPlace                                      AS mesto_rodjenja,
        vra.PersonalIdNumber                                AS jmbg,
        vra.Phone                                           AS telefon,
        vra.Email                                           AS email,
        vra.FathersName                                     AS ime_oca,
        vra.MothersName                                     AS ime_majke,
        vra.Address                                         AS adresa,
        vra.PlaceOfResidence                                AS mesto_boravka,
        g.Name                                              AS pol,
        c.Name                                              AS drzavljanstvo,
        c.NameEnglish                                       AS drzavljanstvo_en,
        c.Code                                              AS drzavljanstvo_kod,
        td.DocumentNumber                                   AS broj_dokumenta,
        td.IssuedBy                                         AS izdao_dokument,
        CONVERT(VARCHAR, td.IssueDate, 23)                  AS datum_izdavanja_dokumenta,
        CONVERT(VARCHAR, td.ExpiryDate, 23)                 AS datum_isteka_dokumenta,
        td.PermissionDocumentForReturn                      AS ima_dozvolu_povratka,
        DATEDIFF(DAY, GETDATE(), td.ExpiryDate)             AS pasos_istice_za_dana,
        hd.HostName                                         AS domacin_ime,
        hd.HostTelephone                                    AS domacin_telefon,
        hd.HostEmail                                        AS domacin_email,
        hd.HostAddress                                      AS domacin_adresa,
        hd.MeansOfSupportDescription                        AS nacin_izdrzavanja,
        inv.FullName                                        AS pozivalac_ime,
        inv.Phone                                           AS pozivalac_telefon,
        inv.Email                                           AS pozivalac_email,
        inv.LegalEntityName                                 AS pozivalac_firma,
        inv.RegistrationNumber                              AS pozivalac_maticni_br,
        it.Name                                             AS tip_pozivara,
        it.NameEnglish                                      AS tip_pozivara_en,
        (SELECT COUNT(*) FROM dbo.DocumentUploads du WHERE du.VisaRequestId = vr.Id) AS broj_uploadovanih_dokumenata,
        (SELECT COUNT(*) FROM dbo.VisaRequests vr2
         JOIN dbo.VisaRequestApplicants vra2 ON vra2.VisaRequestId = vr2.Id AND vra2.PeriodEnd > GETDATE()
         WHERE vra2.PersonalIdNumber = vra.PersonalIdNumber AND vr2.Id != vr.Id AND vr2.PeriodEnd > GETDATE()
        )                                                   AS broj_prethodnih_zahteva,
        DATEDIFF(DAY, GETDATE(), vd.ArrivalDate)            AS dana_do_dolaska,
        DATEDIFF(DAY, vd.ArrivalDate, vd.DepartureDate)     AS duzina_boravka_dana,
        DATEDIFF(YEAR, vra.BirthDate, GETDATE())            AS starost,
        CASE
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 3  THEN 'URGENCY_CRITICAL'
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 8  THEN 'URGENCY_HIGH'
            WHEN DATEDIFF(DAY, GETDATE(), vd.ArrivalDate) < 15 THEN 'URGENCY_MEDIUM'
            ELSE 'URGENCY_LOW'
        END                                                 AS urgency_kategorija,
        CASE
            WHEN vr.IsMinor = 1                                    THEN 'APPLICANT_MINOR'
            WHEN DATEDIFF(YEAR, vra.BirthDate, GETDATE()) > 70    THEN 'APPLICANT_ELDERLY'
            ELSE 'APPLICANT_STANDARD'
        END                                                 AS starosna_kategorija,
        CASE
            WHEN DATEDIFF(DAY, GETDATE(), td.ExpiryDate) < 0   THEN 'PASSPORT_EXPIRED'
            WHEN DATEDIFF(DAY, GETDATE(), td.ExpiryDate) < 180 THEN 'RISK_PASSPORT_EXPIRING'
            ELSE 'PASSPORT_OK'
        END                                                 AS pasos_status
    FROM dbo.VisaRequests vr
    LEFT JOIN dbo.VisaStatuses          vs  ON vs.Id  = vr.VisaStatusId
    LEFT JOIN dbo.VisaData              vd  ON vd.VisaRequestId = vr.Id AND vd.PeriodEnd > GETDATE()
    LEFT JOIN dbo.VisaCategories        vc  ON vc.Id  = vd.VisaCategoryId
    LEFT JOIN dbo.SubTripPurposes       stp ON stp.Id = vd.SubTripPurposeId
    LEFT JOIN dbo.TripPurposes          tp  ON tp.Id  = stp.TripPurposeId
    LEFT JOIN dbo.NumberOfEntries       noe ON noe.Id = vd.NumberOfEntryId
    LEFT JOIN dbo.BorderCrossings       bc  ON bc.Id  = vd.BorderCrossingId
    LEFT JOIN dbo.VisaRequestApplicants vra ON vra.VisaRequestId = vr.Id AND vra.PeriodEnd > GETDATE()
    LEFT JOIN dbo.Genders               g   ON g.Id   = vra.GenderId
    LEFT JOIN dbo.Citizenships          c   ON c.Id   = vra.NationalityId
    LEFT JOIN dbo.TravelDocuments       td  ON td.VisaRequestId = vr.Id
    LEFT JOIN dbo.HostData              hd  ON hd.VisaRequestId = vr.Id AND hd.PeriodEnd > GETDATE()
    LEFT JOIN dbo.Inviters              inv ON inv.VisaRequestId = vr.Id AND inv.PeriodEnd > GETDATE()
    LEFT JOIN dbo.InviterTypes          it  ON it.Id  = inv.InviterTypeId
    WHERE vr.Id = ? AND vr.PeriodEnd > GETDATE()
    FOR JSON PATH, WITHOUT_ARRAY_WRAPPER
) AS rezultat
"""


def get_db_connection():
    conn_str = (
        f"DRIVER={{{settings.db_driver}}};"
        f"SERVER={settings.db_server};"
        f"DATABASE={settings.db_name};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def fetch_request_data(request_id: int) -> dict:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(SQL_QUERY, (request_id,))
        row = cursor.fetchone()

    if not row or not row[0]:
        raise ValueError(f"No data found for RequestId {request_id}")

    return json.loads(row[0])
```

- [ ] **Step 5: Run tests to verify pass**

```powershell
pytest tests/test_db_fetcher.py -v
```

Expected: Both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/pipeline/ tests/test_db_fetcher.py
git commit -m "feat: DB fetcher with full SQL query for visa request data"
```

---

## Task 6: OCR Engine

**Files:**
- Create: `app/pipeline/ocr_engine.py`
- Create: `tests/test_ocr_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ocr_engine.py
import pytest
from pathlib import Path
from app.pipeline.ocr_engine import parse_mrz, is_photo_only_file


def test_parse_mrz_valid_passport():
    # Simulated clean MRZ text from PaddleOCR
    ocr_text = (
        "REPUBLIC OF CHINA\n"
        "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "E7590351226CHN0103015F2705133<<<<<<<<<<<<<<<4"
    )
    result = parse_mrz(ocr_text)
    assert result is not None
    assert result["mrz_valid"] is True
    assert result["nationality"] == "CHN"
    assert result["surname"] == "ZHANG"
    assert result["given_names"] == "NANA"
    assert result["expiry_date"] == "2027-05-13"


def test_parse_mrz_no_mrz_lines():
    result = parse_mrz("This is an invitation letter with no MRZ zone.")
    assert result is None


def test_parse_mrz_short_lines():
    result = parse_mrz("P<CHN\nE75903512")
    assert result is None


def test_is_photo_only_file_true():
    assert is_photo_only_file(Path("3.jpg")) is True
    assert is_photo_only_file(Path(r"C:\docs\3.JPG")) is True


def test_is_photo_only_file_false():
    assert is_photo_only_file(Path("1.pdf")) is False
    assert is_photo_only_file(Path("7.jpg")) is False
    assert is_photo_only_file(Path("3.pdf")) is False
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_ocr_engine.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/pipeline/ocr_engine.py`**

```python
import re
from pathlib import Path

import fitz  # pymupdf
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

_ocr_instance: PaddleOCR | None = None

PHOTO_STEMS = {"3"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_ocr() -> PaddleOCR:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    return _ocr_instance


def is_photo_only_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in PHOTO_EXTENSIONS and file_path.stem in PHOTO_STEMS


def _ocr_numpy(image: np.ndarray) -> str:
    ocr = get_ocr()
    result = ocr.ocr(image, cls=True)
    if not result or not result[0]:
        return ""
    return "\n".join(line[1][0] for line in result[0] if line and len(line) >= 2)


def _pdf_to_images(pdf_path: Path) -> list[np.ndarray]:
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    return images


def extract_text_from_file(file_path: Path) -> tuple[str, int]:
    """Return (ocr_text, page_count). Skips photo-only files."""
    if is_photo_only_file(file_path):
        return "", 0

    suffix = file_path.suffix.lower()
    if suffix in PHOTO_EXTENSIONS:
        img = np.array(Image.open(file_path))
        return _ocr_numpy(img), 1

    if suffix == ".pdf":
        images = _pdf_to_images(file_path)
        texts = [_ocr_numpy(img) for img in images]
        return "\n\n--- PAGE BREAK ---\n\n".join(texts), len(images)

    raise ValueError(f"Unsupported file type: {suffix}")


def parse_mrz(text: str) -> dict | None:
    """Parse MRZ zone from OCR text. Returns None if no MRZ found."""
    mrz_re = re.compile(r"[A-Z0-9<]{44}")
    lines = [ln.replace(" ", "").upper() for ln in text.splitlines()]
    mrz_lines = [ln for ln in lines if mrz_re.fullmatch(ln)]

    if len(mrz_lines) < 2:
        return None

    line1, line2 = mrz_lines[-2], mrz_lines[-1]

    def yymmdd_to_iso(s: str) -> str:
        yy, mm, dd = int(s[:2]), s[2:4], s[4:6]
        year = 2000 + yy if yy < 30 else 1900 + yy
        return f"{year:04d}-{mm}-{dd}"

    name_field = line1[5:44]
    parts = name_field.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

    return {
        "mrz_valid": True,
        "document_number": line2[0:9].replace("<", ""),
        "nationality": line2[10:13].replace("<", ""),
        "birth_date": yymmdd_to_iso(line2[13:19]),
        "expiry_date": yymmdd_to_iso(line2[21:27]),
        "sex": line2[20],
        "surname": surname,
        "given_names": given,
    }
```

- [ ] **Step 4: Run tests to verify pass**

```powershell
pytest tests/test_ocr_engine.py -v
```

Expected: All 5 tests PASS. (PaddleOCR loads on first real use — unit tests only test `parse_mrz` and `is_photo_only_file` which don't require the model.)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/ocr_engine.py tests/test_ocr_engine.py
git commit -m "feat: OCR engine with PaddleOCR wrapper and MRZ parser"
```

---

## Task 7: Ollama Client

**Files:**
- Create: `app/services/ollama_client.py`
- Create: `tests/test_ollama_client.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ollama_client.py
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_generate_returns_parsed_json(respx_mock=None):
    client = OllamaClient()
    mock_response = {"response": json.dumps({"document_type": "PASSPORT", "confidence": 0.9})}

    with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        result = await client.generate("test prompt")
        assert result["document_type"] == "PASSPORT"
    await client.aclose()


@pytest.mark.asyncio
async def test_generate_retries_on_invalid_json():
    client = OllamaClient()
    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if call_count < 2:
            resp.json.return_value = {"response": "not json at all"}
        else:
            resp.json.return_value = {"response": json.dumps({"ok": True})}
        return resp

    with patch.object(client._client, "post", side_effect=mock_post):
        result = await client.generate("test prompt", max_retries=3)
        assert result["ok"] is True
        assert call_count == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_health_check_true():
    client = OllamaClient()
    with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert await client.health_check() is True
    await client.aclose()
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_ollama_client.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/services/ollama_client.py`**

```python
import json
import httpx
from app.config import settings


class OllamaClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=180.0)
        self._model = settings.ollama_model
        self._base_url = settings.ollama_base_url

    async def generate(self, prompt: str, temperature: float = 0.1, max_retries: int = 3) -> dict:
        """Call Ollama /api/generate with JSON format enforcement. Retries on parse failure."""
        for attempt in range(max_retries):
            actual_prompt = (
                prompt
                if attempt == 0
                else prompt + "\n\nIMPORTANT: Return ONLY raw JSON. No markdown. No explanation."
            )
            response = await self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": actual_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            raw = response.json()["response"]
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    raise ValueError(f"Ollama returned invalid JSON after {max_retries} attempts: {raw[:200]}")

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify pass**

```powershell
pytest tests/test_ollama_client.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: async Ollama client with retry logic and JSON enforcement"
```

---

## Task 8: Prompt Registry

**Files:**
- Create: `app/prompts/__init__.py`
- Create: `app/prompts/registry.py`
- Create: `tests/test_prompt_registry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompt_registry.py
import pytest
from pathlib import Path
import yaml
from app.prompts.registry import PromptRegistry


@pytest.fixture
def registry_with_fixtures(tmp_path, monkeypatch):
    structuring_dir = tmp_path / "structuring"
    structuring_dir.mkdir()
    yaml_content = {
        "name": "test_v1",
        "version": "1.0",
        "temperature": 0.1,
        "system": "You are a test system.",
        "few_shots": [
            {"input": "sample ocr text", "output": '{"document_type": "PASSPORT"}'}
        ],
        "template": "{system}\nEXAMPLES:\n{few_shots_formatted}\nNOW ANALYZE:\n{ocr_text}",
    }
    (structuring_dir / "test_v1.yaml").write_text(yaml.dump(yaml_content))
    reg = PromptRegistry(base_dir=tmp_path)
    return reg


def test_load_prompt(registry_with_fixtures):
    config = registry_with_fixtures.load("structuring", "test_v1")
    assert config["name"] == "test_v1"
    assert config["temperature"] == 0.1


def test_format_prompt_includes_ocr_text(registry_with_fixtures):
    prompt, temp = registry_with_fixtures.format("structuring", "test_v1", ocr_text="my ocr text")
    assert "my ocr text" in prompt
    assert temp == 0.1


def test_format_prompt_includes_few_shots(registry_with_fixtures):
    prompt, _ = registry_with_fixtures.format("structuring", "test_v1", ocr_text="anything")
    assert "sample ocr text" in prompt
    assert "PASSPORT" in prompt
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_prompt_registry.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/prompts/__init__.py`** (empty)

- [ ] **Step 4: Create `app/prompts/registry.py`**

```python
from pathlib import Path
import yaml

_DEFAULT_BASE = Path(__file__).parent


class PromptRegistry:
    def __init__(self, base_dir: Path = _DEFAULT_BASE) -> None:
        self._base = base_dir
        self._cache: dict[str, dict] = {}

    def load(self, prompt_type: str, variant: str) -> dict:
        key = f"{prompt_type}/{variant}"
        if key not in self._cache:
            path = self._base / prompt_type / f"{variant}.yaml"
            with open(path, encoding="utf-8") as f:
                self._cache[key] = yaml.safe_load(f)
        return self._cache[key]

    def format(self, prompt_type: str, variant: str, **kwargs) -> tuple[str, float]:
        """Return (formatted_prompt_string, temperature)."""
        config = self.load(prompt_type, variant)

        few_shots_text = ""
        if "few_shots" in config:
            parts = [
                f"INPUT:\n{ex['input'].strip()}\n\nOUTPUT:\n{ex['output'].strip()}"
                for ex in config["few_shots"]
            ]
            few_shots_text = "\n\n---\n\n".join(parts)

        prompt = config["template"].format(
            system=config.get("system", ""),
            few_shots_formatted=few_shots_text,
            **kwargs,
        )
        return prompt, float(config.get("temperature", 0.1))


registry = PromptRegistry()
```

- [ ] **Step 5: Run tests to verify pass**

```powershell
pytest tests/test_prompt_registry.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/prompts/__init__.py app/prompts/registry.py tests/test_prompt_registry.py
git commit -m "feat: YAML-based prompt registry with few-shot formatting"
```

---

## Task 9: Prompt YAML Files

**Files:**
- Create: `app/prompts/structuring/v1_basic.yaml`
- Create: `app/prompts/structuring/v2_few_shot.yaml`
- Create: `app/prompts/categorization/v1_basic.yaml`
- Create: `app/prompts/categorization/v2_few_shot.yaml`

No tests for data files. Run a smoke test instead.

- [ ] **Step 1: Create `app/prompts/structuring/v1_basic.yaml`**

```yaml
name: "structuring_v1_basic"
version: "1.0"
temperature: 0.1
system: |
  You are a document analysis expert for visa processing in Serbia.
  Extract structured data from OCR text of scanned visa application documents.
  Supported document types: PASSPORT, INVITATION_LETTER, VISA_APPLICATION,
  RESIDENCE_PERMIT, INSURANCE, ACCOMMODATION_CONFIRMATION,
  EMAIL_CORRESPONDENCE, DECLARATION, BANK_STATEMENT, OTHER.
  Rules:
  - Return ONLY valid JSON, no markdown, no explanation.
  - Use null for fields that cannot be read.
  - All dates in ISO format YYYY-MM-DD.
  - confidence: float 0.0-1.0 (readability of the OCR text).
few_shots: []
template: |
  {system}

  OCR TEXT TO ANALYZE:
  {ocr_text}

  Return JSON:
  {
    "document_type": "<TYPE>",
    "confidence": <float>,
    "language": "<detected language>",
    "fields": { <relevant fields> }
  }

  Field guide:
  - PASSPORT: first_name, last_name, birth_date, expiry_date, issue_date, nationality, document_number, issuing_country, mrz_line1, mrz_line2
  - INVITATION_LETTER: inviting_org, inviting_person, event, period_from, period_to, purpose, host_address
  - ACCOMMODATION_CONFIRMATION: hotel_name, guest_name, check_in, check_out, address, confirmation_number
  - INSURANCE: insured_name, policy_number, valid_from, valid_to, coverage_amount, insurer
  - BANK_STATEMENT: account_holder, bank_name, balance, currency, statement_date
  - RESIDENCE_PERMIT: holder_name, permit_number, valid_until, permit_type, issuing_country
  - DECLARATION: declarant_name, declaration_date, declaration_subject
  - VISA_APPLICATION: applicant_name, nationality, purpose, entry_date, exit_date, destination
  - EMAIL_CORRESPONDENCE: sender, recipient, subject, date, key_points
```

- [ ] **Step 2: Create `app/prompts/structuring/v2_few_shot.yaml`**

```yaml
name: "structuring_v2_few_shot"
version: "2.0"
temperature: 0.1
system: |
  You are a document analysis expert for visa processing in Serbia.
  Extract structured data from OCR text of scanned visa application documents.
  Supported document types: PASSPORT, INVITATION_LETTER, VISA_APPLICATION,
  RESIDENCE_PERMIT, INSURANCE, ACCOMMODATION_CONFIRMATION,
  EMAIL_CORRESPONDENCE, DECLARATION, BANK_STATEMENT, OTHER.
  Rules:
  - Return ONLY valid JSON, no markdown, no explanation.
  - Use null for fields that cannot be read.
  - All dates in ISO format YYYY-MM-DD.
  - confidence: float 0.0-1.0 (readability of the OCR text).
few_shots:
  - input: |
      REPUBLIC OF CHINA
      P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<
      E7590351226CHN0103015F2705133<<<<<<<<<<<<<<<4
      Surname: ZHANG
      Given names: NA NA
      Nationality: CHINESE
      Date of birth: 01 MAR 2001
      Date of expiry: 13 MAY 2027
    output: |
      {
        "document_type": "PASSPORT",
        "confidence": 0.95,
        "language": "English",
        "fields": {
          "first_name": "NA NA",
          "last_name": "ZHANG",
          "birth_date": "2001-03-01",
          "expiry_date": "2027-05-13",
          "nationality": "CHN",
          "document_number": "E75903512",
          "issuing_country": "CHN",
          "mrz_line1": "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<",
          "mrz_line2": "E7590351226CHN0103015F2705133<<<<<<<<<<<<<<<4"
        }
      }

  - input: |
      Udruženje kreativno-edukativni centar "Muzički Atelje"
      Koče Kolarova 3, Novi Sad
      Association of Muzicki Atelje invites Na-Na Zhang on the
      International Gathering of Violinists in Serbia, Sremski Karlovci,
      Novi Sad in period from 04. to 08. 04. 2019.
      Since NaNa Zhang is less than 18 years old, we invite her
      mother Chen Shuting to accompany her to come to the competition.
      Director: Milan Cizmic
    output: |
      {
        "document_type": "INVITATION_LETTER",
        "confidence": 0.88,
        "language": "English",
        "fields": {
          "inviting_org": "Muzički Atelje",
          "inviting_person": "Milan Cizmic",
          "event": "International Gathering of Violinists",
          "period_from": "2019-04-04",
          "period_to": "2019-04-08",
          "purpose": "Cultural event - violin competition",
          "host_address": "Koče Kolarova 3, Novi Sad"
        }
      }

  - input: |
      TURISTIČKA ORGANIZACIJA OPŠTINE SREMSKI KARLOVCI
      Ms. Zhang Na Na
      Reservation confirmation
      I confirm booking of accommodation for Zhang Na Na and Chen Shuting
      in guest house "Zeravica b&b", Sremski Karlovci, Serbia,
      for period 05.04.2019 to 08.04.2019.
      This confirmation is made for the visa application.
    output: |
      {
        "document_type": "ACCOMMODATION_CONFIRMATION",
        "confidence": 0.92,
        "language": "English",
        "fields": {
          "hotel_name": "Zeravica b&b",
          "guest_name": "Zhang Na Na",
          "check_in": "2019-04-05",
          "check_out": "2019-04-08",
          "address": "Sremski Karlovci, Serbia",
          "confirmation_number": null
        }
      }

  - input: |
      Allianz Tiriac
      Polita de asigurare de calatorie in strainatate nr. VCJ/860063618
      Asigurat: ZHANG NANA
      Perioada asigurata: 04.04.2019 - 08.04.2019
      Suma asigurata: 30,000 EUR
    output: |
      {
        "document_type": "INSURANCE",
        "confidence": 0.90,
        "language": "Romanian",
        "fields": {
          "insured_name": "ZHANG NANA",
          "policy_number": "VCJ/860063618",
          "valid_from": "2019-04-04",
          "valid_to": "2019-04-08",
          "coverage_amount": "30000 EUR",
          "insurer": "Allianz Tiriac"
        }
      }

template: |
  {system}

  EXAMPLES:
  {few_shots_formatted}

  NOW ANALYZE THIS DOCUMENT:
  {ocr_text}

  Return JSON:
  {
    "document_type": "<TYPE>",
    "confidence": <float>,
    "language": "<detected language>",
    "fields": { <relevant fields> }
  }
```

- [ ] **Step 3: Create `app/prompts/categorization/v1_basic.yaml`**

```yaml
name: "categorization_v1_basic"
version: "1.0"
temperature: 0.1
system: |
  You are a visa risk assessment system for the Republic of Serbia.
  Analyze applicant data and assign relevant categories.
  Rules:
  - Only assign categories that clearly apply based on the data.
  - Do NOT repeat categories listed in ALREADY_ASSIGNED.
  - For genuinely novel situations not covered by predefined categories,
    propose a new category using format DOMAIN_KEYWORD (e.g. APPLICANT_JOURNALIST).
  - Reason must be in English, concise (max 15 words).
  - Return ONLY valid JSON.
few_shots: []
template: |
  {system}

  PREDEFINED CATEGORIES (assign from these):
  {categories_json}

  ALREADY ASSIGNED (do NOT repeat):
  {rule_based_categories}

  APPLICANT DATA FROM DATABASE:
  {db_json}

  EXTRACTED DOCUMENT DATA:
  {documents_json}

  Return JSON:
  {
    "categories": [
      {"code": "<PREDEFINED_CODE>", "reason": "<English reason max 15 words>"}
    ],
    "new_categories": [
      {"code": "<NEW_CODE>", "reason": "<why this applies>", "description": "<what this means>"}
    ]
  }
```

- [ ] **Step 4: Create `app/prompts/categorization/v2_few_shot.yaml`**

```yaml
name: "categorization_v2_few_shot"
version: "2.0"
temperature: 0.1
system: |
  You are a visa risk assessment system for the Republic of Serbia.
  Analyze applicant data and assign relevant categories.
  Rules:
  - Only assign categories that clearly apply based on the data.
  - Do NOT repeat categories listed in ALREADY_ASSIGNED.
  - For genuinely novel situations not covered by predefined categories,
    propose a new category using format DOMAIN_KEYWORD (e.g. APPLICANT_JOURNALIST).
  - Reason must be in English, concise (max 15 words).
  - Return ONLY valid JSON.
few_shots:
  - input: |
      DB: {"drzavljanstvo_en": "Chinese", "starost": 17, "svrha_putovanja_en": "Cultural"}
      DOCS: [{"document_type": "INVITATION_LETTER", "fields": {"event": "violin competition", "inviting_org": "Muzički Atelje"}}]
      ALREADY_ASSIGNED: ["APPLICANT_MINOR", "URGENCY_HIGH", "STAY_4_7_DAYS"]
    output: |
      {
        "categories": [
          {"code": "PURPOSE_CULTURAL", "reason": "Invitation to international violin competition"},
          {"code": "FINANCE_SPONSOR", "reason": "Costs covered by inviting organization Muzički Atelje"}
        ],
        "new_categories": [
          {"code": "APPLICANT_UNACCOMPANIED_MINOR", "reason": "Minor traveling with mother, not registered guardian", "description": "Minor applicant accompanied by parent not listed as official guardian"}
        ]
      }
template: |
  {system}

  PREDEFINED CATEGORIES (assign from these):
  {categories_json}

  ALREADY ASSIGNED (do NOT repeat):
  {rule_based_categories}

  EXAMPLE:
  {few_shots_formatted}

  APPLICANT DATA FROM DATABASE:
  {db_json}

  EXTRACTED DOCUMENT DATA:
  {documents_json}

  Return JSON:
  {
    "categories": [
      {"code": "<PREDEFINED_CODE>", "reason": "<English reason max 15 words>"}
    ],
    "new_categories": [
      {"code": "<NEW_CODE>", "reason": "<why this applies>", "description": "<what this means>"}
    ]
  }
```

- [ ] **Step 5: Smoke test — verify YAML loads without error**

```powershell
python -c "
from app.prompts.registry import registry
p, t = registry.format('structuring', 'v2_few_shot', ocr_text='test text')
print('structuring v2_few_shot OK, temp=', t)
p2, t2 = registry.format('categorization', 'v1_basic', categories_json='[]', rule_based_categories='[]', db_json='{}', documents_json='[]')
print('categorization v1_basic OK, temp=', t2)
"
```

Expected: Two OK lines printed, no errors.

- [ ] **Step 6: Commit**

```bash
git add app/prompts/
git commit -m "feat: prompt YAML files with few-shot examples for structuring and categorization"
```

---

## Task 10: Document Structurer

**Files:**
- Create: `app/pipeline/document_structurer.py`
- Create: `tests/test_document_structurer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_document_structurer.py
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
async def test_structure_passport(mock_ollama, monkeypatch):
    mock_ollama.generate.return_value = {
        "document_type": "PASSPORT",
        "confidence": 0.95,
        "language": "English",
        "fields": {"first_name": "NA NA", "last_name": "ZHANG", "expiry_date": "2027-05-13"}
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    ocr_text = (
        "P<CHNZHANG<<NANA<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
        "E7590351226CHN0103015F2705133<<<<<<<<<<<<<<<4"
    )
    doc = await structure_document(Path("1.pdf"), ocr_text, mock_ollama)

    assert isinstance(doc, ExtractedDocument)
    assert doc.document_type == "PASSPORT"
    assert doc.fields["mrz_valid"] is True
    assert doc.fields["nationality"] == "CHN"


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
async def test_structure_uses_confidence_clamp(mock_ollama, monkeypatch):
    mock_ollama.generate.return_value = {
        "document_type": "OTHER", "confidence": 2.5, "language": "unknown", "fields": {}
    }
    monkeypatch.setattr("app.pipeline.document_structurer.settings.active_structuring_prompt", "v1_basic")

    doc = await structure_document(Path("9.pdf"), "random text", mock_ollama)
    assert doc.confidence <= 1.0
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_document_structurer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/pipeline/document_structurer.py`**

```python
from pathlib import Path

from app.config import settings
from app.models.document import ExtractedDocument
from app.pipeline.ocr_engine import parse_mrz
from app.prompts.registry import registry
from app.services.ollama_client import OllamaClient


async def structure_document(
    file_path: Path,
    ocr_text: str,
    ollama: OllamaClient,
) -> ExtractedDocument:
    """Call LLM to classify document type and extract fields from OCR text."""
    variant = settings.active_structuring_prompt
    prompt, temp = registry.format("structuring", variant, ocr_text=ocr_text)
    data = await ollama.generate(prompt, temperature=temp)

    doc = ExtractedDocument(
        file=file_path.name,
        document_type=data.get("document_type", "OTHER"),
        confidence=float(data.get("confidence", 0.0)),
        language=data.get("language", "unknown"),
        fields=data.get("fields") or {},
    )

    if doc.document_type == "PASSPORT":
        mrz = parse_mrz(ocr_text)
        if mrz:
            doc.fields.update(mrz)

    return doc
```

- [ ] **Step 4: Run tests to verify pass**

```powershell
pytest tests/test_document_structurer.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/document_structurer.py tests/test_document_structurer.py
git commit -m "feat: document structurer - LLM classifies and extracts fields per document"
```

---

## Task 11: Categorizer (Rule-based + LLM)

**Files:**
- Create: `app/pipeline/categorizer.py`
- Create: `tests/test_categorizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_categorizer.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
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


def test_ocr_rules_mrz_valid():
    docs = [
        ExtractedDocument(file="1.pdf", document_type="PASSPORT", confidence=0.9,
                          fields={"mrz_valid": True})
    ]
    cats = apply_ocr_rules(docs)
    codes = [c.code for c in cats]
    assert "DOC_MRZ_VALID" in codes
    assert "DOC_COMPLETE" in codes


def test_ocr_rules_missing_critical():
    docs = [
        ExtractedDocument(file="2.pdf", document_type="INVITATION_LETTER", confidence=0.8)
    ]
    cats = apply_ocr_rules(docs)
    codes = [c.code for c in cats]
    assert "DOC_MISSING_CRITICAL" in codes


def test_ocr_rules_low_confidence():
    docs = [
        ExtractedDocument(file="1.pdf", document_type="PASSPORT", confidence=0.3,
                          fields={"mrz_valid": True})
    ]
    cats = apply_ocr_rules(docs)
    codes = [c.code for c in cats]
    assert "DOC_LOW_CONFIDENCE" in codes


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
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_categorizer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/pipeline/categorizer.py`**

```python
import json
from datetime import datetime, timezone

from app.config import settings
from app.models.category import Category
from app.models.document import ExtractedDocument
from app.prompts.registry import registry
from app.services.category_registry import load_categories, save_pending_category
from app.services.ollama_client import OllamaClient


def apply_db_rules(db: dict) -> list[Category]:
    cats: list[Category] = []

    urgency = db.get("urgency_kategorija")
    if urgency:
        days = db.get("dana_do_dolaska", 0)
        cats.append(Category(code=urgency, source="db", reason=f"Arrival in {days} days"))

    starosna = db.get("starosna_kategorija", "APPLICANT_STANDARD")
    if starosna != "APPLICANT_STANDARD":
        age = db.get("starost", 0)
        label = "under 18" if starosna == "APPLICANT_MINOR" else "over 70"
        cats.append(Category(code=starosna, source="db", reason=f"Applicant is {age} years old ({label})"))

    pasos = db.get("pasos_status")
    if pasos and pasos != "PASSPORT_OK":
        days = db.get("pasos_istice_za_dana", 0)
        if pasos == "PASSPORT_EXPIRED":
            cats.append(Category(code="PASSPORT_EXPIRED", source="db",
                                 reason=f"Passport expired {abs(days)} days ago"))
        else:
            cats.append(Category(code="RISK_PASSPORT_EXPIRING", source="db",
                                 reason=f"Passport expires in {days} days"))

    stay = db.get("duzina_boravka_dana", 0)
    if stay:
        if stay <= 3:
            code = "STAY_1_3_DAYS"
        elif stay <= 7:
            code = "STAY_4_7_DAYS"
        elif stay <= 14:
            code = "STAY_8_14_DAYS"
        elif stay <= 30:
            code = "STAY_15_30_DAYS"
        else:
            code = "STAY_OVER_30"
        cats.append(Category(code=code, source="db", reason=f"Stay duration: {stay} days"))

    prev = db.get("broj_prethodnih_zahteva", 0)
    if prev:
        cats.append(Category(code="ADMIN_RESUBMITTED", source="db",
                             reason=f"Applicant has {prev} previous request(s)"))

    submitted = db.get("datum_podnosenja")
    if submitted:
        days_since = (datetime.now(timezone.utc).date() -
                      datetime.fromisoformat(submitted).date()).days
        if days_since > settings.sla_breach_days:
            cats.append(Category(code="ADMIN_SLA_BREACH", source="db",
                                 reason=f"Request pending {days_since} days, SLA breach"))
        elif days_since > settings.sla_warning_days:
            cats.append(Category(code="ADMIN_SLA_WARNING", source="db",
                                 reason=f"Request pending {days_since} days, SLA warning"))

    return cats


def apply_ocr_rules(docs: list[ExtractedDocument]) -> list[Category]:
    cats: list[Category] = []

    has_passport = any(d.document_type == "PASSPORT" for d in docs)
    mrz_docs = [d for d in docs if d.document_type == "PASSPORT" and d.fields.get("mrz_valid")]
    low_conf = [d for d in docs if d.confidence < 0.5]

    if mrz_docs:
        cats.append(Category(code="DOC_MRZ_VALID", source="ocr",
                             reason="Passport MRZ parsed successfully"))

    if low_conf:
        names = ", ".join(d.file for d in low_conf)
        cats.append(Category(code="DOC_LOW_CONFIDENCE", source="ocr",
                             reason=f"Low OCR confidence on: {names}"))

    if not has_passport:
        cats.append(Category(code="DOC_MISSING_CRITICAL", source="ocr",
                             reason="No passport found in submitted documents"))
    elif len(docs) >= 2:
        cats.append(Category(code="DOC_COMPLETE", source="ocr",
                             reason="Passport and supporting documents present"))

    return cats


async def apply_llm_categories(
    db_data: dict,
    documents: list[dict],
    existing_codes: list[str],
    ollama: OllamaClient,
) -> list[Category]:
    active = load_categories()
    categories_json = json.dumps(
        [{"code": c["code"], "description": c["description"]} for c in active],
        indent=2,
    )

    variant = settings.active_categorization_prompt
    prompt, temp = registry.format(
        "categorization",
        variant,
        categories_json=categories_json,
        rule_based_categories=json.dumps(existing_codes),
        db_json=json.dumps(db_data, ensure_ascii=False, indent=2),
        documents_json=json.dumps(documents, ensure_ascii=False, indent=2),
    )

    result = await ollama.generate(prompt, temperature=temp)
    cats: list[Category] = []

    for item in result.get("categories", []):
        code = item.get("code", "").strip()
        if code and code not in existing_codes:
            cats.append(Category(code=code, source="llm", reason=item.get("reason", "")))

    for item in result.get("new_categories", []):
        code = item.get("code", "").strip()
        if code:
            save_pending_category({
                "code": code,
                "reason": item.get("reason", ""),
                "description": item.get("description", ""),
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            })
            cats.append(Category(code=code, source="llm_new", reason=item.get("reason", "")))

    return cats
```

- [ ] **Step 4: Run tests to verify pass**

```powershell
pytest tests/test_categorizer.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/categorizer.py tests/test_categorizer.py
git commit -m "feat: categorizer with rule-based DB rules, OCR rules, and LLM categorization"
```

---

## Task 12: Pipeline Orchestrator

**Files:**
- Create: `app/pipeline/orchestrator.py`
- Create: `tests/test_orchestrator.py` (integration test with all mocks)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import json
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
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_orchestrator.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `app/pipeline/orchestrator.py`**

```python
import time
from pathlib import Path

from app.config import settings
from app.models.job import JobStatus
from app.pipeline.categorizer import apply_db_rules, apply_llm_categories, apply_ocr_rules
from app.pipeline.db_fetcher import fetch_request_data
from app.pipeline.document_structurer import structure_document
from app.pipeline.ocr_engine import extract_text_from_file, is_photo_only_file
from app.services.job_store import set_job_result, update_job_status
from app.services.ollama_client import OllamaClient

_SUPPORTED = {".pdf", ".jpg", ".jpeg", ".png"}


async def run_pipeline(job_id: str, request_id: int) -> None:
    start = time.time()
    ollama = OllamaClient()

    try:
        await update_job_status(job_id, JobStatus.RUNNING)

        db_data = fetch_request_data(request_id)

        docs_path = Path(settings.docs_base_path) / str(request_id)
        if not docs_path.exists():
            raise FileNotFoundError(f"Document folder not found: {docs_path}")

        files = sorted(
            f for f in docs_path.iterdir()
            if f.suffix.lower() in _SUPPORTED and not is_photo_only_file(f)
        )

        extracted_docs = []
        for f in files:
            text, _ = extract_text_from_file(f)
            doc = await structure_document(f, text, ollama)
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

    except Exception as exc:
        await update_job_status(job_id, JobStatus.FAILED, error=str(exc))
        raise
    finally:
        await ollama.aclose()
```

- [ ] **Step 4: Run tests to verify pass**

```powershell
pytest tests/test_orchestrator.py -v
```

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: pipeline orchestrator coordinates all processing steps"
```

---

## Task 13: FastAPI Routes

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/routes/__init__.py`
- Create: `app/api/routes/analyze.py`
- Create: `app/api/routes/status.py`
- Create: `app/api/routes/categories.py`
- Create: `app/api/routes/health.py`
- Modify: `app/main.py` (already has imports, just verify)
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_routes.py
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import app.services.job_store as store
from pathlib import Path


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")


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
```

- [ ] **Step 2: Run test to verify failure**

```powershell
pytest tests/test_routes.py -v
```

Expected: `ModuleNotFoundError` or import errors

- [ ] **Step 3: Create `app/api/__init__.py`** and `app/api/routes/__init__.py`** (both empty)

- [ ] **Step 4: Create `app/api/routes/analyze.py`**

```python
from fastapi import APIRouter, BackgroundTasks
from app.pipeline.orchestrator import run_pipeline
from app.services.job_store import create_job

router = APIRouter()


@router.post("/analyze/{request_id}")
async def analyze(request_id: int, background_tasks: BackgroundTasks):
    job = await create_job(request_id)
    background_tasks.add_task(run_pipeline, job.job_id, request_id)
    return {"job_id": job.job_id}
```

- [ ] **Step 5: Create `app/api/routes/status.py`**

```python
from fastapi import APIRouter, HTTPException
from app.services.job_store import get_job

router = APIRouter()


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.model_dump()
```

- [ ] **Step 6: Create `app/api/routes/categories.py`**

```python
from fastapi import APIRouter, HTTPException
from app.services.category_registry import (
    approve_pending_category,
    load_categories,
    load_pending_categories,
)

router = APIRouter()


@router.get("/categories")
def list_categories():
    return load_categories()


@router.get("/categories/pending")
def list_pending():
    return load_pending_categories()


@router.post("/categories/pending/{code}/approve")
def approve_category(code: str):
    ok = approve_pending_category(code)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Pending category '{code}' not found")
    return {"approved": code}
```

- [ ] **Step 7: Create `app/api/routes/health.py`**

```python
from fastapi import APIRouter
from app.pipeline.db_fetcher import get_db_connection
from app.services.ollama_client import OllamaClient

router = APIRouter()


@router.get("/health")
async def health():
    ollama = OllamaClient()
    ollama_ok = await ollama.health_check()
    await ollama.aclose()

    db_ok = False
    try:
        with get_db_connection() as conn:
            conn.cursor().execute("SELECT 1")
        db_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if ollama_ok and db_ok else "degraded",
        "ollama": ollama_ok,
        "database": db_ok,
    }
```

- [ ] **Step 8: Run tests to verify pass**

```powershell
pytest tests/test_routes.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 9: Run all tests together**

```powershell
pytest tests/ -v
```

Expected: All tests PASS with no failures.

- [ ] **Step 10: Commit**

```bash
git add app/api/ tests/test_routes.py app/main.py
git commit -m "feat: FastAPI routes for analyze, status, categories, and health"
```

---

## Task 14: Eval Framework

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/metrics.py`
- Create: `eval/run_eval.py`
- Create: `eval/test_cases/structuring/passport_001.json`
- Create: `eval/test_cases/structuring/invitation_001.json`
- Create: `eval/test_cases/categorization/case_001.json`

- [ ] **Step 1: Create `eval/__init__.py`** (empty)

- [ ] **Step 2: Create `eval/metrics.py`**

```python
from typing import Any


def json_valid_rate(results: list[dict | None]) -> float:
    """Fraction of results that are valid dicts (non-None)."""
    if not results:
        return 0.0
    return sum(1 for r in results if isinstance(r, dict)) / len(results)


def field_completeness(results: list[dict], expected_fields: list[str]) -> float:
    """Average fraction of expected fields that are populated (non-null) across results."""
    if not results or not expected_fields:
        return 0.0
    scores = []
    for r in results:
        fields = r.get("fields", {})
        filled = sum(1 for f in expected_fields if fields.get(f) is not None)
        scores.append(filled / len(expected_fields))
    return sum(scores) / len(scores)


def doc_type_accuracy(results: list[dict], expected_types: list[str]) -> float:
    """Fraction of results where document_type matches expected."""
    if not results:
        return 0.0
    correct = sum(
        1 for r, exp in zip(results, expected_types)
        if isinstance(r, dict) and r.get("document_type") == exp
    )
    return correct / len(results)


def category_precision(assigned: list[str], expected: list[str]) -> float:
    """Fraction of assigned categories that are in the expected set."""
    if not assigned:
        return 0.0
    return sum(1 for c in assigned if c in expected) / len(assigned)


def summarize(variant: str, results: list[dict], expected_types: list[str],
              expected_fields: list[str], latencies: list[float]) -> dict:
    return {
        "variant": variant,
        "json_valid_rate": round(json_valid_rate(results), 3),
        "field_completeness": round(field_completeness(results, expected_fields), 3),
        "doc_type_accuracy": round(doc_type_accuracy(results, expected_types), 3),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "n_cases": len(results),
    }
```

- [ ] **Step 3: Create `eval/run_eval.py`**

```python
"""
Usage:
  python -m eval.run_eval --prompt-variants v1_basic,v2_few_shot --test-set structuring
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

from app.prompts.registry import PromptRegistry
from app.services.ollama_client import OllamaClient
from eval.metrics import summarize

EVAL_DIR = Path(__file__).parent


def load_test_cases(test_set: str) -> list[dict]:
    case_dir = EVAL_DIR / "test_cases" / test_set
    cases = []
    for f in sorted(case_dir.glob("*.json")):
        cases.append(json.loads(f.read_text(encoding="utf-8")))
    return cases


async def eval_structuring_variant(variant: str, cases: list[dict]) -> dict:
    reg = PromptRegistry()
    ollama = OllamaClient()
    results, latencies, expected_types = [], [], []

    for case in cases:
        prompt, temp = reg.format("structuring", variant, ocr_text=case["input"])
        t0 = time.time()
        try:
            result = await ollama.generate(prompt, temperature=temp)
        except Exception:
            result = None
        latencies.append(time.time() - t0)
        results.append(result)
        expected_types.append(case.get("expected_type", "OTHER"))

    await ollama.aclose()
    expected_fields = cases[0].get("expected_fields", []) if cases else []
    return summarize(variant, results, expected_types, expected_fields, latencies)


async def main(variants: list[str], test_set: str, output: str) -> None:
    cases = load_test_cases(test_set)
    if not cases:
        print(f"No test cases found in eval/test_cases/{test_set}/")
        return

    print(f"Running eval on {len(cases)} cases for test set '{test_set}'")
    all_results = {}

    for variant in variants:
        print(f"  Evaluating variant: {variant}...")
        if test_set == "structuring":
            result = await eval_structuring_variant(variant, cases)
        else:
            print(f"  Skipping unsupported test set '{test_set}'")
            continue
        all_results[variant] = result
        print(f"    json_valid={result['json_valid_rate']:.0%} "
              f"field_completeness={result['field_completeness']:.0%} "
              f"doc_type={result['doc_type_accuracy']:.0%} "
              f"latency={result['avg_latency_sec']}s")

    if all_results:
        winner = max(all_results, key=lambda v: all_results[v]["doc_type_accuracy"])
        report = {"test_set": test_set, "variants": all_results, "winner": winner}
        out_path = Path(output) / f"eval_{test_set}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nWinner: {winner}")
        print(f"Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-variants", default="v1_basic,v2_few_shot")
    parser.add_argument("--test-set", default="structuring")
    parser.add_argument("--output", default="eval/results")
    args = parser.parse_args()
    variants = [v.strip() for v in args.prompt_variants.split(",")]
    asyncio.run(main(variants, args.test_set, args.output))
```

- [ ] **Step 4: Create `eval/test_cases/structuring/passport_001.json`**

Extract the OCR text from a real passport PDF using the OCR engine. Run this once manually:

```powershell
python -c "
from app.pipeline.ocr_engine import extract_text_from_file
from pathlib import Path
text, pages = extract_text_from_file(Path(r'C:\Users\strahinja\Desktop\Test\109537\1.pdf'))
print(text[:1000])
"
```

Copy the printed OCR text into the file:

```json
{
  "input": "<paste OCR text of passport from 109537/1.pdf here>",
  "expected_type": "PASSPORT",
  "expected_fields": ["first_name", "last_name", "expiry_date", "nationality", "document_number"]
}
```

- [ ] **Step 5: Create `eval/test_cases/structuring/invitation_001.json`**

```powershell
python -c "
from app.pipeline.ocr_engine import extract_text_from_file
from pathlib import Path
text, _ = extract_text_from_file(Path(r'C:\Users\strahinja\Desktop\Test\109537\2.pdf'))
print(text[:1000])
"
```

```json
{
  "input": "<paste OCR text of invitation letter from 109537/2.pdf here>",
  "expected_type": "INVITATION_LETTER",
  "expected_fields": ["inviting_org", "period_from", "period_to"]
}
```

- [ ] **Step 6: Create `eval/test_cases/categorization/case_001.json`**

```json
{
  "input": {
    "db_json": {
      "urgency_kategorija": "URGENCY_HIGH",
      "starosna_kategorija": "APPLICANT_MINOR",
      "pasos_status": "PASSPORT_OK",
      "duzina_boravka_dana": 4,
      "dana_do_dolaska": 5,
      "drzavljanstvo_en": "Chinese",
      "svrha_putovanja_en": "Cultural"
    },
    "documents_json": [
      {"document_type": "INVITATION_LETTER", "fields": {"inviting_org": "Muzički Atelje", "event": "violin competition"}}
    ]
  },
  "expected_categories": ["PURPOSE_CULTURAL", "FINANCE_SPONSOR"]
}
```

- [ ] **Step 7: Smoke test eval CLI (requires Ollama running with qwen2.5:7b pulled)**

```powershell
python -m eval.run_eval --prompt-variants v1_basic --test-set structuring --output eval/results
```

Expected: Runs through test cases and prints a summary. Results saved to `eval/results/eval_structuring.json`.

- [ ] **Step 8: Commit**

```bash
git add eval/ 
git commit -m "feat: eval framework for comparing prompt variants with metrics"
```

---

## Task 15: End-to-End Smoke Test

This task verifies the entire pipeline works against a real request.

**Prerequisites:**
- Ollama running: `ollama serve`
- Model pulled: `ollama pull qwen2.5:7b`
- SQL Server running with NoviVis database accessible

- [ ] **Step 1: Start the API server**

```powershell
uvicorn app.main:app --reload --port 8000
```

- [ ] **Step 2: Check health endpoint**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET | ConvertTo-Json
```

Expected:
```json
{ "status": "ok", "ollama": true, "database": true }
```

- [ ] **Step 3: Submit a real request**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/analyze/109537" -Method POST | ConvertTo-Json
```

Expected: `{ "job_id": "some-uuid-..." }`

- [ ] **Step 4: Poll status until DONE**

```powershell
$job_id = "<job_id from step 3>"
do {
    $result = Invoke-RestMethod -Uri "http://localhost:8000/status/$job_id" -Method GET
    Write-Host "Status: $($result.status)"
    Start-Sleep -Seconds 30
} while ($result.status -eq "RUNNING" -or $result.status -eq "PENDING")
$result | ConvertTo-Json -Depth 10
```

Expected after ~5-7 minutes:
```json
{
  "status": "DONE",
  "categories": [ ... ],
  "extracted_documents": [ ... ],
  "processing_time_seconds": 350
}
```

- [ ] **Step 5: Verify categories list endpoint**

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/categories" -Method GET | ConvertTo-Json
```

Expected: List of all predefined categories from `data/categories.json`.

- [ ] **Step 6: Final commit**

```bash
git add .
git commit -m "feat: complete VISAOcr pipeline - OCR + LLM structuring + categorization"
```

---

## Quick Reference

**Start server:**
```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

**Run all tests:**
```powershell
pytest tests/ -v
```

**Run eval:**
```powershell
python -m eval.run_eval --prompt-variants v1_basic,v2_few_shot --test-set structuring
```

**Switch active prompt variant** (edit `.env`):
```ini
ACTIVE_STRUCTURING_PROMPT=v2_few_shot
ACTIVE_CATEGORIZATION_PROMPT=v2_few_shot
```

**Approve a pending LLM category:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/categories/pending/APPLICANT_JOURNALIST/approve" -Method POST
```
