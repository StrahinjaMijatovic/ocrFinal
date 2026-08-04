# VISAOcr — System Design

**Date:** 2026-06-09  
**Status:** Approved  
**Stack:** Python, FastAPI, PaddleOCR, Qwen2.5:7B via Ollama, pyodbc, SQLite

---

## 1. Overview

A three-stage offline pipeline that processes visa applications by combining structured data from SQL Server with OCR-extracted data from scanned documents, producing a categorized JSON result used by consular officers.

**Input:** `RequestId` (integer)  
**Output:** JSON with categories (code + source + reason) and extracted document data  
**Constraint:** Fully offline, CPU-only (Lenovo ThinkPad, i7 11th Gen, 32GB RAM, no GPU)

---

## 2. Architecture

**Pattern:** Monolithic FastAPI with SQLite-backed async background tasks.

```
POST /analyze/{request_id}
        │
        ├── Create job (SQLite) → return job_id immediately
        └── Background Task:
              [1] DB Fetch      → pyodbc → localhost\NoviVis SQL query
              [2] Doc Discovery → scan C:\...\Test\{RequestId}\*.pdf/*.jpg
                                   (skip photo-only files: *.jpg without text content)
              [3] OCR           → PaddleOCR per file → raw_text
              [4] Structuring   → Qwen2.5:7B per document → typed JSON
              [5] Rule Categories → Python rules from DB fields
              [6] LLM Categories  → Qwen2.5:7B → categories + new discovery
              [7] Store result    → SQLite job.result = final JSON

GET /status/{job_id} → returns status + result when DONE
```

---

## 3. Project Structure

```
visa-ocr/
├── app/
│   ├── main.py                       # FastAPI app, lifespan
│   ├── config.py                     # Settings: paths, DB conn, Ollama URL, active prompts
│   ├── api/
│   │   └── routes/
│   │       ├── analyze.py            # POST /analyze/{request_id}
│   │       └── status.py             # GET /status/{job_id}
│   ├── pipeline/
│   │   ├── orchestrator.py           # Runs full pipeline for one job
│   │   ├── db_fetcher.py             # pyodbc → SQL Server → db_json
│   │   ├── ocr_engine.py             # PaddleOCR wrapper, PDF→image→text
│   │   ├── document_structurer.py    # LLM: raw_text → structured document JSON
│   │   └── categorizer.py            # Rule-based + LLM → final categories
│   ├── models/
│   │   ├── job.py                    # Job Pydantic model (id, status, result)
│   │   ├── document.py               # ExtractedDocument model
│   │   └── category.py               # Category model (code, source, reason)
│   ├── services/
│   │   ├── job_store.py              # SQLite CRUD for jobs
│   │   ├── ollama_client.py          # HTTP client for Ollama, retry logic
│   │   └── category_registry.py     # Load/save categories.json + pending
│   └── prompts/
│       ├── registry.py               # PromptRegistry: loads YAML, formats templates
│       ├── structuring/
│       │   ├── v1_basic.yaml
│       │   └── v2_few_shot.yaml
│       └── categorization/
│           ├── v1_basic.yaml
│           └── v2_few_shot.yaml
├── eval/
│   ├── run_eval.py                   # CLI: compare prompt variants
│   ├── metrics.py                    # json_valid_rate, field_completeness, etc.
│   └── test_cases/
│       ├── structuring/              # {input: OCR text, expected: JSON}
│       └── categorization/           # {input: merged data, expected: categories}
├── data/
│   ├── categories.json               # Active category pool
│   └── pending_categories.json       # LLM-discovered, awaiting manual review
├── jobs.db                           # SQLite job persistence
├── requirements.txt
└── .env                              # DOCS_BASE_PATH, DB_CONN_STR, OLLAMA_URL, active prompt versions
```

---

## 4. Models

### OCR: PaddleOCR v2 (CPU)
- Multilingual: Serbian (Latin + Cyrillic), Romanian, English, Chinese
- ~2-5 seconds per page on i7 CPU
- `mrz` library for MRZ zone parsing on passports (deterministic, 100% accurate)

### LLM: Qwen2.5:7B-Instruct via Ollama (Q4_K_M)
- ~4.5GB RAM usage
- ~20-40 seconds per LLM call on CPU
- Same model for both structuring and categorization (different prompts)
- `ollama pull qwen2.5:7b` for installation

### Processing time estimate per request (8 documents)
| Step | Time |
|------|------|
| DB fetch | <1s |
| OCR (8 docs) | ~30s |
| Structuring (8 LLM calls) | ~4-6 min |
| Categorization (1 LLM call) | ~30-60s |
| **Total** | **~5-7 min** |

---

## 5. Data Models

### Job (SQLite)
```json
{
  "job_id": "uuid",
  "request_id": 438281,
  "status": "PENDING | RUNNING | DONE | FAILED",
  "created_at": "ISO datetime",
  "finished_at": "ISO datetime",
  "error": null,
  "result": {}
}
```

### ExtractedDocument
```json
{
  "file": "1.pdf",
  "document_type": "PASSPORT",
  "confidence": 0.92,
  "language": "English",
  "fields": {
    "first_name": "Zhang",
    "last_name": "Na Na",
    "birth_date": "2001-03-01",
    "expiry_date": "2027-05-13",
    "nationality": "CHN",
    "document_number": "E75903512",
    "mrz_valid": true
  }
}
```

### API Response (GET /status/{job_id})
```json
{
  "job_id": "uuid",
  "request_id": 438281,
  "status": "DONE",
  "processing_time_seconds": 347,
  "categories": [
    { "code": "URGENCY_HIGH",     "source": "db",  "reason": "Arrival in 5 days" },
    { "code": "DOC_MRZ_VALID",    "source": "ocr", "reason": "MRZ parsed successfully" },
    { "code": "STAY_4_7_DAYS",    "source": "db",  "reason": "Stay duration: 6 days" },
    { "code": "PURPOSE_CULTURAL", "source": "llm", "reason": "Invitation to violin competition" }
  ],
  "extracted_documents": [ ... ]
}
```

Category `source` values:
- `db` — derived from SQL Server data via Python rules
- `ocr` — derived from OCR processing (e.g., MRZ validation, document completeness check)
- `llm` — assigned by Qwen2.5:7B from predefined list
- `llm_new` — new category proposed by LLM, saved to `pending_categories.json`

---

## 6. LLM Prompts

### Prompt Architecture
Prompts stored as YAML files in `app/prompts/structuring/` and `app/prompts/categorization/`. Active variant controlled by `config.py`:

```python
ACTIVE_STRUCTURING_PROMPT = "v2_few_shot"
ACTIVE_CATEGORIZATION_PROMPT = "v1_basic"
```

### Prompt YAML Structure
```yaml
name: "structuring_v2_few_shot"
version: "2.0"
temperature: 0.1
system: |
  You are a document analysis expert for visa processing...
few_shots:
  - input: |
      <OCR text of passport>
    output: |
      { "document_type": "PASSPORT", ... }
  - input: |
      <OCR text of invitation letter>
    output: |
      { "document_type": "INVITATION_LETTER", ... }
template: |
  {system}
  EXAMPLES:
  {few_shots_formatted}
  NOW ANALYZE:
  {ocr_text}
```

### Document Structuring Prompt (core)
- Detects document type from: `PASSPORT, INVITATION_LETTER, VISA_APPLICATION, RESIDENCE_PERMIT, INSURANCE, ACCOMMODATION_CONFIRMATION, EMAIL_CORRESPONDENCE, DECLARATION, BANK_STATEMENT, OTHER`
- Returns `confidence` float (0.0-1.0)
- All dates in ISO format `YYYY-MM-DD`
- Ollama `format: "json"` enforced
- Retry up to 2x on JSON parse failure with stronger enforcement message
- `temperature: 0.1`

### Categorization Prompt (core)
- Receives: predefined categories list + rule-based categories already assigned + DB JSON + all extracted documents JSON
- Assigns predefined categories that apply
- Proposes new categories only for genuinely novel situations (`DOMAIN_KEYWORD` format)
- Does NOT repeat already-assigned rule-based categories
- Returns: `{ "categories": [...], "new_categories": [...] }`
- `temperature: 0.1`

---

## 7. Category System

### Predefined Categories

**Svrha putovanja**
`PURPOSE_HUMANITARIAN_DEATH`, `PURPOSE_DIPLOMATIC`, `PURPOSE_HUMANITARIAN_MED`, `PURPOSE_EXPO2027`, `PURPOSE_BUSINESS_CRITICAL`, `PURPOSE_WORK`, `PURPOSE_FAMILY_REUNION`, `PURPOSE_STUDY`, `PURPOSE_TOURISM`, `PURPOSE_TRANSIT`, `PURPOSE_SPORTS`, `PURPOSE_CULTURAL`, `PURPOSE_RELIGIOUS`, `PURPOSE_CONFERENCE`

**Podnosilac**
`APPLICANT_VIP`, `APPLICANT_MINOR`, `APPLICANT_DISABILITY`, `APPLICANT_ELDERLY`, `APPLICANT_RETURNING`, `APPLICANT_REFUSED_BEFORE`, `APPLICANT_UNACCOMPANIED_MINOR`, `APPLICANT_GROUP`

**Pasoš**
`PASSPORT_DIPLOMATIC`, `PASSPORT_OFFICIAL`, `PASSPORT_EXPIRED`, `RISK_PASSPORT_EXPIRING`

**Dokumenta**
`DOC_COMPLETE`, `DOC_MRZ_VALID`, `DOC_MISSING_MINOR`, `DOC_LOW_CONFIDENCE`, `DOC_MISSING_CRITICAL`, `DOC_TRANSLATED`, `DOC_MULTIPLE_PASSPORTS`

**Dužina boravka**
`STAY_1_3_DAYS`, `STAY_4_7_DAYS`, `STAY_8_14_DAYS`, `STAY_15_30_DAYS`, `STAY_OVER_30`

**Finansije**
`FINANCE_FULL_COVERAGE`, `FINANCE_SELF_SUFFICIENT`, `FINANCE_INSUFFICIENT`, `FINANCE_SPONSOR`, `ADMIN_PENDING_INFO`

**Administrativno**
`ADMIN_SLA_BREACH`, `ADMIN_SLA_WARNING`, `ADMIN_RESUBMITTED`

**Urgentnost**
`URGENCY_CRITICAL`, `URGENCY_HIGH`, `URGENCY_MEDIUM`, `URGENCY_LOW`

**Rizik**
`OVERSTAY_RISK`, `RISK_SUSPICIOUS_PATTERN`, `RISK_COUNTRY_MEDIUM`, `RISK_COUNTRY_HIGH`, `RISK_INCOMPLETE_DOCS`, `RISK_BLACKLIST_MATCH`, `RISK_OVERSTAY_HISTORY`, `RISK_FORGED_DOC_SUSPECTED`

### Rule-based Assignment (from DB JSON, Python)
| Category | Rule |
|----------|------|
| `URGENCY_*` | From `urgency_kategorija` SQL field |
| `APPLICANT_MINOR/ELDERLY` | From `starosna_kategorija` SQL field |
| `PASSPORT_EXPIRED` / `RISK_PASSPORT_EXPIRING` | From `pasos_status` SQL field |
| `STAY_*` | `duzina_boravka_dana` thresholds |
| `ADMIN_RESUBMITTED` | `broj_prethodnih_zahteva > 0` |
| `ADMIN_SLA_BREACH` | `datum_podnosenja` + configured SLA days |

### New Category Handling
When LLM proposes a new category:
1. Added to `data/pending_categories.json` with code, reason, description, timestamp
2. Returned in response with `source: "llm_new"`
3. NOT automatically added to active pool — requires manual review
4. After manual approval: move entry from `pending_categories.json` to `categories.json`

---

## 8. Evaluation Framework

```bash
python -m eval.run_eval \
  --prompt-variants v1_basic,v2_few_shot \
  --test-set structuring \
  --output eval/results/
```

### Metrics
| Metric | Description |
|--------|-------------|
| `json_valid_rate` | % of calls returning valid JSON |
| `field_completeness` | % of expected fields populated |
| `doc_type_accuracy` | % correct document type detections |
| `category_precision` | % of assigned categories that are correct |
| `avg_latency_sec` | Average LLM call duration |

Test cases stored in `eval/test_cases/structuring/` and `eval/test_cases/categorization/` as JSON files with `input` and `expected` fields, derived from real documents in `C:\Users\strahinja\Desktop\Test\`.

---

## 9. Configuration (.env)

```env
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

---

## 10. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze/{request_id}` | Start pipeline, returns `{ "job_id": "uuid" }` |
| `GET` | `/status/{job_id}` | Returns job status + result when DONE |
| `GET` | `/health` | Checks Ollama + DB connectivity |
| `GET` | `/categories` | Lists all active categories |
| `GET` | `/categories/pending` | Lists pending LLM-discovered categories |
| `POST` | `/categories/pending/{code}/approve` | Moves category from pending to active |

---

## 11. Dependencies (requirements.txt)

```
fastapi
uvicorn
paddlepaddle
paddleocr
pymupdf          # PDF → image rendering
mrz              # MRZ zone parsing for passports
pyodbc           # SQL Server connection
httpx            # Async HTTP client for Ollama
pydantic-settings
python-dotenv
pyyaml           # Prompt YAML loading
aiosqlite        # Async SQLite for job store
```
