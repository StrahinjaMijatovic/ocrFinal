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

    # Passport/ID-card documents are skipped upstream (their data comes from the DB),
    # so passport-presence/MRZ rules no longer apply. Only generic OCR-quality signals
    # remain here.
    low_conf = [d for d in docs if d.confidence < 0.5]
    if low_conf:
        names = ", ".join(d.file for d in low_conf)
        cats.append(Category(code="DOC_LOW_CONFIDENCE", source="ocr",
                             reason=f"Low OCR confidence on: {names}"))

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
