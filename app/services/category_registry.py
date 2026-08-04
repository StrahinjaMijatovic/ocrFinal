import json
import threading
from pathlib import Path

CATEGORIES_PATH = Path("data/categories.json")
PENDING_PATH = Path("data/pending_categories.json")

# Guards the read-modify-write cycles below against concurrent jobs/requests
# clobbering each other's writes to the shared JSON files.
_write_lock = threading.Lock()


def load_categories() -> list[dict]:
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)["categories"]


def load_pending_categories() -> list[dict]:
    with open(PENDING_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_pending_category(entry: dict) -> None:
    with _write_lock:
        pending = load_pending_categories()
        if not any(p["code"] == entry["code"] for p in pending):
            pending.append(entry)
            with open(PENDING_PATH, "w", encoding="utf-8") as f:
                json.dump(pending, f, indent=2, ensure_ascii=False)


def approve_pending_category(code: str) -> bool:
    with _write_lock:
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
