"""Persist completed job results to a single JSON file.

Every finished /analyze run is appended to one file (settings.results_file_path) as
an entry holding the request id, the extracted documents and the categories. Writes
are serialized with an asyncio lock so concurrent background jobs don't corrupt the
file, and the file always stays a valid JSON array."""

import asyncio
import json
from pathlib import Path

_lock = asyncio.Lock()


def _load_existing(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else [data]


async def append_result(result: dict, file_path: str) -> None:
    """Append one job result to the shared results file. A blank file_path disables
    saving. The actual read/modify/write runs off the event loop, guarded by a lock
    so overlapping jobs can't interleave their writes."""
    if not file_path:
        return

    path = Path(file_path)

    async with _lock:
        await asyncio.to_thread(_write_append, path, result)


def _write_append(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_existing(path)
    data.append(result)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
