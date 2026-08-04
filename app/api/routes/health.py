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
