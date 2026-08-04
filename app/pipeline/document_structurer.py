from pathlib import Path

from app.config import settings
from app.models.document import ExtractedDocument
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

    # The LLM may emit keys with an explicit null value, so `.get(key, default)`
    # isn't enough — coalesce None to the fallback with `or`.
    return ExtractedDocument(
        file=file_path.name,
        document_type=data.get("document_type") or "OTHER",
        confidence=float(data.get("confidence") or 0.0),
        language=data.get("language") or "unknown",
        fields=data.get("fields") or {},
    )
