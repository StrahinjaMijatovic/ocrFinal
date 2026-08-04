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
