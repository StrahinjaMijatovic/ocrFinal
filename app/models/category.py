from typing import Literal
from pydantic import BaseModel


class Category(BaseModel):
    code: str
    source: Literal["db", "ocr", "llm", "llm_new"]
    reason: str
