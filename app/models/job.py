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
