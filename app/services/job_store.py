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


async def fail_stale_jobs() -> None:
    """Mark jobs left PENDING/RUNNING by a previous (crashed or restarted) process as
    FAILED. Background tasks live in-process, so they cannot survive a restart."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE jobs SET status = ?, error = ?, finished_at = ? "
            "WHERE status IN (?, ?)",
            (
                JobStatus.FAILED.value,
                "Interrupted by server restart",
                datetime.now(timezone.utc).isoformat(),
                JobStatus.PENDING.value,
                JobStatus.RUNNING.value,
            ),
        )
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
