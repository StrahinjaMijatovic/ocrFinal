from fastapi import APIRouter, HTTPException
from app.services.job_store import get_job

router = APIRouter()


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.model_dump()
