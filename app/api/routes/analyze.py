from fastapi import APIRouter, BackgroundTasks
from app.pipeline.orchestrator import run_pipeline
from app.services.job_store import create_job

router = APIRouter()


@router.post("/analyze/{request_id}")
async def analyze(request_id: int, background_tasks: BackgroundTasks):
    job = await create_job(request_id)
    background_tasks.add_task(run_pipeline, job.job_id, request_id)
    return {"job_id": job.job_id}
