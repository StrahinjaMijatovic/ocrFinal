import pytest
import app.services.job_store as store
from app.models.job import JobStatus


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_jobs.db")


@pytest.mark.asyncio
async def test_create_and_get_job():
    await store.init_db()
    job = await store.create_job(request_id=12345)
    assert job.request_id == 12345
    assert job.status == JobStatus.PENDING

    fetched = await store.get_job(job.job_id)
    assert fetched is not None
    assert fetched.job_id == job.job_id


@pytest.mark.asyncio
async def test_get_nonexistent_job():
    await store.init_db()
    result = await store.get_job("nonexistent-uuid")
    assert result is None


@pytest.mark.asyncio
async def test_update_job_status():
    await store.init_db()
    job = await store.create_job(request_id=999)
    await store.update_job_status(job.job_id, JobStatus.RUNNING)
    updated = await store.get_job(job.job_id)
    assert updated.status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_set_job_result():
    await store.init_db()
    job = await store.create_job(request_id=555)
    result = {"categories": [{"code": "URGENCY_HIGH", "source": "db", "reason": "test"}]}
    await store.set_job_result(job.job_id, result)
    done = await store.get_job(job.job_id)
    assert done.status == JobStatus.DONE
    assert done.result == result
    assert done.finished_at is not None
