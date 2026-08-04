from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.services.job_store import init_db, fail_stale_jobs
from app.api.routes import analyze, status, categories, health

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await fail_stale_jobs()
    yield


app = FastAPI(title="VISAOcr", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


app.include_router(analyze.router)
app.include_router(status.router)
app.include_router(categories.router)
app.include_router(health.router)
