from fastapi import APIRouter, HTTPException
from app.services.category_registry import (
    approve_pending_category,
    load_categories,
    load_pending_categories,
)

router = APIRouter()


@router.get("/categories")
def list_categories():
    return load_categories()


@router.get("/categories/pending")
def list_pending():
    return load_pending_categories()


@router.post("/categories/pending/{code}/approve")
def approve_category(code: str):
    ok = approve_pending_category(code)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Pending category '{code}' not found")
    return {"approved": code}
