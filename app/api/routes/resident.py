from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_sync_user
from app.data.resident_context_cache import store_residents_in_sqlite_cache

router = APIRouter()

@router.post("/resident/sync")
def sync_resident(
    data: dict,
    user=Depends(get_sync_user)
):
    """
    Immediate cache invalidation endpoint.
    Engineering calls this whenever a resident's baseline vitals are updated.
    """
    rid = data.get("resident_id") or data.get("id")
    if not rid:
        raise HTTPException(status_code=400, detail="resident_id or id required")

    # Store directly in SQLite, instantly available to the pipeline
    store_residents_in_sqlite_cache([data])

    return {
        "status": "updated",
        "resident_id": rid
    }
