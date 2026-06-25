import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
workspace_root = Path(__file__).parent.parent
env_path = workspace_root / "env1.env"
load_dotenv(env_path)

# Import routes
from app.api.routes import predict, resident

app = FastAPI(
    title="CareMP Fall Risk Unified ML API",
    description="Unified service managing baseline syncs (Engineering webhook) and real-time BLE bandlog ingestion.",
    version="2.0.0"
)

# Register routers
app.include_router(predict.router, tags=["Predictions"])
app.include_router(resident.router, tags=["Residents"])

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.data.resident_context_cache import (
    fetch_residents_from_engineering_api,
    fetch_residents_from_sqlite_cache
)

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()

async def fetch_and_store():
    """
    Syncs resident setting parameters from Engineering API. Falls back to local SQLite cache if unreachable.
    """
    import asyncio
    logger.info("[SYNC] Starting resident sync from Engineering...")
    # This function is synchronous (uses requests), so run in thread to not block FastAPI
    df = await asyncio.to_thread(fetch_residents_from_engineering_api)
    if df is not None and not df.empty:
        logger.info(f"[SYNC] Success — {len(df)} residents synced")
    else:
        cached_df = await asyncio.to_thread(fetch_residents_from_sqlite_cache)
        cached_count = len(cached_df) if cached_df is not None else 0
        if cached_count > 0:
            logger.warning(
                f"[SYNC] Engineering API unreachable. "
                f"Falling back to {cached_count} residents from SQLite cache. "
                f"Data may be stale — will retry in 12 hours."
            )
        else:
            logger.error(
                "[SYNC] Engineering API unreachable AND SQLite cache is empty. "
                "Resident context unavailable. Predictions will have no baseline data."
            )

@app.on_event("startup")
async def start_schedulers():
    # Run sync immediately on startup
    await fetch_and_store()
    
    # Schedule the 12-hour sync
    scheduler.add_job(fetch_and_store, 'interval', hours=12, id='sync_residents')
    
    # Schedule the 15-minute Smart Batch ML Pipeline
    import asyncio
    scheduler.add_job(
        lambda: asyncio.create_task(asyncio.to_thread(predict.run_batch_ml_predictions)),
        'interval',
        minutes=15,
        id='batch_ml_predictions'
    )
    
    scheduler.start()
    logger.info("[SCHEDULER] Started background jobs (12h Resident Sync, 15m ML Batch).")

@app.on_event("shutdown")
async def stop_schedulers():
    scheduler.shutdown()


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "redis_connected": predict.redis_buffer is not None
    }

if __name__ == "__main__":
    print("Starting CareMP Fall Risk Unified ML API on port 8000...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
