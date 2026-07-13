from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.data.redis_ingestion_buffer import RedisBLEBuffer
import pandas as pd
from pathlib import Path
import joblib
import os
import asyncio
from dotenv import load_dotenv

from app.core.predict_fall_risk_v1 import (
    build_prediction_features,
    predict_rows,
    write_prediction_history_sqlite,
    prediction_model_version,
    DEFAULT_TRAINING_START_AT,
    DEFAULT_TRAINING_END_AT,
    get_resident_baselines
)
from app.core.fall_risk_layered_pipeline_v2 import build_pipeline_record, public_pipeline_record, build_shap_explanations
from app.api.dependencies import get_sync_user

# ---------------------------------------------------------------------------
# ARTIFACT_PATH anchored to this file's location
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
# Navigate up from app/api/routes to the root V2_Fall_risk folder
_ROOT_DIR = _MODULE_DIR.parent.parent.parent
_DEFAULT_ARTIFACT = _ROOT_DIR / "models_v2_grouped_real_holdout" / "fall_risk_xgboost_v23_final.pkl"
ARTIFACT_PATH = _DEFAULT_ARTIFACT

# ---------------------------------------------------------------------------
# Redis initialised from env vars
# ---------------------------------------------------------------------------
load_dotenv(_MODULE_DIR / "env1.env")

_redis_host = os.getenv("REDIS_HOST", "localhost")
_redis_port = int(os.getenv("REDIS_PORT", "6379"))
_redis_db   = int(os.getenv("REDIS_DB",   "0"))
_redis_password = os.getenv("REDIS_PASSWORD", None)

try:
    redis_buffer = RedisBLEBuffer(
        host=_redis_host,
        port=_redis_port,
        db=_redis_db,
        password=_redis_password,
    )
except Exception as e:
    print(f"Warning: Redis connection failed on startup: {e}")
    redis_buffer = None


COLUMN_RENAME = {
    "bandlogId":           "id",
    "bandId":              "band_id",
    "residentId":          "resident_id",
    "heartRate":           "heart_rate",
    "systolicBP":          "systolic_bp",
    "diastolicBP":         "diastolic_bp",
    "oxygenSaturation":    "oxygen_saturation",
    "oxygenSaturationValid": "oxygen_saturation_valid",
    "bodyTemperature":     "body_temperature",
    "skinTemperature":     "skin_temperature",
    "temperatureUnit":     "temperature_unit",
    "stressValid":         "stress_valid",
    "deepSleepTime":       "deep_sleep_time",
    "lightSleepTime":      "light_sleep_time",
    "fatigueLevel":        "fatigue_level",
    "wearingIndicator":    "wearing_indicator",
    "stepCount":           "step_count",
    "generatedAt":         "generated_at",
}

# ---------------------------------------------------------------------------
# Load ML Artifact
# ---------------------------------------------------------------------------
try:
    if ARTIFACT_PATH.exists():
        model_dir = ARTIFACT_PATH.parent
        suffix = ARTIFACT_PATH.name.replace("fall_risk_xgboost_", "")
        ml_artifact = {
            "model":           joblib.load(ARTIFACT_PATH),
            "label_encoder":   joblib.load(model_dir / f"risk_label_encoder_{suffix}"),
            "feature_columns": joblib.load(model_dir / f"model_features_{suffix}"),
            "model_version":   suffix.replace(".pkl", ""),
        }
        print(f"Loaded ML artifact {ml_artifact['model_version']}")
    else:
        print(f"Warning: ML artifact not found at {ARTIFACT_PATH}")
        ml_artifact = None
except Exception as e:
    print(f"Failed to load ML artifact: {e}")
    ml_artifact = None

router = APIRouter()


class BandLog(BaseModel):
    bandlogId:                  int
    bandId:                     Optional[int]   = None
    residentId:                 Optional[int]   = None
    heartRate:                  Optional[float] = None
    hrv:                        Optional[float] = None
    systolicBP:                 Optional[float] = None
    diastolicBP:                Optional[float] = None
    oxygenSaturation:           Optional[float] = None
    oxygenSaturationValid:      Optional[bool]  = True
    bodyTemperature:            Optional[float] = None
    skinTemperature:            Optional[float] = None
    temperatureUnit:            Optional[int]   = 2
    rssi:                       Optional[int]   = 0
    stress:                     Optional[float] = None
    stressValid:                Optional[bool]  = True
    stepCount:                  Optional[int]   = None
    calories:                   Optional[float] = None
    deepSleepTime:              Optional[int]   = None
    lightSleepTime:             Optional[int]   = None
    remSleepTime:               Optional[int]   = None
    extraSleepComputationState: Optional[int]   = None
    sleepConfidence:            Optional[int]   = None
    fatigueLevel:               Optional[int]   = None
    breathing:                  Optional[int]   = None
    wearingIndicator:           Optional[bool]  = True
    generatedAt:                datetime


def _redis_packets_to_df(resident_id: int) -> pd.DataFrame | None:
    """Read the full Redis sliding-window queue for one resident."""
    import json
    key = redis_buffer._get_key(resident_id)
    raw_packets = redis_buffer.r.zrange(key, 0, -1)
    if not raw_packets:
        return None
    packets = [json.loads(p) for p in raw_packets]
    df = pd.DataFrame(packets).rename(columns=COLUMN_RENAME)
    df["generated_at"] = pd.to_datetime(df["generated_at"], format="ISO8601")
    return df.sort_values("generated_at").drop_duplicates(subset=["resident_id", "generated_at"], keep="last")


def _trim_redis_queue(resident_id: int, max_hours: int = 12) -> None:
    """Trim Redis queue to the last `max_hours` so it never grows unboundedly."""
    cutoff_unix = (
        pd.Timestamp.utcnow() - pd.Timedelta(hours=max_hours)
    ).timestamp()
    key = redis_buffer._get_key(resident_id)
    redis_buffer.r.zremrangebyscore(key, "-inf", cutoff_unix)


# ---------------------------------------------------------------------------
# Core ML pipeline — DECOUPLED FROM POSTGRESQL
# ---------------------------------------------------------------------------
def process_ready_packets_for_resident(resident_id: int):
    """
    Fast-track background worker.
    Uses REDIS ONLY for historical telemetry (no Postgres connection).
    Uses SQLITE ONLY for baseline context (no Postgres connection).
    """
    if not redis_buffer:
        print("[ML PIPELINE] Skipping: Redis not connected.")
        return
    if not ml_artifact:
        print("[ML PIPELINE] Skipping: ML artifact not loaded.")
        return

    # ready_df now acts as our complete sliding window (up to 12 hours of history in Redis)
    ready_df = _redis_packets_to_df(resident_id)
    if ready_df is None:
        print(f"[ML PIPELINE] No packets in Redis for Resident {resident_id}.")
        return

    print(f"[ML PIPELINE] Resident {resident_id}: Processing {len(ready_df)} packets purely from Redis history.")

    try:
        # Fetch baselines purely from local SQLite cache (populated by main.py background task)
        # We explicitly pass engine=None to guarantee no Postgres connection attempt is made.
        baseline_df = get_resident_baselines(
            engine=None,
            skip_engineering_api=True,
            use_sqlite_cache=True,
            use_database_fallback=False, 
        )

        features_df = build_prediction_features(
            ready_df, baseline_df, hours=12, use_latest_available=True
        )
        if features_df.empty:
            print(f"[ML PIPELINE] No features generated for Resident {resident_id}.")
            return

        # Predict all rows to build history, but we only output the absolute latest packet to the user
        predictions = predict_rows(features_df, ml_artifact, mode="history")

        # Simply take the most recent prediction generated from the sliding window
        new_predictions = predictions.tail(1).copy()

        output_results = []
        if not new_predictions.empty:
            print(
                f"\n[ML PIPELINE OUTPUT] Generated prediction for latest packet "
                f"(Resident {resident_id}):"
            )
            print(
                new_predictions[
                    ["generated_at", "predicted_risk_label", "prediction_confidence"]
                ].to_string(index=False)
            )

            raw_index = ready_df.set_index(["resident_id", "generated_at"]).sort_index()

            features_for_shap = features_df.loc[
                features_df.index.isin(new_predictions.index)
            ].copy()
            shap_explanations, _ = build_shap_explanations(
                feature_rows=features_for_shap,
                prediction_rows=new_predictions,
                artifact=ml_artifact,
                top_n=3,
            )

            for idx, feature_row in features_for_shap.iterrows():
                key = (feature_row["resident_id"], feature_row["generated_at"])
                if key not in raw_index.index:
                    resident_raw = ready_df[
                        ready_df["resident_id"] == feature_row["resident_id"]
                    ]
                    if resident_raw.empty:
                        continue
                    raw_row = resident_raw.iloc[-1]
                else:
                    raw_row = raw_index.loc[key]
                if isinstance(raw_row, pd.DataFrame):
                    raw_row = raw_row.iloc[-1]

                prediction_row = new_predictions.loc[idx]
                record = build_pipeline_record(
                    feature_row=feature_row,
                    raw_row=raw_row,
                    prediction_row=prediction_row,
                    history_db="fall_risk_prediction_history.db",
                    shap_payload=shap_explanations.get(key),
                    carenotes_df=None,
                )
                output_results.append(public_pipeline_record(record))

            import json
            print("\n[ML PIPELINE JSON PAYLOAD]")
            print(json.dumps(output_results, indent=2, default=str))

            json_file = str(_ROOT_DIR / "last_pipeline_output.json")
            with open(json_file, "w") as f:
                json.dump(output_results, f, indent=2, default=str)

            sqlite_path = str(_ROOT_DIR / "fall_risk_prediction_history.db")
            write_prediction_history_sqlite(
                new_predictions,
                prediction_model_version(ml_artifact),
                DEFAULT_TRAINING_START_AT,
                DEFAULT_TRAINING_END_AT,
                sqlite_path,
            )
        else:
            print(f"[ML PIPELINE] No new predictions generated for Resident {resident_id}.")

        _trim_redis_queue(resident_id, max_hours=12)

        return output_results

    except Exception as e:
        print(f"[ML PIPELINE ERROR] Resident {resident_id}: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"


fast_track_semaphore = asyncio.Semaphore(4)

async def run_fast_track_ml(resident_id: int):
    async with fast_track_semaphore:
        return await asyncio.to_thread(process_ready_packets_for_resident, resident_id)


def run_batch_ml_predictions():
    """
    Scheduled task every 15 minutes.
    Uses REDIS ONLY for historical telemetry (no Postgres connection).
    """
    if not redis_buffer or not ml_artifact:
        print("[BATCH ML] Skipped: Redis or ML artifact not ready.")
        return

    print("[BATCH ML] Starting scheduled 15-minute sweep purely from Redis...")
    keys = redis_buffer.r.keys(f"{redis_buffer.prefix}*")
    if not keys:
        print("[BATCH ML] No active residents found in Redis.")
        return

    all_combined_df_list = []

    # Baselines fetched locally without Postgres
    baseline_df = get_resident_baselines(
        engine=None,
        skip_engineering_api=True,
        use_sqlite_cache=True,
        use_database_fallback=False,
    )

    for key in keys:
        resident_id_str = key.decode() if isinstance(key, bytes) else key
        resident_id_str = resident_id_str.replace(redis_buffer.prefix, "")
        try:
            resident_id = int(resident_id_str)
        except ValueError:
            continue
            
        ready_df = _redis_packets_to_df(resident_id)
        if ready_df is None or ready_df.empty:
            continue

        features_df = build_prediction_features(
            ready_df, baseline_df, hours=12, use_latest_available=True
        )

        if not features_df.empty:
            latest_feature = features_df.sort_values("generated_at").tail(1)
            all_combined_df_list.append(latest_feature)

        _trim_redis_queue(resident_id, max_hours=12)

    if not all_combined_df_list:
        print("[BATCH ML] No features built for active residents.")
        return

    master_features_df = pd.concat(all_combined_df_list, ignore_index=True)

    try:
        predictions = predict_rows(master_features_df, ml_artifact, mode="latest")
        if not predictions.empty:
            print(
                f"\n[BATCH ML OUTPUT] {len(predictions)} resident predictions:"
            )
            print(
                predictions[
                    ["resident_id", "predicted_risk_label", "prediction_confidence"]
                ].to_string(index=False)
            )

            sqlite_path = str(_ROOT_DIR / "fall_risk_prediction_history.db")
            write_prediction_history_sqlite(
                predictions,
                prediction_model_version(ml_artifact),
                DEFAULT_TRAINING_START_AT,
                DEFAULT_TRAINING_END_AT,
                sqlite_path,
            )
            print(f"[BATCH ML] Saved {len(predictions)} risk scores to SQLite.")

            del master_features_df
            import gc
            gc.collect()

    except Exception as e:
        print(f"[BATCH ML ERROR] {e}")
        import traceback
        traceback.print_exc()


@router.post("/predict")
async def ingest_bandlog(
    payload: BandLog,
    _user=Depends(get_sync_user), 
):
    if not redis_buffer:
        raise HTTPException(status_code=500, detail="Redis buffer is not connected.")

    unix_time = payload.generatedAt.timestamp()
    
    packet_dict = payload.model_dump(mode="json")
    redis_buffer.add_packet(
        resident_id=payload.residentId,
        generated_at_unix=unix_time,
        packet_data=packet_dict,
    )

    results = []
    # Use a 60-second lock so we only trigger Fast-Track once per minute per resident
    lock_key = f"{redis_buffer.prefix}{payload.residentId}:lock"
    if redis_buffer.r.set(lock_key, "1", ex=60, nx=True):
        print(
            f"[FAST TRACK] New data for Resident {payload.residentId}. "
            "Triggering instant ML pipeline!"
        )
        results = await run_fast_track_ml(payload.residentId)

    return {
        "status":     "Ingested",
        "resident_id": payload.residentId,
        "queue_size": redis_buffer.get_queue_size(payload.residentId),
        "results":    results,
    }


@router.post("/test-trigger-batch")
def test_trigger_batch(_user=Depends(get_sync_user)):
    try:
        run_batch_ml_predictions()
        return {"status": "success", "message": "Batch sweep completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
