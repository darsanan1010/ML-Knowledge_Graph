from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.data.redis_ingestion_buffer import RedisBLEBuffer
import pandas as pd
from pathlib import Path
import joblib
import os
from dotenv import load_dotenv

from app.core.predict_fall_risk_v1 import (
    build_engine,
    fetch_band_logs,
    build_prediction_features,
    predict_rows,
    write_prediction_history_sqlite,
    prediction_model_version,
    DEFAULT_TRAINING_START_AT,
    DEFAULT_TRAINING_END_AT,
    get_resident_baselines
)
from app.core.fall_risk_layered_pipeline_v2 import build_pipeline_record, public_pipeline_record

router = APIRouter()

# Initialize the robust Redis buffer
try:
    redis_buffer = RedisBLEBuffer(host='localhost', port=6379, db=0)
except Exception as e:
    print(f"Warning: Redis connection failed on startup: {e}")
    redis_buffer = None

# Load the Model Artifact at startup
ARTIFACT_PATH = Path("models_v2_grouped_real_holdout/fall_risk_xgboost_v23_final.pkl")
try:
    if ARTIFACT_PATH.exists():
        model_dir = ARTIFACT_PATH.parent
        suffix = ARTIFACT_PATH.name.replace("fall_risk_xgboost_", "")
        ml_artifact = {
            "model": joblib.load(ARTIFACT_PATH),
            "label_encoder": joblib.load(model_dir / f"risk_label_encoder_{suffix}"),
            "feature_columns": joblib.load(model_dir / f"model_features_{suffix}"),
            "model_version": suffix.replace(".pkl", "")
        }
        print(f"Loaded ML artifact {ml_artifact['model_version']}")
    else:
        print(f"Warning: ML artifact not found at {ARTIFACT_PATH}")
        ml_artifact = None
except Exception as e:
    print(f"Failed to load ML artifact: {e}")
    ml_artifact = None


class BandLog(BaseModel):
    bandlogId: int
    bandId: Optional[int] = None
    residentId: Optional[int] = None
    heartRate: Optional[float] = None
    hrv: Optional[float] = None
    systolicBP: Optional[float] = None
    diastolicBP: Optional[float] = None
    oxygenSaturation: Optional[float] = None
    oxygenSaturationValid: Optional[bool] = True
    bodyTemperature: Optional[float] = None
    skinTemperature: Optional[float] = None
    temperatureUnit: Optional[int] = 2
    rssi: Optional[int] = 0         # DATA INSIGHT: rssi is int in real data
    stress: Optional[float] = None
    stressValid: Optional[bool] = True
    stepCount: Optional[int] = None
    calories: Optional[float] = None
    deepSleepTime: Optional[int] = None
    lightSleepTime: Optional[int] = None
    fatigueLevel: Optional[int] = None
    breathing: Optional[int] = None # DATA INSIGHT: breathing is int in real data
    wearingIndicator: Optional[bool] = True
    generatedAt: datetime


def process_ready_packets_for_resident(resident_id: int):
    """
    Background worker function that runs after a packet is ingested.
    It fetches the 12-hour sliding window from Redis and runs the ML pipeline.
    """
    if not redis_buffer:
        print("[ML PIPELINE] Skipping ML: Redis not connected.")
        return
        
    if not ml_artifact:
        print("[ML PIPELINE] Skipping ML: ML artifact not loaded.")
        return

    key = redis_buffer._get_key(resident_id)
    
    # Get the full 12-hour sliding window from Redis without deleting it
    raw_packets = redis_buffer.r.zrange(key, 0, -1)
    if not raw_packets:
        return
        
    import json
    ready_packets = [json.loads(p) for p in raw_packets]
    
    print(f"[ML PIPELINE] Triggered for Resident {resident_id}. Processing {len(ready_packets)} sliding window packets.")
    
    ready_df = pd.DataFrame(ready_packets)
    ready_df = ready_df.rename(columns={
        "bandlogId": "id",
        "bandId": "band_id",
        "residentId": "resident_id",
        "heartRate": "heart_rate",
        "systolicBP": "systolic_bp",
        "diastolicBP": "diastolic_bp",
        "oxygenSaturation": "oxygen_saturation",
        "oxygenSaturationValid": "oxygen_saturation_valid",
        "bodyTemperature": "body_temperature",
        "skinTemperature": "skin_temperature",
        "temperatureUnit": "temperature_unit",
        "stressValid": "stress_valid",
        "deepSleepTime": "deep_sleep_time",
        "lightSleepTime": "light_sleep_time",
        "fatigueLevel": "fatigue_level",
        "wearingIndicator": "wearing_indicator",
        "stepCount": "step_count",
        "generatedAt": "generated_at"
    })
    ready_df["generated_at"] = pd.to_datetime(ready_df["generated_at"])
    ready_df = ready_df.sort_values("generated_at")
    
    engine = build_engine()
    baseline_df = get_resident_baselines(engine=engine, skip_engineering_api=True, use_sqlite_cache=True, use_database_fallback=True)

    features_df = build_prediction_features(ready_df, baseline_df, hours=12, use_latest_available=True)
    if features_df.empty:
        return
    
    try:
        # Predict the latest risk state
        predictions = predict_rows(features_df, ml_artifact, mode="latest")
        
        if not predictions.empty:
            print(f"\n[ML PIPELINE OUTPUT] Generated risk predictions for Resident {resident_id}:")
            print(predictions[["resident_id", "predicted_risk_label", "prediction_confidence"]].to_string(index=False))
            
            # Save to SQLite
            sqlite_path = "fall_risk_prediction_history.db"
            write_prediction_history_sqlite(
                predictions,
                prediction_model_version(ml_artifact),
                DEFAULT_TRAINING_START_AT,
                DEFAULT_TRAINING_END_AT,
                sqlite_path
            )
    except Exception as e:
        print(f"[ML PIPELINE ERROR] Failed to run predictions: {e}")
        import traceback
        traceback.print_exc()

    if not ready_packets:
        return

    print(f"[ML PIPELINE] Triggered for Resident {resident_id}. Processing {len(ready_packets)} new packets.")
    
    if not ml_artifact:
        print("[ML PIPELINE] Skipped because ML artifact is not loaded.")
        return

    try:
        # Convert ready packets to DataFrame
        ready_df = pd.DataFrame(ready_packets)
        
        # Convert camelCase to snake_case for the ML pipeline
        ready_df = ready_df.rename(columns={
            "bandlogId": "id",
            "bandId": "band_id",
            "residentId": "resident_id",
            "heartRate": "heart_rate",
            "systolicBP": "systolic_bp",
            "diastolicBP": "diastolic_bp",
            "oxygenSaturation": "oxygen_saturation",
            "oxygenSaturationValid": "oxygen_saturation_valid",
            "bodyTemperature": "body_temperature",
            "skinTemperature": "skin_temperature",
            "temperatureUnit": "temperature_unit",
            "stressValid": "stress_valid",
            "deepSleepTime": "deep_sleep_time",
            "lightSleepTime": "light_sleep_time",
            "fatigueLevel": "fatigue_level",
            "wearingIndicator": "wearing_indicator",
            "stepCount": "step_count",
            "generatedAt": "generated_at"
        })
        ready_df["generated_at"] = pd.to_datetime(ready_df["generated_at"])
        
        # Fetch last 12 hours of data from Postgres to build rolling features
        engine = build_engine()
        historical_band_df = fetch_band_logs(
            engine,
            hours=12,
            sleep_baseline_days=7,
            training_start_at=DEFAULT_TRAINING_START_AT,
            training_end_at=DEFAULT_TRAINING_END_AT,
            use_latest_available=True
        )
        
        # Filter historical data for this resident
        historical_band_df = historical_band_df[historical_band_df["resident_id"] == resident_id]
        
        # Combine historical and new packets
        combined_band_df = pd.concat([historical_band_df, ready_df], ignore_index=True)
        combined_band_df = combined_band_df.drop_duplicates(subset=["resident_id", "generated_at"], keep="last")
        combined_band_df = combined_band_df.sort_values("generated_at")
        
        # Fetch baseline context
        baseline_df = get_resident_baselines(engine=engine, skip_engineering_api=True, use_sqlite_cache=True, use_database_fallback=True)
        
        # Build features
        features_df = build_prediction_features(combined_band_df, baseline_df, hours=12, use_latest_available=True)
        
        if features_df.empty:
            print(f"[ML PIPELINE] No features generated for resident {resident_id}.")
            return
            
        # Predict all rows to ensure rolling context is applied
        predictions = predict_rows(features_df, ml_artifact, mode="history")
        
        # Filter predictions to ONLY the new packets we just ingested
        ready_timestamps = set(ready_df["generated_at"].dt.tz_convert(None)) if ready_df["generated_at"].dt.tz is not None else set(ready_df["generated_at"])
        predictions["generated_at_tz_naive"] = pd.to_datetime(predictions["generated_at"]).dt.tz_convert(None) if predictions["generated_at"].dt.tz is not None else pd.to_datetime(predictions["generated_at"])
        
        new_predictions = predictions[predictions["generated_at_tz_naive"].isin(ready_timestamps)].copy()
        new_predictions = new_predictions.drop(columns=["generated_at_tz_naive"])
        
        output_results = []
        if not new_predictions.empty:
            print(f"\n[ML PIPELINE OUTPUT] Predictions for Resident {resident_id}:")
            print(new_predictions[["generated_at", "predicted_risk_label", "prediction_confidence"]].to_string(index=False))
            
            # Map through Rule Engine to get drivers and recommendations
            raw_index = combined_band_df.set_index(["resident_id", "generated_at"]).sort_index()
            for idx, feature_row in features_df.loc[features_df.index.isin(new_predictions.index)].iterrows():
                key = (feature_row["resident_id"], feature_row["generated_at"])
                if key not in raw_index.index:
                    continue
                raw_row = raw_index.loc[key]
                if isinstance(raw_row, pd.DataFrame):
                    raw_row = raw_row.iloc[-1]
                
                prediction_row = new_predictions.loc[idx]
                
                record = build_pipeline_record(
                    feature_row=feature_row,
                    raw_row=raw_row,
                    prediction_row=prediction_row,
                    history_db="fall_risk_prediction_history.db",
                    shap_payload=None,
                    carenotes_df=None
                )
                output_results.append(public_pipeline_record(record))
            
            # Print to terminal
            import json
            print("\n[ML PIPELINE JSON PAYLOAD]")
            print(json.dumps(output_results, indent=2, default=str))
            
            # Save to JSON file
            json_file = "last_pipeline_output.json"
            with open(json_file, "w") as f:
                json.dump(output_results, f, indent=2, default=str)
            print(f"[ML PIPELINE] Wrote JSON payload to {json_file}")
            
            # Save to SQLite
            sqlite_path = "fall_risk_prediction_history.db"
            write_prediction_history_sqlite(
                new_predictions,
                prediction_model_version(ml_artifact),
                DEFAULT_TRAINING_START_AT,
                DEFAULT_TRAINING_END_AT,
                sqlite_path
            )
            print(f"[ML PIPELINE] Saved {len(new_predictions)} risk scores to SQLite database ({sqlite_path}).")
        else:
            print(f"[ML PIPELINE] Processed but no new predictions generated for output.")
            
        return output_results

    except Exception as e:
        print(f"[ML PIPELINE ERROR] Failed to process packets for Resident {resident_id}: {e}")
        import traceback
        traceback.print_exc()


import asyncio

fast_track_semaphore = asyncio.Semaphore(4)

async def run_fast_track_ml(resident_id: int):
    """
    Wraps the ML pipeline in a Semaphore to protect RAM if 100 residents trigger the fast track simultaneously.
    """
    async with fast_track_semaphore:
        return await asyncio.to_thread(process_ready_packets_for_resident, resident_id)


def run_batch_ml_predictions():
    """
    Scheduled background task that runs every 15 minutes.
    Sweeps Redis for all active residents and runs predictions in a single memory-safe vectorized batch.
    """
    if not redis_buffer or not ml_artifact:
        print("[BATCH ML] Skipped: Redis or ML artifact not ready.")
        return

    print("[BATCH ML] Starting scheduled 15-minute sweep...")
    keys = redis_buffer.r.keys(f"{redis_buffer.prefix}*")
    if not keys:
        print("[BATCH ML] No active residents found in Redis.")
        return

    import json
    all_combined_df_list = []
    
    engine = build_engine()
    baseline_df = get_resident_baselines(engine=engine, skip_engineering_api=True, use_sqlite_cache=True, use_database_fallback=True)

    # Fetch last 12 hours of historical data for ALL residents at once (efficient)
    historical_band_df = fetch_band_logs(
        engine,
        hours=12,
        sleep_baseline_days=7,
        training_start_at=DEFAULT_TRAINING_START_AT,
        training_end_at=DEFAULT_TRAINING_END_AT,
        use_latest_available=True
    )

    for key in keys:
        resident_id_str = key.replace(redis_buffer.prefix, "")
        try:
            resident_id = int(resident_id_str)
        except ValueError:
            continue
            
        raw_packets = redis_buffer.r.zrange(key, 0, -1)
        if not raw_packets:
            continue
            
        ready_packets = [json.loads(p) for p in raw_packets]
        ready_df = pd.DataFrame(ready_packets)
        ready_df = ready_df.rename(columns={
            "bandlogId": "id",
            "bandId": "band_id",
            "residentId": "resident_id",
            "heartRate": "heart_rate",
            "systolicBP": "systolic_bp",
            "diastolicBP": "diastolic_bp",
            "oxygenSaturation": "oxygen_saturation",
            "oxygenSaturationValid": "oxygen_saturation_valid",
            "bodyTemperature": "body_temperature",
            "skinTemperature": "skin_temperature",
            "temperatureUnit": "temperature_unit",
            "stressValid": "stress_valid",
            "deepSleepTime": "deep_sleep_time",
            "lightSleepTime": "light_sleep_time",
            "fatigueLevel": "fatigue_level",
            "wearingIndicator": "wearing_indicator",
            "stepCount": "step_count",
            "generatedAt": "generated_at"
        })
        ready_df["generated_at"] = pd.to_datetime(ready_df["generated_at"])
        
        # Filter historical data for this specific resident
        res_history = historical_band_df[historical_band_df["resident_id"] == resident_id]
        
        combined_band_df = pd.concat([res_history, ready_df], ignore_index=True)
        combined_band_df = combined_band_df.drop_duplicates(subset=["resident_id", "generated_at"], keep="last")
        combined_band_df = combined_band_df.sort_values("generated_at")
        
        features_df = build_prediction_features(combined_band_df, baseline_df, hours=12, use_latest_available=True)
        
        if not features_df.empty:
            # For the batch, we only care about their LATEST state
            latest_feature = features_df.sort_values("generated_at").tail(1)
            all_combined_df_list.append(latest_feature)
            
    if not all_combined_df_list:
        print("[BATCH ML] No features built for active residents.")
        return
        
    master_features_df = pd.concat(all_combined_df_list, ignore_index=True)
    
    try:
        # Run prediction on ALL residents simultaneously in C! (Vectorization)
        predictions = predict_rows(master_features_df, ml_artifact, mode="latest")
        if not predictions.empty:
            print(f"\n[BATCH ML OUTPUT] Generated risk predictions for {len(predictions)} residents:")
            print(predictions[["resident_id", "predicted_risk_label", "prediction_confidence"]].to_string(index=False))
            
            sqlite_path = "fall_risk_prediction_history.db"
            write_prediction_history_sqlite(
                predictions,
                prediction_model_version(ml_artifact),
                DEFAULT_TRAINING_START_AT,
                DEFAULT_TRAINING_END_AT,
                sqlite_path
            )
            print(f"[BATCH ML] Saved {len(predictions)} risk scores to SQLite.")
            
            # Explicit garbage collection to prevent memory creep
            del master_features_df
            import gc
            gc.collect()
            
    except Exception as e:
        print(f"[BATCH ML ERROR] Failed to run batch predictions: {e}")
        import traceback
        traceback.print_exc()


@router.post("/predict")
async def ingest_bandlog(payload: BandLog):
    """
    High-speed synchronous ingestion endpoint with Smart Hybrid Batching!
    """
    if not redis_buffer:
        raise HTTPException(status_code=500, detail="Redis buffer is not connected.")

    # Convert the datetime to a unix timestamp for Redis sorting
    unix_time = payload.generatedAt.timestamp()
    
    # Check if this resident's queue is empty (First-Packet check)
    queue_size_before = redis_buffer.get_queue_size(payload.residentId)
    
    # Convert Pydantic model to a standard dictionary
    packet_dict = payload.model_dump(mode='json')

    # Push to the Redis Sorted Set (Instantly sorts out-of-order packets!)
    redis_buffer.add_packet(
        resident_id=payload.residentId,
        generated_at_unix=unix_time,
        packet_data=packet_dict
    )

    results = []
    # Fast Track Trigger (Synchronous)
    if queue_size_before == 0:
        print(f"[FAST TRACK] New data for Resident {payload.residentId}. Triggering instant ML pipeline!")
        results = await run_fast_track_ml(payload.residentId)

    return {
        "status": "Ingested", 
        "resident_id": payload.residentId, 
        "queue_size": redis_buffer.get_queue_size(payload.residentId),
        "results": results
    }
