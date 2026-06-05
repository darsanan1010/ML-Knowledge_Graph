import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import jwt
import pandas as pd
import requests
from dotenv import load_dotenv


env_path = Path(__file__).parent / "env1.env"
load_dotenv(env_path)

ENGINEERING_API_URL = os.getenv("URL")
CARE_MP_SERVICE_SECRET = os.getenv("CARE_MP_SERVICE_SECRET")
CARE_MP_SERVICE_ISS = os.getenv("CARE_MP_SERVICE_ISS")
CARE_MP_SERVICE_AUD = os.getenv("CARE_MP_SERVICE_AUD", "caremp-api")
ROLE = os.getenv("CARE_MP_SERVICE_ROLE", "caremp-ml-engine")
RESIDENT_PAGE_SIZE = int(os.getenv("RESIDENT_PAGE_SIZE", "200"))
RESIDENT_CACHE_DB_PATH = os.getenv("DB_PATH", "./cache.db")


def first_present(data: dict, keys: list[str]):
    for key in keys:
        value = data.get(key)
        if value not in [None, ""]:
            return value
    return None


def compute_age_from_dob(dob_value):
    if dob_value in [None, ""]:
        return None

    dob = pd.to_datetime(dob_value, errors="coerce", utc=True)
    if pd.isna(dob):
        return None

    today = pd.Timestamp.utcnow().date()
    birth_date = dob.date()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    if age < 0 or age > 130:
        return None
    return age


def normalize_resident_age(resident: dict):
    existing_age = first_present(resident, ["age", "residentAge"])
    if existing_age not in [None, ""]:
        age = pd.to_numeric(existing_age, errors="coerce")
        if pd.notna(age) and 0 < age <= 130:
            return int(age)

    dob = first_present(
        resident,
        [
            "dob",
            "dateOfBirth",
            "date_of_birth",
            "birthDate",
            "birth_date",
            "residentDob",
            "residentDOB",
        ],
    )
    return compute_age_from_dob(dob)


def resident_cache_path() -> Path:
    cache_path = Path(RESIDENT_CACHE_DB_PATH).expanduser()
    if not cache_path.is_absolute():
        cache_path = Path(__file__).parent / cache_path
    return cache_path


def generate_ml_token() -> str:
    if not CARE_MP_SERVICE_SECRET:
        raise ValueError("CARE_MP_SERVICE_SECRET is required for Engineering API access")

    now = int(time.time())
    payload = {
        "iss": CARE_MP_SERVICE_ISS,
        "aud": CARE_MP_SERVICE_AUD,
        "role": ROLE,
        "iat": now,
        "exp": now + 60,
    }
    return jwt.encode(payload, CARE_MP_SERVICE_SECRET, algorithm="HS256")


def normalize_resident_context(resident: dict) -> dict:
    dob = first_present(
        resident,
        [
            "dob",
            "dateOfBirth",
            "date_of_birth",
            "birthDate",
            "birth_date",
            "residentDob",
            "residentDOB",
        ],
    )
    return {
        "resident_id": resident.get("id") or resident.get("resident_id"),
        "resident_age": normalize_resident_age(resident),
        "resident_dob": dob,
        "medical_history": resident.get("medicalHistory"),
        "resident_condition": resident.get("condition"),
        "reference_heart_rate": resident.get("referenceHeartRate"),
        "heart_rate_tolerance": resident.get("heartRateTolerance"),
        "reference_hrv": (
            resident.get("referenceHrv")
            if resident.get("referenceHrv") is not None
            else resident.get("referenceHRV")
        ),
        "hrv_tolerance": resident.get("hrvTolerance"),
        "reference_systolic_bp": resident.get("referenceSystolicBP"),
        "systolic_bp_tolerance": resident.get("systolicBPTolerance"),
        "reference_diastolic_bp": resident.get("referenceDiastolicBP"),
        "diastolic_bp_tolerance": resident.get("diastolicBPTolerance"),
        "reference_oxygen_saturation": resident.get("referenceOxygenSaturation"),
        "oxygen_saturation_tolerance": resident.get("oxygenSaturationTolerance"),
        "reference_body_temperature": (
            resident.get("referenceBodyTemperature")
            if resident.get("referenceBodyTemperature") is not None
            else resident.get("referenceTemperature")
        ),
        "temperature_tolerance_celsius": (
            resident.get("temperatureToleranceCelsius")
            if resident.get("temperatureToleranceCelsius") is not None
            else resident.get("temperatureTolerance")
        ),
        "reference_step_count": resident.get("referenceStepCount"),
        "step_count_tolerance": resident.get("stepCountTolerance"),
        "reference_deep_sleep": resident.get("referenceDeepSleep"),
        "reference_light_sleep": resident.get("referenceLightSleep"),
    }


def extract_resident_page(payload):
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        page = payload.get("data") or payload.get("residents") or payload.get("items") or []
        pagination = payload.get("pagination") or {}
        return page, pagination.get("total")
    return [], None


def init_resident_cache():
    cache_path = resident_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS residents (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )


def store_residents_in_sqlite_cache(residents: list[dict]):
    if not residents:
        return

    try:
        init_resident_cache()
        stored = 0
        cache_path = resident_cache_path()
        with sqlite3.connect(cache_path) as conn:
            for resident in residents:
                resident_id = resident.get("id") or resident.get("resident_id")
                if not resident_id:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO residents (id, data, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        str(resident_id),
                        json.dumps(resident),
                        datetime.now().isoformat(),
                    ),
                )
                stored += 1

        print(f"Cached {stored} residents in SQLite: {cache_path}")
    except sqlite3.Error as exc:
        print(f"SQLite resident cache unavailable; continuing without cache. Reason: {exc}")


def fetch_residents_from_sqlite_cache():
    cache_path = resident_cache_path()
    if not cache_path.exists():
        return None

    try:
        init_resident_cache()
        residents = []
        with sqlite3.connect(cache_path) as conn:
            rows = conn.execute("SELECT data FROM residents").fetchall()
    except sqlite3.Error as exc:
        print(f"SQLite resident cache unavailable. Reason: {exc}")
        return None

    for (data_json,) in rows:
        try:
            residents.append(json.loads(data_json))
        except json.JSONDecodeError:
            continue

    if not residents:
        return None

    print(f"Loaded {len(residents)} residents from SQLite cache: {cache_path}")
    baseline_df = pd.DataFrame(
        normalize_resident_context(resident) for resident in residents
    )
    return baseline_df.dropna(subset=["resident_id"])


def fetch_residents_from_engineering_api():
    if not ENGINEERING_API_URL:
        return None

    token = generate_ml_token()
    headers = {"Authorization": f"Bearer {token}"}
    residents = []
    skip = 0

    while True:
        try:
            response = requests.get(
                ENGINEERING_API_URL,
                headers=headers,
                params={
                    "skip": skip,
                    "limit": RESIDENT_PAGE_SIZE,
                    "showDeleted": False,
                },
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Engineering API unavailable: {exc}")
            return None

        page, total = extract_resident_page(response.json())
        if not page:
            break

        residents.extend(page)
        if total is not None and len(residents) >= int(total):
            break
        if len(page) < RESIDENT_PAGE_SIZE:
            break

        skip += RESIDENT_PAGE_SIZE

    store_residents_in_sqlite_cache(residents)
    baseline_df = pd.DataFrame(
        normalize_resident_context(resident) for resident in residents
    )
    return baseline_df.dropna(subset=["resident_id"])


def fetch_residents_from_database(engine):
    query = """
    SELECT
        id AS resident_id,
        "referenceHeartRate" AS reference_heart_rate,
        "referenceSystolicBP" AS reference_systolic_bp,
        "referenceDiastolicBP" AS reference_diastolic_bp,
        "referenceOxygenSaturation" AS reference_oxygen_saturation,
        "referenceBodyTemperature" AS reference_body_temperature,
        "referenceStepCount" AS reference_step_count
    FROM resident_vitals
    WHERE "isDeleted" = false
    """
    return pd.read_sql(query, engine)


def prepare_baselines(baseline_df):
    baseline_df = baseline_df.copy()
    baseline_df["resident_id"] = pd.to_numeric(
        baseline_df["resident_id"], errors="coerce"
    ).astype("Int64")
    baseline_df = baseline_df.dropna(subset=["resident_id"]).drop_duplicates(
        "resident_id", keep="last"
    )

    default_references = {
        "reference_heart_rate": 75.0,
        "reference_hrv": 40.0,
        "reference_systolic_bp": 120.0,
        "reference_diastolic_bp": 80.0,
        "reference_oxygen_saturation": 98.0,
        "reference_body_temperature": 36.8,
        "reference_step_count": 2000.0,
        "reference_deep_sleep": 100.0,
        "reference_light_sleep": 250.0,
    }
    default_tolerances = {
        "heart_rate_tolerance": 20.0,
        "hrv_tolerance": 10.0,
        "systolic_bp_tolerance": 20.0,
        "diastolic_bp_tolerance": 15.0,
        "oxygen_saturation_tolerance": 4.0,
        "temperature_tolerance_celsius": 1.5,
        "step_count_tolerance": 500.0,
    }

    for column, default_value in {**default_references, **default_tolerances}.items():
        if column not in baseline_df.columns:
            baseline_df[column] = default_value
        baseline_df[column] = pd.to_numeric(baseline_df[column], errors="coerce")
        baseline_df[column] = baseline_df[column].fillna(default_value)
        baseline_df.loc[baseline_df[column] <= 0, column] = default_value

    if "resident_age" not in baseline_df.columns:
        baseline_df["resident_age"] = pd.NA
    baseline_df["resident_age"] = pd.to_numeric(
        baseline_df["resident_age"], errors="coerce"
    )
    baseline_df.loc[
        (baseline_df["resident_age"] <= 0) | (baseline_df["resident_age"] > 130),
        "resident_age",
    ] = pd.NA

    if "resident_dob" not in baseline_df.columns:
        baseline_df["resident_dob"] = pd.NA

    for column in [
        "medical_history",
        "resident_condition",
    ]:
        if column not in baseline_df.columns:
            baseline_df[column] = ""
        baseline_df[column] = baseline_df[column].fillna("").astype(str)

    baseline_df["reference_total_sleep_minutes"] = (
        baseline_df["reference_deep_sleep"] + baseline_df["reference_light_sleep"]
    )
    return baseline_df


def get_resident_baselines(
    engine=None,
    skip_engineering_api=False,
    use_sqlite_cache=True,
    use_database_fallback=True,
):
    baseline_df = None

    if not skip_engineering_api:
        baseline_df = fetch_residents_from_engineering_api()

    if (baseline_df is None or baseline_df.empty) and use_sqlite_cache:
        baseline_df = fetch_residents_from_sqlite_cache()

    if (
        (baseline_df is None or baseline_df.empty)
        and use_database_fallback
        and engine is not None
    ):
        baseline_df = fetch_residents_from_database(engine)

    if baseline_df is None or baseline_df.empty:
        raise ValueError("No resident baseline/context data available")

    return prepare_baselines(baseline_df)


if __name__ == "__main__":
    residents_df = get_resident_baselines(
        engine=None,
        skip_engineering_api=False,
        use_sqlite_cache=True,
        use_database_fallback=False,
    )
    print(residents_df[["resident_id"]].head())
    print(f"Loaded {len(residents_df)} resident baseline rows")
