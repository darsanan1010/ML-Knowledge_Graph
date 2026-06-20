import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_HISTORY_DB = "fall_risk_prediction_history.db"

RISK_RANK = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4,
}

RISK_SCORE = {
    "Low": -1,
    "Moderate": -4,
    "High": -8,
    "Critical": -14,
}



from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.engine import URL
import os
from dotenv import load_dotenv
from pathlib import Path

def load_prediction_history(db_path=None):
    env_path  = Path(__file__).parent / "env1.env"
    load_dotenv(env_path)
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "mydb3"
    DB_USER = "postgres"
    DB_PASSWORD = "Metrok@1357"

    try:
        DB_URL = URL.create(
            "postgresql+psycopg2",
            username=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
        )
        engine = create_engine(DB_URL)
        
        # Check if table exists
        with engine.connect() as conn:
            result = conn.execute(sa_text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'fall_risk_prediction_history')"
            )).scalar()
            
            if not result:
                return pd.DataFrame()

            query = "SELECT * FROM fall_risk_prediction_history"
            df = pd.read_sql(query, conn)
            if not df.empty and "generated_at" in df.columns:
                df["generated_at"] = pd.to_datetime(df["generated_at"], format="ISO8601")
                df["risk_rank"] = df["predicted_risk_label"].map(RISK_RANK)
                df["risk_score_legacy"] = df["predicted_risk_label"].map(RISK_SCORE)
                df["risk_score"] = df["risk_score"].fillna(df["risk_score_legacy"])
            return df
    except Exception as e:
        print(f"Warning: Failed to load prediction history from mydb3: {e}")
        return pd.DataFrame()

def get_resident_risk_trend(resident_id, hours=24, db_path=DEFAULT_HISTORY_DB):
    df = load_prediction_history(db_path)
    if df.empty:
        return []

    resident_df = df[df["resident_id"].astype(str).eq(str(resident_id))].copy()
    if resident_df.empty:
        return []

    latest_time = resident_df["generated_at"].max()
    window_start = latest_time - pd.Timedelta(hours=hours)
    resident_df = resident_df[resident_df["generated_at"] >= window_start]

    if resident_df.empty:
        return []

    # Keep the same shape expected by the existing API/dashboard: hour, score, timestamp.
    trend = (
        resident_df.set_index("generated_at")
        .resample("1h")
        .agg(
            score=("risk_score", "mean"),
            risk_rank=("risk_rank", "mean"),
            prob_high=("prob_high", "mean"),
            prob_critical=("prob_critical", "mean"),
            confidence=("prediction_confidence", "mean"),
        )
        .dropna(subset=["score"])
        .reset_index()
    )

    trend["hour"] = range(len(trend))
    trend["score"] = pd.to_numeric(trend["score"], errors="coerce")
    trend["score"] = trend["score"].fillna(0).round(2)
    trend["timestamp"] = trend["generated_at"].dt.strftime("%H:%M")
    trend["generated_at"] = trend["generated_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return trend[
        [
            "hour",
            "score",
            "timestamp",
            "generated_at",
            "risk_rank",
            "prob_high",
            "prob_critical",
            "confidence",
        ]
    ].to_dict("records")


def analyze_trend_direction(risk_trend, requested_hours=24, available_hours=None, current_drivers=None):
    min_points_needed = 4 if requested_hours <= 6 else 8
    if len(risk_trend) < min_points_needed:
        return {
            "trend_direction": "insufficient_data",
            "message": "Not enough prediction history for trend analysis.",
            "confidence": 0,
            "is_partial_window": True,
        }

    if available_hours is None:
        available_hours = requested_hours
    is_partial_window = available_hours < requested_hours
    window_label = (
        f"available {available_hours:.1f}h"
        if is_partial_window
        else f"{requested_hours}h"
    )

    import pandas as pd
    latest_time = pd.to_datetime(risk_trend[-1]["generated_at"])
    chronological_midpoint = latest_time - pd.Timedelta(hours=requested_hours / 2)

    earlier_scores = [p["score"] for p in risk_trend if pd.to_datetime(p["generated_at"]) < chronological_midpoint]
    later_scores = [p["score"] for p in risk_trend if pd.to_datetime(p["generated_at"]) >= chronological_midpoint]

    min_required = 2 if requested_hours <= 6 else 3

    # Require valid hours of coverage in BOTH chronological halves
    if len(earlier_scores) < min_required or len(later_scores) < min_required:
        return {
            "trend_direction": "insufficient_data",
            "message": "Poor data distribution. Need data from both halves of the day to calculate a trend.",
            "confidence": 0,
            "is_partial_window": is_partial_window,
        }

    earlier_avg = sum(earlier_scores) / len(earlier_scores)
    later_avg = sum(later_scores) / len(later_scores)

    # V2 risk scores are POSITIVE (0-100). Higher score = higher risk.
    # Therefore, a POSITIVE trend_strength means risk is INCREASING (Worsening).
    trend_strength = later_avg - earlier_avg

    if trend_strength <= -20:
        trend_severity = "Strongly Improving"
    elif trend_strength <= -10:
        trend_severity = "Improving"
    elif trend_strength >= 20:
        trend_severity = "Strongly Deteriorating"
    elif trend_strength >= 10:
        trend_severity = "Deteriorating"
    else:
        trend_severity = "Stable"

    # Multi-Domain Deterioration Upgrade
    if current_drivers:
        active_domains = set([d.get("domain") for d in current_drivers if d.get("score", 0) < 0])
        if len(active_domains) >= 2 and trend_severity == "Deteriorating":
            trend_severity = "Strongly Deteriorating"
            
    if later_avg > earlier_avg + 10:
        direction = "worsening"
        message = f"Risk trend is worsening over the {window_label} window."
        confidence = 85
    elif later_avg < earlier_avg - 10:
        direction = "improving"
        message = f"Risk trend is improving over the {window_label} window."
        confidence = 80
    else:
        direction = "stable"
        message = f"Risk trend is stable over the {window_label} window."
        confidence = 75

    return {
        "trend_direction": direction,
        "message": message,
        "confidence": confidence,
        "is_partial_window": is_partial_window,
        "earlier_avg_score": round(earlier_avg, 2),
        "later_avg_score": round(later_avg, 2),
        "trend_strength": round(trend_strength, 2),
        "trend_severity": trend_severity,
    }


def get_window_diagnostics(resident_df, hours):
    latest_time = resident_df["generated_at"].max()
    window_start = latest_time - pd.Timedelta(hours=hours)
    window_df = resident_df[resident_df["generated_at"] >= window_start]

    first_prediction = pd.NaT
    last_prediction = pd.NaT
    available_window_hours = 0.0
    hour_buckets = 0

    if not window_df.empty:
        first_prediction = window_df["generated_at"].min()
        last_prediction = window_df["generated_at"].max()
        available_window_hours = (
            last_prediction - first_prediction
        ).total_seconds() / 3600
        hour_buckets = (
            window_df.set_index("generated_at")
            .resample("1h")
            .size()
            .gt(0)
            .sum()
        )

    return {
        "window_start": window_start,
        "window_rows": len(window_df),
        "window_hour_buckets": int(hour_buckets),
        "window_available_hours": round(available_window_hours, 2),
        "window_first_prediction": first_prediction,
        "window_last_prediction": last_prediction,
    }


def prefix_keys(values, prefix):
    return {f"{prefix}{key}": value for key, value in values.items()}


def save_csv_with_fallback(df, output_path):
    try:
        df.to_csv(output_path, index=False)
        print(f"Saved trend summary to {output_path}")
    except PermissionError:
        path = Path(output_path)
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, index=False)
        print(f"{output_path} is locked. Saved trend summary to {fallback_path}")


def build_trend_summary(hours=24, db_path=DEFAULT_HISTORY_DB, fallback_hours=None):
    df = load_prediction_history(db_path)
    if df.empty:
        return pd.DataFrame()

    summaries = []
    for resident_id in sorted(df["resident_id"].unique()):
        trend = get_resident_risk_trend(resident_id, hours=hours, db_path=db_path)
        resident_df = df[df["resident_id"].eq(resident_id)]
        latest = resident_df.sort_values("generated_at").iloc[-1]

        available_hours = (
            resident_df["generated_at"].max() - resident_df["generated_at"].min()
        ).total_seconds() / 3600
        window_diagnostics = get_window_diagnostics(resident_df, hours)
        analysis = analyze_trend_direction(
            trend,
            requested_hours=hours,
            available_hours=window_diagnostics["window_available_hours"],
        )

        summary = {
            "resident_id": resident_id,
            "latest_time": latest["generated_at"],
            "latest_risk_label": latest["predicted_risk_label"],
            "latest_confidence": latest["prediction_confidence"],
            "available_history_hours": round(available_hours, 2),
            "requested_trend_hours": hours,
            "latest_window_start": window_diagnostics["window_start"],
            "latest_window_rows": window_diagnostics["window_rows"],
            "latest_window_hour_buckets": window_diagnostics["window_hour_buckets"],
            "latest_window_available_hours": window_diagnostics[
                "window_available_hours"
            ],
            "latest_window_first_prediction": window_diagnostics[
                "window_first_prediction"
            ],
            "latest_window_last_prediction": window_diagnostics[
                "window_last_prediction"
            ],
            "trend_points": len(trend),
            **analysis,
        }

        if fallback_hours and analysis["trend_direction"] == "insufficient_data":
            fallback_trend = get_resident_risk_trend(
                resident_id,
                hours=fallback_hours,
                db_path=db_path,
            )
            fallback_window = get_window_diagnostics(resident_df, fallback_hours)
            fallback_analysis = analyze_trend_direction(
                fallback_trend,
                requested_hours=fallback_hours,
                available_hours=fallback_window["window_available_hours"],
            )
            summary.update(
                {
                    "fallback_trend_used": True,
                    "fallback_requested_trend_hours": fallback_hours,
                    "fallback_trend_points": len(fallback_trend),
                    **prefix_keys(fallback_window, "fallback_"),
                    **prefix_keys(fallback_analysis, "fallback_"),
                }
            )
        else:
            summary["fallback_trend_used"] = False

        summaries.append(summary)

    return pd.DataFrame(summaries)


def get_resident_trend_context(
    resident_id,
    hours=24,
    fallback_hours=168,
    db_path=DEFAULT_HISTORY_DB,
    current_drivers=None,
):
    df = load_prediction_history(db_path)
    if df.empty:
        return {
            "available": False,
            "reason": "prediction_history_empty",
            "riskTrend": [],
            "analysis": None,
            "fallbackAnalysis": None,
        }

    resident_df = df[df["resident_id"].astype(str).eq(str(resident_id))].copy()
    if resident_df.empty:
        return {
            "available": False,
            "reason": "resident_history_not_found",
            "riskTrend": [],
            "analysis": None,
            "fallbackAnalysis": None,
        }

    trend = get_resident_risk_trend(resident_id, hours=hours, db_path=db_path)
    window = get_window_diagnostics(resident_df, hours)
    analysis = analyze_trend_direction(
        trend,
        requested_hours=hours,
        available_hours=window["window_available_hours"],
        current_drivers=current_drivers,
    )

    fallback_trend = []
    fallback_analysis = None
    fallback_window = None
    if fallback_hours and analysis["trend_direction"] == "insufficient_data":
        fallback_trend = get_resident_risk_trend(
            resident_id,
            hours=fallback_hours,
            db_path=db_path,
        )
        fallback_window = get_window_diagnostics(resident_df, fallback_hours)
        fallback_analysis = analyze_trend_direction(
            fallback_trend,
            requested_hours=fallback_hours,
            available_hours=fallback_window["window_available_hours"],
            current_drivers=current_drivers,
        )

    rapid_hours = 6
    rapid_trend = get_resident_risk_trend(resident_id, hours=rapid_hours, db_path=db_path)
    rapid_window = get_window_diagnostics(resident_df, rapid_hours)
    rapid_analysis = analyze_trend_direction(
        rapid_trend,
        requested_hours=rapid_hours,
        available_hours=rapid_window["window_available_hours"],
        current_drivers=current_drivers,
    )

    previous_score = None
    if len(trend) > 1:
        previous_score = trend[-2].get("risk_score")
        
    latest = resident_df.sort_values("generated_at").iloc[-1]
    return {
        "available": True,
        "residentId": resident_id,
        "latestPrediction": {
            "generatedAt": latest["generated_at"].isoformat(),
            "riskLabel": latest["predicted_risk_label"],
            "confidence": latest["prediction_confidence"],
            "probCritical": latest["prob_critical"],
            "probHigh": latest["prob_high"],
            "probLow": latest["prob_low"],
            "probModerate": latest["prob_moderate"],
        },
        "riskTrend": trend,
        "analysis": analysis,
        "rapidAnalysis": rapid_analysis,
        "fallbackAnalysis": fallback_analysis,
        "latest_score": trend[-1]["score"] if trend else 0,
        "latest_numeric_score": trend[-1].get("risk_score") if trend else None,
        "previous_numeric_score": previous_score,
        "trend_summary": f"{analysis['trend_direction']} ({analysis['confidence']} confidence)",
        "window": window,
        "fallbackRiskTrend": fallback_trend,
        "fallbackWindow": fallback_window,
    }


if __name__ == "__main__":
    summary = build_trend_summary(hours=24, fallback_hours=168)
    print("\n=== V1 24H Trend Summary ===")
    print(summary.to_string(index=False))
    save_csv_with_fallback(summary, "fall_risk_trend_summary_v1.csv")

    weekly_summary = build_trend_summary(hours=168)
    save_csv_with_fallback(weekly_summary, "fall_risk_trend_summary_7d_v1.csv")
