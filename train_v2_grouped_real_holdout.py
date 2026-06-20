import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


DEFAULT_DATASET = "fall_risk_training_dataset_v23.csv"
DEFAULT_FEATURE_MANIFEST = "model_training_features.csv"
DEFAULT_OUTPUT_DIR = Path("models_v2_grouped_real_holdout")
LABEL_ORDER = ["Critical", "High", "Low", "Moderate"]


DEFAULT_TEST_FOLDS = [
    [5, 10],
    [2, 8],
    [6],
    [1, 3],
    [7, 15],
]


def load_features(feature_manifest, df):
    feature_df = pd.read_csv(feature_manifest)
    features = feature_df["feature"].dropna().astype(str).tolist()
    features = [feature for feature in features if feature in df.columns]

    blocked_features = {
        "risk_score",
        "risk_label",
        "class_weight",
        "is_synthetic",
    }
    features = [feature for feature in features if feature not in blocked_features]

    return features


def prepare_matrix(df, features):
    x = df[features].replace([np.inf, -np.inf], np.nan)
    return x.fillna(0)


def resident_balanced_sample(df, max_rows_per_resident, seed):
    if not max_rows_per_resident:
        return df

    parts = []
    for _, group in df.groupby("resident_id", sort=False):
        if len(group) > max_rows_per_resident:
            parts.append(group.sample(n=max_rows_per_resident, random_state=seed))
        else:
            parts.append(group)

    return pd.concat(parts, ignore_index=True)


def choose_validation_residents(train_residents, fold_index):
    if len(train_residents) <= 2:
        return train_residents[-1:]

    validation_count = max(1, min(2, len(train_residents) // 4))
    start = (fold_index * validation_count) % len(train_residents)
    rotated = train_residents[start:] + train_residents[:start]
    return rotated[:validation_count]


def build_model(seed, max_depth=5, min_child_weight=3):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABEL_ORDER),
        eval_metric="mlogloss",
        n_estimators=600,
        learning_rate=0.04,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        early_stopping_rounds=40,
    )


def evaluate_fold(
    fold_name,
    model,
    label_encoder,
    x_test,
    y_test,
    output_dir,
):
    y_pred = model.predict(x_test)
    labels = np.arange(len(label_encoder.classes_))

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "fold": fold_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "critical_recall": recall_score(
            y_test,
            y_pred,
            labels=[label_encoder.transform(["Critical"])[0]],
            average="macro",
            zero_division=0,
        ),
        "high_recall": recall_score(
            y_test,
            y_pred,
            labels=[label_encoder.transform(["High"])[0]],
            average="macro",
            zero_division=0,
        ),
        "high_precision": precision_score(
            y_test,
            y_pred,
            labels=[label_encoder.transform(["High"])[0]],
            average="macro",
            zero_division=0,
        ),
    }

    (output_dir / f"{fold_name}_classification_report.txt").write_text(report)
    pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).to_csv(output_dir / f"{fold_name}_confusion_matrix.csv")

    return metrics, report, matrix


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train V2 XGBoost using grouped real-resident holdout folds. "
            "Synthetic rows are train-only in every fold."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--feature-manifest", default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-rows-per-resident", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--synth-weight", type=float, default=0.5)
    parser.add_argument("--high-class-weight", type=float, default=2.0)
    parser.add_argument(
        "--save-final-model",
        action="store_true",
        help="Train one final model on all real rows plus synthetic rows after CV.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset, parse_dates=["generated_at"])
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = 0
    df["is_synthetic"] = df["is_synthetic"].fillna(0).astype(int)

    features = load_features(args.feature_manifest, df)
    
    if not features:
        raise ValueError("No model features found.")

    real_df = df[df["is_synthetic"].eq(0)].copy()
    synthetic_df = df[df["is_synthetic"].eq(1)].copy()
    real_residents = sorted(real_df["resident_id"].dropna().astype(int).unique())

    test_folds = [
        [resident for resident in fold if resident in real_residents]
        for fold in DEFAULT_TEST_FOLDS
    ]
    test_folds = [fold for fold in test_folds if fold]

    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)

    fold_metrics = []

    print("Real residents:", real_residents)
    print("Synthetic rows:", len(synthetic_df))
    print("Model feature count:", len(features))
    print("Test folds:", test_folds)

    for fold_index, test_residents in enumerate(test_folds, start=1):
        fold_name = f"fold_{fold_index}"
        train_residents = [
            resident for resident in real_residents if resident not in test_residents
        ]
        validation_residents = choose_validation_residents(train_residents, fold_index)
        train_residents = [
            resident
            for resident in train_residents
            if resident not in validation_residents
        ]

        real_train_df = real_df[real_df["resident_id"].isin(train_residents)].copy()
        validation_df = real_df[real_df["resident_id"].isin(validation_residents)].copy()
        test_df = real_df[real_df["resident_id"].isin(test_residents)].copy()

        train_df = pd.concat([real_train_df, synthetic_df], ignore_index=True)
        train_df = resident_balanced_sample(
            train_df,
            max_rows_per_resident=args.max_rows_per_resident,
            seed=args.seed + fold_index,
        )

        zero_variance_features = [
            feature
            for feature in features
            if train_df[feature].nunique(dropna=False) <= 1
        ]
        fold_features = [
            feature for feature in features if feature not in zero_variance_features
        ]

        x_train = prepare_matrix(train_df, fold_features)
        y_train = label_encoder.transform(train_df["risk_label"])
        x_val = prepare_matrix(validation_df, fold_features)
        y_val = label_encoder.transform(validation_df["risk_label"])
        x_test = prepare_matrix(test_df, fold_features)
        y_test = label_encoder.transform(test_df["risk_label"])

        sample_weight = train_df.get(
            "class_weight",
            pd.Series(1.0, index=train_df.index),
        ).copy()
        sample_weight = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0)
        
        # Apply synthetic data weight penalty
        synth_mask = train_df["is_synthetic"] == 1
        sample_weight[synth_mask] = sample_weight[synth_mask] * args.synth_weight
        
        # Apply High class balancing multiplier
        high_mask = train_df["risk_label"] == "High"
        sample_weight[high_mask] = sample_weight[high_mask] * args.high_class_weight

        model = build_model(
            args.seed + fold_index,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            verbose=50,
        )

        metrics, report, matrix = evaluate_fold(
            fold_name,
            model,
            label_encoder,
            x_test,
            y_test,
            output_dir,
        )
        metrics.update(
            {
                "train_residents": ",".join(map(str, train_residents)),
                "validation_residents": ",".join(map(str, validation_residents)),
                "test_residents": ",".join(map(str, test_residents)),
                "real_train_rows": len(real_train_df),
                "synthetic_train_rows": len(synthetic_df),
                "validation_rows": len(validation_df),
                "test_rows": len(test_df),
                "feature_count": len(fold_features),
                "dropped_zero_variance_features": ",".join(zero_variance_features),
            }
        )
        fold_metrics.append(metrics)

        print("\n" + "=" * 80)
        print(f"{fold_name}")
        print("Train residents:", train_residents)
        print("Validation residents:", validation_residents)
        print("Test residents:", test_residents)
        print("Rows:", {
            "real_train": len(real_train_df),
            "synthetic_train": len(synthetic_df),
            "validation": len(validation_df),
            "test": len(test_df),
        })
        print(report)
        print(matrix)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_dir / "grouped_holdout_metrics.csv", index=False)

    summary_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "critical_recall",
        "high_recall",
        "high_precision",
    ]
    summary = metrics_df[summary_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(output_dir / "grouped_holdout_summary.csv")

    print("\nGrouped holdout summary:")
    print(summary)

    if args.save_final_model:
        final_train_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        final_train_df = resident_balanced_sample(
            final_train_df,
            max_rows_per_resident=args.max_rows_per_resident,
            seed=args.seed,
        )

        final_zero_variance_features = [
            feature
            for feature in features
            if final_train_df[feature].nunique(dropna=False) <= 1
        ]
        final_features = [
            feature
            for feature in features
            if feature not in final_zero_variance_features
        ]

        x_final = prepare_matrix(final_train_df, final_features)
        y_final = label_encoder.transform(final_train_df["risk_label"])
        sample_weight = final_train_df.get(
            "class_weight",
            pd.Series(1.0, index=final_train_df.index),
        ).copy()
        sample_weight = pd.to_numeric(sample_weight, errors="coerce").fillna(1.0)
        
        # Apply synthetic data weight penalty
        synth_mask = final_train_df["is_synthetic"] == 1
        sample_weight[synth_mask] = sample_weight[synth_mask] * args.synth_weight
        
        # Apply High class balancing multiplier
        high_mask = final_train_df["risk_label"] == "High"
        sample_weight[high_mask] = sample_weight[high_mask] * args.high_class_weight

        final_model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABEL_ORDER),
            eval_metric="mlogloss",
            n_estimators=300,
            learning_rate=0.04,
            max_depth=args.max_depth,
            min_child_weight=args.min_child_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=args.seed,
        )
        final_model.fit(x_final, y_final, sample_weight=sample_weight, verbose=False)

        joblib.dump(final_model, output_dir / "fall_risk_xgboost_v23_final.pkl")
        joblib.dump(label_encoder, output_dir / "risk_label_encoder_v23_final.pkl")
        joblib.dump(final_features, output_dir / "model_features_v23_final.pkl")
        pd.DataFrame({"feature": final_features}).to_csv(
            output_dir / "model_features_v23_final.csv",
            index=False,
        )
        print(f"\nSaved final model artifacts to {output_dir}")


if __name__ == "__main__":
    main()
