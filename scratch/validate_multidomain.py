import sys
import os
import pandas as pd
import numpy as np
import joblib
import json

def run_validation():
    print("--- Running Multi-Domain Validation Test ---")
    
    model_dir = "models_v2_grouped_real_holdout"
    model = joblib.load(os.path.join(model_dir, "fall_risk_xgboost_v23_final.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "risk_label_encoder_v23_final.pkl"))
    feature_columns = joblib.load(os.path.join(model_dir, "model_features_v23_final.pkl"))
    
    def create_base_row():
        row = {col: 0.0 for col in feature_columns}
        # Set some neutral baselines
        row['activity_ratio'] = 1.0
        row['sleep_ratio'] = 1.0
        return row
        
    row_a = create_base_row()
    row_a['activity_ratio'] = 0.2
    row_a['abnormal_domain_count'] = 1
    row_a['severe_domain_count'] = 1
    
    row_b = create_base_row()
    row_b['activity_ratio'] = 0.2
    row_b['sleep_ratio'] = 0.4
    row_b['abnormal_domain_count'] = 2
    row_b['severe_domain_count'] = 2
    
    row_c = create_base_row()
    row_c['activity_ratio'] = 0.2
    row_c['sleep_ratio'] = 0.4
    # Instead of 'deviation' concepts, let's just trigger the domains
    row_c['abnormal_domain_count'] = 5
    row_c['severe_domain_count'] = 5
    
    # We also need to manipulate specific features the model relies on
    # A massive HR spike and SpO2 drop:
    row_c['avg_hr'] = 120.0
    row_c['avg_spo2'] = 92.0
    
    df = pd.DataFrame([row_a, row_b, row_c])
    probs = model.predict_proba(df)
    classes = label_encoder.classes_
    
    scenarios = [
        "A (Mobility \u2193 | Everything else normal)",
        "B (Mobility \u2193 | Sleep \u2193)",
        "C (Mobility \u2193 | Sleep \u2193 | HR \u2191 | HRV \u2193 | SpO2 \u2193)"
    ]
    
    results = []
    
    for i in range(3):
        prob_dict = {classes[j]: float(probs[i][j]) for j in range(len(classes))}
        risk_score = (prob_dict.get('Low', 0) * 12.5) + (prob_dict.get('Moderate', 0) * 37.5) + (prob_dict.get('High', 0) * 62.5) + (prob_dict.get('Critical', 0) * 87.5)
        label = max(prob_dict, key=prob_dict.get)
        
        results.append({
            "scenarioName": scenarios[i],
            "predictedRiskLabel": label,
            "riskScore": round(risk_score, 1),
            "probabilities": {k: round(v, 2) for k, v in prob_dict.items()},
            "keyFeaturesSupplied": {
                "activity_ratio": df.iloc[i]['activity_ratio'],
                "sleep_ratio": df.iloc[i]['sleep_ratio'],
                "abnormal_domain_count": df.iloc[i]['abnormal_domain_count'],
                "severe_domain_count": df.iloc[i]['severe_domain_count']
            }
        })
        
    output_file = "multidomain_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully ran scenarios and saved output to {output_file}")

if __name__ == "__main__":
    run_validation()
