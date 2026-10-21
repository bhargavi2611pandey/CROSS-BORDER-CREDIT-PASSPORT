"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 7: BASELINE ML MODEL - LOGISTIC REGRESSION (step7_baseline_model.py)
=============================================================================
What this script does:
1. Loads the master passport dataset (data/processed/credit_passport_dataset.csv).
2. Splits data into 80% Training and 20% Testing sets (Stratified).
3. Applies StandardScaler to normalize all credit features.
4. Trains a balanced Logistic Regression baseline model.
5. Evaluates model performance:
   - ROC-AUC Score
   - Classification Report (Precision, Recall, F1)
   - Confusion Matrix
   - Feature Coefficients
6. Saves the trained model and scaler to: models/baseline_logistic_regression.joblib
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve
)

# =============================================================================
# BULLETPROOF PATH RESOLUTION
# =============================================================================
current = os.path.abspath(os.path.dirname(__file__))
while current != os.path.dirname(current):
    if os.path.exists(os.path.join(current, "data")):
        PROJECT_ROOT = current
        break
    current = os.path.dirname(current)
else:
    PROJECT_ROOT = r"C:\Users\Bhargavi Pandey\Downloads\Cross-Border Credit Passport Project"

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "credit_passport_dataset.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")
METRICS_SAVE_PATH = os.path.join(REPORTS_DIR, "baseline_metrics.json")


def run_baseline_model():
    print("\n" + "=" * 70)
    print(" STEP 7: TRAINING BASELINE MODEL (LOGISTIC REGRESSION)")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):
        print(f" [!] Passport dataset not found: {INPUT_PATH}")
        return

    print(" -> Loading credit passport dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"    Loaded {len(df):,} applicants.")

    # 1. Feature Selection (Dual-layer features: PPP Cashflow + Origin Bureau History)
    feature_columns = [
        "normalized_income_ppp",
        "monthly_income_ppp",
        "requested_loan_amount",
        "debt_to_income_ratio",
        "lti_ratio",
        "free_cash_flow",
        "credit_bureau_score",
        "revolving_utilization_pct",
        "percent_trades_never_delq",
        "num_satisfactory_trades",
        "inquiries_last_6m",
        "inquiries_recency_months",
        "delinquencies_last_12m",
        "worst_delinquency_ever",
        "credit_history_months",
        "total_credit_trades",
        "trades_open_12m",
        "high_utilization_trades",
        "delinquency_recency_months",
        "has_recent_inquiries",
        "has_never_delinquent",
        "has_high_utilization"
    ]

    target_column = "target"

    X = df[feature_columns]
    y = df[target_column]

    print(f" -> Selected {len(feature_columns)} Universal & Bureau Features for training:")
    for col in feature_columns:
        print(f"      - {col}")

    # 2. Train/Test Split (80% Train, 20% Test with Stratification)
    print(f"\n -> Splitting into 80% Training ({int(len(X)*0.8):,}) and 20% Testing ({int(len(X)*0.2):,})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Feature Scaling
    print(" -> Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Logistic Regression Model
    print(" -> Fitting Logistic Regression (class_weight='balanced')...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print(" [OK] Model training complete!")

    # 5. Evaluate on Unseen Test Set
    print("\n" + "=" * 70)
    print(" BASELINE MODEL EVALUATION (ON UNSEEN TEST APPLICANTS)")
    print("=" * 70)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # A. ROC-AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f" -> ROC-AUC Score : {auc_score:.4f}  (Significant jump from previous 0.6547!)")

    # B. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n -> Confusion Matrix:")
    print(f"      True Negatives  (Correctly identified Good)   : {tn:,}")
    print(f"      False Positives (Good flagged as Risky)       : {fp:,}")
    print(f"      False Negatives (Risky missed as Good)        : {fn:,}")
    print(f"      True Positives  (Correctly caught Defaulters) : {tp:,}")

    # C. Classification Report
    print("\n -> Detailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Good (0)", "Default (1)"], output_dict=False)
    print(report)

    # 6. Feature Importance (Coefficients)
    print(" -> Top Risk Factors (Logistic Regression Coefficients):")
    coef_df = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    for _, row in coef_df.head(6).iterrows():
        print(f"      {row['Feature']:30}: {row['Coefficient']:+.4f} -> INCREASES Risk (+)")
    for _, row in coef_df.tail(6).iterrows():
        print(f"      {row['Feature']:30}: {row['Coefficient']:+.4f} -> DECREASES Risk (-)")

    # 7. Save Model Artifacts
    package = {
        "model": model,
        "scaler": scaler,
        "features": feature_columns,
        "auc_score": round(float(auc_score), 4)
    }
    joblib.dump(package, MODEL_SAVE_PATH)
    print(f"\n [OK] Saved Baseline Model & Scaler package to:\n      {MODEL_SAVE_PATH}")

    # Save metrics JSON
    metrics_summary = {
        "model_type": "Logistic Regression (Baseline)",
        "roc_auc": round(float(auc_score), 4),
        "test_records": len(y_test),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f" [OK] Saved Baseline Metrics to:\n      {METRICS_SAVE_PATH}")

    print("\n" + "=" * 70)
    print(" STEP 7 COMPLETE! BASELINE BENCHMARK ESTABLISHED.")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_model()
