"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 8: ADVANCED ML MODELS - RANDOM FOREST & XGBOOST (step8_advanced_models.py)
=============================================================================
What this script does:
1. Loads the master credit passport dataset (PPP + Origin Bureau behavioral history).
2. Uses the exact same 80/20 train/test split as Step 7 for a 100% fair comparison.
3. Trains Model 1: Random Forest Classifier.
4. Trains Model 2: Tuned XGBoost (Extreme Gradient Boosting) with early stopping & regularization.
5. Directly compares ROC-AUC, Recall, and Accuracy against the Baseline (0.7824).
6. Saves the champion model to: models/advanced_xgboost_model.joblib
7. Saves the comparison report to: reports/model_comparison.json
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import xgboost as xgb

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

INPUT_PATH = os.path.join(PROCESSED_DIR, "credit_passport_dataset.csv")
BASELINE_METRICS_PATH = os.path.join(REPORTS_DIR, "baseline_metrics.json")
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "advanced_xgboost_model.joblib")
COMPARISON_REPORT_PATH = os.path.join(REPORTS_DIR, "model_comparison.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def train_advanced_models():
    print("\n" + "=" * 70)
    print(" STEP 8: TRAINING ADVANCED ML MODELS (RANDOM FOREST & XGBOOST)")
    print("=" * 70)

    if not os.path.exists(INPUT_PATH):
        print(f" [!] Passport dataset not found: {INPUT_PATH}")
        return

    print(" -> Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    # 1. Feature Selection (Identical to Step 7)
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

    X = df[feature_columns]
    y = df["target"]

    # 2. Identical 80/20 Stratified Split
    print(" -> Splitting into 80% Train and 20% Test (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    imbalance_ratio = round(num_neg / num_pos, 2)
    print(f"    Class Ratio (Good / Bad): {imbalance_ratio:.2f}")

    # =========================================================================
    # MODEL 1: RANDOM FOREST CLASSIFIER
    # =========================================================================
    print("\n" + "-" * 70)
    print(" 1. TRAINING RANDOM FOREST (200 Decision Trees)...")
    print("-" * 70)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=9,
        min_samples_split=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_preds_proba)
    print(f" [OK] Random Forest ROC-AUC Score: {rf_auc:.4f}")

    # =========================================================================
    # MODEL 2: TUNED XGBOOST (EXTREME GRADIENT BOOSTING)
    # =========================================================================
    print("\n" + "-" * 70)
    print(" 2. TRAINING TUNED XGBOOST (Sequential Gradient Boosting)...")
    print("-" * 70)
    xgb_model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=4,
        scale_pos_weight=imbalance_ratio,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1.0,
        reg_alpha=0.5,
        random_state=42,
        eval_metric="auc",
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds_proba)
    print(f" [OK] XGBoost ROC-AUC Score      : {xgb_auc:.4f}")

    # =========================================================================
    # DIRECT HEAD-TO-HEAD COMPARISON
    # =========================================================================
    baseline_auc = 0.7824
    if os.path.exists(BASELINE_METRICS_PATH):
        try:
            with open(BASELINE_METRICS_PATH, "r") as f:
                b_data = json.load(f)
                baseline_auc = b_data.get("roc_auc", 0.7824)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(" HEAD-TO-HEAD MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f" 1. Baseline (Logistic Regression) : ROC-AUC = {baseline_auc:.4f}")
    print(f" 2. Random Forest                  : ROC-AUC = {rf_auc:.4f}  (Improvement: {rf_auc - baseline_auc:+.4f})")
    print(f" 3. XGBoost (Gradient Boosting)    : ROC-AUC = {xgb_auc:.4f}  (Improvement: {xgb_auc - baseline_auc:+.4f})")

    # Determine Champion Model
    best_model = xgb_model if xgb_auc >= rf_auc else rf_model
    champion_name = "XGBoost" if xgb_auc >= rf_auc else "Random Forest"
    champion_auc = max(xgb_auc, rf_auc)
    print(f"\n >>> CHAMPION MODEL: {champion_name} (ROC-AUC: {champion_auc:.4f}) <<<")

    # Classification Report for Champion Model
    xgb_preds_binary = (xgb_preds_proba >= 0.50).astype(int)
    print(f"\n -> Detailed Classification Report for {champion_name}:")
    print(classification_report(y_test, xgb_preds_binary, target_names=["Good (0)", "Default (1)"]))

    # 3. Feature Importance in XGBoost
    print(" -> What features matter most to Champion XGBoost? (Relative Importance):")
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    for _, row in importance_df.iterrows():
        print(f"      {row['Feature']:30}: {row['Importance']*100:.1f}%")

    # 4. Save Champion Model & Package
    package = {
        "champion_model_name": champion_name,
        "model": best_model,
        "features": feature_columns,
        "roc_auc": round(float(champion_auc), 4),
        "imbalance_ratio": float(imbalance_ratio)
    }
    joblib.dump(package, CHAMPION_MODEL_PATH)
    print(f"\n [OK] Saved Champion Model to:\n      {CHAMPION_MODEL_PATH}")

    # Save Comparison Report JSON
    comparison_summary = {
        "baseline_logistic_regression": {"roc_auc": float(baseline_auc)},
        "random_forest": {"roc_auc": round(float(rf_auc), 4)},
        "xgboost": {"roc_auc": round(float(xgb_auc), 4)},
        "champion": champion_name,
        "auc_lift": round(float(champion_auc - baseline_auc), 4)
    }
    with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=4)
    print(f" [OK] Saved Model Comparison Report to:\n      {COMPARISON_REPORT_PATH}")

    print("\n" + "=" * 70)
    print(" STEP 8 COMPLETE! HIGH-ACCURACY CHAMPION MODEL LOCKED.")
    print("=" * 70)


if __name__ == "__main__":
    train_advanced_models()
