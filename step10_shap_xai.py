"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 10: EXPLAINABLE AI (XAI) WITH SHAP (step10_shap_xai.py)
=============================================================================
What this script does:
1. Loads the champion model (models/advanced_xgboost_model.joblib).
2. Uses TreeExplainer to compute exact SHAP values for credit applicants.
3. Generates Global Interpretability Plots:
   - SHAP Beeswarm Summary Plot (how features drive risk up or down)
   - SHAP Bar Plot (overall feature importance ranking)
4. Generates Local Interpretability (Individual Passport Explanation):
   - Waterfall Plot for an individual cross-border applicant
   - Automatically generates legal 'Adverse Action' reason codes
5. Saves all visual explanation artifacts to: reports/visualizations/
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Set clean font style
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

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
OUTPUT_VIZ_DIR = os.path.join(REPORTS_DIR, "visualizations")

os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "credit_passport_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "advanced_xgboost_model.joblib")


def run_shap_explainability():
    print("\n" + "=" * 70)
    print(" STEP 10: EXPLAINABLE AI (SHAP) & REGULATORY REASON ENGINE")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(INPUT_PATH):
        print(f" [!] Model or dataset missing. Please run Step 6 and Step 8 first.")
        return

    # 1. Load Model & Features
    print(" -> Loading Champion Model and credit dataset...")
    pkg = joblib.load(MODEL_PATH)
    model = pkg["model"]
    feature_cols = pkg["features"]

    df = pd.read_csv(INPUT_PATH)
    X = df[feature_cols]

    # Sample 2,000 applicants for fast, high-precision SHAP computation
    print(" -> Sampling 2,000 applicants for SHAP TreeExplainer...")
    X_sample = X.sample(2000, random_state=42)

    # 2. Initialize TreeExplainer
    print(" -> Computing exact game-theoretic Shapley values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # =========================================================================
    # A. GLOBAL EXPLANATIONS: BEESWARM SUMMARY PLOT
    # =========================================================================
    print(" -> Generating Graph: SHAP Beeswarm Summary Plot...")
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Summary: How Financial Features Drive Credit Risk", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    beeswarm_path = os.path.join(OUTPUT_VIZ_DIR, "08_shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Saved Beeswarm Plot to:\n      {beeswarm_path}")

    # =========================================================================
    # B. LOCAL EXPLANATION: INDIVIDUAL PASSPORT WATERFALL PLOT
    # =========================================================================
    print(" -> Generating Graph: Individual Applicant SHAP Waterfall Plot...")
    
    # Pick a sample applicant (e.g. Applicant #0 in the sample)
    applicant_idx = 0
    applicant_data = X_sample.iloc[applicant_idx]
    applicant_shap = shap_values[applicant_idx]
    
    # Predict their exact default probability
    pred_prob = model.predict_proba(X_sample.iloc[[applicant_idx]])[0, 1]

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(applicant_shap, max_display=9, show=False)
    plt.title(f"Credit Passport Decision Breakdown (Risk Probability: {pred_prob*100:.1f}%)", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()

    waterfall_path = os.path.join(OUTPUT_VIZ_DIR, "09_shap_individual_waterfall.png")
    plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" [OK] Saved Waterfall Plot to:\n      {waterfall_path}")

    # =========================================================================
    # C. AUTOMATED ADVERSE ACTION REASON GENERATOR
    # =========================================================================
    print("\n" + "=" * 70)
    print(" DEMO: AUTOMATED REGULATORY 'ADVERSE ACTION' REPORT")
    print("=" * 70)
    print(f" Applicant Profile:")
    for col, val in applicant_data.items():
        print(f"      {col:26}: {val:,.2f}")
    
    print(f"\n Predicted Risk of Default : {pred_prob*100:.1f}%")
    print(f" Credit Passport Risk Score: {round((1.0 - pred_prob) * 100):.0f} / 100")

    # Deconstruct individual feature contributions
    feature_contributions = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP_Impact": applicant_shap.values,
        "Actual_Value": applicant_data.values
    }).sort_values(by="SHAP_Impact", ascending=False)

    # Top factors increasing risk (Positive SHAP = Higher risk)
    top_risk_drivers = feature_contributions[feature_contributions["SHAP_Impact"] > 0].head(3)
    # Top factors decreasing risk (Negative SHAP = Safer)
    top_strengths = feature_contributions[feature_contributions["SHAP_Impact"] < 0].tail(3)

    print("\n -> TOP RISK DRIVERS (Regulatory Adverse Action Reasons):")
    if len(top_risk_drivers) > 0:
        for i, (_, row) in enumerate(top_risk_drivers.iterrows(), 1):
            print(f"      {i}. {row['Feature']} ({row['Actual_Value']:,.2f}) pushed risk UP by +{row['SHAP_Impact']:.3f} points")
    else:
        print("      No major risk drivers detected (Applicant is very safe).")

    print("\n -> TOP APPLICANT STRENGTHS (Credit Positive Factors):")
    for i, (_, row) in enumerate(top_strengths.iterrows(), 1):
        print(f"      {i}. {row['Feature']} ({row['Actual_Value']:,.2f}) pulled risk DOWN by {row['SHAP_Impact']:.3f} points")

    print("\n" + "=" * 70)
    print(" STEP 10 COMPLETE! EXPLAINABILITY ENGINE OPERATIONAL.")
    print("=" * 70)


if __name__ == "__main__":
    run_shap_explainability()