"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 9: MODEL EVALUATION & OPTIMAL THRESHOLD ANALYSIS (step9_model_evaluation.py)
=============================================================================
What this script does:
1. Loads the champion trained model (models/advanced_xgboost_model.joblib).
2. Generates test set default probabilities.
3. Computes and plots the ROC Curve and Precision-Recall Curve.
4. Performs Financial Cost-Benefit Optimization:
   - Loss per Default (False Negative) = $10,000
   - Loss per Rejected Good Applicant (False Positive) = $1,000
5. Finds the optimal business threshold that minimizes bank losses.
6. Saves decision graphs into reports/visualizations/
7. Saves the optimal threshold configuration into reports/optimal_threshold_config.json
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix
)

# Set clean visualization style
sns.set_theme(style="whitegrid")
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
THRESHOLD_CONFIG_PATH = os.path.join(REPORTS_DIR, "optimal_threshold_config.json")

# Business Cost Assumptions
COST_OF_DEFAULT = 10000.0       # Direct loss when a borrower defaults (Unpaid Principal)
COST_OF_REJECTION = 1000.0      # Opportunity loss when turning away a good borrower (Lost Interest)


def run_evaluation_and_threshold_analysis():
    print("\n" + "=" * 70)
    print(" STEP 9: MODEL EVALUATION & FINANCIAL THRESHOLD OPTIMIZATION")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(INPUT_PATH):
        print(f" [!] Model or dataset missing. Please run Step 6 and Step 8 first.")
        return

    # 1. Load Model & Features
    print(" -> Loading Champion Model and test data...")
    pkg = joblib.load(MODEL_PATH)
    model = pkg["model"]
    feature_cols = pkg["features"]
    champion_name = pkg.get("champion_model_name", "Champion Model")

    df = pd.read_csv(INPUT_PATH)
    X = df[feature_cols]
    y = df["target"]

    # Recreate the exact 20,000 test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Get Predicted Probabilities of Default
    y_prob = model.predict_proba(X_test)[:, 1]

    # =========================================================================
    # A. PLOT ROC & PRECISION-RECALL CURVES
    # =========================================================================
    print(" -> Generating Graph: ROC Curve & Precision-Recall Curve...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f'{champion_name} (AUC = {roc_auc_val:.3f})')
    axes[0].plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Random Guess (0.50)")
    axes[0].set_title(f"ROC Curve ({champion_name})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate (Good applicants falsely flagged)")
    axes[0].set_ylabel("True Positive Rate (Defaulters caught)")
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc_val = average_precision_score(y_test, y_prob)
    axes[1].plot(recall, precision, color="#8e44ad", lw=2.5, label=f'PR-AUC = {pr_auc_val:.3f}')
    axes[1].set_title("Precision-Recall Curve (Critical for Imbalance)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Recall (Fraction of all defaulters caught)")
    axes[1].set_ylabel("Precision (Accuracy when flagging a default)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    roc_pr_save_path = os.path.join(OUTPUT_VIZ_DIR, "06_roc_and_pr_curves.png")
    plt.savefig(roc_pr_save_path, dpi=300)
    plt.close()
    print(f" [OK] Saved curves to:\n      {roc_pr_save_path}")

    # =========================================================================
    # B. FINANCIAL THRESHOLD OPTIMIZATION (Minimizing Dollar Loss)
    # =========================================================================
    print("\n -> Performing Financial Cost-Benefit Optimization...")
    print(f"    Assumed Cost per Default (Unpaid Loan) : ${COST_OF_DEFAULT:,.0f}")
    print(f"    Assumed Lost Profit per Good Rejection : ${COST_OF_REJECTION:,.0f}")

    threshold_candidates = np.linspace(0.10, 0.80, 71)
    cost_records = []

    for thresh in threshold_candidates:
        # If predicted probability >= thresh -> REJECT loan (flagged as risky)
        # If predicted probability < thresh  -> APPROVE loan (considered safe)
        y_pred_binary = (y_prob >= thresh).astype(int)
        
        # Confusion matrix:
        # tn: Good applicant approved (Great!)
        # fp: Good applicant rejected (Lost profit: $1,000)
        # fn: Defaulted applicant approved (Default loss: $10,000!)
        # tp: Defaulted applicant rejected (Saved money!)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()

        total_dollar_cost = (fn * COST_OF_DEFAULT) + (fp * COST_OF_REJECTION)
        approval_rate = (tn + fn) / len(y_test) * 100.0
        
        # Portfolio default rate among approved applicants
        approved_defaults = fn
        total_approved = tn + fn
        portfolio_bad_rate = (approved_defaults / total_approved * 100.0) if total_approved > 0 else 0.0

        cost_records.append({
            "threshold": round(thresh, 3),
            "total_dollar_cost_millions": round(total_dollar_cost / 1e6, 3),
            "approval_rate": round(approval_rate, 1),
            "portfolio_bad_rate": round(portfolio_bad_rate, 1),
            "defaulters_caught_tp": int(tp),
            "defaulters_missed_fn": int(fn),
            "good_rejected_fp": int(fp)
        })

    cost_df = pd.DataFrame(cost_records)

    # Find the threshold with the absolute lowest dollar loss
    best_idx = cost_df["total_dollar_cost_millions"].idxmin()
    best_row = cost_df.loc[best_idx]
    optimal_thresh = best_row["threshold"]

    # Also compare with the naive 0.50 threshold
    naive_50_row = cost_df.iloc[(cost_df['threshold'] - 0.50).abs().argsort()[:1]].iloc[0]

    # Plot Cost Curve
    plt.figure(figsize=(10, 5))
    plt.plot(cost_df["threshold"], cost_df["total_dollar_cost_millions"], color="#c0392b", lw=2.5, label="Total Bank Loss ($ Millions)")
    plt.axvline(optimal_thresh, color="#27ae60", linestyle="--", lw=2, label=f'Optimal Cutoff ({optimal_thresh:.2f})')
    plt.axvline(0.50, color="#7f8c8d", linestyle=":", lw=2, label='Naive 0.50 Cutoff')
    plt.title("Financial Cost Curve: Finding the Profit-Maximizing Underwriting Cutoff", fontsize=13, fontweight='bold')
    plt.xlabel("Risk Cutoff Threshold (Reject if Probability >= Cutoff)")
    plt.ylabel("Total Loss ($ Millions)")
    plt.legend()
    plt.tight_layout()

    cost_save_path = os.path.join(OUTPUT_VIZ_DIR, "07_cost_benefit_threshold.png")
    plt.savefig(cost_save_path, dpi=300)
    plt.close()
    print(f" [OK] Saved Cost-Benefit Curve to:\n      {cost_save_path}")

    # =========================================================================
    # C. PRINT SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FINANCIAL UNDERWRITING THRESHOLD FINDINGS")
    print("=" * 70)
    print(f" 1. Naive Cutoff (0.50)   : Total Loss = ${naive_50_row['total_dollar_cost_millions']}M | Approval Rate = {naive_50_row['approval_rate']}% | Bad Rate = {naive_50_row['portfolio_bad_rate']}%")
    print(f" 2. OPTIMAL CUTOFF ({optimal_thresh:.2f}) : Total Loss = ${best_row['total_dollar_cost_millions']}M | Approval Rate = {best_row['approval_rate']}% | Bad Rate = {best_row['portfolio_bad_rate']}%")
    
    savings = (naive_50_row['total_dollar_cost_millions'] - best_row['total_dollar_cost_millions'])
    print(f"\n >>> BY USING THE OPTIMAL THRESHOLD ({optimal_thresh:.2f}), THE BANK SAVES ${savings*1000:,.0f} THOUSAND DOLLARS! <<<")

    # Save threshold config for Step 12 (API)
    threshold_config = {
        "optimal_threshold": float(optimal_thresh),
        "naive_threshold": 0.50,
        "cost_assumptions": {
            "default_loss": COST_OF_DEFAULT,
            "rejection_cost": COST_OF_REJECTION
        },
        "optimal_metrics": {
            "approval_rate_pct": float(best_row["approval_rate"]),
            "portfolio_default_rate_pct": float(best_row["portfolio_bad_rate"]),
            "dollar_loss_millions": float(best_row["total_dollar_cost_millions"])
        }
    }

    with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=4)
    print(f"\n [OK] Saved Optimal Threshold Configuration to:\n      {THRESHOLD_CONFIG_PATH}")

    print("\n" + "=" * 70)
    print(" STEP 9 COMPLETE! OPTIMAL UNDERWRITING THRESHOLD LOCKED.")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation_and_threshold_analysis()