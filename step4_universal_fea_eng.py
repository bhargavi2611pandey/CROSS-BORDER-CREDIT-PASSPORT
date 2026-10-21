"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 4: UNIVERSAL FEATURE ENGINEERING (step4_universal_fea_eng.py)
=============================================================================
What this script does:
1. Loads the cleaned dataset from data/processed/lending_club_clean.csv.
2. Engineers universal, currency-independent financial ratios:
   - monthly_income: Annual income converted to monthly
   - lti_ratio: Loan-to-Income ratio (Loan / Annual Income)
   - monthly_debt_burden: Absolute monthly debt burden
   - free_cash_flow: Remaining disposable monthly income (safety buffer)
   - financial_stability_score: Composite index (0-100) from employment & housing
   - credit_risk_tier: Universal credit rating (Poor, Fair, Good, Excellent)
   - repayment_burden_ratio: Ratio of monthly debt payment to monthly income
   - disposable_income_ratio: Free cash flow relative to monthly income
   - fico_to_dti_ratio: Credit score scaled by debt burden
   - loan_to_monthly_income: Principal exposure measured in months of earnings
   - leverage_buffer: Buffer capacity combining credit rating and cash buffer
   - Risk purpose and home ownership indicators
3. Generates a formal Universal Feature Schema specification (JSON).
4. Saves the resulting passport-ready dataset into data/processed/universal_features.csv.
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np

# Bulletproof path resolution
current = os.path.abspath(os.path.dirname(__file__))
while current != os.path.dirname(current):
    if os.path.exists(os.path.join(current, "data")):
        PROJECT_ROOT = current
        break
    current = os.path.dirname(current)
else:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

INPUT_CLEAN_PATH = os.path.join(PROCESSED_DIR, "lending_club_clean.csv")
OUTPUT_UNIVERSAL_PATH = os.path.join(PROCESSED_DIR, "universal_features.csv")
SCHEMA_OUTPUT_PATH = os.path.join(REPORTS_DIR, "universal_feature_schema.json")


def compute_stability_score(row):
    """
    Computes a universal Financial Stability Score between 0 and 100.
    Based on employment history (up to 60 pts) and housing tenure (up to 40 pts).
    """
    years = float(row.get("emp_length_years", 0))
    emp_score = min(years * 6.0, 60.0)

    home = str(row.get("home_ownership", "rent")).lower()
    if "own" in home:
        house_score = 40.0
    elif "mortgage" in home:
        house_score = 30.0
    elif "rent" in home:
        house_score = 15.0
    else:
        house_score = 10.0

    return round(emp_score + house_score, 1)


def categorize_credit_score(fico):
    """Maps any numerical credit score into universal global credit bands."""
    if fico >= 750:
        return "Excellent"
    elif fico >= 700:
        return "Good"
    elif fico >= 650:
        return "Fair"
    else:
        return "Poor"


def engineer_universal_features():
    print("\n" + "=" * 70)
    print(" STEP 4: ENGINEERING UNIVERSAL CROSS-BORDER FEATURES (ENRICHED)")
    print("=" * 70)

    if not os.path.exists(INPUT_CLEAN_PATH):
        print(f" [!] Processed input file not found: {INPUT_CLEAN_PATH}")
        print("     Please ensure Step 3 completed successfully.")
        return

    print(" -> Loading cleaned dataset...")
    df = pd.read_csv(INPUT_CLEAN_PATH)
    print(f"    Loaded {len(df):,} records.")

    # 1. Universal Income Ratios
    print(" -> Calculating Monthly Income & Loan-to-Income (LTI) ratio...")
    df["monthly_income"] = (df["annual_income"] / 12.0).round(2)
    df["annual_income_safe"] = df["annual_income"].replace(0, 1.0)
    df["lti_ratio"] = (df["loan_amnt"] / df["annual_income_safe"]).clip(0, 5.0).round(4)

    # 2. Free Cash Flow & Monthly Debt Burden
    print(" -> Calculating Monthly Debt Burden & Free Cash Flow (Buffer)...")
    df["monthly_debt_burden"] = (df["monthly_income"] * (df["dti"] / 100.0)).round(2)
    df["free_cash_flow"] = (df["monthly_income"] - df["monthly_debt_burden"]).round(2)

    # 3. Non-Linear Interaction Financial Ratios (High-Signal Predictive Ratios)
    print(" -> Engineering Advanced Predictive Ratios (Burden, Leverage Buffer, Ratios)...")
    df["repayment_burden_ratio"] = (df["monthly_debt_burden"] / (df["monthly_income"] + 1.0)).clip(0, 2.0).round(4)
    df["disposable_income_ratio"] = (df["free_cash_flow"] / (df["monthly_income"] + 1.0)).clip(-1.0, 1.0).round(4)
    df["fico_to_dti_ratio"] = (df["fico_n"] / (df["dti"] + 1.0)).clip(0, 1000.0).round(4)
    df["loan_to_monthly_income"] = (df["loan_amnt"] / (df["monthly_income"] + 1.0)).clip(0, 60.0).round(4)
    df["leverage_buffer"] = (df["fico_n"] * (df["free_cash_flow"].clip(lower=0) / (df["loan_amnt"] + 1.0))).clip(0, 5000.0).round(4)

    # 4. Loan Purpose & Categorical Signals
    print(" -> Encoding Purpose Risk Signals & Housing Types...")
    high_risk_purposes = ["small_business", "moving", "renewable_energy", "house"]
    low_risk_purposes = ["car", "wedding", "major_purchase", "home_improvement"]
    
    df["loan_purpose_str"] = df["loan_purpose"].fillna("other").astype(str).str.lower()
    df["is_high_risk_purpose"] = df["loan_purpose_str"].isin(high_risk_purposes).astype(int)
    df["is_low_risk_purpose"] = df["loan_purpose_str"].isin(low_risk_purposes).astype(int)

    home_str = df["home_ownership"].fillna("rent").astype(str).str.lower()
    df["is_home_mortgage"] = (home_str == "mortgage").astype(int)
    df["is_home_own"] = (home_str == "own").astype(int)
    df["is_home_rent"] = (home_str == "rent").astype(int)

    # 5. Universal Financial Stability Index (0 to 100)
    print(" -> Calculating Universal Financial Stability Index (0 to 100)...")
    df["financial_stability_score"] = df.apply(compute_stability_score, axis=1)

    # 6. Universal Credit Risk Tier
    print(" -> Mapping credit scores into Universal Credit Tiers...")
    df["credit_risk_tier"] = df["fico_n"].apply(categorize_credit_score)

    # Clean up temporary column
    df = df.drop(columns=["annual_income_safe", "loan_purpose_str"])

    # 7. Save the Universal Dataset
    print(f" -> Saving Universal Features Dataset ({len(df):,} rows)...")
    df.to_csv(OUTPUT_UNIVERSAL_PATH, index=False)
    print(f" [OK] Saved successfully to:\n      {OUTPUT_UNIVERSAL_PATH}")

    # 8. Generate and save the Universal Schema Definition
    schema_spec = {
        "project": "Cross-Border Credit Passport System",
        "version": "2.0 (Enriched)",
        "description": "Standardized currency-invariant financial features for cross-border applicants.",
        "features": {
            "annual_income": "Total annual earnings in base currency.",
            "monthly_income": "Annual income divided by 12.",
            "loan_amnt": "Requested loan amount.",
            "dti": "Debt-to-Income ratio percentage (Monthly debt payments / Monthly income * 100).",
            "lti_ratio": "Loan-to-Income ratio (Requested loan amount / Annual income).",
            "monthly_debt_burden": "Total monthly debt payments in base currency.",
            "free_cash_flow": "Remaining uncommitted monthly disposable income (Safety buffer).",
            "repayment_burden_ratio": "Proportion of monthly income committed to servicing debt.",
            "disposable_income_ratio": "Proportion of uncommitted income available after existing debt.",
            "fico_to_dti_ratio": "Origin credit score relative to existing debt burden.",
            "loan_to_monthly_income": "Principal loan exposure measured in monthly income units.",
            "leverage_buffer": "Composite safety index combining creditworthiness and cash cushion.",
            "is_high_risk_purpose": "Binary indicator for historically higher-risk loan purposes.",
            "is_low_risk_purpose": "Binary indicator for prime/collateralized loan purposes.",
            "is_home_mortgage": "Binary indicator if applicant holds a residential mortgage.",
            "is_home_own": "Binary indicator if applicant owns residential property outright.",
            "is_home_rent": "Binary indicator if applicant is renting.",
            "emp_length_years": "Total years of documented employment.",
            "financial_stability_score": "Composite stability index (0 to 100) based on employment & housing.",
            "fico_n": "Standardized credit bureau score.",
            "credit_risk_tier": "Universal risk category: Poor (<650), Fair (650-699), Good (700-749), Excellent (750+).",
            "target": "Binary credit outcome (0 = Good / Paid, 1 = Risky / Defaulted)."
        }
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SCHEMA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_spec, f, indent=4)

    print(f" [OK] Universal Feature Schema Specification saved to:\n      {SCHEMA_OUTPUT_PATH}")

    print("\n -> Summary of Newly Engineered Universal Features:")
    summary_cols = [
        "lti_ratio", "free_cash_flow", "repayment_burden_ratio",
        "disposable_income_ratio", "fico_to_dti_ratio", "leverage_buffer"
    ]
    print(df[summary_cols].describe().round(2).to_string())


if __name__ == "__main__":
    engineer_universal_features()
