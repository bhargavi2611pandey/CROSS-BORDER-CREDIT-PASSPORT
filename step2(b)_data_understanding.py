"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 2(B): DATA UNDERSTANDING (02_data_understanding.py)
=============================================================================
What this script does:
1. Analyzes the target variables (Good vs Bad) across datasets.
2. Checks numeric distributions (median, percentiles, min, max).
3. Identifies extreme outliers in Income, Loan Amount, and DTI.
4. Detects special coded values (like FICO's -7, -8, -9).
5. Computes correlation between credit features and default risk.
=============================================================================
"""

import os
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

FICO_PATH = os.path.join(DATA_DIR, "fico", "heloc_dataset_v1.csv")
LC_PATH = os.path.join(DATA_DIR, "lending_club", "LC_loans_granting_model_dataset.csv")
HC_PATH = os.path.join(DATA_DIR, "home_credit", "application_train.csv")


def understand_fico():
    print("\n" + "=" * 70)
    print(" 1. UNDERSTANDING FICO HELOC (Credit Bureau Metrics & Special Codes)")
    print("=" * 70)
    
    if not os.path.exists(FICO_PATH):
        print(f" [!] FICO file not found at: {FICO_PATH}")
        return

    df = pd.read_csv(FICO_PATH)
    
    # 1. Target check
    target_counts = df["RiskPerformance"].value_counts()
    print(" -> Target ('RiskPerformance'):")
    print(f"      'Bad' (90+ days past due): {target_counts.get('Bad', 0):,} ({target_counts.get('Bad', 0)/len(df)*100:.1f}%)")
    print(f"      'Good' (Paid as agreed) : {target_counts.get('Good', 0):,} ({target_counts.get('Good', 0)/len(df)*100:.1f}%)")
    
    # 2. Investigate the famous FICO negative special codes (-7, -8, -9)
    print("\n -> Investigating Special Negative Bureau Codes (-7, -8, -9):")
    special_codes = [-7, -8, -9]
    sample_cols = ["ExternalRiskEstimate", "MSinceOldestTradeOpen", "MSinceMostRecentDelq", "NetFractionRevolvingBurden"]
    
    for col in sample_cols:
        if col in df.columns:
            counts_neg = df[col].isin(special_codes).sum()
            pct_neg = (counts_neg / len(df)) * 100
            print(f"      {col}: {counts_neg:,} rows ({pct_neg:.1f}%) contain special codes (-7/-8/-9)")
            
    # 3. Numeric statistics on valid positive numbers
    print("\n -> Distribution of Key Credit Features (Excluding negative codes):")
    for col in ["ExternalRiskEstimate", "NumSatisfactoryTrades"]:
        if col in df.columns:
            valid_data = df[df[col] > 0][col]
            print(f"      {col}: Min={valid_data.min()}, Median={valid_data.median()}, Max={valid_data.max()}")


def understand_lending_club():
    print("\n" + "=" * 70)
    print(" 2. UNDERSTANDING LENDING CLUB (Income, DTI & Loan Outcomes)")
    print("=" * 70)
    
    if not os.path.exists(LC_PATH):
        print(f" [!] Lending Club file not found at: {LC_PATH}")
        return

    # Load key columns to be fast and memory-efficient
    cols_to_read = ["revenue", "loan_amnt", "dti_n", "fico_n", "emp_length", "home_ownership_n", "Default"]
    df = pd.read_csv(LC_PATH, usecols=cols_to_read, low_memory=False)

    print(f" -> Total Loans Analyzed: {len(df):,}")

    # 1. Target check
    default_counts = df["Default"].value_counts()
    print(" -> Target Variable ('Default'):")
    print(f"      0 (Fully Paid): {default_counts.get(0, 0):,} ({default_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"      1 (Defaulted) : {default_counts.get(1, 0):,} ({default_counts.get(1, 0)/len(df)*100:.1f}%)")

    # 2. Financial Metrics Distribution
    print("\n -> Financial Characteristics of Borrowers:")
    for col, name in [("revenue", "Annual Income ($)"), ("loan_amnt", "Loan Requested ($)"), 
                      ("dti_n", "Debt-to-Income Ratio (%)"), ("fico_n", "Average FICO Score")]:
        series = df[col].dropna()
        q05 = series.quantile(0.05)
        median = series.quantile(0.50)
        q95 = series.quantile(0.95)
        max_val = series.max()
        print(f"      {name:25}: 5th%=${q05:,.0f} | Median=${median:,.0f} | 95th%=${q95:,.0f} | Max=${max_val:,.0f}")

    # 3. How does FICO score affect default rate?
    print("\n -> Risk Relationship: Does a higher FICO score mean fewer defaults?")
    df["fico_bucket"] = pd.cut(df["fico_n"], bins=[600, 660, 700, 750, 850], 
                               labels=["Poor (<660)", "Fair (660-700)", "Good (700-750)", "Excellent (750+)"])
    risk_by_fico = df.groupby("fico_bucket", observed=False)["Default"].mean() * 100
    for bucket, default_rate in risk_by_fico.items():
        print(f"      FICO {bucket:16}: Default Rate = {default_rate:.1f}%")

    # 4. Employment & Housing
    print("\n -> Top Home Ownership Status:")
    for home_type, count in df["home_ownership_n"].value_counts().head(3).items():
        print(f"      {str(home_type).title()}: {count:,} ({count/len(df)*100:.1f}%)")


def understand_home_credit():
    print("\n" + "=" * 70)
    print(" 3. UNDERSTANDING HOME CREDIT (Origin / Emerging Market Profile)")
    print("=" * 70)
    
    if not os.path.exists(HC_PATH):
        print(f" [!] Home Credit file not yet downloaded at: {HC_PATH}")
        print("     (You can run with FICO & Lending Club now, and add Home Credit anytime!)")
        return

    df = pd.read_csv(HC_PATH, nrows=50000, low_memory=False)
    print(f" -> Analyzed sample of {len(df):,} records from application_train.csv")
    
    # Target
    target_counts = df["TARGET"].value_counts()
    print(" -> Target ('TARGET'):")
    print(f"      0 (Paid on time)     : {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"      1 (Payment trouble)  : {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Financial fields
    for col, name in [("AMT_INCOME_TOTAL", "Annual Income"), ("AMT_CREDIT", "Credit Amount"), ("AMT_ANNUITY", "Monthly Payment")]:
        if col in df.columns:
            s = df[col].dropna()
            print(f"      {name:20}: Median={s.median():,.0f} | 95th%={s.quantile(0.95):,.0f} | Max={s.max():,.0f}")


def main():
    print("\n" + "#" * 70)
    print(" STEP 2: DEEP DIVE DATA UNDERSTANDING")
    print("#" * 70)
    
    understand_fico()
    understand_lending_club()
    understand_home_credit()
    
    print("\n" + "=" * 70)
    print(" STEP 2 UNDERSTANDING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()