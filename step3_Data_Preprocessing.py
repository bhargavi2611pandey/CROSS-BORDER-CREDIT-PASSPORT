"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 3: DATA PREPROCESSING (03_data_preprocessing.py)
=============================================================================
What this script does:
1. Cleans FICO HELOC:
   - Maps 'Good' -> 0 and 'Bad' -> 1
   - Converts special negative codes (-7, -8, -9) into clean features and flags
2. Cleans Lending Club:
   - Drops non-predictive/empty columns ('desc', 'id')
   - Converts 'emp_length' strings into numeric integers
   - Caps income and DTI extreme outliers (Winsorization)
   - Imputes any remaining missing values with medians
3. Saves the clean datasets into: data/processed/
=============================================================================
"""

import os
import re
import pandas as pd
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

FICO_RAW_PATH = os.path.join(DATA_DIR, "fico", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\heloc_dataset_v1.csv")
LC_RAW_PATH = os.path.join(DATA_DIR, "lending_club", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\LC_loans_granting_model_dataset.csv")

FICO_CLEAN_PATH = os.path.join(PROCESSED_DIR, "fico_clean.csv")
LC_CLEAN_PATH = os.path.join(PROCESSED_DIR, "lending_club_clean.csv")


def preprocess_fico():
    print("\n" + "=" * 65)
    print(" 1. PREPROCESSING FICO HELOC DATASET")
    print("=" * 65)

    if not os.path.exists(FICO_RAW_PATH):
        print(f" [!] FICO raw file not found at: {FICO_RAW_PATH}")
        return

    df = pd.read_csv(FICO_RAW_PATH)
    print(f" -> Raw shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 1. Target Standardization: 'Good' -> 0 (Safe), 'Bad' -> 1 (Default/Risk)
    target_mapping = {"Good": 0, "Bad": 1}
    df["target"] = df["RiskPerformance"].map(target_mapping)
    df = df.drop(columns=["RiskPerformance"])
    print(" -> Standardized target: 'target' (0 = Good, 1 = Bad)")

    # 2. Handle Special Negative Codes (-7, -8, -9)
    # Feature engineering: create a flag for customers who have NEVER had a delinquency (-8 in delinquency column)
    if "MSinceMostRecentDelq" in df.columns:
        df["has_never_delinquent"] = (df["MSinceMostRecentDelq"] == -8).astype(int)
        # Replace -8 with 999 (meaning virtually infinite months since any late payment)
        df["MSinceMostRecentDelq"] = df["MSinceMostRecentDelq"].replace(-8, 999)
        print(" -> Created 'has_never_delinquent' flag and replaced -8 with 999 in delinquency.")

    # In inquiry column: -7 means condition not met / no inquiries
    if "MSinceMostRecentInqexcl7days" in df.columns:
        df["has_no_inquiries"] = (df["MSinceMostRecentInqexcl7days"] == -7).astype(int)
        df["MSinceMostRecentInqexcl7days"] = df["MSinceMostRecentInqexcl7days"].replace([-7, -8], 999)
        print(" -> Created 'has_no_inquiries' flag and replaced -7/-8 with 999.")

    # Replace any remaining negative codes across all other columns with 0
    numeric_cols = [c for c in df.columns if c != "target"]
    for c in numeric_cols:
        df[c] = df[c].apply(lambda x: 0 if x in [-7, -8, -9] else x)

    # Save cleaned FICO
    df.to_csv(FICO_CLEAN_PATH, index=False)
    print(f" [OK] Saved clean FICO dataset to:\n      {FICO_CLEAN_PATH}")
    print(f"      Clean shape: {df.shape[0]:,} rows, {df.shape[1]} columns")


def clean_emp_length(val):
    """Converts employment text strings ('10+ years', '< 1 year') into numeric integers."""
    if pd.isna(val) or val == "no information":
        return 0
    val_str = str(val).strip()
    if "<" in val_str:
        return 0
    digits = re.findall(r'\d+', val_str)
    if digits:
        return int(digits[0])
    return 0


def preprocess_lending_club():
    print("\n" + "=" * 65)
    print(" 2. PREPROCESSING LENDING CLUB DATASET")
    print("=" * 65)

    if not os.path.exists(LC_RAW_PATH):
        print(f" [!] Lending Club raw file not found at: {LC_RAW_PATH}")
        return

    # Read necessary columns
    cols_to_use = [
        "revenue", "loan_amnt", "dti_n", "fico_n", "emp_length", 
        "home_ownership_n", "purpose", "experience_c", "Default"
    ]
    
    print(" -> Loading Lending Club raw columns...")
    df = pd.read_csv(LC_RAW_PATH, usecols=cols_to_use, low_memory=False)
    print(f" -> Raw shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # 1. Target Standardization
    df = df.rename(columns={"Default": "target"})
    print(" -> Standardized target: 'target' (0 = Paid, 1 = Default)")

    # 2. Clean 'emp_length' to integer numbers
    df["emp_length_years"] = df["emp_length"].apply(clean_emp_length)
    df = df.drop(columns=["emp_length"])
    print(" -> Converted 'emp_length' to numeric 'emp_length_years' (0 to 10)")

    # 3. Outlier Capping (Winsorization)
    # Income cap at 99th percentile to remove extreme distortions
    income_cap = df["revenue"].quantile(0.99)
    df["annual_income_capped"] = df["revenue"].clip(upper=income_cap)
    
    # DTI cap at 100% (any debt ratio > 100% capped to 100%)
    df["dti_capped"] = df["dti_n"].clip(lower=0, upper=100)
    print(f" -> Capped annual income at 99th percentile (${income_cap:,.0f}) and DTI at 100%")

    # 4. Handle Missing Values (Median imputation for numeric features)
    numeric_features = ["annual_income_capped", "loan_amnt", "dti_capped", "fico_n", "emp_length_years"]
    for col in numeric_features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 5. Clean Categorical Features
    df["home_ownership"] = df["home_ownership_n"].fillna("other").str.lower()
    df["loan_purpose"] = df["purpose"].fillna("other").str.lower()

    # Drop intermediate/raw columns
    df_clean = df[[
        "annual_income_capped", "loan_amnt", "dti_capped", "fico_n", 
        "emp_length_years", "home_ownership", "loan_purpose", "experience_c", "target"
    ]].rename(columns={
        "annual_income_capped": "annual_income",
        "dti_capped": "dti"
    })

    # Save cleaned Lending Club
    df_clean.to_csv(LC_CLEAN_PATH, index=False)
    print(f" [OK] Saved clean Lending Club dataset to:\n      {LC_CLEAN_PATH}")
    print(f"      Clean shape: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")


def main():
    print("\n" + "#" * 65)
    print(" STEP 3: DATA PREPROCESSING PIPELINE")
    print("#" * 65)

    preprocess_fico()
    preprocess_lending_club()

    print("\n" + "=" * 65)
    print(f" PREPROCESSING COMPLETE!")
    print(f" Clean datasets are ready inside:\n {PROCESSED_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()