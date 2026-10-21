"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 1: DATA AUDIT (01_data_audit.py)
=============================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Set project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "audit")

# Map of dataset names to their expected file paths
DATASET_PATHS = {
    "FICO HELOC": os.path.join(DATA_DIR, "fico", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\heloc_dataset_v1.csv"),
    "Lending Club": os.path.join(DATA_DIR, "lending_club", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\LC_loans_granting_model_dataset.csv"),
    "Home Credit (Application)": os.path.join(DATA_DIR, "home_credit", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\application_train.csv"),
    "Home Credit (Bureau)": os.path.join(DATA_DIR, "home_credit", "bureau.csv")
}

# Potential target column names across all 3 datasets
CANDIDATE_TARGETS = ["RiskPerformance", "Default", "loan_status", "TARGET", "default"]

# Keywords indicating financial and credit variables
FINANCIAL_KEYWORDS = [
    "inc", "income", "debt", "dti", "amt", "bal", "balance", 
    "util", "limit", "rate", "delinq", "loan", "trade", "inq", "fico"
]


def audit_single_file(dataset_name: str, file_path: str) -> dict:
    """
    Audits a single CSV file without altering or deleting any data.
    """
    print("\n" + "=" * 65)
    print(f" AUDITING: {dataset_name}")
    print(f" File Path: {file_path}")
    print("=" * 65)

    if not os.path.exists(file_path):
        print(f" [!] STATUS: NOT FOUND ON DISK")
        print(f"     Please place the CSV file in: {file_path}")
        return {
            "dataset_name": dataset_name,
            "status": "FILE_NOT_FOUND",
            "file_path": file_path
        }

    print(" -> Loading dataset into memory (this may take a few seconds)...")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f" [X] Error loading file: {str(e)}")
        return {
            "dataset_name": dataset_name,
            "status": "LOAD_ERROR",
            "error": str(e)
        }

    rows, cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    print(f" [OK] Loaded successfully!")
    print(f"      Rows (Applicants / Records): {rows:,}")
    print(f"      Columns (Attributes / Features): {cols:,}")
    print(f"      RAM Memory Footprint: {memory_mb} MB")

    # 1. Inspect Data Types
    dtype_breakdown = df.dtypes.astype(str).value_counts().to_dict()
    print("\n -> Column Data Types:")
    for dtype_name, count in dtype_breakdown.items():
        print(f"      {dtype_name}: {count} columns")

    # 2. Inspect Missing Values (NaN / Blanks)
    missing_series = df.isnull().sum()
    total_cells = rows * cols
    total_missing = int(missing_series.sum())
    missing_pct_overall = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    cols_with_missing = missing_series[missing_series > 0]
    cols_high_missing = (cols_with_missing / rows * 100).loc[lambda x: x > 50].round(2).to_dict()

    print(f"\n -> Missing Values Analysis:")
    print(f"      Total missing data points: {total_missing:,} ({missing_pct_overall}% of all cells)")
    print(f"      Columns with at least 1 missing value: {len(cols_with_missing)} of {cols}")
    print(f"      Columns with MORE than 50% missing data: {len(cols_high_missing)}")
    if cols_high_missing:
        print("      Top columns with severe missingness (>50%):")
        for col_name, pct in list(cols_high_missing.items())[:5]:
            print(f"        - {col_name}: {pct}% missing")

    # 3. Target Variable Detection & Class Distribution
    detected_target = None
    target_summary = {}

    for cand in CANDIDATE_TARGETS:
        if cand in df.columns:
            detected_target = cand
            break

    print(f"\n -> Target Variable (The Outcome We Want to Predict):")
    if detected_target:
        val_counts = df[detected_target].value_counts(dropna=False).to_dict()
        val_pcts = (df[detected_target].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
        print(f"      Detected Target Column: '{detected_target}'")
        print(f"      Class Breakdown (Distribution):")
        for class_val, count in val_counts.items():
            pct = val_pcts[class_val]
            print(f"        - '{class_val}': {count:,} ({pct}%)")
        target_summary = {
            "column_name": detected_target,
            "counts": {str(k): int(v) for k, v in val_counts.items()},
            "percentages": {str(k): float(v) for k, v in val_pcts.items()}
        }
    else:
        print("      [?] No standard target column automatically detected.")

    # 4. Candidate Financial Features Discovery
    matched_financial_cols = [
        c for c in df.columns 
        if any(keyword in c.lower() for keyword in FINANCIAL_KEYWORDS)
    ]
    print(f"\n -> Candidate Financial Columns Found: {len(matched_financial_cols)}")
    print(f"      Sample: {matched_financial_cols[:8]}")

    return {
        "dataset_name": dataset_name,
        "status": "AUDITED_SUCCESSFULLY",
        "file_path": file_path,
        "shape": {"rows": rows, "columns": cols},
        "memory_mb": memory_mb,
        "data_types": dtype_breakdown,
        "missing_summary": {
            "total_missing_cells": total_missing,
            "missing_pct_overall": missing_pct_overall,
            "columns_with_missing_count": len(cols_with_missing),
            "columns_over_50pct_missing": cols_high_missing
        },
        "target_summary": target_summary,
        "candidate_financial_columns": matched_financial_cols
    }


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("\n" + "#" * 65)
    print(" CROSS-BORDER CREDIT PASSPORT: STEP 1 - DATA AUDIT")
    print("#" * 65)

    all_audit_results = {}

    for name, path in DATASET_PATHS.items():
        all_audit_results[name] = audit_single_file(name, path)

    # Save complete JSON summary report
    report_file_path = os.path.join(REPORTS_DIR, "audit_summary_report.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(all_audit_results, f, indent=4)

    print("\n" + "=" * 65)
    print(" AUDIT EXECUTION COMPLETED!")
    print(f" Full audit report saved to:\n {report_file_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()