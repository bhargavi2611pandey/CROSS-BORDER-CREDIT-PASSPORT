"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 6: CREATE PASSPORT DATASET (step6_create_passport_dataset.py)
=============================================================================
What this script does:
1. Loads the high-signal credit bureau dataset (heloc_dataset_v1.csv) combined with
   universal cash flow data.
2. Cleans negative special codes (-7, -8, -9) into clean numerical trade indicators.
3. Maps origin credit estimates into standard 300-850 bureau credit scores.
4. Simulates realistic cross-border migration corridors (IND, GBR, MEX, CAN, USA).
5. Applies World Bank PPP normalization to compute real purchasing power in USD.
6. Combines Cross-Border Financial Health with Origin Bureau Behavioral Trade Lines:
   - Revolving Utilization Burden (Credit card balance / limit)
   - Inquiries in Last 6 Months (Credit seeking velocity)
   - Percent Trades Never Delinquent (Lifetime on-time payment track record)
   - Number of Satisfactory Accounts & Delinquency Recency
7. Generates unique Passport IDs (PASS-XXXXXX).
8. Saves the finalized, high-accuracy dataset to:
   data/processed/credit_passport_dataset.csv
=============================================================================
"""

import os
import pandas as pd
import numpy as np

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

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
HELOC_PATH = os.path.join(PROJECT_ROOT, "heloc_dataset_v1.csv")
OUTPUT_PASSPORT_PATH = os.path.join(PROCESSED_DIR, "credit_passport_dataset.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# World Bank PPP conversion factors relative to USD (from Step 5)
PPP_FACTORS = {
    "USA": 1.0,
    "IND": 23.20,
    "GBR": 0.70,
    "MEX": 9.80,
    "CAN": 1.22
}

FX_FACTORS = {
    "USA": 1.0,
    "IND": 83.50,
    "GBR": 0.78,
    "MEX": 17.50,
    "CAN": 1.36
}


def build_passport_dataset():
    print("\n" + "=" * 70)
    print(" STEP 6: ASSEMBLING MASTER CREDIT PASSPORT DATASET (DUAL-LAYER HIGH ACCURACY)")
    print("=" * 70)

    if not os.path.exists(HELOC_PATH):
        print(f" [!] Raw bureau dataset not found: {HELOC_PATH}")
        return

    print(" -> Loading origin credit bureau dataset (HELOC)...")
    df_raw = pd.read_csv(HELOC_PATH)
    n = len(df_raw)
    print(f"    Loaded {n:,} applicant credit histories.")

    np.random.seed(42)

    # 1. Simulate Realistic Cross-Border Migration Corridors
    countries = ["IND", "GBR", "MEX", "CAN", "USA"]
    probabilities = [0.35, 0.15, 0.20, 0.10, 0.20]
    source_country = np.random.choice(countries, size=n, p=probabilities)
    destination_country = "USA"

    # 2. Vectorized PPP & Currency Math
    print(" -> Applying World Bank Purchasing Power Parity (PPP) conversions...")
    fx_series = pd.Series(source_country).map(FX_FACTORS)
    ppp_series = pd.Series(source_country).map(PPP_FACTORS)

    # Generate realistic base earning power
    base_usd_income = np.random.normal(loc=68000, scale=22000, size=n).clip(min=18000, max=250000)
    local_income = (base_usd_income * fx_series).round(2)
    normalized_income_ppp = (local_income / ppp_series).round(2)
    monthly_income_ppp = (normalized_income_ppp / 12.0).round(2)

    requested_loan = np.random.uniform(3000, 45000, size=n).round(-2)
    
    # 3. Clean Bureau Variables & Handle Special Negative Codes (-7, -8, -9)
    print(" -> Cleaning negative bureau codes and engineering behavioral metrics...")
    
    # Map external estimate to standard 300 - 850 bureau scale
    ext_risk = df_raw["ExternalRiskEstimate"].replace([-7, -8, -9], np.nan)
    ext_risk_clean = ext_risk.fillna(ext_risk.median())
    bureau_score_850 = (300 + (ext_risk_clean - 40) * (550 / 55)).clip(300, 850).round(1)

    # Behavioral credit signals
    util_burden = df_raw["NetFractionRevolvingBurden"].replace([-7, -8, -9], 20).clip(lower=0)
    pct_never_delq = df_raw["PercentTradesNeverDelq"].replace([-7, -8, -9], 100).clip(lower=0)
    sat_trades = df_raw["NumSatisfactoryTrades"].replace([-7, -8, -9], 5).clip(lower=0)
    inq_6m = df_raw["NumInqLast6M"].replace([-7, -8, -9], 0).clip(lower=0)
    inq_recency = df_raw["MSinceMostRecentInqexcl7days"].replace([-7, -8, -9], 12).clip(lower=0)
    delq_12m = df_raw["MaxDelq2PublicRecLast12M"].replace([-7, -8, -9], 0).clip(lower=0)
    delq_ever = df_raw["MaxDelqEver"].replace([-7, -8, -9], 0).clip(lower=0)
    history_m = df_raw["AverageMInFile"].replace([-7, -8, -9], 60).clip(lower=0)
    total_trades = df_raw["NumTotalTrades"].replace([-7, -8, -9], 10).clip(lower=0)
    trades_open_12m = df_raw["NumTradesOpeninLast12M"].replace([-7, -8, -9], 1).clip(lower=0)
    high_util_trades = df_raw["NumBank2NatlTradesWHighUtilization"].replace([-7, -8, -9], 0).clip(lower=0)
    delq_recency = df_raw["MSinceMostRecentDelq"].replace([-7, -8, -9], 99).clip(lower=0)

    # Cash flow & debt burden
    dti = (util_burden * 0.5 + np.random.uniform(8, 22, size=n)).clip(5, 70).round(2)
    monthly_debt = (monthly_income_ppp * (dti / 100.0)).round(2)
    free_cash = (monthly_income_ppp - monthly_debt).round(2)
    lti = (requested_loan / normalized_income_ppp).round(4)

    # Derived indicator flags
    has_recent_inq = (inq_6m > 0).astype(int)
    has_never_delq = (pct_never_delq == 100).astype(int)
    has_high_util = (util_burden > 50).astype(int)

    # Binary target (0 = Good / Paid, 1 = Risky / Default)
    target = (df_raw["RiskPerformance"] == "Bad").astype(int)

    # 4. Generate International Passport IDs
    passport_ids = [f"PASS-{i+100001:06d}" for i in range(n)]

    # 5. Assemble Master DataFrame
    df_passport = pd.DataFrame({
        "passport_id": passport_ids,
        "source_country": source_country,
        "destination_country": destination_country,
        "local_currency_income": local_income,
        "normalized_income_ppp": normalized_income_ppp,
        "monthly_income_ppp": monthly_income_ppp,
        "requested_loan_amount": requested_loan,
        "debt_to_income_ratio": dti,
        "lti_ratio": lti,
        "free_cash_flow": free_cash,
        "credit_bureau_score": bureau_score_850,
        "revolving_utilization_pct": util_burden,
        "percent_trades_never_delq": pct_never_delq,
        "num_satisfactory_trades": sat_trades,
        "inquiries_last_6m": inq_6m,
        "inquiries_recency_months": inq_recency,
        "delinquencies_last_12m": delq_12m,
        "worst_delinquency_ever": delq_ever,
        "credit_history_months": history_m,
        "total_credit_trades": total_trades,
        "trades_open_12m": trades_open_12m,
        "high_utilization_trades": high_util_trades,
        "delinquency_recency_months": delq_recency,
        "has_recent_inquiries": has_recent_inq,
        "has_never_delinquent": has_never_delq,
        "has_high_utilization": has_high_util,
        "target": target
    })

    # Save to CSV
    print(" -> Saving Master Credit Passport Dataset to disk...")
    df_passport.to_csv(OUTPUT_PASSPORT_PATH, index=False)
    print(f" [OK] Saved successfully to:\n      {OUTPUT_PASSPORT_PATH}")

    # Summary report
    print("\n" + "=" * 70)
    print(" DUAL-LAYER CREDIT PASSPORT DATASET SUMMARY")
    print("=" * 70)
    print(f" Total Applicants   : {len(df_passport):,}")
    print(f" Total Features     : {df_passport.shape[1]}")
    
    print("\n -> Breakdown by Migration Corridor:")
    country_counts = df_passport["source_country"].value_counts()
    for c, count in country_counts.items():
        print(f"      {c}: {count:,} ({count/len(df_passport)*100:.1f}%)")

    print("\n -> Target Distribution (Risk Performance):")
    target_counts = df_passport["target"].value_counts()
    print(f"      0 (Good / Non-Default) : {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df_passport)*100:.1f}%)")
    print(f"      1 (Bad / Default Risk) : {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df_passport)*100:.1f}%)")

    print("\n -> First Sample Record:")
    print(df_passport.iloc[0].to_dict())
    print("\n" + "=" * 70)
    print(" STEP 6 COMPLETE! HIGH-ACCURACY DATASET READY.")
    print("=" * 70)


if __name__ == "__main__":
    build_passport_dataset()
