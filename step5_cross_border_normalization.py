"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 5: CROSS-BORDER NORMALIZATION ENGINE (step5_cross_border_normalization.py)
=============================================================================
What this script does:
1. Implements the World Bank Purchasing Power Parity (PPP) conversion table.
2. Builds dual-layer conversion:
   - Market FX (Nominal currency exchange)
   - Real PPP (Purchasing power equivalence)
3. Computes Home-Country Relative Income Percentiles.
4. Normalizes cross-border applicants across major credit corridors:
   - India (INR) -> USA (USD)
   - UK (GBP) -> USA (USD)
   - Mexico (MXN) -> USA (USD)
5. Saves the normalization rules into reports/normalization_config.json
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np

# =============================================================================
# BULLETPROOF PATH RESOLUTION (Finds the root folder automatically)
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
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CONFIG_OUTPUT_PATH = os.path.join(REPORTS_DIR, "normalization_config.json")
OUTPUT_NORMALIZED_PROFILES = os.path.join(PROCESSED_DIR, "cross_border_normalized_profiles.csv")

# =============================================================================
# WORLD BANK ECONOMIC BENCHMARKS & CONVERSION FACTORS (Base: 1 USD)
# =============================================================================
COUNTRY_PROFILES = {
    "USA": {
        "currency": "USD",
        "fx_to_usd": 1.0,         # 1 USD = 1 USD
        "ppp_to_usd": 1.0,        # Base country
        "median_income": 65000,   # Annual middle-class income
        "top_10_income": 130000
    },
    "IND": {
        "currency": "INR",
        "country_name": "India",
        "fx_to_usd": 83.50,       # Market exchange: 83.5 INR per 1 USD
        "ppp_to_usd": 23.20,      # World Bank PPP: 23.2 INR buys what 1 USD buys
        "median_income": 450000,  # Middle class formal banking income
        "top_10_income": 1500000
    },
    "GBR": {
        "currency": "GBP",
        "country_name": "United Kingdom",
        "fx_to_usd": 0.78,        # Market exchange: 0.78 GBP per 1 USD
        "ppp_to_usd": 0.70,       # PPP factor
        "median_income": 35000,
        "top_10_income": 70000
    },
    "MEX": {
        "currency": "MXN",
        "country_name": "Mexico",
        "fx_to_usd": 17.50,       # Market exchange: 17.5 MXN per 1 USD
        "ppp_to_usd": 9.80,       # PPP factor
        "median_income": 180000,
        "top_10_income": 450000
    },
    "CAN": {
        "currency": "CAD",
        "country_name": "Canada",
        "fx_to_usd": 1.36,
        "ppp_to_usd": 1.22,
        "median_income": 60000,
        "top_10_income": 115000
    }
}


def normalize_income(amount_local: float, source_country: str, dest_country: str = "USA") -> dict:
    """
    Applies dual-layer normalization: Nominal FX conversion vs. Real PPP adjustment,
    plus Home-Country Economic Standing Percentile.
    """
    src = COUNTRY_PROFILES.get(source_country, COUNTRY_PROFILES["USA"])
    dst = COUNTRY_PROFILES.get(dest_country, COUNTRY_PROFILES["USA"])

    # 1. Market FX Conversion (Nominal)
    # Convert local currency -> USD -> Destination currency
    amount_in_usd_nominal = amount_local / src["fx_to_usd"]
    amount_dest_nominal = amount_in_usd_nominal * dst["fx_to_usd"]

    # 2. Purchasing Power Parity (PPP) Conversion (Real Living Standard)
    amount_in_usd_ppp = amount_local / src["ppp_to_usd"]
    amount_dest_ppp = amount_in_usd_ppp * dst["ppp_to_usd"]

    # 3. Relative Economic Standing (Home-Country Percentile Estimate)
    # Compares against home country median and top 10%
    med = src["median_income"]
    top = src["top_10_income"]
    
    if amount_local <= med:
        relative_score = (amount_local / med) * 50.0  # 0 to 50th percentile
    else:
        relative_score = 50.0 + ((amount_local - med) / (top - med)) * 40.0
    relative_percentile = min(round(relative_score, 1), 99.0)

    return {
        "source_country": source_country,
        "source_currency": src["currency"],
        "source_amount": round(amount_local, 2),
        "dest_country": dest_country,
        "dest_currency": dst["currency"],
        "nominal_fx_converted": round(amount_dest_nominal, 2),
        "real_ppp_converted": round(amount_dest_ppp, 2),
        "home_country_income_percentile": relative_percentile
    }


def run_cross_border_normalization():
    print("\n" + "=" * 70)
    print(" STEP 5: CROSS-BORDER NORMALIZATION ENGINE")
    print("=" * 70)

    # Test cases representing real-world immigrants
    test_applicants = [
        {"name": "Priya (Tech Lead)", "country": "IND", "salary": 1600000, "debt": 300000},
        {"name": "Rahul (Junior Engineer)", "country": "IND", "salary": 700000, "debt": 150000},
        {"name": "James (Financial Analyst)", "country": "GBR", "salary": 45000, "debt": 8000},
        {"name": "Carlos (Marketing Manager)", "country": "MEX", "salary": 320000, "debt": 50000},
    ]

    print("\n -> Demonstrating Cross-Border Normalization Across Key Corridors:")
    results = []

    for app in test_applicants:
        norm = normalize_income(app["salary"], source_country=app["country"], dest_country="USA")
        
        # Calculate scale-invariant Debt-to-Income
        dti = round((app["debt"] / app["salary"]) * 100, 1)
        
        row = {
            "Applicant": app["name"],
            "Home Country": app["country"],
            "Local Salary": f"{norm['source_currency']} {app['salary']:,}",
            "Nominal FX (USD)": f"${norm['nominal_fx_converted']:,.0f}",
            "Real PPP (USD)": f"${norm['real_ppp_converted']:,.0f}",
            "Home Percentile": f"{norm['home_country_income_percentile']}%",
            "DTI Ratio": f"{dti}%"
        }
        results.append(row)

    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))

    # Save test profiles to CSV
    df_results.to_csv(OUTPUT_NORMALIZED_PROFILES, index=False)
    print(f"\n [OK] Saved normalized demo profiles to:\n      {OUTPUT_NORMALIZED_PROFILES}")

    # Save Configuration JSON
    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(COUNTRY_PROFILES, f, indent=4)
    print(f" [OK] Saved World Bank Normalization Config to:\n      {CONFIG_OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print(" STEP 5 NORMALIZATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_cross_border_normalization()