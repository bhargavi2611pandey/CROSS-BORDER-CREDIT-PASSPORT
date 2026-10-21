"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 11: THE CREDIT PASSPORT ENGINE (step11_passport_engine.py)
=============================================================================
What this script does:
1. Implements the high-accuracy CreditPassportEngine class:
   - Currency & World Bank PPP Normalization
   - Universal Cashflow & Origin Bureau Behavioral Trade Line Inference
   - Real-time AI Inference with Champion XGBoost (ROC-AUC ~0.79)
   - Passport Credit Score Computation (0 - 100)
   - Underwriting Decision (APPROVED / REVIEW / DECLINED)
   - Human-friendly Layman SHAP Adverse Action & Strength Explanations
2. Tests the engine on 3 realistic cross-border immigrants:
   - Priya Sharma (India -> USA)
   - Carlos Mendez (Mexico -> USA)
   - David Smith (UK -> USA)
3. Prints official Credit Passport Cards and saves reports/sample_credit_passports.json
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import shap

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

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "advanced_xgboost_model.joblib")
THRESHOLD_CONFIG_PATH = os.path.join(REPORTS_DIR, "optimal_threshold_config.json")
OUTPUT_PASSPORTS_JSON = os.path.join(REPORTS_DIR, "sample_credit_passports.json")

# World Bank conversion factors (from Step 5)
PPP_FACTORS = {"USA": 1.0, "IND": 23.20, "GBR": 0.70, "MEX": 9.80, "CAN": 1.22}
FX_FACTORS = {"USA": 1.0, "IND": 83.50, "GBR": 0.78, "MEX": 17.50, "CAN": 1.36}


def format_layman_reason(feature: str, val: float, is_strength: bool) -> str:
    """
    Translates raw algorithmic SHAP features and float values into simple,
    intuitive, layman-friendly credit explanations under ECOA/FCRA principles.
    """
    if feature == "credit_bureau_score":
        if is_strength:
            return f"Strong Credit Bureau Rating ({int(val)}): Verified history of high creditworthiness and consistent repayment discipline."
        else:
            return f"Low Credit Bureau Rating ({int(val)}): Below-average credit score indicating higher repayment risk."
    
    if feature == "credit_history_months":
        yrs = round(val / 12.0, 1)
        if is_strength:
            return f"Well-Seasoned Credit History ({int(val)} mos / {yrs} yrs): Long, mature track record of successfully managing credit accounts."
        else:
            return f"Short Credit History ({int(val)} months): Brief credit record, making long-term financial habits harder to evaluate."
    
    if feature == "inquiries_recency_months":
        if is_strength:
            return f"No Urgent Credit Seeking ({int(val)} months clean): Has not aggressively applied for new credit recently, reflecting financial stability."
        else:
            return "Recent Credit Inquiries: Applied for new loans or cards very recently, which often signals urgent borrowing needs."
    
    if feature == "delinquency_recency_months":
        if is_strength:
            return "Clean Payment History: No record of recent missed payments or collection notices on file."
        else:
            return f"Recent Missed Payment ({int(val)} months ago): Recorded an overdue or delayed payment relatively recently."
    
    if feature == "delinquencies_last_12m":
        if is_strength:
            return "Zero Recent Overdue Accounts: Paid 100% of all credit accounts strictly on time throughout the past 12 months."
        else:
            return f"Recent Overdue Accounts ({int(val)} in past year): Had one or more delayed or missed payments in the last 12 months."
    
    if feature == "worst_delinquency_ever":
        if is_strength:
            return "Clean Historical Record: Zero instances of severe defaults or 90+ day charge-offs in credit history."
        else:
            return "Historical Delinquency Severity: Recorded a significant payment delay or default in prior credit history."

    if feature == "percent_trades_never_delq":
        if is_strength:
            return f"Spotless Payment Discipline ({int(val)}% on-time): Overwhelming majority of past accounts maintained zero delinquencies."
        else:
            delayed = max(0, 100 - int(val))
            return f"History of Payment Delays ({delayed}% delayed): Past credit lines reflect multiple missed or late due dates."
    
    if feature == "revolving_utilization_pct":
        if is_strength:
            return f"Low Credit Card Balances ({val:.1f}% used): Uses only a small portion of available credit lines, preserving ample liquidity."
        else:
            return f"Heavily Maxed-Out Credit Cards ({val:.1f}% used): Credit lines are operating near maximum capacity, indicating heavy debt reliance."
    
    if feature == "high_utilization_trades":
        if is_strength:
            return "Disciplined Account Limits: No credit cards or overdrafts are operating near maximum capacity."
        else:
            return f"Maxed-Out Accounts ({int(val)} credit lines): Multiple credit cards or accounts operating near borrowing limits."

    if feature == "num_satisfactory_trades":
        if is_strength:
            return f"Proven Loan Track Record ({int(val)} clean loans): Substantial history of successfully managing and closing credit accounts."
        else:
            return f"Few Completed Credit Lines ({int(val)} accounts): Limited number of successfully closed loans on record."
    
    if feature == "inquiries_last_6m":
        if is_strength:
            return "Zero Recent Credit Inquiries: No hard credit checks in the last 6 months, reflecting stable borrowing habits."
        else:
            return f"Multiple Credit Inquiries ({int(val)} checks): High credit-seeking velocity in the past 6 months signals potential cash stress."

    if feature == "debt_to_income_ratio":
        if is_strength:
            return f"Comfortable Debt Burden ({val:.1f}% DTI): Existing monthly loan payments consume a modest portion of income."
        else:
            return f"Elevated Debt Burden ({val:.1f}% DTI): Monthly loan and debt obligations consume a high portion of earnings."

    if feature == "lti_ratio":
        if is_strength:
            return f"Conservative Loan Size ({val:.2f}x annual salary): Requested loan amount is easily serviceable relative to yearly income."
        else:
            return f"Large Loan Request ({val:.2f}x annual salary): Requested loan is large compared to yearly earnings."

    if feature == "free_cash_flow":
        if is_strength:
            return f"Strong Cash Cushion (${val:,.0f}/mo buffer): Generous disposable income buffer left each month after all debts are paid."
        else:
            return f"Tight Monthly Cash Cushion (${val:,.0f}/mo buffer): Limited disposable income remaining after monthly debt payments."

    if feature in ["normalized_income_ppp", "monthly_income_ppp"]:
        if is_strength:
            return f"Strong Purchasing Power (${val:,.0f} PPP): High real earning power equivalent in USD under World Bank living standards."
        else:
            return f"Modest Real Purchasing Power (${val:,.0f} PPP): Earning power is tighter relative to the requested loan size."

    clean_name = feature.replace("_", " ").title()
    return f"{clean_name}: strong verified level ({val:,.1f})" if is_strength else f"{clean_name}: elevated risk impact"


class CreditPassportEngine:
    """
    Central Cross-Border Credit Underwriting Engine (Dual-Layer High Accuracy).
    Orchestrates currency normalization, universal feature derivation,
    origin bureau trade lines, ML scoring (ROC-AUC ~0.79), and SHAP explainability.
    """
    def __init__(self):
        print(" -> Initializing High-Accuracy Credit Passport Engine...")
        if not os.path.exists(CHAMPION_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {CHAMPION_MODEL_PATH}. Please run Step 8 first.")

        # Load Model Package
        pkg = joblib.load(CHAMPION_MODEL_PATH)
        self.model = pkg["model"]
        self.feature_cols = pkg["features"]
        self.model_name = pkg.get("champion_model_name", "XGBoost")
        self.model_auc = pkg.get("roc_auc", 0.7933)

        # Initialize SHAP TreeExplainer for instant explanations
        self.explainer = shap.TreeExplainer(self.model)
        print(f" [OK] Engine ready! Model: {self.model_name} (ROC-AUC: {self.model_auc:.4f})")

    def evaluate_applicant(self, raw_input: dict) -> dict:
        """
        Takes raw international applicant profile and generates a full Credit Passport.
        """
        source_country = str(raw_input.get("source_country", "USA")).upper()
        local_income = float(raw_input["local_annual_income"])
        requested_loan = float(raw_input["requested_loan_amount"])
        monthly_debt = float(raw_input.get("monthly_debt_payments", 0.0))
        emp_years = float(raw_input.get("employment_years", 3.0))
        bureau_score = float(raw_input.get("credit_bureau_score", 680.0))

        # 1. Cross-Border PPP & Currency Normalization
        ppp_div = PPP_FACTORS.get(source_country, 1.0)
        fx_div = FX_FACTORS.get(source_country, 1.0)

        # Real purchasing power in USD
        normalized_income_ppp = round(local_income / ppp_div, 2)
        nominal_usd_income = round(local_income / fx_div, 2)
        monthly_income_ppp = round(normalized_income_ppp / 12.0, 2)

        # 2. Universal Cash Flow Metrics
        monthly_debt_usd = round(monthly_debt / ppp_div, 2)
        dti_ratio = round((monthly_debt_usd / monthly_income_ppp) * 100.0, 2) if monthly_income_ppp > 0 else 35.0
        lti_ratio = round(requested_loan / normalized_income_ppp, 4) if normalized_income_ppp > 0 else 0.20
        free_cash_flow = round(monthly_income_ppp - monthly_debt_usd, 2)

        # 3. Origin Credit Bureau Behavioral Trade Lines (Smart realistic fallbacks)
        util_pct = float(raw_input.get("revolving_utilization_pct", min(85.0, max(5.0, dti_ratio * 0.9))))
        pct_never_delq = float(raw_input.get("percent_trades_never_delq", 100.0 if bureau_score >= 720 else (80.0 if bureau_score >= 640 else 55.0)))
        sat_trades = float(raw_input.get("num_satisfactory_trades", 22.0 if bureau_score >= 720 else (12.0 if bureau_score >= 640 else 4.0)))
        inq_6m = float(raw_input.get("inquiries_last_6m", 0.0 if bureau_score >= 720 else (1.0 if bureau_score >= 650 else 3.0)))
        inq_recency = float(raw_input.get("inquiries_recency_months", 12.0 if inq_6m == 0 else 1.0))
        delq_12m = float(raw_input.get("delinquencies_last_12m", 0.0 if pct_never_delq == 100 else 2.0))
        delq_ever = float(raw_input.get("worst_delinquency_ever", 0.0 if pct_never_delq == 100 else 4.0))
        history_m = float(raw_input.get("credit_history_months", max(36.0, emp_years * 12.0)))
        total_trades = float(raw_input.get("total_credit_trades", max(sat_trades + 2, 8.0)))
        trades_open_12m = float(raw_input.get("trades_open_12m", 1.0))
        high_util_trades = float(raw_input.get("high_utilization_trades", 1.0 if util_pct > 50 else 0.0))
        delq_recency = float(raw_input.get("delinquency_recency_months", 99.0 if pct_never_delq == 100 else 6.0))

        has_recent_inq = 1 if inq_6m > 0 else 0
        has_never_delq = 1 if pct_never_delq >= 100 else 0
        has_high_util = 1 if util_pct > 50 else 0

        # 4. Assemble Exact Feature Vector for Model
        feature_dict = {
            "normalized_income_ppp": normalized_income_ppp,
            "monthly_income_ppp": monthly_income_ppp,
            "requested_loan_amount": requested_loan,
            "debt_to_income_ratio": dti_ratio,
            "lti_ratio": lti_ratio,
            "free_cash_flow": free_cash_flow,
            "credit_bureau_score": bureau_score,
            "revolving_utilization_pct": util_pct,
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
            "has_high_utilization": has_high_util
        }

        feature_df = pd.DataFrame([feature_dict])[self.feature_cols]

        # 5. Predict Risk Probability with Champion XGBoost
        prob_default = float(self.model.predict_proba(feature_df)[0, 1])

        # 6. Calculate Passport Score (0 to 100 scale, higher is safer)
        passport_score = int(round((1.0 - prob_default) * 100.0))
        passport_score = max(5, min(98, passport_score))

        # Assign Credit Tier
        if passport_score >= 80:
            credit_tier = "Tier A (Prime / Low Risk)"
        elif passport_score >= 70:
            credit_tier = "Tier B (Near-Prime / Moderate Risk)"
        elif passport_score >= 60:
            credit_tier = "Tier C (Fair / Guarded Risk)"
        else:
            credit_tier = "Tier D (Subprime / High Risk)"

        # 7. Underwriting Decision Thresholds (Strict 40% Default Risk Cutoff)
        if prob_default < 0.20:
            decision = "APPROVED (Instant Pass)"
            decision_color = "GREEN"
        elif prob_default <= 0.30:
            decision = "APPROVED (Standard Terms)"
            decision_color = "LIGHT_GREEN"
        elif prob_default <= 0.40:
            decision = "MANUAL REVIEW (Refer to Underwriter)"
            decision_color = "YELLOW"
        else:
            decision = "DECLINED (High Risk Profile - Exceeds 40% Risk Cutoff)"
            decision_color = "RED"

        # 8. Compute Individual SHAP Reasons (Converted to Layman Explanations)
        shap_res = self.explainer(feature_df)
        shap_vals = shap_res.values[0]

        impact_df = pd.DataFrame({
            "feature": self.feature_cols,
            "actual_val": feature_df.iloc[0].values,
            "shap_impact": shap_vals
        })

        # Strengths (Negative SHAP pulls risk down) -> Layman simple language
        strengths = []
        for _, r in impact_df.sort_values(by="shap_impact").head(3).iterrows():
            if r["shap_impact"] < 0:
                strengths.append(format_layman_reason(r["feature"], r["actual_val"], is_strength=True))

        # Risk Factors / Adverse Action (Positive SHAP pushes risk up) -> Layman simple language
        risk_factors = []
        for _, r in impact_df.sort_values(by="shap_impact", ascending=False).head(3).iterrows():
            if r["shap_impact"] > 0:
                risk_factors.append(format_layman_reason(r["feature"], r["actual_val"], is_strength=False))

        return {
            "applicant_name": raw_input.get("applicant_name", "Cross-Border Applicant"),
            "source_country": source_country,
            "destination_country": raw_input.get("destination_country", "USA"),
            "passport_id": f"PASS-{np.random.randint(100000, 999999)}",
            "passport_score": passport_score,
            "credit_tier": credit_tier,
            "default_probability_pct": round(prob_default * 100.0, 1),
            "underwriting_decision": decision,
            "decision_color": decision_color,
            "financial_profile": {
                "local_currency_salary": f"{local_income:,.0f} ({source_country})",
                "nominal_usd_salary": f"${nominal_usd_income:,.0f}",
                "real_ppp_usd_salary": f"${normalized_income_ppp:,.0f}",
                "monthly_disposable_buffer": f"${free_cash_flow:,.0f}",
                "dti_ratio": f"{dti_ratio:.1f}%",
                "lti_ratio": f"{lti_ratio:.2f}x",
                "bureau_credit_score": f"{bureau_score:.0f}",
                "revolving_utilization": f"{util_pct:.1f}%",
                "percent_clean_trades": f"{pct_never_delq:.0f}%",
                "inquiries_last_6m": f"{inq_6m:.0f}"
            },
            "top_credit_strengths": strengths,
            "top_risk_drivers": risk_factors
        }


def print_passport_card(passport: dict):
    """Prints a beautiful, formatted passport card in the terminal."""
    print("\n" + "=" * 65)
    print(f"       CROSS-BORDER CREDIT PASSPORT  [{passport['passport_id']}]")
    print("=" * 65)
    print(f" Applicant Name        : {passport['applicant_name']}")
    print(f" Migration Corridor    : {passport['source_country']} -> {passport['destination_country']}")
    print(f" Credit Passport Score : {passport['passport_score']} / 100  ({passport['credit_tier']})")
    print(f" Default Probability   : {passport['default_probability_pct']}%")
    print(f" Underwriting Decision : {passport['underwriting_decision']}")
    print("-" * 65)
    print(" FINANCIAL HEALTH (PPP NORMALIZED):")
    fp = passport["financial_profile"]
    print(f"   Local Salary       : {fp['local_currency_salary']}")
    print(f"   Real PPP (USD)     : {fp['real_ppp_usd_salary']} / year (Buying Power)")
    print(f"   Monthly Free Cash  : {fp['monthly_disposable_buffer']} (Buffer after debt)")
    print(f"   Debt-to-Income     : {fp['dti_ratio']}")
    print(f"   Credit Utilization : {fp['revolving_utilization']}")
    print(f"   Clean Trade Lines  : {fp['percent_clean_trades']}")
    print(f"   Recent Inquiries   : {fp['inquiries_last_6m']}")
    print("-" * 65)
    print(" TOP CREDIT STRENGTHS (LAYMAN EXPLANATIONS):")
    for s in passport["top_credit_strengths"]:
        print(f"   [+] {s}")
    print(" ADVERSE ACTION / RISK FACTORS (LAYMAN EXPLANATIONS):")
    for r in passport["top_risk_drivers"]:
        print(f"   [-] {r}")
    print("=" * 65)


def main():
    engine = CreditPassportEngine()

    test_cases = [
        {
            "applicant_name": "Priya Sharma (Senior Software Engineer)",
            "source_country": "IND",
            "destination_country": "USA",
            "local_annual_income": 2200000,
            "requested_loan_amount": 12000,
            "monthly_debt_payments": 20000,
            "credit_bureau_score": 755,
            "employment_years": 7,
            "revolving_utilization_pct": 14.0,
            "percent_trades_never_delq": 100.0,
            "num_satisfactory_trades": 25.0,
            "inquiries_last_6m": 0.0
        },
        {
            "applicant_name": "Carlos Mendez (Project Manager)",
            "source_country": "MEX",
            "destination_country": "USA",
            "local_annual_income": 380000,
            "requested_loan_amount": 8000,
            "monthly_debt_payments": 4500,
            "credit_bureau_score": 670,
            "employment_years": 4,
            "revolving_utilization_pct": 42.0,
            "percent_trades_never_delq": 85.0,
            "num_satisfactory_trades": 12.0,
            "inquiries_last_6m": 1.0
        },
        {
            "applicant_name": "David Smith (High Debt Risk Profile)",
            "source_country": "GBR",
            "destination_country": "USA",
            "local_annual_income": 24000,
            "requested_loan_amount": 25000,
            "monthly_debt_payments": 900,
            "credit_bureau_score": 580,
            "employment_years": 1,
            "revolving_utilization_pct": 78.0,
            "percent_trades_never_delq": 60.0,
            "num_satisfactory_trades": 3.0,
            "inquiries_last_6m": 4.0
        }
    ]

    evaluated_passports = []
    for tc in test_cases:
        passport = engine.evaluate_applicant(tc)
        print_passport_card(passport)
        evaluated_passports.append(passport)

    with open(OUTPUT_PASSPORTS_JSON, "w", encoding="utf-8") as f:
        json.dump(evaluated_passports, f, indent=4)
    print(f"\n [OK] Saved all demo passports to:\n      {OUTPUT_PASSPORTS_JSON}")

    print("\n" + "=" * 70)
    print(" STEP 11 COMPLETE! CREDIT PASSPORT ENGINE FULLY OPERATIONAL.")
    print("=" * 70)


if __name__ == "__main__":
    main()
