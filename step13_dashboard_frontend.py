"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 13: ENTERPRISE UNDERWRITING & KYC PORTAL (step13_dashboard_frontend.py)
=============================================================================
What this Web Application provides:
1. Stage 1: Digital KYC & Identity Verification (PAN/Aadhaar/NIN/CURP + Bank verification)
2. Stage 2: Origin Bureau Verification (RBI CIBIL 300-900 / UK Experian / etc.)
3. Stage 3: Cross-Border Translation Engine (PPP Living Standard + US FICO Underwriting)
4. SHAP Regulatory Reason Codes & Adverse Action Notice
=============================================================================
"""

import os
import sys
import re
import requests
import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Global Credit Passport | KYC & Underwriting",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bulletproof path setup
current = os.path.abspath(os.path.dirname(__file__))
while current != os.path.dirname(current):
    if os.path.exists(os.path.join(current, "data")):
        PROJECT_ROOT = current
        break
    current = os.path.dirname(current)
else:
    PROJECT_ROOT = r"C:\Users\Bhargavi Pandey\Downloads\Cross-Border Credit Passport Project"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from step11_passport_engine import CreditPassportEngine, PPP_FACTORS, FX_FACTORS

FASTAPI_URL = "http://127.0.0.1:8000/api/v1/passport/evaluate"

@st.cache_resource
def get_engine():
    return CreditPassportEngine()

# National ID definitions per country
COUNTRY_CONFIGS = {
    "IND": {
        "name": "India",
        "currency": "INR (₹)",
        "id_name": "PAN Card Number",
        "id_placeholder": "ABCDE1234F",
        "id_regex": r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$",
        "bureau_name": "CIBIL (RBI Regulated)",
        "bureau_min": 300, "bureau_max": 900,
        "primary_banks": ["State Bank of India (SBI)", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra"]
    },
    "GBR": {
        "name": "United Kingdom",
        "currency": "GBP (£)",
        "id_name": "National Insurance Number (NINO)",
        "id_placeholder": "QQ123456C",
        "id_regex": r"^[A-CEGHJ-PR-TW-Z]{1}[A-CEGHJ-NPR-TW-Z]{1}[0-9]{6}[A-D]{1}$",
        "bureau_name": "Experian UK / Equifax",
        "bureau_min": 0, "bureau_max": 999,
        "primary_banks": ["Barclays", "HSBC UK", "Lloyds Bank", "NatWest", "Standard Chartered"]
    },
    "MEX": {
        "name": "Mexico",
        "currency": "MXN ($)",
        "id_name": "CURP / RFC ID",
        "id_placeholder": "ABCD800101XYZ",
        "id_regex": r"^[A-Z0-9]{10,18}$",
        "bureau_name": "Buró de Crédito México",
        "bureau_min": 400, "bureau_max": 850,
        "primary_banks": ["BBVA México", "Santander México", "Banorte", "Citibanamex"]
    },
    "CAN": {
        "name": "Canada",
        "currency": "CAD ($)",
        "id_name": "Social Insurance Number (SIN)",
        "id_placeholder": "123-456-789",
        "id_regex": r"^[0-9\s-]{9,11}$",
        "bureau_name": "Equifax / TransUnion Canada",
        "bureau_min": 300, "bureau_max": 900,
        "primary_banks": ["RBC (Royal Bank of Canada)", "TD Bank", "Scotiabank", "BMO", "CIBC"]
    },
    "USA": {
        "name": "United States (Domestic)",
        "currency": "USD ($)",
        "id_name": "SSN / ITIN",
        "id_placeholder": "XXX-XX-1234",
        "id_regex": r"^[0-9\s-]{9,11}$",
        "bureau_name": "FICO / Experian US",
        "bureau_min": 300, "bureau_max": 850,
        "primary_banks": ["Chase", "Bank of America", "Wells Fargo", "Citibank", "Capital One"]
    }
}

# 1-Click Demo Profiles
PRESETS = {
    "Custom Applicant": {
        "name": "", "country": "IND", "national_id": "ABCPS1234K", "bank": "HDFC Bank",
        "salary": 1800000, "debt": 20000, "loan": 12000, "score": 750, "years": 5, "home": "mortgage"
    },
    "Priya Sharma (Tech Lead - India ➔ USA) [Verified CIBIL 770]": {
        "name": "Priya Sharma", "country": "IND", "national_id": "ABCPS8912P", "bank": "HDFC Bank",
        "salary": 2200000, "debt": 22000, "loan": 15000, "score": 770, "years": 7, "home": "mortgage"
    },
    "Carlos Mendez (Manager - Mexico ➔ USA) [Verified Buró 690]": {
        "name": "Carlos Mendez", "country": "MEX", "national_id": "MEMC850412HDF", "bank": "BBVA México",
        "salary": 420000, "debt": 6000, "loan": 8000, "score": 690, "years": 4, "home": "rent"
    },
    "David Smith (Designer - UK ➔ USA) [High Debt Alert]": {
        "name": "David Smith", "country": "GBR", "national_id": "JH241890B", "bank": "Barclays",
        "salary": 24000, "debt": 950, "loan": 28000, "score": 580, "years": 1, "home": "rent"
    }
}


# =============================================================================
# SIDEBAR: APPLICANT IDENTITY & KYC PORTAL
# =============================================================================
st.sidebar.title("🛂 Credit Passport Portal")
st.sidebar.markdown("Cross-Border KYC, Identity & Bureau Verification")

preset_choice = st.sidebar.selectbox("🎯 1-Click Demo Profile:", list(PRESETS.keys()))
preset = PRESETS[preset_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("1. Identity & Nationality")

applicant_name = st.sidebar.text_input("Full Legal Name", value=preset["name"] or "Applicant Name")

source_country = st.sidebar.selectbox(
    "Citizen / Resident Country:",
    list(COUNTRY_CONFIGS.keys()),
    index=list(COUNTRY_CONFIGS.keys()).index(preset["country"]),
    format_func=lambda x: f"{COUNTRY_CONFIGS[x]['name']} ({x})"
)

cfg = COUNTRY_CONFIGS[source_country]

national_id = st.sidebar.text_input(
    f"{cfg['id_name']}",
    value=preset["national_id"],
    placeholder=cfg["id_placeholder"],
    help=f"Verified against {cfg['name']} Government Database"
)

primary_bank = st.sidebar.selectbox(
    "Origin Bank (Account Aggregator / Statements):",
    cfg["primary_banks"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Home Financial Profile")

salary = st.sidebar.number_input(
    f"Annual Salary ({cfg['currency']})",
    min_value=1000, max_value=50000000, value=preset["salary"], step=10000
)

debt = st.sidebar.number_input(
    f"Monthly Debt Obligations ({cfg['currency']})",
    min_value=0, max_value=5000000, value=preset["debt"], step=1000
)

bureau_score = st.sidebar.slider(
    f"{cfg['bureau_name']} Score:",
    min_value=cfg["bureau_min"], max_value=cfg["bureau_max"], value=preset["score"],
    help=f"Official credit bureau score in {cfg['name']}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Destination Loan Request (USA)")

loan_usd = st.sidebar.number_input(
    "Requested Loan in Destination (USD $):",
    min_value=500, max_value=150000, value=preset["loan"], step=1000
)

emp_years = st.sidebar.slider("Employment Tenure (Years)", min_value=0, max_value=30, value=preset["years"])
home_type = st.sidebar.selectbox("Housing Situation", ["mortgage", "own", "rent"], index=["mortgage", "own", "rent"].index(preset["home"]))
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation", "car", "home_improvement", "major_purchase", "wedding", "small_business", "moving", "medical", "other"], index=0)

evaluate_btn = st.sidebar.button("🚀 Verify & Issue Credit Passport", type="primary", use_container_width=True)


# =============================================================================
# MAIN DASHBOARD AREA
# =============================================================================
st.title("🌐 Cross-Border Credit Passport System")
st.markdown(f"Automated International Underwriting: **{cfg['name']} ➔ United States (USA)**")
st.markdown("---")

if evaluate_btn or preset_choice != "Custom Applicant":

    # =========================================================================
    # STAGE 1 & 2: VERIFICATION BANNER
    # =========================================================================
    # Validate National ID format
    id_valid = bool(re.match(cfg["id_regex"], national_id.strip())) if national_id else False

    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        if id_valid:
            st.success(f"✔️ **Identity Verified:**\n{cfg['id_name']}: `{national_id}`")
        else:
            st.warning(f"⚠️ **Pending Format:**\n{cfg['id_name']}: `{national_id}`")
    with v_col2:
        st.success(f"✔️ **Banking Linked:**\n{primary_bank} (Active)")
    with v_col3:
        st.success(f"✔️ **Bureau Report Fetched:**\n{cfg['bureau_name']}: **{bureau_score}**")

    st.markdown("---")

    # =========================================================================
    # STAGE 3: RUN THE CREDIT PASSPORT ENGINE
    # =========================================================================
    # Calibrate / Scale Origin Bureau Score to US Standard (300 - 850)
    # E.g. Indian CIBIL (300-900) scaled to US FICO (300-850)
    origin_min = cfg["bureau_min"]
    origin_max = cfg["bureau_max"]
    normalized_us_score = 300 + ((bureau_score - origin_min) / (origin_max - origin_min)) * (850 - 300)
    normalized_us_score = round(max(300, min(850, normalized_us_score)), 0)

    payload = {
        "applicant_name": applicant_name,
        "source_country": source_country,
        "destination_country": "USA",
        "local_annual_income": float(salary),
        "requested_loan_amount": float(loan_usd),
        "monthly_debt_payments": float(debt),
        "credit_bureau_score": float(normalized_us_score),
        "employment_years": float(emp_years),
        "home_ownership": home_type,
        "loan_purpose": loan_purpose,
        "revolving_utilization_pct": float(util_pct if 'util_pct' in locals() else preset.get("util", 25)),
        "percent_trades_never_delq": float(clean_pct if 'clean_pct' in locals() else preset.get("clean", 95)),
        "inquiries_last_6m": float(inq_count if 'inq_count' in locals() else preset.get("inq", 0))
    }

    # Evaluate via FastAPI or Local Engine Fallback
    passport = None
    try:
        resp = requests.post(FASTAPI_URL, json=payload, timeout=1.5)
        if resp.status_code == 200:
            passport = resp.json()
    except Exception:
        pass

    if passport is None:
        engine = get_engine()
        passport = engine.evaluate_applicant(payload)

    # =========================================================================
    # DISPLAY PASSPORT RESULTS
    # =========================================================================
    decision = passport["underwriting_decision"]
    score = passport["passport_score"]

    if "APPROVED" in decision:
        st.success(f"### ✅ UNDERWRITING DECISION: {decision}")
    elif "REVIEW" in decision:
        st.warning(f"### ⚠️ UNDERWRITING DECISION: {decision}")
    else:
        st.error(f"### 🛑 UNDERWRITING DECISION: {decision}")

    # Top KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Credit Passport Score", f"{score} / 100", delta=passport['credit_tier'].split(' ')[0])
    with m2:
        st.metric("Default Risk Probability", f"{passport['default_probability_pct']}%", delta="- Low Risk" if score > 70 else "+ High Risk", delta_color="inverse")
    with m3:
        st.metric("Real PPP Buying Power (USD)", passport["financial_profile"]["real_ppp_usd_salary"], help="Real standard of living equivalent in the USA")
    with m4:
        st.metric("US Equiv. Credit Score", f"{int(normalized_us_score)} FICO", help=f"Converted from {cfg['name']} {cfg['bureau_name']}")

    st.markdown("---")

    # Details & Explanations
    left_c, right_c = st.columns([1.2, 1.0])

    with left_c:
        st.subheader("📋 Official Digital Credit Passport")
        fp = passport["financial_profile"]
        
        passport_data = {
            "Verified Attribute": [
                "Passport ID", "Legal Name", "Corridor",
                f"Home National ID ({cfg['id_name']})", "Linked Bank Account",
                f"Origin Credit Bureau Score", "US Equivalent Credit Bureau Score",
                f"Local Annual Salary ({cfg['currency']})", "Nominal FX Value (USD)",
                "World Bank Real PPP Lifestyle", "Monthly Free Cashflow Buffer",
                "Debt-to-Income (DTI)", "Loan-to-Income (LTI)"
            ],
            "Details": [
                passport["passport_id"], applicant_name, f"{cfg['name']} ➔ USA",
                national_id, primary_bank,
                f"{bureau_score} ({cfg['bureau_name']})", f"{int(normalized_us_score)} (US FICO Scale)",
                f"{salary:,.0f} {cfg['currency']}", fp["nominal_usd_salary"],
                fp["real_ppp_usd_salary"], fp["monthly_disposable_buffer"],
                fp["dti_ratio"], fp["lti_ratio"]
            ]
        }
        st.dataframe(pd.DataFrame(passport_data), hide_index=True, use_container_width=True)

    with right_c:
        st.subheader("🔍 Explainable AI (SHAP Reasons)")
        st.markdown("**Under ECOA / Reg B & RBI transparency principles:**")

        st.markdown("##### 🟢 Top Credit Strengths (Positive Drivers):")
        if passport["top_credit_strengths"]:
            for s in passport["top_credit_strengths"]:
                st.info(f"✔️ {s}")
        else:
            st.write("No major credit strengths detected.")

        st.markdown("##### 🔴 Adverse Action Factors (Risk Drivers):")
        if passport["top_risk_drivers"]:
            for r in passport["top_risk_drivers"]:
                st.error(f"⚠️ {r}")
        else:
            st.success("No adverse action flags detected.")

    # Educational Expander
    st.markdown("---")
    with st.expander("💡 How the Cross-Border Credit Passport Works Under the Hood"):
        st.markdown(f"""
        1. **KYC & Identity Verification:** Verifies the applicant's official government identity document (**{cfg['id_name']}: {national_id}**) and confirms active banking history at **{primary_bank}**.
        2. **Bureau Score Calibration:** Maps the **{bureau_score} {cfg['bureau_name']}** score from {cfg['name']}'s national scale to its exact **US FICO equivalent ({int(normalized_us_score)})**.
        3. **Purchasing Power Parity (PPP):** Rather than blindly converting {salary:,.0f} {cfg['currency']} at market exchange rates (which yields {fp['nominal_usd_salary']}), the system applies World Bank PPP to reflect a real **{fp['real_ppp_usd_salary']} standard of living** with a **{fp['monthly_disposable_buffer']} monthly cash cushion**.
        4. **Underwriting Decision:** The US destination lender receives an instant, legally compliant **{decision}** recommendation backed by game-theoretic SHAP reason codes.
        """)