"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 12: PRODUCTION FASTAPI MICROSERVICE (step12_api_backend.py)
=============================================================================
What this API provides:
1. GET  /                   -> API Welcome & Metadata
2. GET  /health             -> Microservice Health & Liveness Probe
3. GET  /api/v1/corridors   -> Supported Migration Corridors & World Bank PPP rates
4. POST /api/v1/passport/evaluate -> Live Credit Passport Evaluation & SHAP XAI
5. GET  /docs               -> Interactive Swagger UI for live browser testing
=============================================================================
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =============================================================================
# BULLETPROOF PATH RESOLUTION & ENGINE IMPORT
# =============================================================================
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

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================
app = FastAPI(
    title="Cross-Border Credit Passport API",
    description="Enterprise Underwriting Microservice powered by ML (ROC-AUC ~0.79), World Bank PPP Normalization, and SHAP Explainability.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(" -> Initializing High-Accuracy Credit Passport Engine for API service...")
engine = CreditPassportEngine()


# =============================================================================
# PYDANTIC DATA VALIDATION SCHEMAS
# =============================================================================
class ApplicantRequestSchema(BaseModel):
    applicant_name: str = Field(..., example="Priya Sharma", description="Full legal name of the applicant")
    source_country: str = Field(..., example="IND", description="ISO 3-letter origin country code (IND, GBR, MEX, CAN, USA)")
    destination_country: str = Field(default="USA", example="USA", description="Target lending market")
    local_annual_income: float = Field(..., gt=0, example=2200000.0, description="Annual income in local home currency")
    requested_loan_amount: float = Field(..., gt=0, example=12000.0, description="Requested loan principal in destination currency (USD)")
    monthly_debt_payments: float = Field(default=0.0, ge=0, example=20000.0, description="Current monthly debt obligations in home currency")
    credit_bureau_score: float = Field(default=720.0, ge=300, le=850, example=755.0, description="Origin country bureau score (e.g. CIBIL, Experian)")
    employment_years: float = Field(default=5.0, ge=0, le=50, example=7.0, description="Documented continuous employment tenure")
    home_ownership: str = Field(default="mortgage", example="mortgage", description="'rent', 'own', or 'mortgage'")
    loan_purpose: str = Field(default="car", example="car", description="Loan purpose")
    # Optional origin bureau trade line signals
    revolving_utilization_pct: Optional[float] = Field(default=None, ge=0, le=100, example=14.0, description="Revolving credit card utilization percentage")
    percent_trades_never_delq: Optional[float] = Field(default=None, ge=0, le=100, example=100.0, description="Percentage of historical trade accounts with zero late payments")
    num_satisfactory_trades: Optional[float] = Field(default=None, ge=0, example=22.0, description="Number of fully repaid, satisfactory trade accounts")
    inquiries_last_6m: Optional[float] = Field(default=None, ge=0, example=0.0, description="Number of hard credit inquiries in the past 6 months")


class PassportResponseSchema(BaseModel):
    status: str
    applicant_name: str
    source_country: str
    destination_country: str
    passport_id: str
    passport_score: int
    credit_tier: str
    default_probability_pct: float
    underwriting_decision: str
    decision_color: str
    financial_profile: Dict[str, Any]
    top_credit_strengths: List[str]
    top_risk_drivers: List[str]


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.get("/", tags=["General"])
def root():
    """Welcome endpoint with system metadata."""
    return {
        "system": "Cross-Border Credit Passport API",
        "status": "online",
        "version": "3.0.0",
        "champion_model": engine.model_name,
        "champion_roc_auc": engine.model_auc,
        "interactive_documentation": "/docs"
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Microservice health and liveness probe."""
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "champion_roc_auc": engine.model_auc,
        "explainer_loaded": engine.explainer is not None
    }


@app.get("/api/v1/corridors", tags=["Configuration"])
def get_supported_corridors():
    """Returns supported migration corridors and World Bank PPP rates."""
    return {
        "supported_countries": list(PPP_FACTORS.keys()),
        "world_bank_ppp_factors": PPP_FACTORS,
        "market_fx_rates": FX_FACTORS
    }


@app.post("/api/v1/passport/evaluate", response_model=PassportResponseSchema, tags=["Underwriting"])
def evaluate_credit_passport(applicant: ApplicantRequestSchema):
    """
    Main Underwriting Endpoint:
    Normalizes local currency via World Bank PPP, derives universal cashflow ratios,
    evaluates origin bureau behavioral trade lines with XGBoost (AUC ~0.79), and computes SHAP reason codes.
    """
    try:
        input_data = {k: v for k, v in applicant.model_dump().items() if v is not None}
        result = engine.evaluate_applicant(input_data)
        result["status"] = "success"
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Underwriting evaluation failed: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" LAUNCHING HIGH-ACCURACY CREDIT PASSPORT FASTAPI SERVER")
    print("=" * 70)
    print(" Server URL       : http://127.0.0.1:8000")
    print(" Interactive Docs : http://127.0.0.1:8000/docs")
    print("=" * 70 + "\n")
    uvicorn.run("step12_api_backend:app", host="127.0.0.1", port=8000, reload=True)
