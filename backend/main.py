from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict
import uvicorn

from predictor import StartupPredictor
#from multiagent.router import router as multiagent_router, init_predictor

app = FastAPI(
    title="Startup Success Prediction API",
    description="Single-model prediction + 9-agent LangGraph pipeline for comprehensive startup analysis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = StartupPredictor()
#init_predictor(predictor)
#app.include_router(multiagent_router)


class StartupInput(BaseModel):
    funding_rounds: int = Field(..., ge=0, le=20, example=3)
    founder_experience_years: float = Field(..., ge=0, le=50, example=10)
    team_size: int = Field(..., ge=1, le=1000, example=45)
    market_size_billion: float = Field(..., gt=0, example=25.0)
    product_traction_users: int = Field(..., ge=0, example=500000)
    burn_rate_million: float = Field(..., gt=0, example=8.5)
    revenue_million: float = Field(..., ge=0, example=1200000)
    investor_type: str = Field(..., example="tier1_vc")
    sector: str = Field(..., example="AI")
    founder_background: str = Field(..., example="ex_bigtech")


class PredictionResponse(BaseModel):
    predicted_outcome: str
    outcome_probabilities: Dict[str, float]
    revenue_prediction: float
    burn_rate_risk: float
    burn_rate_risk_label: str
    predicted_funding_rounds: int
    acquisition_likelihood: float
    ipo_potential_score: float
    valuation_estimate_M: float
    investor_risk_score: float
    investor_risk_label: str


@app.get("/")
def root():
    return {
        "message": "Startup Success Prediction API v2",
        "docs": "/docs",
        "predict": "/predict  [POST]",
        "multiagent": "/multiagent/analyze  [POST, SSE]",
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": predictor.is_ready(), "version": "2.0.0"}


@app.get("/options")
def options():
    return {
        "investor_type": ["none", "angel", "tier2_vc", "tier1_vc"],
        "sector": ["AI", "Fintech", "Health", "SaaS", "Ecommerce", "EdTech", "CleanTech", "BioTech"],
        "founder_background": ["first_time", "academic", "ex_bigtech", "serial"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(startup: StartupInput):
    try:
        return predictor.predict(startup.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)