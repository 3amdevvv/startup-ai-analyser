"""
predictor.py
------------
Loads all saved models from saved_models/ and exposes a single
`predict(startup_dict) -> dict` function that mirrors the notebook
logic exactly.
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "saved_models"

# Feature names (must match notebook order)
BASE_FEATURES = [
    "funding_rounds",
    "founder_experience_years",
    "team_size",
    "market_size_billion",
    "product_traction_users",
    "burn_rate_million",
    "revenue_million",
    "investor_type_enc",
    "sector_enc",
    "founder_background_enc",
    "revenue_per_employee",
    "burn_efficiency",
    "traction_per_employee",
    "market_penetration",
    "funding_per_round",
    "exp_team_ratio",
    "revenue_to_market",
    "log_revenue",
    "log_traction",
]

CAT_COLS = ["investor_type", "sector", "founder_background"]

MULTIPLE_MAP = {"none": 3, "angel": 5, "tier2_vc": 8, "tier1_vc": 15}

VALID_INVESTOR_TYPES = {"none", "angel", "tier2_vc", "tier1_vc"}
VALID_FOUNDER_BACKGROUNDS = {"first_time", "academic", "ex_bigtech", "serial"}


class StartupPredictor:
    """Loads all models once and provides a predict() method."""

    def __init__(self):
        self._ready = False
        self._load_models()

    def _load_models(self):
        try:
            self.outcome_model = joblib.load(MODELS_DIR / "outcome_classifier.pkl")
            self.rev_model     = joblib.load(MODELS_DIR / "revenue_regressor.pkl")
            self.burn_model    = joblib.load(MODELS_DIR / "burn_risk_classifier.pkl")
            self.fund_model    = joblib.load(MODELS_DIR / "funding_rounds_regressor.pkl")
            self.acq_model     = joblib.load(MODELS_DIR / "acquisition_classifier.pkl")
            self.ipo_model     = joblib.load(MODELS_DIR / "ipo_classifier.pkl")
            self.scaler        = joblib.load(MODELS_DIR / "scaler.pkl")
            self.le_dict       = joblib.load(MODELS_DIR / "label_encoders.pkl")
            self.le_outcome    = joblib.load(MODELS_DIR / "outcome_label_encoder.pkl")
            self._ready = True
            logger.info("✅ All models loaded successfully from %s", MODELS_DIR)
        except FileNotFoundError as e:
            logger.error(
                "❌ Model file not found: %s\n"
                "   Place the saved_models/ folder next to this file.\n"
                "   Run the notebook to generate models first.",
                e,
            )

    def is_ready(self) -> bool:
        return self._ready

    # ── Validation ─────────────────────────────────────────────────────────────
    def _validate(self, d: dict):
        if d["investor_type"] not in VALID_INVESTOR_TYPES:
            raise ValueError(
                f"investor_type must be one of {sorted(VALID_INVESTOR_TYPES)}, "
                f"got '{d['investor_type']}'"
            )
        if d["founder_background"] not in VALID_FOUNDER_BACKGROUNDS:
            raise ValueError(
                f"founder_background must be one of {sorted(VALID_FOUNDER_BACKGROUNDS)}, "
                f"got '{d['founder_background']}'"
            )
        # sector is validated by LabelEncoder — will raise if unknown

    # ── Feature Engineering (mirrors notebook) ─────────────────────────────────
    @staticmethod
    def _engineer(row: pd.DataFrame) -> pd.DataFrame:
        row = row.copy()
        row["revenue_per_employee"]  = row["revenue_million"] / (row["team_size"] + 1)
        row["burn_efficiency"]       = row["revenue_million"] / (row["burn_rate_million"] + 0.01)
        row["traction_per_employee"] = row["product_traction_users"] / (row["team_size"] + 1)
        row["market_penetration"]    = row["product_traction_users"] / (row["market_size_billion"] * 1e6 + 1)
        row["funding_per_round"]     = row["burn_rate_million"] * 12 / (row["funding_rounds"] + 1)
        row["exp_team_ratio"]        = row["founder_experience_years"] / (row["team_size"] + 1)
        row["revenue_to_market"]     = row["revenue_million"] / (row["market_size_billion"] * 1e6 + 1)
        row["log_revenue"]           = np.log1p(row["revenue_million"])
        row["log_traction"]          = np.log1p(row["product_traction_users"])
        return row

    # ── Main predict method ────────────────────────────────────────────────────
    def predict(self, startup_dict: dict) -> dict:
        if not self._ready:
            raise RuntimeError(
                "Models not loaded. Ensure saved_models/ folder is present and "
                "contains all .pkl files."
            )

        self._validate(startup_dict)

        row = pd.DataFrame([startup_dict])

        # Encode categoricals
        for c in CAT_COLS:
            try:
                row[c + "_enc"] = self.le_dict[c].transform(row[c])
            except ValueError:
                known = list(self.le_dict[c].classes_)
                raise ValueError(f"Unknown value '{row[c].iloc[0]}' for '{c}'. Known: {known}")

        # Feature engineering
        row = self._engineer(row)

        # Scale
        X = self.scaler.transform(row[BASE_FEATURES])

        # ── Model inference ────────────────────────────────────────────────────
        outcome_proba   = self.outcome_model.predict_proba(X)[0]
        predicted_class = self.le_outcome.inverse_transform([outcome_proba.argmax()])[0]

        rev_log     = self.rev_model.predict(X)[0]
        revenue_pred = float(np.expm1(rev_log))

        burn_risk   = float(self.burn_model.predict_proba(X)[0][1])

        fund_pred   = int(np.clip(round(self.fund_model.predict(X)[0]), 1, None))

        acq_prob    = float(self.acq_model.predict_proba(X)[0][1])
        ipo_prob    = float(self.ipo_model.predict_proba(X)[0][1])

        # Valuation estimate
        mult          = MULTIPLE_MAP.get(startup_dict["investor_type"], 5)
        valuation_est = (startup_dict["revenue_million"] / 1e6) * mult

        # Investor risk score
        failure_idx = list(self.le_outcome.classes_).index("Failure")
        val_norm    = 0.5  # placeholder, same as notebook
        risk_score  = (
            0.5 * outcome_proba[failure_idx]
            + 0.3 * burn_risk
            + 0.2 * (1 - val_norm)
        )

        # Risk labels
        burn_label = "HIGH" if burn_risk > 0.5 else "LOW"
        if risk_score < 0.3:
            risk_label = "Low"
        elif risk_score < 0.5:
            risk_label = "Moderate"
        elif risk_score < 0.7:
            risk_label = "High"
        else:
            risk_label = "Very High"

        return {
            "predicted_outcome":        str(predicted_class),
            "outcome_probabilities":    {
                cls: round(float(p), 4)
                for cls, p in zip(self.le_outcome.classes_, outcome_proba)
            },
            "revenue_prediction":       round(revenue_pred, 2),
            "burn_rate_risk":           round(burn_risk, 4),
            "burn_rate_risk_label":     burn_label,
            "predicted_funding_rounds": fund_pred,
            "acquisition_likelihood":   round(acq_prob, 4),
            "ipo_potential_score":      round(ipo_prob, 4),
            "valuation_estimate_M":     round(float(valuation_est), 2),
            "investor_risk_score":      round(float(risk_score), 4),
            "investor_risk_label":      risk_label,
        }