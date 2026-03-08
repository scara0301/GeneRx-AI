"""
server.py — FastAPI backend for GeneRx-AI
Combines ML predictions with clinical rules engine.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from clinical_engine import (
    DRUG_CATALOG,
    evaluate_drug,
    check_drug_interactions,
)
from ml_model import get_predictor

app = FastAPI(
    title="GeneRx-AI API",
    description="Clinical drug safety assessment powered by ML and evidence-based rules",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PatientProfile(BaseModel):
    name: str = "Patient"
    age: int = 50
    sex: str = "F"
    weight_kg: float = 70.0
    conditions: List[str] = []
    egfr: float = 90.0
    alt: float = 25.0
    ast: float = 25.0
    hba1c: float = 5.5
    systolic_bp: int = 120
    diastolic_bp: int = 80
    ldl: float = 3.0
    inr: float = 1.0
    current_meds: List[str] = []
    allergies: List[str] = []


class AssessmentRequest(BaseModel):
    patient: PatientProfile
    drugs: List[str]


class InteractionRequest(BaseModel):
    drugs: List[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    predictor = get_predictor()
    return {
        "status": "ok",
        "ml_model_loaded": predictor.loaded,
        "drugs_available": len(DRUG_CATALOG),
    }


@app.get("/api/drugs")
def get_drugs():
    """Return drug catalog with basic info."""
    drugs = []
    for name, info in DRUG_CATALOG.items():
        drugs.append({
            "name": name,
            "category": info.get("category", ""),
            "description": info.get("description", ""),
        })
    return {"drugs": drugs}


@app.post("/api/assess")
def assess_drugs(request: AssessmentRequest):
    """Assess drugs for a patient — combines ML model + clinical rules."""
    patient = request.patient
    predictor = get_predictor()

    # Build patient profile dict for rules engine
    patient_dict = {
        "name": patient.name,
        "patient_id": "API",
        "age": patient.age,
        "sex": patient.sex,
        "weight_kg": patient.weight_kg,
        "bmi": patient.weight_kg / (1.70 ** 2),
        "conditions": patient.conditions,
        "egfr": patient.egfr,
        "alt": patient.alt,
        "ast": patient.ast,
        "hba1c": patient.hba1c,
        "systolic_bp": patient.systolic_bp,
        "diastolic_bp": patient.diastolic_bp,
        "ldl": patient.ldl,
        "inr": patient.inr,
        "current_meds": patient.current_meds,
        "allergies": patient.allergies,
    }

    results = []
    for drug_name in request.drugs:
        # Clinical rules assessment
        rule_result = evaluate_drug(patient_dict, drug_name)

        # ML prediction
        ml_result = predictor.predict(
            drug_name=drug_name,
            patient_age=patient.age,
            patient_sex=patient.sex,
            patient_weight=patient.weight_kg,
            num_concomitant_drugs=len(patient.current_meds),
        )

        # Combine results
        combined = {
            "drug_name": drug_name,
            "suitability": rule_result["suitability"],
            "risk_level": rule_result["risk_level"],
            "reasons": rule_result["reasons"],
            "warnings": rule_result["warnings"],
            "dose_notes": rule_result["dose_notes"],
            "monitoring": rule_result["monitoring"],
            "side_effects": rule_result["side_effects"],
            "ml_prediction": ml_result,
        }
        results.append(combined)

    # Drug interactions
    all_drugs = request.drugs + patient.current_meds
    interactions = check_drug_interactions(all_drugs)

    return {
        "patient_name": patient.name,
        "assessments": results,
        "interactions": interactions,
    }


@app.post("/api/interactions")
def check_interactions(request: InteractionRequest):
    """Check drug-drug interactions."""
    interactions = check_drug_interactions(request.drugs)
    return {"interactions": interactions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
