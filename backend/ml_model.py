"""
ml_model.py — Load trained model and make predictions.
Combines ML predictions with clinical rules for robust assessments.
"""
import os
import pickle
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DrugRiskPredictor:
    """Loads the trained drug risk model and provides predictions."""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.risk_labels = ["Low Risk", "Moderate", "High Risk", "Critical"]
        self.loaded = False
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, "drug_risk_model.pkl")
        if not os.path.exists(model_path):
            print(f"[ML Model] No trained model found at {model_path}")
            return

        try:
            with open(model_path, "rb") as f:
                artifacts = pickle.load(f)
            self.model = artifacts["model"]
            self.label_encoder = artifacts["label_encoder"]
            self.feature_cols = artifacts["feature_cols"]
            self.risk_labels = artifacts.get("risk_labels", self.risk_labels)
            self.loaded = True
            print(f"[ML Model] Loaded successfully (accuracy: {artifacts.get('accuracy', 'N/A')})")
        except Exception as e:
            print(f"[ML Model] Error loading model: {e}")

    def predict(self, drug_name, patient_age=50, patient_sex="F",
                patient_weight=70, reaction_count=0, num_concomitant_drugs=0,
                known_side_effects=0, mean_se_freq=0, max_se_freq=0):
        """
        Predict risk category for a drug-patient combination.
        Returns: dict with score, category, confidence, and label.
        """
        if not self.loaded:
            return {
                "score": None,
                "category": None,
                "label": "Model not available",
                "confidence": 0,
                "available": False,
            }

        # Encode drug name
        drug_lower = drug_name.lower().strip()
        # Handle paracetamol/acetaminophen alias
        if drug_lower == "paracetamol":
            drug_lower = "acetaminophen"

        try:
            drug_encoded = self.label_encoder.transform([drug_lower])[0]
        except ValueError:
            # Drug not in training data
            return {
                "score": None,
                "category": None,
                "label": f"Drug '{drug_name}' not in training data",
                "confidence": 0,
                "available": False,
            }

        # Encode sex
        sex_encoded = 1 if patient_sex.upper() in ("F", "FEMALE") else 0

        # Build feature vector matching the 11-feature schema
        # For new assessments, we use neutral defaults for reaction-based features
        # since the reactions haven't happened yet.
        weighted_reaction_score = 1.0  # Neutral baseline
        is_primary_suspect = 1        # Assessed as the target drug

        features = np.array([[
            drug_encoded,
            patient_age,
            sex_encoded,
            patient_weight,
            reaction_count,
            weighted_reaction_score,
            is_primary_suspect,
            num_concomitant_drugs,
            known_side_effects,
            mean_se_freq,
            max_se_freq,
        ]])

        # Predict
        try:
            pred_class = int(self.model.predict(features)[0])
            pred_proba = self.model.predict_proba(features)[0]
            confidence = float(max(pred_proba)) * 100

            return {
                "score": float(pred_class) / 3.0,  # Normalize to 0-1
                "category": pred_class,
                "label": self.risk_labels[pred_class],
                "confidence": round(float(confidence), 1),
                "probabilities": {
                    self.risk_labels[i]: round(float(p) * 100.0, 1)
                    for i, p in enumerate(pred_proba)
                    if i < len(self.risk_labels)
                },
                "available": True,
            }
        except Exception as e:
            return {
                "score": None,
                "category": None,
                "label": f"Prediction error: {str(e)}",
                "confidence": 0,
                "available": False,
            }


# Singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DrugRiskPredictor()
    return _predictor
