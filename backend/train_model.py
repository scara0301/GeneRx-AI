"""
train_model.py — Train XGBoost drug risk prediction model.
Uses merged FAERS + SIDER training data.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train():
    print("=" * 60)
    print("Training Optimized Drug Risk Prediction Model")
    print("=" * 60)

    # Load training data
    data_path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} training samples")

    # Encode drug names
    le_drug = LabelEncoder()
    df["drug_encoded"] = le_drug.fit_transform(df["drug_name"])

    # Optimized Feature columns
    feature_cols = [
        "drug_encoded",
        "patient_age",
        "sex_encoded",
        "patient_weight",
        "reaction_count",
        "weighted_reaction_score",
        "is_primary_suspect",
        "num_concomitant_drugs",
        "known_side_effects_count",
        "mean_se_frequency",
        "max_se_frequency",
    ]

    X = df[feature_cols].copy()
    y = df["risk_category"].copy()

    # Fill NaNs
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if HAS_XGBOOST:
        print("\nTuning XGBoost hyperparameters...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb = XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False)
        random_search = RandomizedSearchCV(
            xgb, param_distributions=param_dist, n_iter=10, 
            scoring='accuracy', n_jobs=-1, cv=3, random_state=42, verbose=1
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        print(f"Best params: {random_search.best_params_}")
    else:
        print("\nXGBoost not found, using baseline GradientBoosting...")
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    risk_labels = ["Low Risk", "Moderate", "High Risk", "Critical"]
    present_labels = sorted(y_test.unique())
    target_names = [risk_labels[i] for i in present_labels]

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, labels=present_labels))

    # Save model artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "drug_risk_model.pkl")

    artifacts = {
        "model": model,
        "label_encoder": le_drug,
        "feature_cols": feature_cols,
        "risk_labels": risk_labels,
        "accuracy": accuracy,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nModel saved to: {model_path}")
    return model, le_drug


if __name__ == "__main__":
    train()
