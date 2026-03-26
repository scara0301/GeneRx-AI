"""
build_model.py — Synthetic training pipeline for HF Spaces deployment.

Generates a realistic training dataset from the known drug catalog
and trains the drug risk model. No external data downloads required.
Run at Docker build time: python backend/build_model.py
"""
import os
import pickle
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ──────────────────────────────────────────────────────────────────────────────
# Drug profiles: (base_risk_category, side_effect_count, se_mean_freq, se_max_freq)
# risk_category: 0=Low, 1=Moderate, 2=High, 3=Critical
# ──────────────────────────────────────────────────────────────────────────────
DRUG_PROFILES = {
    "metformin":      (0, 5,  0.18, 0.30),
    "atorvastatin":   (0, 5,  0.05, 0.10),
    "amlodipine":     (0, 5,  0.11, 0.20),
    "ramipril":       (1, 5,  0.07, 0.20),
    "metoprolol":     (1, 5,  0.12, 0.20),
    "warfarin":       (2, 5,  0.16, 0.35),
    "amoxicillin":    (0, 5,  0.12, 0.20),
    "ibuprofen":      (1, 5,  0.13, 0.20),
    "acetaminophen":  (0, 3,  0.01, 0.02),  # paracetamol alias
    "omeprazole":     (0, 5,  0.04, 0.07),
}

DRUG_NAMES = list(DRUG_PROFILES.keys())
N_SAMPLES = 12_000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_dataset():
    rows = []
    drug_idx = {d: i for i, d in enumerate(DRUG_NAMES)}

    for _ in range(N_SAMPLES):
        drug = random.choice(DRUG_NAMES)
        base_risk, se_count, se_mean, se_max = DRUG_PROFILES[drug]

        age    = np.clip(np.random.normal(55, 18), 18, 95)
        sex    = random.choice([0, 1])          # 0=M, 1=F
        weight = np.clip(np.random.normal(75, 18), 40, 150)
        n_conc = random.randint(0, 8)           # concomitant drugs
        n_reac = random.randint(0, 6)           # number of reactions reported
        w_reac = round(np.clip(np.random.normal(1.2, 0.6), 0.1, 4.0), 2)
        is_ps  = random.choice([0, 1])          # primary suspect flag
        se_cnt = int(se_count + random.randint(-1, 2))
        se_mf  = round(np.clip(se_mean + np.random.normal(0, 0.05), 0, 1), 3)
        se_xf  = round(np.clip(se_max  + np.random.normal(0, 0.05), 0, 1), 3)

        # Riskmodifiers
        risk = base_risk
        if age > 75:        risk = min(3, risk + 1)
        if n_conc >= 5:     risk = min(3, risk + 1)
        if n_reac >= 4:     risk = min(3, risk + 1)
        if w_reac > 2.5:    risk = min(3, risk + 1)
        if is_ps == 1 and n_reac > 2: risk = min(3, risk + 1)
        # Add noise to prevent trivial overfitting
        if random.random() < 0.10:
            risk = max(0, min(3, risk + random.choice([-1, 1])))

        rows.append([
            drug_idx[drug],   # drug_encoded
            age,
            sex,
            weight,
            n_reac,           # reaction_count
            w_reac,           # weighted_reaction_score
            is_ps,            # is_primary_suspect
            n_conc,           # num_concomitant_drugs
            se_cnt,           # known_side_effects_count
            se_mf,            # mean_se_frequency
            se_xf,            # max_se_frequency
            risk,             # risk_category (target)
        ])

    return rows, drug_idx


def train():
    print("=" * 60)
    print(" GeneRx-AI — Building Synthetic Training Dataset")
    print("=" * 60)

    rows, drug_idx = generate_dataset()

    import numpy as np
    data = np.array(rows, dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    feature_cols = [
        "drug_encoded", "patient_age", "sex_encoded", "patient_weight",
        "reaction_count", "weighted_reaction_score", "is_primary_suspect",
        "num_concomitant_drugs", "known_side_effects_count",
        "mean_se_frequency", "max_se_frequency",
    ]

    # Build a LabelEncoder-compatible object from drug_idx
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(sorted(drug_idx, key=lambda d: drug_idx[d]))

    # Train / test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Try XGBoost, fall back to GradientBoosting
    try:
        from xgboost import XGBClassifier
        print("\nTraining XGBoost classifier...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("\nXGBoost not found — using GradientBoosting fallback...")
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_SEED
        )

    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "drug_risk_model.pkl")

    artifacts = {
        "model":         model,
        "label_encoder": le,
        "feature_cols":  feature_cols,
        "risk_labels":   ["Low Risk", "Moderate", "High Risk", "Critical"],
        "accuracy":      accuracy,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Model saved → {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
