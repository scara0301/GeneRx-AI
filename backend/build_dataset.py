"""
build_dataset.py — Merge FAERS + SIDER data into ML training dataset.
Combines adverse event reports with side effect profiles to create
feature-rich training data for drug risk prediction.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FAERS_DIR = os.path.join(DATA_DIR, "faers")
SIDER_DIR = os.path.join(DATA_DIR, "sider")


def load_faers():
    """Load FAERS adverse events data."""
    path = os.path.join(FAERS_DIR, "faers_events.csv")
    if not os.path.exists(path):
        print(f"FAERS data not found at {path}")
        print("Run fetch_faers.py first.")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} FAERS events")
    return df


def load_sider():
    """Load SIDER side effects data."""
    se_path = os.path.join(SIDER_DIR, "side_effects.csv")
    freq_path = os.path.join(SIDER_DIR, "side_effect_frequencies.csv")

    df_se = None
    df_freq = None

    if os.path.exists(se_path):
        df_se = pd.read_csv(se_path)
        print(f"Loaded {len(df_se)} SIDER side-effect pairs")
    else:
        print(f"SIDER side effects not found at {se_path}")

    if os.path.exists(freq_path):
        df_freq = pd.read_csv(freq_path)
        print(f"Loaded {len(df_freq)} SIDER frequency records")
    else:
        print(f"SIDER frequencies not found at {freq_path}")

    return df_se, df_freq


def compute_drug_risk_profile(faers_df):
    """Compute per-drug risk profile from FAERS data."""
    # Normalize drug names
    faers_df["drug_name"] = faers_df["drug_name"].str.lower().str.strip()

    # Aggregate per drug
    drug_stats = faers_df.groupby("drug_name").agg(
        total_events=("drug_name", "count"),
        mean_reactions=("reaction_count", "mean"),
        serious_rate=("serious", "mean"),
        death_rate=("death", "mean"),
        hospital_rate=("hospitalization", "mean"),
        lt_rate=("life_threatening", "mean"),
        mean_age=("patient_age", "mean"),
        mean_concomitant=("num_concomitant_drugs", "mean"),
    ).reset_index()

    print("\nDrug risk profiles:")
    print(drug_stats.to_string(index=False))
    return drug_stats


def compute_sider_features(df_se, df_freq):
    """Compute per-drug side effect features from SIDER."""
    features = {}

    if df_se is not None and not df_se.empty:
        se_counts = df_se.groupby("drug_name")["side_effect"].count().reset_index()
        se_counts.columns = ["drug_name", "known_side_effects_count"]
        for _, row in se_counts.iterrows():
            features[row["drug_name"]] = {"known_side_effects_count": row["known_side_effects_count"]}

    if df_freq is not None and not df_freq.empty:
        freq_stats = df_freq.groupby("drug_name").agg(
            mean_frequency=("frequency_pct", "mean"),
            max_frequency=("frequency_pct", "max"),
        ).reset_index()
        for _, row in freq_stats.iterrows():
            if row["drug_name"] not in features:
                features[row["drug_name"]] = {}
            features[row["drug_name"]]["mean_se_frequency"] = row["mean_frequency"]
            features[row["drug_name"]]["max_se_frequency"] = row["max_frequency"]

    return features


try:
    from backend.utils import get_reaction_severity
except ImportError:
    from utils import get_reaction_severity

def build_training_data(faers_df, sider_features):
    """Build training dataset by combining FAERS events with SIDER features."""
    faers_df = faers_df.copy()
    faers_df["drug_name"] = faers_df["drug_name"].str.lower().str.strip()

    # Clean demographics
    faers_df["patient_age"] = pd.to_numeric(faers_df["patient_age"], errors="coerce")
    faers_df["patient_weight"] = pd.to_numeric(faers_df["patient_weight"], errors="coerce")

    # Filter valid ages (0-100) and weights (10-250 kg)
    faers_df.loc[(faers_df["patient_age"] < 0) | (faers_df["patient_age"] > 100), "patient_age"] = np.nan
    faers_df.loc[(faers_df["patient_weight"] < 10) | (faers_df["patient_weight"] > 250), "patient_weight"] = np.nan

    # Fill missing demographics with medians
    faers_df["patient_age"] = faers_df["patient_age"].fillna(faers_df["patient_age"].median())
    faers_df["patient_weight"] = faers_df["patient_weight"].fillna(faers_df["patient_weight"].median())

    # Encode sex: 0=M, 1=F, 0.5=Unknown
    faers_df["sex_encoded"] = faers_df["patient_sex"].map({"M": 0, "F": 1}).fillna(0.5)

    # New Feature: Weighted Reaction Severity
    # Reactions are stored as | separated strings
    def compute_severity(react_str):
        if not react_str or pd.isna(react_str): return 1.0
        reacts = str(react_str).split("|")
        return get_reaction_severity(reacts)

    faers_df["weighted_reaction_score"] = faers_df["reactions"].apply(compute_severity)

    # New Feature: Primary Suspect Flag
    faers_df["is_primary_suspect"] = (faers_df["drug_role"] == "Primary Suspect").astype(int)

    # Add SIDER features
    faers_df["known_side_effects_count"] = faers_df["drug_name"].map(
        lambda x: sider_features.get(x, {}).get("known_side_effects_count", 0)
    )
    faers_df["mean_se_frequency"] = faers_df["drug_name"].map(
        lambda x: sider_features.get(x, {}).get("mean_se_frequency", 0)
    )
    faers_df["max_se_frequency"] = faers_df["drug_name"].map(
        lambda x: sider_features.get(x, {}).get("max_se_frequency", 0)
    )

    # Compute risk score (target variable)
    # Weighted combination of seriousness indicators
    faers_df["risk_score"] = (
        faers_df["serious"] * 0.20 +
        faers_df["hospitalization"] * 0.20 +
        faers_df["life_threatening"] * 0.30 +
        faers_df["death"] * 0.30
    )

    # Risk categories
    def categorize_risk(score):
        if score <= 0.1: return 0  # Low
        elif score <= 0.3: return 1  # Moderate
        elif score <= 0.6: return 2  # High
        else: return 3  # Critical

    faers_df["risk_category"] = faers_df["risk_score"].apply(categorize_risk)

    # Select training features
    feature_cols = [
        "drug_name",
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
        "risk_category",
    ]

    return faers_df[feature_cols].copy()


def main():
    print("=" * 60)
    print("Building ML Training Dataset (FAERS + SIDER)")
    print("=" * 60)

    # Load data
    faers_df = load_faers()
    if faers_df is None:
        return

    df_se, df_freq = load_sider()

    # Compute features
    drug_profiles = compute_drug_risk_profile(faers_df)
    sider_features = compute_sider_features(df_se, df_freq)

    # Build training data
    df_train = build_training_data(faers_df, sider_features)

    # Save
    output_path = os.path.join(DATA_DIR, "training_data.csv")
    df_train.to_csv(output_path, index=False)

    print(f"\nTraining dataset: {len(df_train)} rows")
    print(f"Saved to: {output_path}")
    print(f"\nFeature columns: {list(df_train.columns)}")
    print(f"\nRisk category distribution:")
    print(df_train["risk_category"].value_counts().sort_index().to_string())
    print(f"\nMissing values:")
    print(df_train.isnull().sum().to_string())

    # Save drug profiles
    profiles_path = os.path.join(DATA_DIR, "drug_risk_profiles.csv")
    drug_profiles.to_csv(profiles_path, index=False)
    print(f"\nDrug profiles saved to: {profiles_path}")

    return df_train


if __name__ == "__main__":
    main()
