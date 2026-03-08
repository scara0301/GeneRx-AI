"""
fetch_sider.py — Download and process SIDER 4.1 side effects database.
SIDER maps drugs to known side effects with frequency information.
Source: http://sideeffects.embl.de/
"""
import os
import gzip
import io
import requests
import pandas as pd

SIDER_BASE = "http://sideeffects.embl.de/media/download/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sider")

# Our target drugs (lowercase for matching)
TARGET_DRUGS = {
    "metformin", "atorvastatin", "amlodipine", "ramipril",
    "metoprolol", "warfarin", "amoxicillin", "ibuprofen",
    "acetaminophen", "paracetamol", "omeprazole",
}

# Map SIDER flat-file drug IDs to names
# SIDER uses PubChem Compound IDs (CID) prefixed with CID
DRUG_NAME_MAP = {}


def download_file(filename):
    """Download a file from SIDER, handle gzip if needed."""
    url = SIDER_BASE + filename
    print(f"  Downloading {url}...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        if filename.endswith(".gz"):
            return gzip.decompress(resp.content).decode("utf-8", errors="replace")
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading {filename}: {e}")
        return None


def parse_drug_names(content):
    """Parse drug_names.tsv: maps SIDER compound IDs to drug names."""
    name_map = {}
    for line in content.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            compound_id = parts[0].strip()
            drug_name = parts[1].strip().lower()
            name_map[compound_id] = drug_name
    return name_map


def parse_side_effects(content, name_map):
    """Parse meddra_all_se.tsv: drug-side-effect pairs."""
    records = []
    for line in content.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 6:
            compound_id = parts[0].strip()
            # parts[1] = compound_id2 (stereo)
            # parts[2] = UMLS CUI from label
            # parts[3] = MedDRA concept type (LLT or PT)
            # parts[4] = UMLS CUI from MedDRA
            side_effect = parts[5].strip() if len(parts) > 5 else parts[4].strip()

            drug_name = name_map.get(compound_id, "")
            if drug_name and drug_name in TARGET_DRUGS:
                records.append({
                    "drug_name": drug_name,
                    "side_effect": side_effect,
                    "compound_id": compound_id,
                })
    return records


def parse_frequencies(content, name_map):
    """Parse meddra_freq.tsv: side effect frequencies."""
    records = []
    for line in content.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 5:
            compound_id = parts[0].strip()
            # parts[1] = UMLS CUI
            # parts[2] = frequency type (postmarketing, clinical trial, etc.)
            freq_value = parts[3].strip() if len(parts) > 3 else ""
            freq_lower = parts[4].strip() if len(parts) > 4 else ""
            side_effect = parts[-1].strip()

            drug_name = name_map.get(compound_id, "")
            if drug_name and drug_name in TARGET_DRUGS:
                # Parse frequency
                freq_pct = None
                try:
                    if "%" in freq_value:
                        freq_pct = float(freq_value.replace("%", ""))
                    elif freq_lower and freq_lower.replace(".", "").isdigit():
                        freq_pct = float(freq_lower) * 100
                except (ValueError, TypeError):
                    pass

                records.append({
                    "drug_name": drug_name,
                    "side_effect": side_effect,
                    "frequency_description": freq_value,
                    "frequency_pct": freq_pct,
                })
    return records


def main():
    print("=" * 60)
    print("SIDER 4.1 Data Fetcher — Drug Side Effects")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Download drug names mapping
    names_content = download_file("drug_names.tsv")
    if not names_content:
        print("Failed to download drug names. Exiting.")
        return

    name_map = parse_drug_names(names_content)
    print(f"  Loaded {len(name_map)} drug name mappings")

    # Check which of our target drugs exist in SIDER
    found_drugs = {v for v in name_map.values() if v in TARGET_DRUGS}
    print(f"  Our drugs found in SIDER: {found_drugs}")

    # 2. Download side effects
    se_content = download_file("meddra_all_se.tsv.gz")
    if se_content:
        se_records = parse_side_effects(se_content, name_map)
        df_se = pd.DataFrame(se_records)
        if not df_se.empty:
            # Deduplicate
            df_se = df_se.drop_duplicates(subset=["drug_name", "side_effect"])
            se_path = os.path.join(OUTPUT_DIR, "side_effects.csv")
            df_se.to_csv(se_path, index=False)
            print(f"\n  Side effects saved: {len(df_se)} unique drug-SE pairs")
            print(f"  Per drug:")
            print(df_se["drug_name"].value_counts().to_string(header=False))
        else:
            print("  No side effect records matched our drugs.")
    else:
        print("  Failed to download side effects file.")

    # 3. Download frequencies
    freq_content = download_file("meddra_freq.tsv.gz")
    if freq_content:
        freq_records = parse_frequencies(freq_content, name_map)
        df_freq = pd.DataFrame(freq_records)
        if not df_freq.empty:
            freq_path = os.path.join(OUTPUT_DIR, "side_effect_frequencies.csv")
            df_freq.to_csv(freq_path, index=False)
            print(f"\n  Frequency data saved: {len(df_freq)} records")
        else:
            print("  No frequency records matched our drugs.")
    else:
        print("  Failed to download frequency file.")

    print("\nSIDER data fetch complete.")


if __name__ == "__main__":
    main()
