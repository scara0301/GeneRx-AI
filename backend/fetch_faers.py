import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.fda.gov/drug/event.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faers")

TARGET_DRUGS = [
    "metformin",
    "atorvastatin",
    "amlodipine",
    "ramipril",
    "metoprolol",
    "warfarin",
    "amoxicillin",
    "ibuprofen",
    "acetaminophen",
    "omeprazole",
]


def fetch_drug_events(drug_name, max_records=2000):
    all_records = []
    batch_size = 100
    skip = 0

    print(f"  Fetching {drug_name}...")

    while skip < max_records:
        params = {
            "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
            "limit": batch_size,
            "skip": skip,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 404:
                print(f"    No results for {drug_name}")
                break
            if resp.status_code != 200:
                print(f"    API error {resp.status_code}, retrying...")
                time.sleep(2)
                continue

            results = resp.json().get("results", [])
            if not results:
                break

            for event in results:
                patient = event.get("patient", {})
                drugs = patient.get("drug", [])
                reactions = patient.get("reaction", [])

                age = None
                raw_age = patient.get("patientonsetage")
                age_unit = patient.get("patientonsetageunit")
                if raw_age:
                    try:
                        age = float(raw_age)
                        if age_unit == "802":
                            age /= 12
                        elif age_unit == "803":
                            age /= 365
                        elif age_unit == "800":
                            age *= 10
                    except (ValueError, TypeError):
                        age = None

                sex = {"1": "M", "2": "F"}.get(patient.get("patientsex"), "Unknown")

                weight = None
                raw_weight = patient.get("patientweight")
                if raw_weight:
                    try:
                        weight = float(raw_weight)
                    except (ValueError, TypeError):
                        weight = None

                serious = event.get("serious", "0")
                death = event.get("seriousnessdeath", "0")
                hospital = event.get("seriousnesshospitalization", "0")
                disability = event.get("seriousnessdisabling", "0")
                life_threat = event.get("seriousnesslifethreatening", "0")

                reaction_list = [r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")]
                reaction_outcomes = [r.get("reactionoutcome", "0") for r in reactions]

                drug_role = "Unknown"
                indication = ""
                for d in drugs:
                    gnames = d.get("openfda", {}).get("generic_name", [])
                    if any(drug_name.lower() in g.lower() for g in gnames):
                        drug_role = {
                            "1": "Primary Suspect",
                            "2": "Secondary Suspect",
                            "3": "Concomitant",
                        }.get(d.get("drugcharacterization", "0"), "Unknown")
                        indication = d.get("drugindication", "")
                        break

                all_records.append({
                    "drug_name": drug_name,
                    "patient_age": age,
                    "patient_sex": sex,
                    "patient_weight": weight,
                    "reactions": "|".join(reaction_list),
                    "reaction_count": len(reaction_list),
                    "serious": int(serious == "1") if serious else 0,
                    "death": int(death == "1") if death else 0,
                    "hospitalization": int(hospital == "1") if hospital else 0,
                    "disability": int(disability == "1") if disability else 0,
                    "life_threatening": int(life_threat == "1") if life_threat else 0,
                    "drug_role": drug_role,
                    "indication": indication,
                    "reaction_outcomes": "|".join(reaction_outcomes),
                    "num_concomitant_drugs": len(drugs) - 1,
                })

            skip += batch_size
            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            print(f"    Network error: {e}, retrying in 3s...")
            time.sleep(3)

    print(f"    {len(all_records)} events retrieved")
    return all_records


def main():
    print("=" * 60)
    print("FAERS Data Fetcher — openFDA Adverse Event Reports")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_events = []

    for drug in TARGET_DRUGS:
        all_events.extend(fetch_drug_events(drug, max_records=5000))
        time.sleep(1)

    df = pd.DataFrame(all_events)
    output_path = os.path.join(OUTPUT_DIR, "faers_events.csv")
    df.to_csv(output_path, index=False)

    print(f"\nTotal events: {len(df)}")
    print(f"Saved to: {output_path}")
    print(f"\nPer drug:\n{df['drug_name'].value_counts().to_string()}")
    print(f"\nSerious: {df['serious'].sum()}  |  Deaths: {df['death'].sum()}  |  Hospitalizations: {df['hospitalization'].sum()}")

    return df


if __name__ == "__main__":
    main()
