"""
GeneRx-AI — Clinical Rules Engine
Drug recommendation and risk assessment based on clinical parameters:
  - Demographics (age, sex, weight/BMI)
  - Medical conditions (diabetes, CKD, heart failure, etc.)
  - Lab results (eGFR, ALT, AST, HbA1c, Blood Pressure, LDL, INR)
  - Current medications (for drug-drug interaction checks)
  - Allergies
No genetic data required.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DRUG CATALOG
# ─────────────────────────────────────────────────────────────────────────────
DRUG_CATALOG = {
    "Metformin": {
        "category": "Diabetes",
        "class": "Biguanide",
        "description": "First-line oral diabetes medication that lowers blood sugar",
        "mechanism": "Reduces glucose production in the liver; improves insulin sensitivity",
        "monitoring": ["eGFR (every 6–12 months)", "Vitamin B12 (annually)", "HbA1c (every 3 months)"],
        "side_effects": {
            "Nausea / Upset Stomach": 0.30,
            "Diarrhea": 0.25,
            "Metallic Taste": 0.15,
            "Vitamin B12 Deficiency": 0.10,
            "Lactic Acidosis (rare)": 0.01,
        },
    },
    "Atorvastatin": {
        "category": "Cardiovascular",
        "class": "Statin",
        "description": "Cholesterol-lowering medication to reduce heart disease risk",
        "mechanism": "Inhibits HMG-CoA reductase to reduce LDL cholesterol",
        "monitoring": ["LDL Cholesterol (annually)", "Liver enzymes (ALT/AST) baseline then if symptoms"],
        "side_effects": {
            "Muscle Aches": 0.10,
            "Headache": 0.08,
            "Nausea": 0.06,
            "Liver Enzyme Elevation": 0.01,
            "Muscle Breakdown (rare)": 0.001,
        },
    },
    "Amlodipine": {
        "category": "Cardiovascular",
        "class": "Calcium Channel Blocker",
        "description": "Blood pressure and chest pain medication",
        "mechanism": "Relaxes blood vessels by blocking calcium entry into arterial muscle cells",
        "monitoring": ["Blood Pressure (monthly until stable)", "Ankle swelling"],
        "side_effects": {
            "Ankle Swelling": 0.20,
            "Headache": 0.12,
            "Flushing": 0.10,
            "Dizziness": 0.08,
            "Palpitations": 0.05,
        },
    },
    "Ramipril": {
        "category": "Cardiovascular",
        "class": "ACE Inhibitor",
        "description": "Blood pressure medication that also protects the kidneys in diabetes",
        "mechanism": "Blocks angiotensin-converting enzyme to reduce blood pressure",
        "monitoring": ["eGFR + Potassium (within 2 weeks of starting)", "Blood Pressure (monthly)"],
        "side_effects": {
            "Dry Cough": 0.20,
            "Dizziness": 0.10,
            "Elevated Potassium": 0.05,
            "Kidney Function Change": 0.05,
            "Angioedema (rare)": 0.005,
        },
    },
    "Metoprolol": {
        "category": "Cardiovascular",
        "class": "Beta Blocker",
        "description": "Heart rate and blood pressure medication",
        "mechanism": "Blocks beta-adrenergic receptors to slow heart rate and reduce blood pressure",
        "monitoring": ["Heart rate", "Blood Pressure", "Blood sugar (masks hypoglycaemia symptoms)"],
        "side_effects": {
            "Fatigue": 0.20,
            "Cold Hands/Feet": 0.15,
            "Slow Heart Rate": 0.10,
            "Dizziness": 0.08,
            "Sleep Disturbance": 0.06,
        },
    },
    "Warfarin": {
        "category": "Anticoagulant",
        "class": "Vitamin K Antagonist",
        "description": "Blood thinner to prevent clots and stroke",
        "mechanism": "Inhibits Vitamin K-dependent clotting factors",
        "monitoring": ["INR (weekly initially, then monthly)", "Signs of bleeding"],
        "side_effects": {
            "Bleeding": 0.35,
            "Bruising": 0.25,
            "Hair Thinning": 0.08,
            "Nausea": 0.07,
            "Skin Necrosis (rare)": 0.005,
        },
    },
    "Amoxicillin": {
        "category": "Antibiotic",
        "class": "Penicillin",
        "description": "Common antibiotic for bacterial infections",
        "mechanism": "Inhibits bacterial cell wall synthesis",
        "monitoring": ["Signs of allergic reaction", "Resolution of infection"],
        "side_effects": {
            "Diarrhea": 0.20,
            "Nausea": 0.15,
            "Rash": 0.08,
            "Allergic Reaction": 0.05,
            "Yeast Infection": 0.10,
        },
    },
    "Ibuprofen": {
        "category": "Pain & Inflammation",
        "class": "NSAID",
        "description": "Anti-inflammatory painkiller for pain, fever, and swelling",
        "mechanism": "Inhibits COX enzymes to reduce prostaglandin synthesis",
        "monitoring": ["Kidney function (if prolonged use)", "Blood pressure", "GI symptoms"],
        "side_effects": {
            "Stomach Pain / Ulcer Risk": 0.20,
            "Nausea": 0.15,
            "Blood Pressure Rise": 0.12,
            "Kidney Function Change": 0.10,
            "Fluid Retention": 0.08,
        },
    },
    "Paracetamol": {
        "category": "Pain & Fever",
        "class": "Analgesic",
        "description": "Safe, mild painkiller and fever reducer",
        "mechanism": "Central pain relief; mechanism not fully understood",
        "monitoring": ["Liver enzymes (if chronic high-dose use)", "Avoid alcohol"],
        "side_effects": {
            "Liver Damage (overdose)": 0.001,
            "Nausea (rare)": 0.02,
            "Rash (rare)": 0.01,
        },
    },
    "Omeprazole": {
        "category": "Gastrointestinal",
        "class": "Proton Pump Inhibitor",
        "description": "Reduces stomach acid for heartburn, ulcers, and acid reflux",
        "mechanism": "Irreversibly inhibits gastric H+/K+-ATPase pump",
        "monitoring": ["Magnesium (long-term use)", "Vitamin B12 (long-term use)"],
        "side_effects": {
            "Headache": 0.07,
            "Nausea": 0.05,
            "Diarrhea": 0.04,
            "Low Magnesium (long-term)": 0.03,
            "Bone Density Decrease (long-term)": 0.02,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DRUG-DRUG INTERACTIONS
# ─────────────────────────────────────────────────────────────────────────────
DRUG_DRUG_INTERACTIONS = {
    ("Warfarin", "Ibuprofen"): {
        "severity": "CRITICAL",
        "message": "Warfarin + Ibuprofen: Greatly increased bleeding risk. Avoid combination.",
    },
    ("Warfarin", "Amoxicillin"): {
        "severity": "MODERATE",
        "message": "Warfarin + Amoxicillin: Antibiotics may alter INR. Monitor INR closely.",
    },
    ("Warfarin", "Omeprazole"): {
        "severity": "MODERATE",
        "message": "Warfarin + Omeprazole: May increase Warfarin levels. Monitor INR.",
    },
    ("Ramipril", "Metoprolol"): {
        "severity": "LOW",
        "message": "Ramipril + Metoprolol: Both lower blood pressure — risk of hypotension. Monitor BP.",
    },
    ("Ramipril", "Ibuprofen"): {
        "severity": "HIGH",
        "message": "Ramipril + Ibuprofen: NSAIDs reduce ACE inhibitor effectiveness and worsen kidney function.",
    },
    ("Metformin", "Ibuprofen"): {
        "severity": "MODERATE",
        "message": "Metformin + Ibuprofen: NSAIDs can reduce kidney function, risking Metformin accumulation.",
    },
    ("Amlodipine", "Metoprolol"): {
        "severity": "LOW",
        "message": "Amlodipine + Metoprolol: Additive blood pressure lowering. Generally well-tolerated; monitor BP.",
    },
    ("Metoprolol", "Ibuprofen"): {
        "severity": "MODERATE",
        "message": "Metoprolol + Ibuprofen: NSAIDs can blunt the effect of beta-blockers and raise blood pressure.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ALLERGY MAP  (allergy category → contraindicated drugs)
# ─────────────────────────────────────────────────────────────────────────────
ALLERGY_CONTRAINDICATIONS = {
    "Penicillin": ["Amoxicillin"],
    "NSAIDs": ["Ibuprofen"],
    "Sulfa drugs": [],
    "ACE Inhibitors": ["Ramipril"],
    "Statins": ["Atorvastatin"],
    "Beta Blockers": ["Metoprolol"],
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_drug(patient: dict, drug_name: str) -> dict:
    """
    Evaluate how suitable a drug is for a given patient.

    Args:
        patient: dict with keys:
            age (int), sex (str), weight_kg (float), bmi (float),
            conditions (list[str]), egfr (float), alt (float), ast (float),
            hba1c (float), systolic_bp (int), diastolic_bp (int),
            ldl (float), inr (float),
            current_meds (list[str]), allergies (list[str])
        drug_name: Name of the drug to evaluate

    Returns:
        dict: {
            drug_name, suitability, risk_level, reasons, dose_notes,
            side_effects, monitoring, warnings
        }
    """
    reasons = []
    dose_notes = []
    warnings = []
    suitability = "Suitable"   # Suitable | Caution | Avoid | Contraindicated

    age         = patient.get("age", 40)
    sex         = patient.get("sex", "M")
    bmi         = patient.get("bmi", 25.0)
    conditions  = [c.lower() for c in patient.get("conditions", [])]
    egfr        = patient.get("egfr", 90.0)
    alt         = patient.get("alt", 30.0)
    ast         = patient.get("ast", 30.0)
    hba1c       = patient.get("hba1c", 5.5)
    systolic_bp = patient.get("systolic_bp", 120)
    ldl         = patient.get("ldl", 3.0)
    inr         = patient.get("inr", 1.0)
    allergies   = [a.lower() for a in patient.get("allergies", [])]

    # ── Universal age modifier ────────────────────────────────────────────────
    if age >= 65:
        dose_notes.append("Patient ≥65 years: Consider lower starting doses and closer monitoring.")
    if age >= 80:
        warnings.append("Patient ≥80 years: High caution — increased sensitivity to most medications.")

    # ── BMI note ─────────────────────────────────────────────────────────────
    if bmi > 30:
        dose_notes.append(f"BMI {bmi:.1f}: Obesity may affect drug distribution — verify weight-based dosing.")

    # ─────────────────────────────────────────────────────────────────────────
    # Per-drug clinical rules
    # ─────────────────────────────────────────────────────────────────────────
    if drug_name == "Metformin":
        # CKD contraindication
        if egfr < 30:
            suitability = "Contraindicated"
            reasons.append(f"eGFR {egfr} mL/min/1.73m² is <30 → Metformin contraindicated (lactic acidosis risk).")
        elif egfr < 45:
            suitability = "Caution"
            reasons.append(f"eGFR {egfr} mL/min/1.73m² is 30–44 → Reduce Metformin dose; monitor renal function closely.")
        else:
            reasons.append(f"eGFR {egfr} mL/min/1.73m² is adequate for Metformin.")

        if "diabetes" in conditions or hba1c >= 6.5:
            reasons.append("Indicated: Diabetes diagnosed (or HbA1c ≥6.5%). First-line choice per guidelines.")
        else:
            warnings.append("Metformin is used for diabetes — patient does not appear to have a diabetes diagnosis.")

        if hba1c > 9.0:
            dose_notes.append(f"HbA1c {hba1c}% is very high — may need combination therapy alongside Metformin.")

    elif drug_name == "Atorvastatin":
        # Liver disease
        if alt > 90 or ast > 90:   # >3× ULN (ULN ~30 U/L)
            suitability = "Contraindicated"
            reasons.append(f"ALT {alt} or AST {ast} >3× upper limit of normal → Statins contraindicated in active liver disease.")
        elif alt > 45 or ast > 45:
            suitability = "Caution"
            reasons.append(f"ALT {alt} / AST {ast} mildly elevated → Use Atorvastatin with close liver monitoring.")
        else:
            reasons.append("Liver enzymes normal — Atorvastatin safe from hepatic perspective.")

        if ldl > 3.0 or "heart disease" in conditions or "hypertension" in conditions or "diabetes" in conditions:
            reasons.append("Indicated: Elevated LDL or cardiovascular risk factors present.")
        else:
            warnings.append("LDL appears well-controlled — confirm whether statin therapy is clinically indicated.")

    elif drug_name == "Amlodipine":
        if "heart failure" in conditions:
            suitability = "Caution"
            reasons.append("Caution in heart failure: Amlodipine may worsen fluid retention in some patients.")
        if systolic_bp >= 140 or "hypertension" in conditions:
            reasons.append("Indicated: Hypertension present. Amlodipine is a first-line antihypertensive.")
        elif systolic_bp < 100:
            suitability = "Avoid"
            reasons.append(f"BP {systolic_bp} mmHg is already low → Amlodipine may cause dangerous hypotension.")
        else:
            reasons.append("Blood pressure within range; confirm clinical indication before prescribing.")

    elif drug_name == "Ramipril":
        if egfr < 30:
            suitability = "Caution"
            reasons.append(f"eGFR {egfr}: ACE inhibitors may worsen kidney function if eGFR <30. Use only with nephrology input.")
            dose_notes.append("Start at lowest dose; recheck eGFR and potassium within 1–2 weeks.")
        elif egfr < 45:
            suitability = "Caution"
            reasons.append(f"eGFR {egfr}: Reduce dose; monitor eGFR and potassium closely.")
        else:
            reasons.append(f"eGFR {egfr}: renal function adequate for Ramipril.")

        if "diabetes" in conditions:
            reasons.append("Beneficial: ACE inhibitors are kidney-protective in patients with diabetes — strongly indicated.")
        if "heart failure" in conditions:
            reasons.append("Beneficial: ACE inhibitors reduce mortality in heart failure — strongly indicated.")
        if systolic_bp >= 140 or "hypertension" in conditions:
            reasons.append("Indicated for hypertension.")
        if "hyperkalemia" in conditions:
            suitability = "Avoid"
            reasons.append("Avoid: Ramipril elevates potassium — dangerous in patients with hyperkalemia.")

    elif drug_name == "Metoprolol":
        if "asthma" in conditions or "copd" in conditions:
            suitability = "Contraindicated"
            reasons.append("Contraindicated: Beta-blockers cause bronchoconstriction — dangerous in asthma/COPD.")
        elif "heart failure" in conditions:
            reasons.append("Beneficial in stable heart failure (reduces mortality). Use with cardiology guidance.")
        elif systolic_bp >= 140 or "hypertension" in conditions:
            reasons.append("Indicated for hypertension and heart rate control.")
        else:
            reasons.append("Suitable if clinical indication confirmed.")

        if "diabetes" in conditions:
            dose_notes.append("Caution in diabetes: Beta-blockers may mask hypoglycaemia symptoms (except sweating).")
        if "heart block" in conditions:
            suitability = "Contraindicated"
            reasons.append("Contraindicated in heart block: Will worsen conduction abnormality.")

    elif drug_name == "Warfarin":
        if inr > 3.5:
            suitability = "Caution"
            reasons.append(f"Current INR {inr} is supratherapeutic — patient may be over-anticoagulated. Review dose.")
        elif inr < 2.0 and "atrial fibrillation" in conditions:
            reasons.append(f"INR {inr} is sub-therapeutic for AF — may need dose adjustment.")
        
        if "atrial fibrillation" in conditions or "dvt" in conditions or "pulmonary embolism" in conditions:
            reasons.append("Indicated for anticoagulation in AF / VTE.")
        
        if "peptic ulcer" in conditions or "gi bleeding" in conditions:
            suitability = "Avoid"
            reasons.append("Avoid: Active GI pathology significantly increases bleeding risk with anticoagulation.")
        
        dose_notes.append("Numerous food and drug interactions — regular INR monitoring is essential.")
        dose_notes.append("Vitamin K-rich foods (e.g. leafy greens) affect INR — dietary counselling needed.")

    elif drug_name == "Amoxicillin":
        # Penicillin allergy
        if "penicillin" in allergies:
            suitability = "Contraindicated"
            reasons.append("Contraindicated: Documented Penicillin allergy — Amoxicillin is a Penicillin antibiotic.")
        else:
            reasons.append("No penicillin allergy documented — Amoxicillin structurally appropriate.")

        if egfr < 30:
            dose_notes.append(f"eGFR {egfr}: Reduce Amoxicillin dose and extend dosing interval in severe CKD.")

    elif drug_name == "Ibuprofen":
        # CKD
        if egfr < 60:
            if egfr < 30:
                suitability = "Contraindicated"
                reasons.append(f"Contraindicated: eGFR {egfr} — NSAIDs seriously worsen kidney function in CKD.")
            else:
                suitability = "Avoid"
                reasons.append(f"Avoid: eGFR {egfr} — NSAIDs reduce renal blood flow and worsen CKD.")
        # Heart failure
        if "heart failure" in conditions:
            suitability = "Avoid" if suitability == "Suitable" else suitability
            reasons.append("Avoid: Ibuprofen causes fluid retention and worsens heart failure.")
        # Peptic ulcer
        if "peptic ulcer" in conditions or "gi bleeding" in conditions:
            suitability = "Avoid" if suitability == "Suitable" else suitability
            reasons.append("Avoid: NSAIDs cause GI mucosal injury — dangerous in peptic ulcer disease.")
        # Elderly
        if age >= 65:
            suitability = "Caution" if suitability == "Suitable" else suitability
            reasons.append("Caution in elderly: Increased GI bleeding and kidney impairment risk.")
        if suitability == "Suitable":
            reasons.append("No major contraindications identified. Use lowest effective dose for shortest duration.")
        
        dose_notes.append("Always take with food. If long-term, consider co-prescribing Omeprazole for GI protection.")

    elif drug_name == "Paracetamol":
        # Very rarely contraindicated
        if "liver failure" in conditions or "hepatitis" in conditions:
            suitability = "Caution"
            reasons.append("Caution: Active liver disease — use lowest effective dose; avoid chronic high-dose use.")
        else:
            reasons.append("Paracetamol is generally the safest analgesic. Suitable for most patients.")
        dose_notes.append("Do not exceed 4g/day (2g/day in frail elderly or liver disease). Avoid alcohol.")

    elif drug_name == "Omeprazole":
        reasons.append("Suitable for acid suppression, GERD, peptic ulcer, or GI protection with NSAIDs/steroids.")
        if "peptic ulcer" in conditions or "gerd" in conditions:
            reasons.append("Indicated: Active peptic ulcer or GERD diagnosis.")
        if alt > 90 or ast > 90:
            dose_notes.append("Hepatic impairment: Reduce Omeprazole dose.")
        dose_notes.append("For long-term use (>1 year), monitor Magnesium and Vitamin B12 levels.")

    # ── Allergy check for all drugs ───────────────────────────────────────────
    for allergy_cat, contraindicated_drugs in ALLERGY_CONTRAINDICATIONS.items():
        if drug_name in contraindicated_drugs and allergy_cat.lower() in allergies:
            suitability = "Contraindicated"
            reasons.insert(0, f"⚠️ ALLERGY ALERT: Patient has documented {allergy_cat} allergy — {drug_name} is contraindicated.")

    # ── Compute risk level ────────────────────────────────────────────────────
    risk_level = compute_risk_level(suitability)

    return {
        "drug_name":   drug_name,
        "suitability": suitability,
        "risk_level":  risk_level,
        "reasons":     reasons,
        "dose_notes":  dose_notes,
        "side_effects": DRUG_CATALOG[drug_name]["side_effects"],
        "monitoring":  DRUG_CATALOG[drug_name]["monitoring"],
        "warnings":    warnings,
        "category":    DRUG_CATALOG[drug_name]["category"],
        "description": DRUG_CATALOG[drug_name]["description"],
    }


def compute_risk_level(suitability: str) -> str:
    """Map suitability to a risk level label."""
    return {
        "Suitable":       "Low",
        "Caution":        "Moderate",
        "Avoid":          "High",
        "Contraindicated":"Critical",
    }.get(suitability, "Unknown")


def check_drug_interactions(drug_list: list) -> list:
    """
    Check all pairwise drug-drug interactions for a given list of drugs.

    Returns:
        List of interaction dicts: {drug_a, drug_b, severity, message}
    """
    interactions = []
    drug_list = list(drug_list)
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            pair = (drug_list[i], drug_list[j])
            pair_rev = (drug_list[j], drug_list[i])
            info = DRUG_DRUG_INTERACTIONS.get(pair) or DRUG_DRUG_INTERACTIONS.get(pair_rev)
            if info:
                interactions.append({
                    "drug_a":   drug_list[i],
                    "drug_b":   drug_list[j],
                    "severity": info["severity"],
                    "message":  info["message"],
                })
    return interactions


def generate_patient_summary(patient: dict, results: list, interactions: list) -> str:
    """
    Generate a plain text clinical report for a patient assessment.
    """
    name = patient.get("name", "Unknown")
    pid  = patient.get("patient_id", "N/A")
    age  = patient.get("age", "?")
    sex  = patient.get("sex", "?")
    conditions = ", ".join(patient.get("conditions", [])) or "None reported"

    lines = [
        "=" * 65,
        "          GeneRx-AI — CLINICAL DRUG ASSESSMENT REPORT",
        "=" * 65,
        f"Patient Name  : {name}",
        f"Patient ID    : {pid}",
        f"Age / Sex     : {age} / {sex}",
        f"Conditions    : {conditions}",
        f"eGFR          : {patient.get('egfr', 'N/A')} mL/min/1.73m²",
        f"ALT / AST     : {patient.get('alt', 'N/A')} / {patient.get('ast', 'N/A')} U/L",
        f"HbA1c         : {patient.get('hba1c', 'N/A')}%",
        f"Blood Pressure: {patient.get('systolic_bp', 'N/A')}/{patient.get('diastolic_bp', 'N/A')} mmHg",
        f"LDL           : {patient.get('ldl', 'N/A')} mmol/L",
        f"Current Meds  : {', '.join(patient.get('current_meds', [])) or 'None'}",
        f"Allergies     : {', '.join(patient.get('allergies', [])) or 'None'}",
        "",
        "─" * 65,
        "DRUG ASSESSMENT SUMMARY",
        "─" * 65,
    ]

    for r in results:
        lines.append(f"\n▶ {r['drug_name']}  [{r['suitability'].upper()}]  Risk: {r['risk_level']}")
        for reason in r["reasons"]:
            lines.append(f"   • {reason}")
        for note in r["dose_notes"]:
            lines.append(f"   📋 Dose Note: {note}")
        for warn in r["warnings"]:
            lines.append(f"   ⚠️  {warn}")

    if interactions:
        lines += ["", "─" * 65, "DRUG-DRUG INTERACTIONS", "─" * 65]
        for ix in interactions:
            lines.append(f"\n[{ix['severity']}] {ix['drug_a']} ↔ {ix['drug_b']}")
            lines.append(f"   {ix['message']}")

    lines += [
        "",
        "─" * 65,
        "⚠️  DISCLAIMER: This report is for clinical decision support",
        "    only and does not replace professional medical judgment.",
        "=" * 65,
    ]

    return "\n".join(lines)


def simulate_response_over_time(suitability: str, steps: int = 12) -> list:
    """
    Generate a drug response curve based on suitability score.
    Used for visualising expected therapeutic trajectory.
    """
    import numpy as np

    target_map = {
        "Suitable":        0.80,
        "Caution":         0.55,
        "Avoid":           0.25,
        "Contraindicated": 0.10,
    }
    target = target_map.get(suitability, 0.5)
    series = [0.0]

    for i in range(1, steps):
        progress = i / steps
        current = series[-1]
        step = (target - current) * progress * 0.5
        noise = np.random.normal(0, 0.02)
        series.append(max(0.0, min(1.0, current + step + noise)))

    series.append(target)
    return series