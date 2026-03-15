SEVERITY_WEIGHTS = {
    "cardiac arrest": 10.0,
    "myocardial infarction": 9.0,
    "ventricular tachycardia": 8.5,
    "heart failure": 8.0,
    "atrial fibrillation": 5.0,
    "bradycardia": 4.0,
    "tachycardia": 3.0,

    "renal failure": 9.0,
    "renal failure acute": 9.5,
    "renal impairment": 6.0,
    "hepatic failure": 9.5,
    "liver function test abnormal": 4.0,
    "hepatitis": 7.0,

    "anaphylactic shock": 10.0,
    "anaphylactic reaction": 9.0,
    "agranulocytosis": 8.5,
    "pancytopenia": 8.0,
    "haemorrhage": 7.0,
    "gastrointestinal haemorrhage": 8.5,

    "stevens-johnson syndrome": 10.0,
    "toxic epidermal necrolysis": 10.0,
    "drug reaction with eosinophilia and systemic symptoms": 9.0,

    "convulsion": 8.0,
    "stroke": 9.0,
    "cerebrovascular accident": 9.0,
    "loss of consciousness": 7.0,
    "syncope": 5.0,

    "nausea": 1.0,
    "vomiting": 1.5,
    "diarrhoea": 1.5,
    "headache": 1.0,
    "dizziness": 1.5,
    "fatigue": 1.0,
    "rash": 2.0,
    "pruritus": 1.5,
}


def get_reaction_severity(reactions):
    """
    Returns a weighted severity score for a list of MedDRA adverse reaction terms.
    Combines 60% max severity with 40% average severity across all terms.
    """
    if not reactions:
        return 0.0

    total_score = 0.0
    for r in reactions:
        r_lower = r.lower().strip()
        weight = 1.0
        for term, w in SEVERITY_WEIGHTS.items():
            if term in r_lower:
                weight = max(weight, w)
        total_score += weight

    n = len(reactions)
    max_weight = max(SEVERITY_WEIGHTS.get(r.lower().strip(), 1.0) for r in reactions)
    avg_weight = total_score / n
    return (0.6 * max_weight) + (0.4 * avg_weight)
