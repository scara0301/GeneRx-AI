"""
GeneRx AI — MedDRA Reaction Weighting Utilities
Provides clinical severity scores for common adverse event terms.
"""

# Dictionary of high-severity MedDRA terms and their impact (1.0 to 10.0 scale)
# Based on common clinical severity indicators (renal, hepatic, cardiac, neurological)
SEVERITY_WEIGHTS = {
    # Cardiac
    "cardiac arrest": 10.0,
    "myocardial infarction": 9.0,
    "ventricular tachycardia": 8.5,
    "heart failure": 8.0,
    "atrial fibrillation": 5.0,
    "bradycardia": 4.0,
    "tachycardia": 3.0,
    
    # Renal / Hepatic
    "renal failure": 9.0,
    "renal failure acute": 9.5,
    "renal impairment": 6.0,
    "hepatic failure": 9.5,
    "liver function test abnormal": 4.0,
    "hepatitis": 7.0,
    
    # Blood / Immune
    "anaphylactic shock": 10.0,
    "anaphylactic reaction": 9.0,
    "agranulocytosis": 8.5,
    "pancytopenia": 8.0,
    "haemorrhage": 7.0,
    "gastrointestinal haemorrhage": 8.5,
    
    # Skin (Severe)
    "stevens-johnson syndrome": 10.0,
    "toxic epidermal necrolysis": 10.0,
    "drug reaction with eosinophilia and systemic symptoms": 9.0,
    
    # Neurological
    "convulsion": 8.0,
    "stroke": 9.0,
    "cerebrovascular accident": 9.0,
    "loss of consciousness": 7.0,
    "syncope": 5.0,
    
    # Common / Low Severity (for normalization baseline)
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
    Calculates a weighted severity score for a list of reaction terms.
    Args:
        reactions (list): List of MedDRA Preferred Terms (strings)
    Returns:
        float: Normalized severity score
    """
    if not reactions:
        return 0.0
    
    total_score = 0.0
    valid_count = 0
    
    for r in reactions:
        r_lower = r.lower().strip()
        # Direct match or substring match for high-severity terms
        weight = 1.0 # Default weight for unknown terms
        
        for term, w in SEVERITY_WEIGHTS.items():
            if term in r_lower:
                weight = max(weight, w)
        
        total_score += weight
        valid_count += 1
        
    if valid_count == 0:
        return 0.0
        
    # We use a combination of average severity and max severity to capture "one bad event"
    # while also considering the overall burden.
    max_weight = max([SEVERITY_WEIGHTS.get(r.lower().strip(), 1.0) for r in reactions] + [1.0])
    avg_weight = total_score / valid_count
    
    # Heuristic: 60% max weight + 40% average weight
    return (0.6 * max_weight) + (0.4 * avg_weight)
