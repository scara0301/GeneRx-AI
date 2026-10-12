# 09 — Clinical rules engine: every rule, threshold and the medicine behind it

> Source: `backend/clinical_engine.py`. Verdict ladder: **Suitable → Caution → Avoid → Contraindicated**
> mapped to risk **Low → Moderate → High → Critical**.
> Disclaimer for the interview: thresholds follow common textbook/label guidance, but the rules are a
> **teaching implementation**, not a validated formulary.

## 1. Inputs the rules read

| Field | Used for |
|-------|----------|
| `age` | ≥65 dose note, ≥80 warning, Ibuprofen elderly caution |
| `bmi` | >30 dosing note (BMI computed server-side with a fixed 1.70 m height) |
| `conditions` (lower-cased list) | diabetes, hypertension, heart failure, asthma, copd, heart block, atrial fibrillation, dvt, pulmonary embolism, peptic ulcer, gi bleeding, hyperkalemia, liver failure, hepatitis, gerd |
| `egfr` (mL/min/1.73 m²) | kidney function → Metformin, Ramipril, Ibuprofen, Amoxicillin |
| `alt`, `ast` (U/L) | liver function → Atorvastatin, Omeprazole |
| `hba1c` (%) | diabetes control → Metformin |
| `systolic_bp` (mmHg) | hypertension/hypotension → Amlodipine, Ramipril, Metoprolol |
| `ldl` (mmol/L) | cholesterol → Atorvastatin |
| `inr` | anticoagulation → Warfarin |
| `allergies` (lower-cased list) | allergy-class contraindications |
| `sex`, `diastolic_bp`, `weight_kg` | read/passed but **not used** by any rule |

## 2. Universal rules (every drug)
| Condition | Output |
|-----------|--------|
| age ≥ 65 | dose note: lower starting doses, closer monitoring |
| age ≥ 80 | **warning**: increased sensitivity to most medications (Beers-style caution) |
| BMI > 30 | dose note: obesity may affect distribution – verify weight-based dosing |
| allergy-class match | verdict forced to **Contraindicated**, `⚠️ ALLERGY ALERT` inserted as reason #1 |

## 3. Per-drug rules

### Metformin (biguanide, first-line T2DM)
| Rule | Verdict | Medical reason |
|------|---------|----------------|
| eGFR < 30 | **Contraindicated** | accumulates in renal failure → lactic acidosis |
| eGFR 30–44 | Caution | reduce dose, monitor renal function (matches FDA label guidance: don't *initiate* at 30–45) |
| eGFR ≥ 45 | (reason: adequate) | – |
| diabetes in conditions **or** HbA1c ≥ 6.5 % | reason: indicated | 6.5 % is the ADA diagnostic threshold |
| otherwise | warning: no diabetes diagnosis | Metformin used for diabetes (verdict unchanged!) |
| HbA1c > 9 % | dose note | may need combination therapy |

### Atorvastatin (statin)
| Rule | Verdict | Reason |
|------|---------|--------|
| ALT or AST > 90 (≈ 3× ULN with ULN ~30) | **Contraindicated** | active liver disease |
| ALT or AST > 45 (≈ 1.5× ULN) | Caution | monitor liver |
| LDL > 3.0 or heart disease / hypertension / diabetes | reason: indicated | CV risk reduction |
| otherwise | warning: LDL controlled, confirm indication | – |

### Amlodipine (dihydropyridine CCB)
| Rule | Verdict | Reason |
|------|---------|--------|
| heart failure | Caution | fluid retention/oedema |
| SBP ≥ 140 or hypertension | reason: indicated | first-line antihypertensive |
| SBP < 100 | **Avoid** | hypotension |
| else | reason: confirm indication | – |

### Ramipril (ACE inhibitor)
| Rule | Verdict | Reason |
|------|---------|--------|
| eGFR < 30 | Caution (+ dose note: lowest dose, recheck K⁺/eGFR in 1–2 wks) | can worsen renal function |
| eGFR 30–44 | Caution | reduce dose, monitor |
| diabetes | reason: beneficial | renoprotective |
| heart failure | reason: beneficial | mortality benefit |
| SBP ≥ 140 / hypertension | reason: indicated | – |
| "hyperkalemia" | **Avoid** | ACE-I raises potassium |

### Metoprolol (β-blocker)
| Rule | Verdict | Reason |
|------|---------|--------|
| asthma or COPD | **Contraindicated** | bronchoconstriction |
| heart failure | reason: beneficial in stable HF | mortality benefit |
| SBP ≥ 140 / hypertension | reason: indicated | – |
| diabetes | dose note | may mask hypoglycaemia (except sweating) |
| "heart block" | **Contraindicated** | worsens AV conduction |

### Warfarin (vitamin-K antagonist)
| Rule | Verdict | Reason |
|------|---------|--------|
| INR > 3.5 | Caution | already over-anticoagulated |
| INR < 2 and AF | reason: sub-therapeutic | dose adjustment |
| AF / DVT / PE | reason: indicated | anticoagulation |
| peptic ulcer or GI bleeding | **Avoid** | bleeding risk |
| always | dose notes: many food/drug interactions; vitamin-K diet | regular INR monitoring |

### Amoxicillin (penicillin)
| Rule | Verdict | Reason |
|------|---------|--------|
| "penicillin" allergy | **Contraindicated** | cross-class |
| eGFR < 30 | dose note | reduce dose/extend interval |

### Ibuprofen (NSAID)
| Rule | Verdict | Reason |
|------|---------|--------|
| eGFR < 30 | **Contraindicated** | renal haemodynamics (prostaglandin-dependent perfusion) |
| eGFR 30–59 | Avoid | worsens CKD |
| heart failure | Avoid (if still Suitable) | fluid retention |
| peptic ulcer / GI bleed | Avoid (if still Suitable) | mucosal injury |
| age ≥ 65 | Caution (if still Suitable) | GI bleed/renal risk |
| always | dose note: take with food; consider PPI for long-term use | – |
The `x if suitability == "Suitable" else suitability` pattern ensures a *milder* rule never downgrades a *stronger* one.

### Paracetamol / acetaminophen
liver failure or hepatitis ⇒ Caution; else "generally the safest analgesic". Dose note: ≤ 4 g/day (2 g/day in frail elderly/liver disease), avoid alcohol.

### Omeprazole (PPI)
Always suitable (reason text); indication note for peptic ulcer/GERD; ALT/AST > 90 ⇒ dose-reduction note; long-term monitoring of Mg²⁺ and B12.

## 4. Drug–drug interaction table (8 pairs)

| Pair | Severity | Mechanism / message |
|------|----------|---------------------|
| Warfarin + Ibuprofen | **CRITICAL** | NSAID: platelet inhibition + GI injury + (displacement) ⇒ major bleeding |
| Ramipril + Ibuprofen | HIGH | NSAID blunts ACE-I effect, worsens kidney function ("triple whammy" with diuretics) |
| Warfarin + Amoxicillin | MODERATE | antibiotics change gut flora/vitamin K ⇒ INR changes |
| Warfarin + Omeprazole | MODERATE | CYP2C19 inhibition may raise (R-)warfarin levels |
| Metformin + Ibuprofen | MODERATE | NSAID reduces GFR ⇒ metformin accumulation |
| Metoprolol + Ibuprofen | MODERATE | NSAID blunts β-blocker antihypertensive effect |
| Ramipril + Metoprolol | LOW | additive hypotension |
| Amlodipine + Metoprolol | LOW | additive BP lowering, usually well tolerated |

## 5. Allergy map
| Allergy key (lower-cased in UI) | Blocks |
|---------------------------------|--------|
| penicillin | Amoxicillin |
| nsaids | Ibuprofen |
| ace inhibitors | Ramipril |
| statins | Atorvastatin |
| beta blockers | Metoprolol (no UI checkbox) |
| sulfa drugs | – (none of the 10 drugs) |

## 6. Why `if/elif` per drug, and why rules?
* **Transparent & testable:** each branch can be unit-tested with a patient dict and the returned `reasons` explain the decision.
* **Safe default:** unmatched cases stay "Suitable" with reassuring reason text; hard stops are explicit.
* **Cons:** O(drugs) code growth, duplicated threshold logic, free-text condition matching, no versioning/provenance of guidelines. In production: rule tables in data (JSON/YAML/DB) with guideline citations, standard vocabularies (SNOMED CT for conditions, LOINC for labs, RxNorm/ATC for drugs), and a rules engine / CDS Hooks / FHIR integration.

## 7. Rule-engine test ideas (shows testing mindset)
```python
def test_metformin_contraindicated_low_egfr():
    r = evaluate_drug({"egfr": 28, "conditions": ["diabetes"]}, "Metformin")
    assert r["suitability"] == "Contraindicated" and r["risk_level"] == "Critical"

def test_penicillin_allergy_blocks_amoxicillin():
    r = evaluate_drug({"allergies": ["penicillin"]}, "Amoxicillin")
    assert r["suitability"] == "Contraindicated" and r["reasons"][0].startswith("⚠️ ALLERGY")

def test_interaction_is_order_independent():
    assert check_drug_interactions(["Ibuprofen", "Warfarin"])[0]["severity"] == "CRITICAL"
```
(There are currently **no automated tests** in the repo — say so, and say these are what you'd add first.)
