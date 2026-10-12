# 07 — Data, feature engineering and the target variable

## 1. Data sources

### 1.1 openFDA FAERS — FDA Adverse Event Reporting System
* **What it is:** a public database of adverse-event *reports* submitted by doctors, pharmacists, patients and manufacturers. Accessed via the REST endpoint `https://api.fda.gov/drug/event.json`.
* **One record = one report** (a "safety report") containing: patient (age, sex, weight), a list of drugs (with a *characterisation*: 1 primary suspect, 2 secondary suspect, 3 concomitant), a list of reactions (MedDRA *preferred terms*), and seriousness flags (`serious`, `seriousnessdeath`, `seriousnesshospitalization`, `seriousnesslifethreatening`, `seriousnessdisabling`, …).
* **What I fetched:** for each of 10 drugs, up to 5,000 reports filtered on `patient.drug.openfda.generic_name:"<drug>"` → up to **50,000 rows** (README: "50,000 events").
* **Why FAERS:** the biggest free source of real-world safety data; JSON API, no key required for light use; contains exactly what a "risk" model needs (severity outcomes).
* **Known limitations (say them!):**
  1. **No denominator** – we don't know how many people took the drug, so we can't compute incidence/risk rates. Only "among reports, how serious".
  2. **Reporting bias** – voluntary; serious/novel events over-reported; media attention, litigation, drug age affect volumes (stimulated reporting).
  3. **Confounding** – patients on Warfarin are already sick; seriousness may reflect the disease not the drug. A report ≠ causation.
  4. **Duplicates/missing fields** – age/weight often missing (handled by median imputation).
  5. **Sampling** – first N records returned by the API, not a random sample.

### 1.2 SIDER 4.1 — Side Effect Resource (EMBL)
* **What it is:** side-effects mined from public **drug labels**, mapped to MedDRA terms, some with **frequencies** (percent or ranges).
* **Files used:** `drug_names.tsv` (compound-ID → name), `meddra_all_se.tsv.gz` (drug–side-effect pairs), `meddra_freq.tsv.gz` (frequencies).
* **Why SIDER:** gives a *label-based, drug-level* view (how many side effects are known and how frequent) which is independent of reporting behaviour in FAERS. README: "1,934 pairs" for the 10 drugs.
* **Limitations:** SIDER 4.1 dates from ~2015 (not updated since), frequencies exist for only a subset of pairs, name matching is by exact lowercase generic name.

### 1.3 Synthetic data (`build_model.py`) — what actually ships in Docker
Used because the Space had no network at build time and rejected binaries. See §6.

### 1.4 `data/drug_catalog.json` (legacy)
Three drugs (Warfarin, Clopidogrel, Simvastatin) with pharmacogenes (CYP2C9/VKORC1, CYP2C19, SLCO1B1), molecular formulae and BERT-style `input_ids`/`attention_mask` – the remnant of the original gene/NLP-based concept. Not read by any code.

---

## 2. Raw schema produced by `fetch_faers.py` (15 columns)

| Column | Source field in openFDA | Notes |
|--------|------------------------|-------|
| `drug_name` | search term | lower-case generic name |
| `patient_age` | `patient.patientonsetage` + `patientonsetageunit` | converted to years (see bug note: unit 803 = weeks, code divides by 365) |
| `patient_sex` | `patient.patientsex` | 1→"M", 2→"F", else "Unknown" |
| `patient_weight` | `patient.patientweight` | kg |
| `reactions` | `patient.reaction[].reactionmeddrapt` | joined with `\|` |
| `reaction_count` | derived | number of reaction terms |
| `serious` | `serious` | 0/1 |
| `death` | `seriousnessdeath` | 0/1 |
| `hospitalization` | `seriousnesshospitalization` | 0/1 |
| `disability` | `seriousnessdisabling` | 0/1 — collected, **not used** in the target |
| `life_threatening` | `seriousnesslifethreatening` | 0/1 |
| `drug_role` | `patient.drug[].drugcharacterization` of the matched drug | Primary Suspect / Secondary Suspect / Concomitant / Unknown |
| `indication` | `drugindication` | not used by the model |
| `reaction_outcomes` | `reactionoutcome` | not used |
| `num_concomitant_drugs` | `len(patient.drug) - 1` | how many *other* drugs were on the report |

---

## 3. Cleaning (in `build_training_data`)

| Step | Code idea | Why |
|------|-----------|-----|
| Normalise drug names | `.str.lower().str.strip()` | consistent keys for joins & encoding |
| Coerce numerics | `pd.to_numeric(errors="coerce")` | bad strings → NaN instead of crashing |
| Plausibility filter | age ∉ [0,100] → NaN; weight ∉ [10,250] kg → NaN | FAERS contains typos/outliers (e.g. age 999, weight 0.5) |
| Median imputation | `fillna(median)` for age & weight | robust to outliers; keeps all rows; trees don't need scaling |
| Sex encoding | M→0, F→1, unknown→0.5 | numeric, keeps "unknown" distinct instead of dropping rows |
| Safety net in training | `X.fillna(0)` in `train_model.py` | any residual NaN (e.g. drug missing in SIDER) → 0 |

**Why median not mean / not dropping rows?** Age and weight are right-skewed with outliers → median is robust; dropping rows with missing demographics would lose a large chunk of FAERS.
**Why not scale?** Tree models are invariant to monotonic feature scaling.

---

## 4. The 11 model features

Order matters — it's the order of `feature_cols` and of the vector built in `ml_model.predict`.

| # | Feature | Type | How it is computed | Why it is useful | At serving time |
|---|---------|------|--------------------|------------------|-----------------|
| 1 | `drug_encoded` | int (0–9) | `LabelEncoder` on drug name | the drug is the strongest prior for severity | from the request |
| 2 | `patient_age` | float | cleaned FAERS age | older ⇒ more severe outcomes | from the request |
| 3 | `sex_encoded` | 0/1/0.5 | map M/F | sex-linked ADR differences | from the request (0.5 never used) |
| 4 | `patient_weight` | float kg | cleaned FAERS weight | dose/exposure proxy | from the request |
| 5 | `reaction_count` | int | number of MedDRA terms in the report | more reactions ⇒ more complex/severe case | **default 0** |
| 6 | `weighted_reaction_score` | float 1–10 | `utils.get_reaction_severity` (0.6·max + 0.4·avg of term weights) | encodes *clinical gravity* of reactions, not just count | **default 1.0** |
| 7 | `is_primary_suspect` | 0/1 | `drug_role == "Primary Suspect"` | whether this drug is the main suspected cause | **default 1** |
| 8 | `num_concomitant_drugs` | int | `len(drugs) − 1` | polypharmacy ⇒ interactions ⇒ risk | `len(current_meds)` |
| 9 | `known_side_effects_count` | int | SIDER: count of distinct side-effects per drug | breadth of known toxicity | **default 0** |
| 10 | `mean_se_frequency` | float % | SIDER: mean frequency over pairs that have a %; | typical frequency | **default 0** |
| 11 | `max_se_frequency` | float % | SIDER: max frequency | worst-case frequency | **default 0** |

### Engineering highlights (the parts to brag about)
1. **Domain-knowledge feature** — `weighted_reaction_score`: turning free-text MedDRA reaction names into a clinically weighted severity number. Uses *substring* matching so "Acute renal failure" hits "renal failure"; combines worst-case (60 %) and average (40 %). It is a hand-crafted prior, not learned.
2. **Cross-source join** — SIDER's drug-level statistics are joined onto each FAERS row by drug name (`map` with a lambda; missing ⇒ 0).
3. **Role feature** — `is_primary_suspect` separates "drug probably caused it" from "was just on the list".
4. **Polypharmacy proxy** — `num_concomitant_drugs`.

### Features *not* used and why
* `death`, `hospitalization`, `life_threatening`, `serious` — **they define the target** ⇒ using them would be **target leakage** (100 % accuracy for the wrong reason).
* `disability`, `indication`, `reaction_outcomes`, `reactions` (raw text) — not engineered (future work: indication as a feature; TF-IDF/embeddings of reactions).
* Lab values (eGFR, ALT, …) — **FAERS doesn't contain them**, which is why labs only drive the rules engine.
* Per-drug aggregate rates from `compute_drug_risk_profile` (serious_rate, death_rate, …) — computed and saved, but deliberately not fed in (they'd be a leaky target encoding computed on the full data).

### Feature importance in the shipped (synthetic) model — verified
| Feature | Importance |
|---------|-----------:|
| reaction_count | 0.249 |
| is_primary_suspect | 0.184 |
| num_concomitant_drugs | 0.178 |
| drug_encoded | 0.165 |
| patient_age | 0.066 |
| max_se_frequency | 0.053 |
| weighted_reaction_score | 0.026 |
| mean_se_frequency | 0.023 |
| patient_weight | 0.021 |
| known_side_effects_count | 0.018 |
| sex_encoded | 0.017 |

Interpretation: mirrors the generator's rules (reaction_count ≥ 4, primary-suspect with > 2 reactions, ≥ 5 co-meds, drug base risk, age > 75). Weight and sex are near-noise by construction. **Do not present these as clinical insights** — they reflect the synthetic data-generating process. For the real FAERS model, re-run and read `model.feature_importances_` (or SHAP) yourself.

---

## 5. The target variable (`risk_category`)

```python
risk_score = 0.20*serious + 0.20*hospitalization + 0.30*life_threatening + 0.30*death

categorize_risk(score):  ≤0.1 → 0 Low | ≤0.3 → 1 Moderate | ≤0.6 → 2 High | else → 3 Critical
```

What the classes mean in practice (all flags are 0/1; in FAERS the other seriousness flags normally come with `serious=1` — I didn't verify this on the real download):

| Report has… | risk_score | Class |
|-------------|-----------:|-------|
| nothing serious | 0.0 | 0 – Low Risk |
| serious only | 0.2 | 1 – Moderate |
| serious + hospitalisation | 0.4 | 2 – High Risk |
| serious + life-threatening | 0.5 | 2 – High Risk |
| serious + death | 0.5 | 2 – High Risk |
| serious + 2 of {hosp, LT, death} | 0.7–0.8 | 3 – Critical |
| all four | 1.0 | 3 – Critical |

* **Why a weighted sum, not just `death`?** Death alone is rare (class imbalance) and ignores severity gradations; a graded ordinal target uses more of the signal.
* **Why these weights?** Hand-chosen clinical judgement: life-threatening and death are the gravest (0.30 each), hospitalisation and "serious" less (0.20 each). They are not tuned — a limitation; a sensitivity analysis would be the improvement.
* **Why 4 classes?** Matches the UI's risk levels (Low/Moderate/High/Critical) and the rules engine's levels.
* **It is an *ordinal* target treated as nominal** – multi-class softmax ignores that 3 is "worse" than 1. Alternatives: ordinal regression, or regression on `risk_score`.
* **What the label really measures:** *seriousness of a reported event*, not the probability that this drug harms this patient.

---

## 6. Synthetic dataset (`build_model.generate_dataset`) — exact recipe

```
12,000 rows. For each row:
  drug      ~ uniform over 10 drugs
  age       ~ Normal(55, 18) clipped to [18, 95]
  sex       ~ {0,1}
  weight    ~ Normal(75, 18) clipped to [40, 150]
  n_conc    ~ randint(0, 8)          concomitant drugs
  n_reac    ~ randint(0, 6)          reaction count
  w_reac    ~ Normal(1.2, 0.6) clipped [0.1, 4.0]   weighted reaction score
  is_ps     ~ {0,1}
  se_cnt    = profile.se_count + randint(-1, 2)
  se_mf/xf  = profile.mean/max + Normal(0, 0.05) clipped [0,1]

  risk = profile.base_risk                                     # hand-assigned per drug
  +1 if age > 75; +1 if n_conc ≥ 5; +1 if n_reac ≥ 4; +1 if w_reac > 2.5; +1 if is_ps and n_reac > 2
  (each +1 capped at 3)
  with 10 % probability: risk += ±1 (clipped to 0..3)         # label noise
```
Base risks: Warfarin 2 (High); Ramipril, Metoprolol, Ibuprofen 1 (Moderate); the other six 0 (Low).
Class distribution (verified): Low 1,787 · Moderate 3,442 · High 3,445 · Critical 3,326.
Split: 9,600 train / 2,400 test (stratified, seed 42). **Test accuracy 91.4 %** — which is essentially the ceiling set by the 10 % label noise (a perfect learner of the rule gets ≈ 90–95 %). Meaning: the model has *rediscovered the generator*, which tells you nothing about real clinical risk.

**Why build this at all?** It's a pragmatic deployment fallback: deterministic, offline, ships a working ML component in the container. **Why it's defensible:** the base risks encode real clinical priors (anticoagulants are riskier than PPIs; elderly and polypharmacy raise risk), and the artifact format is identical to the real pipeline so swapping in the FAERS-trained model is a one-file change. **Why it's a weakness:** it's circular; be upfront.

---

## 7. Real-data pipeline numbers (from README — unverified by me)
* 50,000 FAERS events → 40,000 train / 10,000 test (80/20 stratified).
* Tuned XGBoost → **63.1 %** test accuracy on 4 classes.
* Why accuracy is modest: features are mostly drug-level constants plus coarse demographics; the label is noisy (voluntary reports); no patient-level clinical covariates; four overlapping ordinal classes. To judge it you need the **majority-class baseline** and **per-class precision/recall/F1** (`train_model.py` prints a `classification_report`) – check those before the interview.
