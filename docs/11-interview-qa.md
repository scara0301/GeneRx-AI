# 11 — Interview Q&A, part 1: project, architecture, data, feature engineering

How to use: read the **question**, say your answer out loud, *then* read the model answer. Answers are
written in first person so you can adapt them. Cross-references point to the detailed docs.
Part 2 → [11b (modelling & evaluation)](11b-interview-qa-modelling.md) · Part 3 → [11c (engineering, deployment, ethics, behavioural, curveballs)](11c-interview-qa-engineering.md)

---

## A. The project & motivation

**Q1. Tell me about this project in one minute.**
A clinical drug-safety assistant. You enter a patient's demographics, conditions, labs, current meds and allergies and choose candidate drugs; it returns Suitable/Caution/Avoid/Contraindicated per drug with reasons, dose notes, monitoring, side-effects, drug–drug interactions, plus an XGBoost risk signal trained on FDA adverse-event data. FastAPI backend, vanilla-JS frontend with clinician and patient modes, Docker on Hugging Face Spaces. (See README pitch.)

**Q2. What problem does it solve and who is the user?**
Medication errors and adverse drug events are a major, preventable cause of harm. Prescribers must mentally check renal/hepatic function, comorbidities, allergies and interactions for every drug. The tool automates that checklist. Two users: clinicians (full detail) and patients (plain language, "discuss with your doctor").

**Q3. Why did you build it / what did you want to learn?**
(Make this yours.) Safe framing: "I wanted a project that covers the whole ML lifecycle on real public data — API ingestion, cleaning, feature engineering, tuning, serving and deployment — in a domain where explainability matters, so I had to think about *how* ML and rules should coexist rather than just maximise accuracy."

**Q4. Why healthcare/drug safety?**
Public, high-quality data (openFDA, SIDER); high real-world stakes that force good engineering judgement (explainability, safety, honesty about limits).

**Q5. Why is it called GeneRx if it doesn't use genetics?**
It began as a pharmacogenomics idea (leftover `drug_catalog.json` has CYP2C9/VKORC1…). Individual genotype data isn't something a clinician usually has at the point of care and I had no patient-level genomic dataset, so v2 pivoted to routinely available clinical parameters. The name stayed; adding CPIC gene–drug rules is on the roadmap.

**Q6. What is the single most important design decision?**
Rules decide, ML advises. In clinical software a contraindication must be explainable and deterministic; a probabilistic score can't be allowed to outvote "eGFR < 30 ⇒ no Metformin".

**Q7. Is this a medical device? Could a doctor rely on it?**
No. It's educational/decision-support-style demo, not validated, covers 10 drugs, and carries a disclaimer in README and footer. A real product would need clinical validation, regulatory clearance (software as a medical device), curated knowledge bases and governance.

**Q8. What's novel about your approach?**
Honestly: not the algorithm. The value is integration — combining a transparent rules layer, a data-driven risk model built from two heterogeneous sources (FAERS reports + SIDER labels), a hand-engineered MedDRA severity feature, and a usable two-audience UI, end-to-end and deployed.

**Q9. What was the hardest part?**
Two: (1) turning FAERS — noisy, voluntary, no denominator — into a sensible supervised target and being honest about what it means; (2) deployment: Vercel didn't suit it, then Hugging Face rejected the binary model, which led to training at Docker build time.

**Q10. What would you do differently?**
Start with a test suite and the evaluation protocol, pin dependencies, design the serving feature set *first* (to avoid train/serve skew), and use patient-level data for the model.

**Q11. How long did it take / team size?** (Answer truthfully. The git history shows an initial build in April 2025, a rebuild and deployment in March 2026.)

**Q12. How did the project evolve?**
Streamlit prototype (Apr 2025) → FastAPI + SPA rebuild (Mar 2026) → deployment on Vercel failed → Hugging Face Spaces with Docker → fix API routing for the HF iframe path → solve model-file problem via build-time training. (Details: [01](01-overview-and-why.md) §5.)

---

## B. Architecture & design

**Q13. Draw/describe the architecture.** See [03](03-architecture.md): browser SPA → FastAPI → {rules engine, ML predictor (pickle)}; offline pipeline fetch → build_dataset → train → pickle; Docker/HF deployment.

**Q14. Why a hybrid system rather than pure ML?** Explainability, safety (hard contraindications), and because FAERS has no lab values — rules are the only way to use eGFR/ALT/INR. ML adds a data-derived population signal.

**Q15. Why not pure rules?** Rules know only what was coded; they give no probabilities or data-driven signal; ML can surface patterns (polypharmacy, age effects) rules don't capture.

**Q16. Does the ML ever change the verdict?** No. `suitability` and `risk_level` come only from `evaluate_drug`; the ML result is attached as `ml_prediction`. I verified they can disagree (Metformin at eGFR 28: Contraindicated vs ML Low Risk).

**Q17. Isn't it a problem that ML and rules can disagree?** They answer different questions (patient-specific suitability vs. typical severity of reported events). The disagreement is shown, not hidden. The improvement is a patient-level model with labs as features, or an explicit reconciliation rule (e.g., show a banner when ML is Low but rules say Contraindicated).

**Q18. Why a stateless API with no database?** No PHI persisted (privacy), trivial deployment/scaling, the problem is a pure function of the request. A DB would be needed for accounts, audit logs, saved patients.

**Q19. Why one endpoint `/api/assess` for everything?** One round trip, consistent snapshot of the patient, simpler UI; internally it composes small functions so they remain testable.

**Q20. How does the system degrade if the ML model is missing?** `_load_model` catches the missing file/exception, `loaded=False`; `predict` returns `{"available": false}`; `/api/health` reports `ml_model_loaded:false`; UI shows "Rules Engine Only" and hides the chart.

**Q21. Why a singleton predictor?** Loading a pickle per request would be slow; `get_predictor()` lazily creates one instance reused by all requests.

**Q22. Is the singleton thread-safe?** Reads are safe once built. The lazy init has a benign race (two threads could both construct it on the first request) — worst case it loads twice. Fix: initialise at app startup (`@app.on_event("startup")` / lifespan).

**Q23. Why do you serve the frontend from FastAPI?** One container, one port, no CORS in production, matches HF Spaces' single-port model.

**Q24. Why enable CORS then?** For local development with the UI on :3000 and API on :8000. In production restrict origins.

**Q25. How would you scale it?** Stateless ⇒ run N replicas behind a load balancer; model loaded per worker (`uvicorn --workers`/gunicorn); predictions are µs; the cost is mostly JSON/Python overhead. Cache by request hash if needed. For heavy models, move inference to a separate service.

**Q26. How would you version the model and rules?** Model: artifact with metadata (version, training data hash, metrics), stored in a registry (MLflow/HF Hub). Rules: move to data files with guideline citations and semantic versions; log both versions with every assessment.

**Q27. How would you integrate with an EHR?** FHIR resources (Patient, Observation for labs, MedicationStatement, AllergyIntolerance) → map LOINC/SNOMED/RxNorm codes to the engine's inputs; expose as a CDS Hooks service (`order-select`).

**Q28. What are the failure modes?** Missing/invalid inputs (defaults hide missingness), unknown drug (500), model absent (handled), CDN unavailable (UI icons/chart), pickle incompatibility. Defaults are *normal* lab values, which can mask "unknown" as "normal" — better to track which values were actually provided.

**Q29. Why Pydantic defaults to normal values?** So partial input doesn't trigger false alarms; downside is silent assumption. A safer UX: mark fields as "assumed normal" in the response.

**Q30. Why return `reasons` as a list of strings?** Explainability + the UI can show the first as "Key concern". Better: structured reasons `{code, severity, text, evidence}`.

---

## C. Data

**Q31. What data did you use and why?** openFDA FAERS (real-world adverse event reports) and SIDER 4.1 (label side-effects with frequencies) for 10 common drugs. FAERS gives outcomes/seriousness; SIDER gives a label-based per-drug profile. Both are free and public.

**Q32. What is FAERS?** The FDA Adverse Event Reporting System: spontaneous reports from clinicians, consumers and manufacturers; each report lists patient info, drugs (suspect/concomitant), MedDRA-coded reactions and outcomes (death, hospitalisation, life-threatening, disability…).

**Q33. What is MedDRA?** The Medical Dictionary for Regulatory Activities, a standardised hierarchy of medical terms (LLT → PT → HLT → HLGT → SOC). FAERS reactions are *preferred terms* (PT), which is what my severity dictionary is keyed on.

**Q34. What is SIDER?** A database of drug side effects extracted from package inserts, with MedDRA mapping and (for some) frequencies. v4.1 (2015). Static.

**Q35. How did you collect FAERS?** `fetch_drug_events` pages the openFDA endpoint (`limit=100`, `skip`) with a `generic_name` search for each of 10 drugs, up to 5,000 each; retries on errors, sleeps to respect rate limits, parses nested JSON into 15 flat columns.

**Q36. Rate limiting & robustness?** 0.3 s between calls, 1 s between drugs, retry loops for non-200 and network errors, 404 ⇒ stop. Weakness: no retry cap.

**Q37. How many records / classes?** Up to 50,000 events (10 drugs × 5,000), README says 50,000 → 40k train/10k test. Class distribution is printed by `build_dataset.py` (check it before the interview).

**Q38. What are the limitations of FAERS?** No denominator, reporting bias, duplicates, missing fields, confounding by indication, causality not established. So it supports *signal detection*, not incidence estimates.

**Q39. How did you handle missing data?** Age/weight: invalid values → NaN, then median imputation. Sex: unknown → 0.5. SIDER features: missing → 0. Trees are robust to scaling; median is outlier-robust.

**Q40. Why median imputation and not model-based imputation?** Simple, fast, robust to outliers; fraction of missing values modest. Better: add `age_missing`/`weight_missing` indicators or use XGBoost's native missing handling (leave NaN).

**Q41. Outlier handling?** Plausibility filters: age 0–100, weight 10–250 kg → treated as missing.

**Q42. Did you check for duplicates?** Not explicitly — FAERS has duplicate/follow-up reports for the same case. Improvement: dedupe on `safetyreportid`/case ID (the fetcher doesn't store it).

**Q43. Class imbalance?** I print the class distribution and use a stratified split. I didn't apply class weights/SMOTE. For rare Critical class I'd use `sample_weight`, macro-F1 and probability thresholds.

**Q44. Why those 10 drugs?** Common drugs spanning diabetes, cardiovascular, anticoagulant, antibiotic, NSAID, analgesic and GI classes, with well-documented renal/hepatic/allergy/interaction logic that I could encode as rules, and enough FAERS volume.

**Q45. How do you join FAERS and SIDER?** By lower-cased generic name: compute SIDER stats per drug into a dict and `map` it onto each FAERS row. Paracetamol/acetaminophen aliasing: SIDER target set includes both; FAERS uses acetaminophen.

**Q46. Is there data leakage in the pipeline?** I deliberately exclude the four seriousness flags from the features because they define the target. Other risk: the weighted reaction score is derived from outcomes recorded in the same report (valid for retrospective classification, but unavailable at prescribing time → the serving-default problem). Also random split with drug-level duplicates is mildly optimistic.

**Q47. How do you ensure reproducibility?** Fixed seeds (`random_state=42`, `random.seed`, `np.random.seed`), deterministic data generator, Docker build.

**Q48. How would you get better data?** MIMIC-IV / EHR cohorts with labs and outcomes, claims databases, FAERS deduplicated with disproportionality statistics (PRR/ROR/EBGM), DrugBank/DDInter for interactions, PharmGKB/CPIC for genes.

---

## D. Feature engineering

**Q49. Walk me through your features.** Eleven: drug (label-encoded), age, sex, weight, reaction_count, weighted_reaction_score, is_primary_suspect, num_concomitant_drugs, and three SIDER stats (count, mean freq, max freq). Table in [07](07-data-and-feature-engineering.md) §4.

**Q50. What is `weighted_reaction_score` and why?** A clinical-severity score for the reaction list: each MedDRA term gets a weight 1–10 (cardiac arrest 10 … nausea 1); score = 0.6·max + 0.4·mean. It converts text to a numeric signal that reflects *how bad* reactions are, which count alone can't.

**Q51. Why 60/40 max/mean?** Heuristic: the worst reaction should dominate (one SJS case matters more than ten headaches) but breadth should still count. Not tuned; I'd run a sensitivity analysis or learn the weights.

**Q52. How did you pick the severity weights?** Clinical judgement grouped by organ system (cardiac, renal/hepatic, haematologic/anaphylaxis, severe cutaneous, neurologic, mild). A limitation: not clinician-validated; could be replaced by an established scale (e.g., CTCAE grades) or learned embeddings.

**Q53. Any bug in that function?** Yes — `max_weight` uses an exact dictionary lookup while per-term weights use substring match, so long terms (e.g. "Cardiac arrest during surgery") get max=1.0. I'd collect per-term weights and take their max. (✅ reproduced: 4.6 vs expected 10.)

**Q54. Why substring matching?** MedDRA terms vary ("Renal failure acute", "Acute kidney injury"); substring catches variants without a full ontology. Risk: false matches (e.g., "hepatitis" inside "hepatitis B vaccination reaction").

**Q55. What is `is_primary_suspect`?** 1 if FAERS characterised the drug as the primary suspect (`drugcharacterization = 1`), meaning the reporter believes this drug caused the event; vs secondary suspect/concomitant.

**Q56. Why `num_concomitant_drugs`?** Polypharmacy increases interaction risk and complexity; in the API it's approximated by the count of the patient's current meds.

**Q57. Why SIDER features and what do they capture?** `known_side_effects_count` (breadth of documented toxicity), `mean/max_se_frequency` (how often side effects occur). They're drug-level constants — essentially a drug fingerprint with clinical meaning.

**Q58. Aren't SIDER features redundant with `drug_encoded`?** Yes, within the training set they are deterministic functions of drug. They add interpretability and would let the model generalise to unseen drugs in principle; for 10 drugs they are redundant. Trees handle collinearity fine.

**Q59. Why label-encode the drug rather than one-hot?** With 10 values trees can still isolate any drug via successive splits; one-hot adds 10 columns. Label encoding imposes an arbitrary order — for linear models that would be wrong. Better options: one-hot, target encoding, native categorical support.

**Q60. Why not scale features?** Tree-based models are invariant to monotonic transforms.

**Q61. Why was sex encoded 0/1/0.5?** To keep rows with unknown sex without inventing a class. For trees it's just a 3-valued numeric feature.

**Q62. Which features did you not use and why?** The four seriousness flags (target leakage), `disability` (not in the target definition), `indication`, `reaction_outcomes` and raw reaction text (not engineered), and per-drug aggregate rates (would be a leaky target encoding).

**Q63. How did you construct the target?** `risk_score = 0.2·serious + 0.2·hospitalization + 0.3·life_threatening + 0.3·death`, bins ≤0.1/0.3/0.6 → Low/Moderate/High/Critical. See table in [07](07-data-and-feature-engineering.md) §5.

**Q64. Why not predict death directly?** Too rare/imbalanced and ignores severity gradations; a graded outcome uses more information and maps to the UI's four levels.

**Q65. Are the target weights justified?** They're clinical-judgement heuristics. A different weighting changes class boundaries; I'd run a sensitivity analysis or have clinicians rank outcome severities.

**Q66. Is the target ordinal? Did you treat it that way?** It's ordinal but trained as nominal multiclass. Ordinal approaches (cumulative-link models, regression + thresholds, or ordinal-aware loss) would exploit the order.

**Q67. Could you engineer more features?** Interaction counts (known DDIs among concomitant drugs), age bins, BMI, reaction-system-organ-class one-hots, time-to-onset, indication flags, text embeddings of reaction terms, drug class, reporter type, country, report year (drift).

**Q68. How do you validate that a feature helps?** Ablation (drop and compare CV score), permutation importance, SHAP, and checking for leakage by looking at suspiciously dominant features.

**Q69. What did feature importance show?** (Synthetic model, verified) reaction_count 25 %, is_primary_suspect 18 %, num_concomitant_drugs 18 %, drug 17 %, age 7 % … This mirrors my synthetic generator, so it's not a clinical finding.

**Q70. What is the synthetic dataset and why does it exist?** `build_model.py` generates 12,000 rows using per-drug base risk + modifiers (age > 75, ≥5 co-meds, ≥4 reactions, severity > 2.5, primary suspect with >2 reactions) + 10 % noise. It exists so Docker can train a working model offline. Its accuracy (91 %) reflects the noise ceiling; it is a placeholder.
