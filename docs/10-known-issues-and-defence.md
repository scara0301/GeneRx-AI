# 10 — Known issues, limitations, and how to talk about them

Strategy: **say it first, calmly, with the fix.** Interviewers reward self-awareness; they punish
defensiveness. Every item below was found by reading the code and (where marked ✅) reproducing it by
running the code.

Severity key: 🔴 could embarrass you if discovered · 🟠 should know · 🟡 minor

---

## 🔴 1. The deployed ML model is trained on *synthetic* data
**Evidence:** `Dockerfile` → `RUN python backend/build_model.py`; that script generates 12,000 rows from `DRUG_PROFILES` + hand-written modifiers + 10 % noise. ✅ Test accuracy 91.4 % ≈ the ceiling set by the noise.
**README mismatch:** README says "trained on 50,000 real FDA adverse event reports… 63.1 %". That describes the *local research pipeline* (`fetch_* → build_dataset → train_model`), not the Space.
**How to say it:** "There are two training paths. The research path trains on real FAERS+SIDER and gets about 63 % on four classes. For the hosted demo I had a constraint — Hugging Face wouldn't accept the binary model and the build has no data access — so the Docker build trains a model on a synthetic dataset derived from the same drug profiles, with the identical artifact format. The synthetic model is a functional placeholder; its 91 % just means it learned my generator. Swapping in the FAERS model is dropping in a different pickle."
**Fix:** publish the real model via the HF Hub / `huggingface_hub` download at startup (or bake a pre-built pickle into the image from a CI job), and update the README to state which model is deployed.

## 🔴 2. Train/serve skew in `predict()`
**Evidence:** `ml_model.predict` fills `reaction_count=0`, `weighted_reaction_score=1.0`, `is_primary_suspect=1`, `known_side_effects=0`, `mean_se_freq=0`, `max_se_freq=0` and `server.py` never overrides them. The model was trained with real SIDER values per drug. ✅ With serving defaults, Warfarin (base risk High) predicts **Moderate**; with its true SIDER values it predicts **High**.
**How to say it:** "Six of the eleven features describe an adverse event that hasn't happened yet, or come from SIDER which the API doesn't pass, so they're neutral defaults. That causes skew — I measured it flipping Warfarin from High to Moderate. The right fix is to persist the per-drug SIDER stats in the artifact and look them up in `predict`, and ideally train a separate 'pre-event' model using only features available at prescribing time."

## 🔴 3. The ML target doesn't mean "risk to *this* patient"
**Evidence:** target = seriousness of a *reported* event. FAERS has no denominator and is voluntary. ✅ Metformin at eGFR 28: rules → **Contraindicated**, ML → **Low Risk 97 %**.
**How to say it:** "They answer different questions. Rules answer 'is this safe for this patient's labs and history?'; the model answers 'how serious are reported events for this drug in a patient like this?'. That's why the model is advisory and never changes the verdict. A better design feeds labs/comorbidities into the model, using patient-level data."

## 🟠 4. Unknown drug name crashes `/api/assess`
**Evidence:** `DRUG_CATALOG[drug_name]` at the end of `evaluate_drug` ⇒ `KeyError: 'Aspirin'` ⇒ HTTP 500. ✅ Also names are case-sensitive ("metformin" would fail the rule branch silently *and* crash on the catalog lookup).
**Fix:** validate in the endpoint (`HTTPException(422, "Unknown drug")`) or use an `Enum`/`Literal` of catalog names in the Pydantic model; normalise case.

## 🟠 5. Rules cover a toy subset
* 10 drugs; 8 of 45 pairwise interactions; allergies by class; no pregnancy, no dose calculation, no renal dosing tables, no QT, no CYP phenotype.
* **UI ↔ engine vocabulary mismatch** ✅: the UI offers "Liver Disease"; the engine looks for "liver failure"/"hepatitis" ⇒ Paracetamol returns **Suitable** for "Liver Disease". Engine also tests conditions the UI never sends (`heart disease`, `hyperkalemia`, `heart block`, `dvt`, `pulmonary embolism`, `gi bleeding`, `gerd`). Ramipril's hyperkalemia and Metoprolol's heart-block rules are therefore unreachable from the UI.
* Patient mode can never produce allergy alerts (wizard has no allergy step), can't set Atrial Fibrillation (so Warfarin's indication never fires), and fixes ALT/AST/INR at normal values.
* Warfarin with no indication returns "Suitable" with **no** reasons.
**Fix:** a controlled vocabulary shared by UI and engine (SNOMED CT codes), a data-driven rule table, tests for each rule.

## 🟠 6. No automated tests, CI, or pinned dependencies
`requirements.txt` is unpinned (`fastapi`, `xgboost`, …) ⇒ builds are not reproducible; pickles are version-sensitive. **Fix:** pin versions (`pip-compile`/`uv lock`), add `pytest` for rules + API (`TestClient` — I used it to verify the endpoints), GitHub Actions running tests and `ruff`.

## 🟠 7. Pickle for the model
Security (arbitrary code execution on load – only safe because we load our own file) and portability (XGBoost warns about cross-version pickles). **Fix:** `model.save_model("model.ubj")` + JSON for the encoder, or ONNX.

## 🟠 8. Naming vs content ("GeneRx" but no genes)
The header of `clinical_engine.py` says "No genetic data required." `data/drug_catalog.json` is the leftover of the pharmacogenomic idea. **How to say it:** "The project started as a pharmacogenomics idea; public patient-level genotype data wasn't available to build a real model, so v2 pivoted to clinical parameters, which clinicians actually have at the point of care. The name stuck; the roadmap includes PharmGKB/CPIC gene–drug rules as an additional rule layer." (Adjust to your true history.)

## 🟡 9. Small code bugs (all reproducible by reading the code)
| # | Where | Issue | Fix |
|---|-------|-------|-----|
| a | `app.js:171` | `ast` read from the **ALT** input (`labAlt`) – copy/paste; there is no AST field | add an AST input or send `alt` only |
| b | `fetch_faers.py` age unit | code treats `803` as days (÷365); FDA code list: 801 year, 802 month, 803 week, 804 day | `803 → /52`, `804 → /365` |
| c | `utils.get_reaction_severity` | `max_weight` uses an exact dict lookup while the per-term weights use substring match ✅ ("Cardiac arrest during surgery" → 4.6 not 10) | collect the per-term weights in a list and use `max(weights)` |
| d | `evaluate_drug` Amoxicillin | penicillin allergy yields two near-duplicate reasons ✅ | drop the drug-specific branch or dedupe |
| e | `server.py` | BMI uses a hard-coded height of 1.70 m | collect height or remove the BMI rule |
| f | `app.js` `switchMode` | assigns undeclared `currentMode` (implicit global) | `let currentMode` |
| g | `app.js` `renderResults` | `data.patient_name` (user-typed) goes into `innerHTML` unescaped (self-XSS) | `textContent`/escape |
| h | `train_model.py` | `use_label_encoder=False` deprecated in new XGBoost | remove |
| i | `app.js` | `API_BASE` is `localhost:8000` on localhost, so serving on `localhost:7860` breaks the UI | use relative URLs: `fetch('/api/...')` always works because the server serves the UI |
| j | `ml_model.predict` | sex "Unknown" → 0 (training used 0.5) | map unknown to 0.5 |
| k | `app.js` / CSS | "High Risk" (`#ef4444`) and "Critical" (`#dc2626`) chart colours are almost identical | distinct palette |
| l | `fetch_faers.py` | unbounded retry loop on persistent non-200 | max retries + backoff |
| m | `server.py` | `allow_origins=["*"]` + `allow_credentials=True` | restrict origins |
| n | repo | `.gitignore` lists `data/` but `data/drug_catalog.json` is tracked; `.gitattributes` still has the LFS filter | clean up |
| o | `clinical_engine.py` | dead code (`generate_patient_summary`, `simulate_response_over_time`) | delete or wire up |

## 🟡 10. Evaluation gaps (real FAERS model)
* Only accuracy is reported in the README; no macro-F1, confusion matrix, calibration, per-drug breakdown, or confidence intervals.
* Possible optimistic bias from random splits when rows share drug-level features.
* Target weights (0.2/0.2/0.3/0.3) and severity weights (1–10) are hand-set, not validated.

---

## What you can honestly claim as strengths
* End-to-end ownership: data acquisition (2 sources, API pagination, retries) → cleaning → feature engineering → tuning → serving → UI → containerised deployment.
* Sound architecture instinct: explainable rules decide, ML advises; fail-soft behaviour when the model is missing; model artifact bundles encoder + feature list; singleton loader.
* Pragmatic problem-solving under deployment constraints (Vercel → HF; binary-file restriction → build-time training).
* Domain-aware feature (MedDRA severity weighting) and leakage-aware feature choice (seriousness flags excluded from features).
* Awareness of limitations (this document) and a concrete roadmap.

## Roadmap (what you'd do next — say it in this order)
1. Make the deployed model real: persist per-drug SIDER stats, fix skew, publish the FAERS-trained artifact; update README.
2. Tests + CI + pinned dependencies + input validation (drug names, ranges).
3. Replace hand-written interactions with DrugBank/RxNorm/DDInter; add SNOMED/LOINC vocabularies.
4. Patient-level model on EHR data (e.g. MIMIC-IV) with labs as features; ordinal/cost-sensitive loss; calibration; SHAP in the UI.
5. Pharmacogenomic layer (CPIC guidelines: CYP2C9/VKORC1 for Warfarin, CYP2C19 for Clopidogrel, SLCO1B1 for Simvastatin) — which would finally justify the name "GeneRx".
6. Governance: audit log, versioned rules with citations, regulatory classification (software as a medical device), privacy (no PHI stored today).
