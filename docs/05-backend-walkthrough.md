# 05 — Backend walkthrough (function by function, with the *why*)

Each entry: **What** it does · **Why** it is written that way · **Watch out** (what an interviewer could poke at).

---

## A. `server.py`

### Imports & `sys.path.insert`
* **What:** adds `backend/` to `sys.path`, then `from clinical_engine import …`, `from ml_model import …` (flat imports, not `backend.clinical_engine`).
* **Why:** the Dockerfile starts `uvicorn backend.server:app`, while a developer may run `python backend/server.py`. The path insert makes both work without relative-import gymnastics.
* **Watch out:** it's a pragmatic hack; the cleaner fix is `from .clinical_engine import …` plus always running as a package.

### FastAPI app + CORS
* **What:** `FastAPI(title, description, version="2.0.0")`; `CORSMiddleware(allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`.
* **Why:** during local dev the static UI is on `:3000` and the API on `:8000` → different origins → browsers block responses without CORS headers.
* **Watch out:** wildcard CORS is fine for a public demo with no cookies/auth; in production restrict to known origins. (Browsers also reject `*` together with credentialed requests; here no credentials are used.)

### `PatientProfile`, `AssessmentRequest`, `InteractionRequest` (Pydantic)
* **What:** typed request schemas with sensible defaults (e.g. eGFR 90, ALT 25, HbA1c 5.5, BP 120/80, LDL 3.0, INR 1.0 — all "normal").
* **Why:** automatic validation (wrong type ⇒ HTTP 422), self-documenting OpenAPI, and defaults mean a clinician can omit values that are unknown; defaults are *normal* values so they don't trigger false alarms.
* **Watch out:** no range validation (e.g. `age=-5` is accepted). Could add `Field(ge=0, le=120)`. `conditions`, `drugs`, `allergies` are free strings and matched with exact/lower-case comparison.

### `health_check()` — `GET /api/health`
* **What:** returns status, whether the ML model loaded, and catalog size.
* **Why:** health probe for the container/hosting platform **and** the UI's status badge ("ML Model Active" vs "Rules Engine Only").

### `get_drugs()` — `GET /api/drugs`
* **What:** list comprehension over `DRUG_CATALOG` → `{name, category, description}`.
* **Why:** the UI builds its drug grid from the server so the front-end and back-end never disagree about supported drugs (single source of truth).

### `assess_drugs()` — `POST /api/assess` ⭐ (the core)
Step by step:
1. `patient = request.patient`; `predictor = get_predictor()` (cached singleton).
2. Build a plain `patient_dict` for the rules engine. It adds `patient_id="API"` and `bmi = weight / 1.70²`.
   *Why BMI with a fixed height?* The form doesn't collect height; BMI is only used for one soft note (BMI > 30 ⇒ "verify weight-based dosing"). **Watch out:** this will mislabel very tall/short patients; better to collect height or drop the rule.
3. For each requested drug: `evaluate_drug(patient_dict, drug_name)` for the verdict and `predictor.predict(...)` for the ML signal. The ML call receives only age, sex, weight and `num_concomitant_drugs = len(current_meds)` (a count of the patient's current drugs — a stand-in for polypharmacy).
4. Append a merged dict (rules fields + `ml_prediction`).
5. `check_drug_interactions(request.drugs + patient.current_meds)` — checks the candidate drugs against each other *and* against current meds.
* **Why one endpoint does everything:** one network round-trip per assessment; the UI needs everything together.
* **Watch out:** (a) unknown drug ⇒ `KeyError` ⇒ 500; (b) duplicates (a drug in both lists) are paired with itself – no rule exists so nothing happens; (c) ML ignores the patient's labs entirely; (d) output includes the ML result *beside* but never *inside* the verdict.

### `check_interactions()` — `POST /api/interactions`
Thin wrapper over `check_drug_interactions`. Handy for tests / other clients; the UI doesn't use it.

### Static mounting + `read_index` + `read_file`
* **What:** `/static` mount for `frontend/`; `/` returns `index.html`; `/{file_name}` returns the file if it exists else `index.html`.
* **Why:** one process serves UI + API → one Docker container, no separate web server, no CORS issue in production.
* **Watch out:** route order matters — `/api/*` routes are registered before the catch-all `/{file_name}` so they win. `file_name` is a single path segment (FastAPI doesn't match `/` in it), so classic `../` traversal doesn't work, but a cleaner approach is to rely on `StaticFiles(html=True)` only.

### `if __name__ == "__main__"`
`uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))` — PORT env var is the PaaS convention; 7860 is Hugging Face's default.

---

## B. `clinical_engine.py`

### Header comment
"Demographics, conditions, labs, current meds, allergies. **No genetic data required.**" → this is the explicit design pivot away from pharmacogenomics.

### `DRUG_CATALOG`
* **What:** 10 drugs × (category, pharmacological class, plain-English description, mechanism, monitoring list, side-effects with probabilities).
* **Why:** the knowledge base for explanations; it is also the **single source of truth for what the app supports**. Side-effect probabilities are approximate label-derived frequencies (e.g. Metformin nausea 0.30, Warfarin bleeding 0.35) used for display tags only.
* **Watch out:** values are curated by hand, not pulled from a formal source – say "approximate, illustrative".

### `DRUG_DRUG_INTERACTIONS`
* **What:** 8 hand-written pairs with severities: Warfarin+Ibuprofen CRITICAL; Ramipril+Ibuprofen HIGH; Warfarin+Amoxicillin / Warfarin+Omeprazole / Metformin+Ibuprofen / Metoprolol+Ibuprofen MODERATE; Ramipril+Metoprolol / Amlodipine+Metoprolol LOW.
* **Why a dict keyed by tuple:** O(1) lookup; the checker tries both orders so you only store each pair once.
* **Watch out:** coverage is tiny (10 drugs have 45 possible pairs; 8 are encoded). Real systems use DrugBank/RxNorm/Lexicomp. Also, tuples are *ordered* so you must remember to check the reverse (the code does).

### `ALLERGY_CONTRAINDICATIONS`
Maps an allergy *category* (Penicillin, NSAIDs, ACE Inhibitors, Statins, Beta Blockers, Sulfa) to the catalog drugs in that class. "Sulfa drugs": `[]` because none of the 10 drugs is a sulfonamide. The UI's allergy checkboxes (penicillin / nsaids / sulfa drugs / ace inhibitors / statins) match these keys after lower-casing.

### `evaluate_drug(patient, drug_name)` ⭐
1. **Extract + normalise** inputs with defaults: lower-case `conditions` and `allergies` so matching is case-insensitive.
2. **Universal modifiers:** age ≥ 65 ⇒ dose note ("lower starting doses"); age ≥ 80 ⇒ warning; BMI > 30 ⇒ dose note.
3. **Per-drug `if/elif`** – each branch sets `suitability` (and appends reasons/warnings/dose notes). Common pattern: *severity ladder* — a hard contraindication (e.g. eGFR < 30) sets "Contraindicated"; a softer threshold (30–44) sets "Caution"; otherwise a reassuring reason is appended. Indication checks ("Indicated: …") add positive reasons or a warning if no indication is evident.
4. **Allergy loop (applies to all drugs):** if the drug is in a contraindicated list for an allergy the patient has ⇒ force `Contraindicated` and insert an `⚠️ ALLERGY ALERT` reason at index 0 (so it shows first).
5. `compute_risk_level(suitability)` and return the assembled dict (plus catalog data: side_effects, monitoring, category, description).
* **Why rules-first:** auditable, deterministic, each sentence cites the lab value that caused it.
* **Why `if/elif` and not a rule DSL:** only 10 drugs; simplest readable thing. At scale you'd move to a data-driven rule table (YAML/DB) or a standard like CDS Hooks / Arden syntax.
* **Watch out:**
  * *Downgrades can happen silently:* branch order matters; e.g. Ibuprofen uses `"Avoid" if suitability == "Suitable" else suitability` so a stronger verdict is never weakened — but Amlodipine's `suitability = "Avoid"` for low BP overrides an earlier "Caution" for heart failure.
  * *Duplicate message:* a penicillin-allergic patient gets both the Amoxicillin branch's "Contraindicated" reason and the generic allergy alert (verified in output).
  * *Warfarin* never reports "Contraindicated"; only "Caution" (INR > 3.5) or "Avoid" (peptic ulcer/GI bleed). With no AF/DVT/PE listed it stays "Suitable" with empty reasons (seen in my test) — it doesn't say "no indication".
  * *Condition strings must match exactly* (lower-cased): the UI offers "Liver Disease" but the Paracetamol rule looks for "liver failure"/"hepatitis" ⇒ that rule can't fire from the UI.

### `compute_risk_level(suitability)`
A lookup: Suitable→Low, Caution→Moderate, Avoid→High, Contraindicated→Critical, default "Unknown". Kept as a separate function so the mapping is in one place and testable.

### `check_drug_interactions(drug_list)`
Double loop over unordered pairs `(i<j)` → O(n²) with tiny n; for each pair it looks up `(a,b)` or `(b,a)` and appends `{drug_a, drug_b, severity, message}`.

### `generate_patient_summary` (dead code)
Builds a formatted text report (header, labs, per-drug bullets, interactions, disclaimer). Left from the CLI/report era; the UI renders its own HTML.

### `simulate_response_over_time` (dead code)
Creates a noisy exponential-ish approach to a target efficacy (0.80/0.55/0.25/0.10 by suitability) to draw a "therapeutic response" curve; a simulated visual, **not** pharmacokinetics — good thing it isn't wired up.

---

## C. `ml_model.py`

### `DrugRiskPredictor.__init__` / `_load_model`
* **What:** tries `models/drug_risk_model.pkl`; loads the artifact dict `{model, label_encoder, feature_cols, risk_labels, accuracy}`; sets `loaded=True`. Missing file or exception ⇒ prints a message and stays in rules-only mode.
* **Why bundle the encoder and feature list with the model:** the fitted `LabelEncoder` is part of the model — drug→integer mapping must be identical at train and serve time; storing `feature_cols` documents the expected column order.
* **Watch out:** `pickle` executes code on load — only load files you built yourself (security). XGBoost recommends its own `save_model` JSON/UBJ format for cross-version portability; a pickle created under one version may not load under another.

### `predict(...)`
1. Not loaded ⇒ `available:false`.
2. `drug_lower = drug_name.lower().strip()`; **alias** `paracetamol → acetaminophen` (the catalog uses the British name, FAERS/openFDA uses the US generic name).
3. `label_encoder.transform([drug_lower])` — unknown ⇒ `ValueError` ⇒ `available:false, label:"'X' not in training data"`.
4. `sex_encoded = 1 if F/FEMALE else 0` (training also had 0.5 for unknown sex in the real pipeline).
5. Build the 11-element feature row in **training order**: `[drug, age, sex, weight, reaction_count, weighted_reaction_score=1.0, is_primary_suspect=1, num_concomitant_drugs, known_side_effects, mean_se_freq, max_se_frequency]`.
6. `model.predict` → class; `model.predict_proba` → 4 probabilities.
7. Return `score=class/3`, `confidence=max prob ×100` (1 decimal), per-label percentages, `available:true`.
* **Why those hard-coded defaults:** at prediction time we have no adverse event yet, so event-level fields (`reaction_count`, `weighted_reaction_score`, `is_primary_suspect`) must be filled with neutral guesses; the SIDER features default to 0 because the API caller doesn't provide them.
* **Watch out:** this is the **train/serve skew** — the model was trained with real SIDER values per drug, and at serve time they are 0. Shown experimentally: Warfarin flips High→Moderate. A fix: look up each drug's SIDER stats (from a saved table) inside `predict` instead of defaulting to 0.

### `get_predictor()`
Lazy global singleton: the pickle is read once on first use (first `/api/health` or `/api/assess`), not per request, so latency stays low.

---

## D. `utils.py`

### `SEVERITY_WEIGHTS`
~40 MedDRA Preferred Terms grouped by organ system (cardiac, renal/hepatic, haematologic/anaphylaxis, severe skin reactions, neurologic, mild symptoms) with weights 1–10 reflecting clinical gravity (cardiac arrest/anaphylactic shock/SJS/TEN = 10, nausea/headache/fatigue = 1). **Hand-assigned domain knowledge**, i.e. a manual feature-engineering/knowledge injection step.

### `get_reaction_severity(reactions)`
1. Empty list ⇒ 0.0.
2. For each term: lowercase; weight = max over all dictionary terms that appear as a **substring** (default 1.0). So "Acute renal failure" contains "renal failure" ⇒ 9.0.
3. `avg_weight = total/n`.
4. `max_weight = max(SEVERITY_WEIGHTS.get(exact_term, 1.0))` ← **exact match only** (inconsistent with step 2).
5. Return `0.6·max + 0.4·avg`.
* **Why 60/40 max/avg:** a report with one catastrophic reaction should score high even if padded with mild ones (max dominates), but breadth of reactions still raises the score (avg). A reasonable heuristic, not a tuned value.
* **Watch out (real bug, verified):** because `max_weight` uses exact lookup, `["Cardiac arrest during surgery"]` scores 4.6 instead of 10 (step 2 found 10 but step 4 used 1.0). Fix: track the per-term weights in a list and take `max(weights)`.

---

## E. `fetch_faers.py`

### `fetch_drug_events(drug_name, max_records)`
* Query: `search=patient.drug.openfda.generic_name:"<drug>"`, `limit=100`, `skip=n`. Loop until `skip >= max_records`.
* **Robustness:** `404` ⇒ no results, stop; other non-200 ⇒ sleep 2 s and retry; network exception ⇒ sleep 3 s and retry; `time.sleep(0.3)` between pages to stay inside openFDA's rate limit (as I remember the docs: 240 requests/min and 1,000/day per IP without an API key — **unverified**, check before quoting; 10 drugs × 50 pages = 500 calls, so one full run fits).
* **Parsing each report:**
  * *Age*: `patientonsetage` with unit codes → years (802 months ÷12; 803 ÷365; 800 decade ×10; default assumes years/801). **Bug:** per the FDA spec, 803 = *week* (÷52) and 804 = *day* (÷365).
  * *Sex*: 1→M, 2→F, else "Unknown".
  * *Weight*: float or None.
  * *Seriousness flags*: `serious`, `seriousnessdeath`, `seriousnesshospitalization`, `seriousnessdisabling`, `seriousnesslifethreatening` → ints 0/1.
  * *Reactions*: MedDRA preferred terms joined with `|`, count, outcomes.
  * *Drug role*: finds the entry in `patient.drug[]` whose `openfda.generic_name` matches; maps `drugcharacterization` 1/2/3 → Primary Suspect / Secondary Suspect / Concomitant; also captures `drugindication`.
  * *`num_concomitant_drugs = len(drugs) - 1`* (total drugs listed on the report minus the studied one).
* **Watch out:** a 5,000-record cap per drug is a convenience cap (not a random sample — `skip` pagination returns the API's default order); infinite retry loop on persistent errors (no max-retries cap); openFDA limits deep `skip` pagination (I recall 25,000 — unverified), so 5,000 per drug is safe.

### `main()`
Loops over the 10 drugs (5,000 each ⇒ up to 50,000 rows), 1 s pause between drugs, saves CSV, prints counts of serious/death/hospitalization.

---

## F. `fetch_sider.py`

* `download_file` — GET, `raise_for_status`, gunzip if `.gz`; returns text or `None` (error is printed, never raised) so the pipeline can continue partially.
* `parse_drug_names` — `drug_names.tsv` (`CID…<TAB>name`) → `{compound_id: name}`.
* `parse_side_effects` — `meddra_all_se.tsv` columns: `[CID_flat, CID_stereo, UMLS_label, MedDRA_type(LLT/PT), UMLS_meddra, side_effect_name]`; takes `parts[5]`, keeps only target drugs.
* `parse_frequencies` — `meddra_freq.tsv`: parses "x%" text or the lower-bound fraction ×100 into `frequency_pct`.
* `main` — dedupes (drug, side effect) pairs → `side_effects.csv`; frequency records → `side_effect_frequencies.csv`.
* **Why SIDER at all?** FAERS says what happened in *reports*; SIDER says what the *label* lists as known side-effects and how frequent they are → gives a per-drug "side-effect burden" feature that doesn't depend on reporting behaviour.
* **Watch out:** SIDER lists drug-name aliases per compound; matching on a lowercase name only catches exact generic names. The README says "1,934 pairs".

---

## G. `build_dataset.py`

See [07](07-data-and-feature-engineering.md) for the full walkthrough; one-line summaries:
* `load_faers`, `load_sider` — read CSVs, tolerate missing files.
* `compute_drug_risk_profile` — per-drug aggregates (saved for analysis only).
* `compute_sider_features` — per-drug `known_side_effects_count`, `mean_se_frequency`, `max_se_frequency`.
* `build_training_data` — cleaning (age 0–100, weight 10–250 → NaN → median impute), sex encoding, severity score, primary-suspect flag, SIDER joins, target construction.
* `main` — orchestrates and prints diagnostics (class distribution, missing values).

## H. `train_model.py` and `build_model.py`

See [08](08-modelling-and-algorithm-choice.md). Summary:
* `train_model.train` — real data; `LabelEncoder` for drug; stratified 80/20 split; `RandomizedSearchCV` (10 iterations × 3-fold CV over 6 hyper-parameters); evaluates accuracy + per-class report; pickles artifact.
* `build_model.generate_dataset/train` — synthetic data and fixed hyper-parameters (200 trees, depth 5, lr 0.1, subsample/colsample 0.8) for deployment.
