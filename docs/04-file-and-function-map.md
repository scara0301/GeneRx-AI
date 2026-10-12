# 04 — File & function map (what each thing calls, and what it returns)

Legend: **Calls →** things this function invokes · **Called by ←** who invokes it · **In** inputs · **Out** return value / side effect.
Line numbers refer to the current files.

## 0. Repo tree

```
GeneRx-AI/
├── README.md                  HF Spaces front-matter + project README
├── LICENSE                    MIT
├── Dockerfile                 image for HF Spaces (trains model at build)
├── requirements.txt           fastapi uvicorn pandas numpy scikit-learn xgboost requests (unpinned)
├── run_app.bat                Windows one-click: install → fetch → build → train → launch
├── .gitignore                 ignores data/, __pycache__, venvs; NOT models/
├── .gitattributes             *.pkl filter=lfs  (left over from the Git-LFS attempt)
├── data/
│   └── drug_catalog.json      LEGACY: 3 drugs with target_genes + tokenised ids. UNUSED.
├── backend/
│   ├── __init__.py            empty → makes `backend` a package (for `uvicorn backend.server:app`, `python -m backend.x`)
│   ├── server.py              FastAPI app: 4 API routes + static serving
│   ├── clinical_engine.py     rule engine + drug catalog + interactions + allergies
│   ├── ml_model.py            DrugRiskPredictor (loads pickle, predicts)
│   ├── utils.py               MedDRA severity weights + get_reaction_severity
│   ├── fetch_faers.py         openFDA → faers_events.csv
│   ├── fetch_sider.py         SIDER → side_effects.csv, side_effect_frequencies.csv
│   ├── build_dataset.py       merge + feature engineering → training_data.csv
│   ├── train_model.py         tune+train XGBoost on real data → pkl
│   └── build_model.py         synthetic data + train → pkl  (used by Dockerfile)
└── frontend/
    ├── index.html             page skeleton
    ├── style.css              dark theme (1,023 lines)
    └── app.js                 SPA logic (713 lines)
```

---

## 1. `backend/server.py` — the web API

Module-level code
| What | Effect |
|------|--------|
| `sys.path.insert(0, dirname(__file__))` (L3) | lets `from clinical_engine import …` work whether started as `uvicorn backend.server:app` or `python backend/server.py` |
| `app = FastAPI(title, description, version="2.0.0")` (L15) | creates the ASGI app; docs at `/docs` |
| `add_middleware(CORSMiddleware, allow_origins=["*"], …)` (L21) | permits cross-origin calls (UI on :3000 → API on :8000) |
| `app.mount("/static", StaticFiles("frontend"))` (L136) | serves frontend files under `/static/…` |

Pydantic models
| Model | Fields (defaults) |
|-------|-------------------|
| `PatientProfile` L30 | name="Patient", age=50, sex="F", weight_kg=70.0, conditions=[], egfr=90.0, alt=25.0, ast=25.0, hba1c=5.5, systolic_bp=120, diastolic_bp=80, ldl=3.0, inr=1.0, current_meds=[], allergies=[] |
| `AssessmentRequest` L48 | patient: PatientProfile, drugs: List[str] |
| `InteractionRequest` L53 | drugs: List[str] |

Route functions
| Function | HTTP | Calls → | Called by ← | In | Out |
|----------|------|---------|-------------|----|-----|
| `health_check()` L57 | GET `/api/health` | `get_predictor()`, `len(DRUG_CATALOG)` | `app.js:checkApiStatus` | – | `{"status":"ok","ml_model_loaded":bool,"drugs_available":10}` |
| `get_drugs()` L67 | GET `/api/drugs` | iterates `DRUG_CATALOG` | `app.js:loadDrugCatalog` | – | `{"drugs":[{name,category,description}…]}` |
| `assess_drugs(request)` L76 | POST `/api/assess` | `get_predictor()`, `evaluate_drug()`, `predictor.predict()`, `check_drug_interactions()` | `app.js:runAssessment`, `runPatientAssessment` | `AssessmentRequest` | `{patient_name, assessments:[{drug_name, suitability, risk_level, reasons, warnings, dose_notes, monitoring, side_effects, ml_prediction}], interactions:[…]}` |
| `check_interactions(request)` L131 | POST `/api/interactions` | `check_drug_interactions()` | (not used by the UI; API only) | `InteractionRequest` | `{"interactions":[{drug_a,drug_b,severity,message}]}` |
| `read_index()` L138 | GET `/` | `FileResponse` | browser | – | `frontend/index.html` |
| `read_file(file_name)` L142 | GET `/{file_name}` | `os.path.exists`, `FileResponse` | browser (`style.css`, `app.js`) | one path segment | the file, else `index.html` |
| `__main__` block L150 | – | `uvicorn.run(app, 0.0.0.0, PORT or 7860)` | `python backend/server.py` | env `PORT` | runs server |

Details inside `assess_drugs`: builds `patient_dict` (adds `patient_id="API"`, **`bmi = weight/1.70²`** — fixed 1.70 m height since height is not collected); ML gets only `drug_name, age, sex, weight, num_concomitant_drugs=len(current_meds)`; interactions are checked on `request.drugs + patient.current_meds` (so candidate-vs-current *and* candidate-vs-candidate *and* current-vs-current pairs are all examined).

---

## 2. `backend/clinical_engine.py` — rules engine (545 lines)

Data structures (module constants)
| Name | Shape | Used by |
|------|-------|---------|
| `DRUG_CATALOG` L13 | `{Drug: {category, class, description, mechanism, monitoring[], side_effects{name: probability}}}` × 10 drugs | `evaluate_drug` (monitoring, side_effects, category, description), `server.get_drugs`, `server.health_check` |
| `DRUG_DRUG_INTERACTIONS` L155 | `{(DrugA, DrugB): {severity, message}}` × 8 pairs (CRITICAL/HIGH/MODERATE/LOW) | `check_drug_interactions` |
| `ALLERGY_CONTRAINDICATIONS` L191 | `{allergy category: [drugs]}` (6 categories; Sulfa → `[]`) | `evaluate_drug` |

Functions
| Function | Calls → | Called by ← | In | Out |
|----------|---------|-------------|----|-----|
| `evaluate_drug(patient, drug_name)` L202 | `compute_risk_level` | `server.assess_drugs` | patient dict (age, sex, bmi, conditions, egfr, alt, ast, hba1c, systolic_bp, ldl, inr, allergies…), drug name (exact catalog key) | `{drug_name, suitability, risk_level, reasons[], dose_notes[], side_effects{}, monitoring[], warnings[], category, description}` — raises `KeyError` for unknown drug |
| `compute_risk_level(suitability)` L431 | – | `evaluate_drug` | "Suitable"/"Caution"/"Avoid"/"Contraindicated" | "Low"/"Moderate"/"High"/"Critical" (else "Unknown") |
| `check_drug_interactions(drug_list)` L441 | – | `server.assess_drugs`, `server.check_interactions` | list of drug names | list of `{drug_a, drug_b, severity, message}` (every unordered pair, looks up (a,b) then (b,a)) |
| `generate_patient_summary(patient, results, interactions)` L465 | – | **nobody (dead code)** | patient dict, results, interactions | multi-line plain-text report string |
| `simulate_response_over_time(suitability, steps=12)` L522 | `numpy.random.normal` | **nobody (dead code)** | suitability | list of floats (13 points) drifting to a target (0.80/0.55/0.25/0.10) with noise |

`evaluate_drug` internal flow: read patient fields (with defaults) → universal age/BMI notes → big `if/elif` on drug name (per-drug rules, see [09](09-clinical-rules-engine.md)) → global allergy loop → `compute_risk_level` → assemble dict.

---

## 3. `backend/ml_model.py` — inference wrapper (84 lines)

| Item | Calls → | Called by ← | In | Out |
|------|---------|-------------|----|-----|
| `MODEL_DIR` L5 | – | `_load_model` | – | `<repo>/models` |
| `class DrugRiskPredictor` L8 | – | `get_predictor` | – | object with `.model, .label_encoder, .feature_cols, .risk_labels, .loaded` |
| `__init__()` L10 | `_load_model` | `get_predictor` | – | sets defaults, `loaded=False` |
| `_load_model()` L18 | `os.path.exists`, `pickle.load` | `__init__` | – | fills attributes from the pickle dict; prints status; on any failure leaves `loaded=False` (rules-only mode) |
| `predict(drug_name, patient_age=50, patient_sex="F", patient_weight=70, reaction_count=0, num_concomitant_drugs=0, known_side_effects=0, mean_se_freq=0, max_se_freq=0)` L36 | `label_encoder.transform`, `model.predict`, `model.predict_proba` | `server.assess_drugs` | patient basics + drug | `{score, category, label, confidence, probabilities, available}`; unavailable cases return `available:false` with a message in `label` |
| `get_predictor()` L80 | `DrugRiskPredictor()` | `server.health_check`, `server.assess_drugs` | – | **singleton** (global `_predictor`) so the pickle is loaded once |

Output example (verified):
```json
{"score":0.0,"category":0,"label":"Low Risk","confidence":97.2,
 "probabilities":{"Low Risk":97.2,"Moderate":2.7,"High Risk":0.0,"Critical":0.0},"available":true}
```
(`score = category/3`; `confidence = max(prob)*100`. Call: `predict("Metformin", 72, "M", 80, 0, 1)`.)
Unknown drug example (verified): `predict("Aspirin", …)` → `{"score":null,"category":null,"label":"'Aspirin' not in training data","confidence":0,"available":false}`.

---

## 4. `backend/utils.py` — severity scoring (65 lines)

| Item | Calls → | Called by ← | In | Out |
|------|---------|-------------|----|-----|
| `SEVERITY_WEIGHTS` L1 | – | `get_reaction_severity` | – | dict of ~40 MedDRA terms → weight 1–10 (cardiac arrest 10, SJS 10, anaphylactic shock 10, … nausea 1, headache 1) |
| `get_reaction_severity(reactions)` L45 | `str.lower/strip` | `build_dataset.build_training_data` (via inner `compute_severity`) | list of reaction-term strings | float: `0.6·max_weight + 0.4·avg_weight`; 0.0 for empty list |

---

## 5. `backend/fetch_faers.py` — openFDA downloader (156 lines)

| Item | Calls → | Called by ← | In | Out |
|------|---------|-------------|----|-----|
| `BASE_URL`, `OUTPUT_DIR`, `TARGET_DRUGS` (10 generic names) | – | – | – | constants |
| `fetch_drug_events(drug_name, max_records=2000)` L23 | `requests.get`, `time.sleep` | `main` | drug (lowercase generic), record cap | `list[dict]`, one dict per adverse-event report (15 keys: drug_name, patient_age, patient_sex, patient_weight, reactions "A\|B\|C", reaction_count, serious, death, hospitalization, disability, life_threatening, drug_role, indication, reaction_outcomes, num_concomitant_drugs). Pages 100 at a time via `skip`, retries on errors/non-200, stops on 404/empty |
| `main()` L131 | `fetch_drug_events` (max_records=5000 each), `pd.DataFrame`, `to_csv` | `__main__` | – | writes `data/faers/faers_events.csv`; prints per-drug counts; returns DataFrame |

---

## 6. `backend/fetch_sider.py` — SIDER downloader (167 lines)

| Function | Calls → | Called by ← | In | Out |
|----------|---------|-------------|----|-----|
| `download_file(filename)` L27 | `requests.get`, `gzip.decompress` | `main` | file name under `http://sideeffects.embl.de/media/download/` | decoded text or `None` on error |
| `parse_drug_names(content)` L42 | – | `main` | TSV text of `drug_names.tsv` | `{compound_id: lowercase drug name}` |
| `parse_side_effects(content, name_map)` L54 | – | `main` | TSV of `meddra_all_se.tsv` | list of `{drug_name, side_effect, compound_id}` (filtered to target drugs) |
| `parse_frequencies(content, name_map)` L77 | – | `main` | TSV of `meddra_freq.tsv` | list of `{drug_name, side_effect, frequency_description, frequency_pct}` (numeric % parsed from "x%" or the lower-bound column ×100) |
| `main()` L111 | all four above, pandas | `__main__` | – | writes `data/sider/side_effects.csv` (deduped) and `data/sider/side_effect_frequencies.csv` |

---

## 7. `backend/build_dataset.py` — merge & feature engineering (221 lines)

| Function | Calls → | Called by ← | In | Out |
|----------|---------|-------------|----|-----|
| `load_faers()` L15 | `pd.read_csv` | `main` | – | FAERS DataFrame or `None` |
| `load_sider()` L27 | `pd.read_csv` | `main` | – | `(df_side_effects, df_frequencies)` (either may be `None`) |
| `compute_drug_risk_profile(faers_df)` L50 | `groupby().agg` | `main` | FAERS df | per-drug stats (total_events, mean_reactions, serious_rate, death_rate, hospital_rate, lt_rate, mean_age, mean_concomitant) — **only saved/printed; not used as model features** |
| `compute_sider_features(df_se, df_freq)` L72 | `groupby` | `main` | the two SIDER dfs | `{drug: {known_side_effects_count, mean_se_frequency, max_se_frequency}}` |
| `build_training_data(faers_df, sider_features)` L101 | `get_reaction_severity` (utils) | `main` | FAERS df + SIDER dict | DataFrame with 11 features + `risk_category` (see [07](07-data-and-feature-engineering.md)) |
| inner `compute_severity(react_str)` L123 | `get_reaction_severity` | `.apply` in `build_training_data` | "A\|B\|C" string | float (1.0 if empty) |
| inner `categorize_risk(score)` L154 | – | `.apply` | risk_score | 0/1/2/3 |
| `main()` L181 | all above | `__main__` | – | writes `data/training_data.csv` and `data/drug_risk_profiles.csv`; prints distribution & missing values |

---

## 8. `backend/train_model.py` — real-data training (126 lines)

| Function | Calls → | Called by ← | In | Out |
|----------|---------|-------------|----|-----|
| `train()` L26 | `LabelEncoder`, `train_test_split`, `RandomizedSearchCV(XGBClassifier)` (or `GradientBoostingClassifier` fallback), `accuracy_score`, `classification_report`, `pickle.dump` | `__main__` | reads `data/training_data.csv` | writes `models/drug_risk_model.pkl` = `{model, label_encoder, feature_cols, risk_labels, accuracy}`; returns `(model, le_drug)` |

---

## 9. `backend/build_model.py` — synthetic training for deployment (172 lines)

| Item | Calls → | Called by ← | In | Out |
|------|---------|-------------|----|-----|
| `DRUG_PROFILES` L21 | – | `generate_dataset` | – | `{drug: (base_risk 0–3, se_count, se_mean_freq, se_max_freq)}` for 10 drugs |
| `generate_dataset()` L42 | `random`, `np.random` | `train` | – | `(rows, drug_idx)` — 12,000 rows × 12 values |
| `train()` L90 | `generate_dataset`, `train_test_split`, `XGBClassifier` (fixed params) / `GradientBoostingClassifier` fallback, `accuracy_score`, `pickle.dump` | `__main__`, **Dockerfile `RUN`** | – | writes `models/drug_risk_model.pkl`; prints accuracy (0.9137) |

---

## 10. `frontend/app.js` — every function (713 lines)

Globals: `basePath`, `API_BASE` (L8–11), `selectedDrugs` (Set), `drugCatalog` (array), `patientWizardStep`, `patientData`, `WIZARD_STEPS`, `PLAIN_CONDITIONS`. (`currentMode` is assigned in `switchMode` without a declaration → implicit global.)

| Function | Calls → | Called by ← | Does | Output / effect |
|----------|---------|-------------|------|-----------------|
| `DOMContentLoaded` handler L19 | `lucide.createIcons`, `checkApiStatus`, `loadDrugCatalog`, `setupEventListeners` | browser | boot | – |
| `setupEventListeners()` L26 | `switchMode`, `runAssessment`, `resetForm` | boot | wires Clinician/Patient/Run/Clear buttons | – |
| `checkApiStatus()` L40 | `fetch /api/health` (5 s timeout) | boot | sets status dot + text: "ML Model Active" / "Rules Engine Only" / "API Offline" | DOM |
| `loadDrugCatalog()` L56 | `fetch /api/drugs`, `renderDrugGrid`, `renderCurrentMedsCheckboxes` | boot | fills `drugCatalog`; hard-coded 10-drug fallback if API is down | state + DOM |
| `switchMode(mode)` L84 | `renderPatientWizard` | click handlers | shows/hides sidebar, drug card, results, wizard | DOM |
| `renderDrugGrid()` L111 | `updateRunButton` | `loadDrugCatalog`, `toggleDrug`, `resetForm` | draws drug cards with `selected` class | DOM |
| `toggleDrug(name)` L123 | `renderDrugGrid` | inline `onclick` | add/remove from `selectedDrugs` | state |
| `updateRunButton()` L132 | – | `renderDrugGrid` | disables Run button when nothing selected | DOM |
| `renderCurrentMedsCheckboxes()` L137 | – | `loadDrugCatalog` | checkbox per catalog drug | DOM |
| `collectPatientProfile()` L149 | DOM reads | `runAssessment` | builds the JSON `patient` object (defaults if blank; `diastolic_bp = 0.65·SBP`; **`ast` copied from the ALT input**) | patient object |
| `runAssessment()` L184 | `collectPatientProfile`, `fetch POST /api/assess`, `renderResults` | Run button | shows "Analyzing…", handles errors with a "Connection Error" card | DOM |
| `renderResults(data)` L222 | `renderSummaryTable`, `renderInteractions`, `renderDetailCards`, `renderMLChart` | `runAssessment` | un-hides results card, fills header, scrolls into view | DOM |
| `riskClass(suitability)` L250 | – | several | map to CSS class safe/caution/avoid/critical | string |
| `renderSummaryTable(a)` L260 | `riskClass` | `renderResults` | table: Drug, Suitability badge, Risk level, ML confidence, Key concern (first reason) | DOM |
| `renderInteractions(list)` L294 | – | `renderResults` | DDI banner or hide | DOM |
| `renderDetailCards(a)` L312 | `riskClass`, `lucide.createIcons` | `renderResults` | expandable card per drug: Reasoning, Dose Notes, Warnings, Side Effects tags, Monitoring | DOM |
| `toggleDetail(i)` L377 | – | inline `onclick` | expand/collapse | DOM |
| `renderMLChart(a)` L382 | `new Chart` | `renderResults` | stacked bar of class probabilities per drug (destroys the previous chart) | canvas |
| `renderPatientWizard()` L477 | `lucide.createIcons`, `runPatientAssessment` (step 6) | `switchMode`, `wizardNext/Back/Restart` | renders progress dots + current step + nav buttons | DOM |
| `wizardSaveStep()` L574 | – | `wizardNext`, `wizardBack` | reads current step inputs into `patientData` | state |
| `wizardNext()` L603 | `wizardSaveStep`, `renderPatientWizard` | button | validates ≥1 drug at step 5; advance | – |
| `wizardBack()` L613 | same | button | go back | – |
| `wizardRestart()` L619 | `renderPatientWizard` | button | reset state | – |
| `runPatientAssessment()` L625 | `fetch POST /api/assess`, `renderPatientResults` | `renderPatientWizard` (step 6) | builds patient object (ALT/AST = 25, INR = 1.0, allergies = [] fixed) | DOM |
| `renderPatientResults(data)` L661 | – | `runPatientAssessment` | plain-language cards (Good Match / Use With Caution / Not Recommended / Do Not Use), first reason, interactions | DOM |
| `resetForm()` L700 | `renderDrugGrid` | Clear button | resets the form to defaults, clears selection, hides results | DOM |

## 11. `frontend/index.html` / `style.css`

* `index.html` — loads Inter font (Google Fonts), Lucide (`unpkg`) and Chart.js 4 (`jsdelivr`) from CDNs, then `style.css` and `app.js`. Regions: `<nav class="topbar">` (brand, mode toggle, API status), `<aside id="sidebar">` (patient form), `<section id="contentArea">` with three cards: `#drugSelectionCard`, `#resultsCard`, `#patientWizard`; footer disclaimer.
* `style.css` — sections: variables/reset, scrollbar, topbar, layout, sidebar/form, content, drug grid, primary button, results table, risk badges, DDI alerts, detail cards, ML section, patient wizard, patient result cards, disclaimer, animations, responsive.

## 12. The complete call graph in one picture

```
app.js ──► GET /api/health ───────────► server.health_check ─► ml_model.get_predictor, len(DRUG_CATALOG)
app.js ──► GET /api/drugs  ───────────► server.get_drugs    ─► DRUG_CATALOG
app.js ──► POST /api/assess ──────────► server.assess_drugs
                                           ├─► clinical_engine.evaluate_drug ─► compute_risk_level
                                           ├─► ml_model.DrugRiskPredictor.predict ─► LabelEncoder / XGBoost
                                           └─► clinical_engine.check_drug_interactions
(API only) POST /api/interactions ────► server.check_interactions ─► clinical_engine.check_drug_interactions

OFFLINE:
fetch_faers.main ─► fetch_drug_events ─► requests ─► data/faers/faers_events.csv
fetch_sider.main ─► download_file ─► parse_drug_names / parse_side_effects / parse_frequencies ─► data/sider/*.csv
build_dataset.main ─► load_faers, load_sider, compute_drug_risk_profile, compute_sider_features,
                      build_training_data ─► utils.get_reaction_severity ─► data/training_data.csv
train_model.train ─► reads training_data.csv ─► RandomizedSearchCV(XGB) ─► models/drug_risk_model.pkl
build_model.train ─► generate_dataset ─► XGB.fit ─► models/drug_risk_model.pkl   (Docker)
```

## 13. Dead / unused code (be ready to say so)

| Item | Status |
|------|--------|
| `clinical_engine.generate_patient_summary` | never called (CLI-report helper from an earlier version) |
| `clinical_engine.simulate_response_over_time` | never called (planned response-curve chart) |
| `server./api/interactions` | works, but the UI doesn't call it |
| `build_dataset.compute_drug_risk_profile` | output saved to CSV only |
| `fetch_sider.DRUG_NAME_MAP` | empty module constant, never used |
| `data/drug_catalog.json` | legacy gene-based catalog; nothing reads it |
| `.gitattributes` `*.pkl filter=lfs` | leftover from the LFS attempt (the pkl isn't tracked now) |
| `diastolic_bp`, `sex` (patient fields) | collected/passed, never used by any rule (`sex` is only used by the ML encoder) |
