# 03 — Architecture

(Mermaid diagrams render on GitHub / most Markdown viewers. ASCII versions are included so you can
redraw them on a whiteboard.)

## 1. System context — the 30,000-ft view

```mermaid
flowchart LR
    U[User<br/>Clinician or Patient] -->|browser| FE[Frontend SPA<br/>index.html + app.js + style.css]
    FE -->|JSON over HTTP| API[FastAPI backend<br/>server.py]
    API --> RULES[Clinical Rules Engine<br/>clinical_engine.py]
    API --> ML[ML Predictor<br/>ml_model.py]
    ML -->|loads| PKL[(models/drug_risk_model.pkl)]
    subgraph OFF[Offline training]
        FAERS[(openFDA FAERS API)] --> F1[fetch_faers.py]
        SIDER[(SIDER 4.1 files)] --> F2[fetch_sider.py]
        F1 --> BD[build_dataset.py]
        F2 --> BD
        U2[utils.py<br/>severity score] --> BD
        BD --> TD[(data/training_data.csv)]
        TD --> TR[train_model.py]
        SYN[build_model.py<br/>synthetic generator] -.->|used in Docker build| PKL
        TR --> PKL
    end
```

```
                         ┌──────────────────────────── ONLINE (runtime) ────────────────────────────┐
  Browser  ──HTTP/JSON──►│ FastAPI (server.py)                                                      │
 (index.html,            │   ├─ /api/health  /api/drugs  /api/assess  /api/interactions             │
  app.js, style.css)     │   ├─ clinical_engine.py  → verdict, reasons, dose notes, DDI  (RULES)   │
                         │   └─ ml_model.py         → risk class + probabilities          (XGBoost) │
                         │         └─ loads models/drug_risk_model.pkl at first request             │
                         └──────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────── OFFLINE (training) ──────────────────────────┐
  openFDA ─► fetch_faers ─┐                                                                         │
                          ├─► build_dataset (+utils.get_reaction_severity) ─► training_data.csv      │
  SIDER ──► fetch_sider ──┘                                       └──► train_model ─► drug_risk_model.pkl
  (Docker/HF path:  build_model.py ── synthetic data ──────────────────────────────► drug_risk_model.pkl)
                         └──────────────────────────────────────────────────────────────────────────┘
```

**Key architectural properties**
* **Stateless API** — nothing is stored; every `/api/assess` request carries the whole patient profile. No DB, no auth, no PHI persisted (good for privacy; also why it's a demo).
* **Single deployable** — FastAPI also serves the static frontend, so one container/one port.
* **Decoupled training** — the web app never trains; it only loads a pickled artifact → fast start, simple.
* **Two interchangeable training paths** that emit the *same artifact schema* (`model, label_encoder, feature_cols, risk_labels, accuracy`).
* **Fail-soft ML** — no model / unknown drug / prediction error ⇒ `{"available": false, ...}`; the rules result is still returned.

## 2. Backend components (module dependency graph)

```mermaid
flowchart TD
    server[server.py<br/>FastAPI app] --> ce[clinical_engine.py]
    server --> mm[ml_model.py]
    bd[build_dataset.py] --> ut[utils.py]
    tm[train_model.py]
    bm[build_model.py]
    ff[fetch_faers.py]
    fs[fetch_sider.py]
    mm -. loads artifact from .-> pkl[(models/*.pkl)]
    tm -. writes .-> pkl
    bm -. writes .-> pkl
    bd -. reads .-> faers[(data/faers/*.csv)]
    bd -. reads .-> sider[(data/sider/*.csv)]
    ff -. writes .-> faers
    fs -. writes .-> sider
    bd -. writes .-> td[(data/training_data.csv)]
    tm -. reads .-> td
```

Only **two** module-to-module imports exist at runtime: `server → clinical_engine`, `server → ml_model`.
`build_dataset → utils` is the only import in the pipeline (done with a `try/except ImportError` so it works as
`python -m backend.build_dataset` *and* `python backend/build_dataset.py`).

## 3. Request sequence — `POST /api/assess` (the core flow)

```mermaid
sequenceDiagram
    participant B as Browser (app.js)
    participant S as server.assess_drugs
    participant C as clinical_engine
    participant M as ml_model.DrugRiskPredictor
    B->>S: POST /api/assess {patient, drugs[]}
    Note over S: Pydantic validates → AssessmentRequest
    S->>S: build patient_dict (adds bmi = weight/1.70², patient_id="API")
    loop for each drug in drugs
        S->>C: evaluate_drug(patient_dict, drug)
        C-->>S: {suitability, risk_level, reasons, warnings, dose_notes, monitoring, side_effects,...}
        S->>M: predict(drug, age, sex, weight, num_concomitant_drugs=len(current_meds))
        M-->>S: {score, category, label, confidence, probabilities, available}
    end
    S->>C: check_drug_interactions(drugs + current_meds)
    C-->>S: [{drug_a, drug_b, severity, message}]
    S-->>B: {patient_name, assessments[], interactions[]}
    B->>B: renderResults → table, DDI banner, detail cards, Chart.js stacked bars
```

ASCII call tree for the same request:

```
POST /api/assess
└─ server.assess_drugs(request)
   ├─ get_predictor()                         → singleton DrugRiskPredictor (loads pickle once)
   ├─ for each drug:
   │   ├─ clinical_engine.evaluate_drug()
   │   │    └─ compute_risk_level()
   │   └─ DrugRiskPredictor.predict()
   │        ├─ label_encoder.transform([drug])
   │        ├─ model.predict(features)
   │        └─ model.predict_proba(features)
   └─ clinical_engine.check_drug_interactions()
```

## 4. Training / data pipeline

```mermaid
flowchart LR
    A[openFDA /drug/event.json<br/>10 drugs × up to 5000 reports] -->|fetch_drug_events| B[faers_events.csv<br/>15 columns]
    C[SIDER drug_names.tsv<br/>meddra_all_se.tsv.gz<br/>meddra_freq.tsv.gz] -->|parse_*| D[side_effects.csv<br/>side_effect_frequencies.csv]
    B --> E[build_training_data]
    D -->|compute_sider_features| E
    U[get_reaction_severity] --> E
    E -->|clean ages/weights, impute medians,<br/>encode sex, severity score,<br/>primary-suspect flag, SIDER joins,<br/>risk_score → risk_category| F[training_data.csv<br/>11 features + target]
    F --> G[train_model.py<br/>LabelEncoder drug,<br/>80/20 stratified split,<br/>RandomizedSearchCV XGBoost]
    G --> H[(drug_risk_model.pkl)]
```

## 5. ML inference path (what happens inside `predict`)

```
drug_name "Paracetamol" ──lowercase/strip──► "paracetamol" ──alias──► "acetaminophen"
   └─ label_encoder.transform → integer id (unknown → return available:false)
patient_sex "F"/"FEMALE" → 1 else 0
features = [drug_id, age, sex, weight, reaction_count=0, weighted_reaction_score=1.0,
            is_primary_suspect=1, num_concomitant_drugs, known_se=0, mean_se_freq=0, max_se_freq=0]
XGBoost.predict → class 0..3        XGBoost.predict_proba → 4 probabilities
return {score=class/3, category, label, confidence=max(prob)*100, probabilities{label:%}, available:true}
```
The 6 hard-coded values are the train/serve-skew issue (see [10](10-known-issues-and-defence.md)).

## 6. Frontend architecture

```
index.html  (structure: topbar, sidebar form, content cards, wizard container)
   ├─ style.css   (dark glassmorphism theme, CSS variables, responsive rules)
   └─ app.js      (state + rendering; no framework)
        state:   selectedDrugs:Set, drugCatalog:[], patientWizardStep:int, patientData:{}
        flow:    DOMContentLoaded → checkApiStatus + loadDrugCatalog + setupEventListeners
        doctor:  collectPatientProfile → runAssessment → renderResults →
                   renderSummaryTable / renderInteractions / renderDetailCards / renderMLChart
        patient: renderPatientWizard (6 steps) → wizardNext/Back/SaveStep →
                   runPatientAssessment → renderPatientResults
```
Rendering approach: build HTML strings with template literals and assign to `innerHTML`
(simple; no virtual DOM). Trade-off: no automatic escaping — safe here because all strings come from our own
backend, but worth mentioning (XSS awareness) if drug names ever became user-supplied.

## 7. Deployment architecture

```mermaid
flowchart LR
    GH[(GitHub repo)] -->|push| HF[Hugging Face Space<br/>sdk: docker]
    HF -->|docker build| IMG[Image: python:3.10-slim<br/>pip install → build_model.py → pkl baked in]
    IMG -->|run| CT[Container :7860<br/>uvicorn backend.server:app]
    User((Browser)) -->|HTTPS iframe /embed/user/space| CT
```

* Model baked **into the image** → no runtime downloads, cold start is fast, reproducible (`random_state=42`).
* One process serves `/` (index.html), `/{file}` (css/js), `/static/*`, and `/api/*`.
* Catch-all `GET /{file_name}`: returns the file if it exists in `frontend/`, otherwise falls back to `index.html` (SPA-style fallback).

## 8. Design decisions worth quoting

| Decision | Alternative | Why this one |
|----------|-------------|--------------|
| Hybrid rules + ML | pure ML / pure rules | explainability + safety + data-driven signal |
| Stateless REST | sessions / DB | no PHI stored, trivial to scale/deploy |
| Pickled artifact dict | MLflow / ONNX | simplest thing that works for one model; carries encoder + feature list + labels together |
| Synthetic-data trainer for deploy | commit pkl / LFS / download at runtime | HF rejected binaries; no network needed at build; deterministic |
| Vanilla JS | React/Vue | no build step; small app; one-container deploy |
| CORS `*` | restricted origins | dev convenience (two ports); acceptable for a public demo with no credentials — tighten for production |
