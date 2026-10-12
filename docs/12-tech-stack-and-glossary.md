# 12 — Tech stack (what & why) and glossary

## 1. Every dependency and where it is used

| Library / tool | Version in my test env | Used in | Purpose | Why this one |
|----------------|-----------------------|---------|---------|--------------|
| Python | 3.13 (Docker: 3.10-slim) | all | language | ML ecosystem |
| **fastapi** | latest (unpinned) | `server.py` | REST API, validation, docs | see [01](01-overview-and-why.md) |
| **uvicorn** | latest | Docker `CMD`, `server.py` | ASGI server | standard for FastAPI |
| **pydantic** | via FastAPI | `server.py` | request schemas | validation |
| **starlette** | via FastAPI | CORS, `StaticFiles`, `FileResponse` | web toolkit | – |
| **pandas** | latest | `fetch_*`, `build_dataset`, `train_model` | DataFrames, groupby, CSV | de-facto data wrangling |
| **numpy** | latest | `ml_model`, `build_model`, `train_model`, `clinical_engine.simulate…` | arrays, random | – |
| **scikit-learn** | 1.9.1 | `train_model`, `build_model` | `LabelEncoder`, `train_test_split`, `RandomizedSearchCV`, metrics, `GradientBoostingClassifier` fallback | standard ML toolkit |
| **xgboost** | 3.4.1 | `train_model`, `build_model`, pickle | the classifier | see [08](08-modelling-and-algorithm-choice.md) |
| **requests** | latest | `fetch_faers`, `fetch_sider` | HTTP client | simple & reliable |
| **pickle / gzip / io / os / sys / time / random** | stdlib | various | persistence, decompress, paths, sleeps | – |
| httpx (test only) | – | `fastapi.testclient` | needed by `TestClient` | only for my verification |
| **Chart.js 4** | CDN | `app.js` | stacked bar chart | tiny, canvas-based |
| **Lucide** | CDN (`@latest`) | `index.html`, `app.js` | icons | lightweight SVG icons |
| **Inter** | Google Fonts | CSS | typography | readable UI font |
| **Docker** | – | `Dockerfile` | containerise | HF Spaces `sdk: docker` |
| **Hugging Face Spaces** | – | README front-matter | hosting | free, ML-friendly |
| **Git / GitHub** | – | repo | VCS | – |

## 2. Language / framework concepts used in the code (quick definitions)

| Concept | Where | One-liner |
|---------|-------|-----------|
| ASGI | uvicorn/FastAPI | async successor to WSGI |
| Dependency-free pure functions | `clinical_engine` | easy to unit test |
| Singleton | `get_predictor` | one shared model instance |
| Pydantic `BaseModel` | `PatientProfile` | typed, validated data classes |
| Lazy loading | `_predictor = None` | create on first use |
| Template literals | `app.js` | JS string interpolation for HTML |
| Optional chaining `?.` | `app.js` | safe property access |
| `AbortSignal.timeout` | `checkApiStatus` | fetch timeout |
| CSS custom properties | `style.css` | theme tokens |
| Docker layer cache | `Dockerfile` | copy requirements before source |

## 3. Medical glossary

| Term | Meaning |
|------|---------|
| **ADR / ADE** | adverse drug reaction / event |
| **FAERS** | FDA Adverse Event Reporting System |
| **openFDA** | FDA's public API portal |
| **MedDRA / PT** | standardised medical terminology; Preferred Term |
| **SIDER** | Side Effect Resource built from drug labels |
| **eGFR** | estimated glomerular filtration rate – kidney function (mL/min/1.73 m²). >90 normal, 60–89 mild ↓, 30–59 moderate ↓, 15–29 severe ↓, <15 failure |
| **CKD** | chronic kidney disease |
| **ALT / AST** | liver enzymes (U/L); raised in liver injury; ULN ≈ 30–40 |
| **ULN** | upper limit of normal |
| **HbA1c** | 3-month average blood glucose %; ≥6.5 % diagnostic for diabetes |
| **SBP / DBP** | systolic / diastolic blood pressure |
| **LDL** | "bad" cholesterol (mmol/L) |
| **INR** | clotting time ratio; Warfarin target usually 2–3 |
| **DDI** | drug–drug interaction |
| **Contraindication** | a situation where a drug must not be used |
| **Polypharmacy** | taking many medicines at once (≥5) |
| **Lactic acidosis** | dangerous acid build-up; Metformin rare risk in renal failure |
| **ACE inhibitor** | e.g. Ramipril; lowers BP, protects kidney; can raise potassium, cause cough |
| **Beta-blocker** | e.g. Metoprolol; avoid in asthma/COPD/heart block |
| **Statin** | e.g. Atorvastatin; lowers LDL; liver/muscle effects |
| **NSAID** | e.g. Ibuprofen; GI, renal, cardiovascular risks |
| **PPI** | proton pump inhibitor (Omeprazole) |
| **Vitamin K antagonist** | Warfarin |
| **CYP450** | liver enzymes metabolising drugs (CYP2C9, CYP2C19…) |
| **Pharmacogenomics (PGx)** | how genes affect drug response |
| **CPIC / PharmGKB** | guidelines / knowledge base for PGx |
| **CDS** | clinical decision support |
| **SaMD** | software as a medical device |
| **Primary suspect** | FAERS role: drug the reporter thinks caused the event |
| **Concomitant** | other drug taken at the same time |

## 4. ML glossary

| Term | Meaning |
|------|---------|
| **Supervised learning** | learn mapping X→y from labelled data |
| **Multi-class** | more than two classes (here four) |
| **Ordinal** | classes with an order |
| **Feature engineering** | creating informative inputs from raw data |
| **Target leakage** | features that contain information about the label that wouldn't exist at prediction time |
| **Train/serve skew** | features differ between training and serving |
| **Imputation** | filling missing values |
| **Label encoding / one-hot** | categorical → integer / binary columns |
| **Stratified split** | split preserving class ratios |
| **Cross-validation** | repeated train/validate splits for robust estimates |
| **Hyper-parameter** | setting chosen before training (depth, lr…) |
| **Grid vs random search** | exhaustive vs sampled hyper-parameter search |
| **Boosting / bagging** | sequential error-correcting ensemble / parallel averaged ensemble |
| **Shrinkage (learning rate)** | scales each tree's contribution |
| **Softmax** | converts scores to probabilities summing to 1 |
| **Log-loss (cross-entropy)** | penalises confident wrong probabilities |
| **Calibration** | predicted probabilities match observed frequencies |
| **Precision / recall / F1** | quality metrics per class |
| **Macro vs weighted average** | unweighted vs support-weighted mean across classes |
| **SHAP** | additive feature attribution for explanations |
| **Drift** | data/model behaviour changing over time |
| **Pickle** | Python object serialisation (unsafe for untrusted input) |
| **Artifact** | the saved bundle: model + encoder + metadata |

## 5. Numbers to memorise

| Fact | Value |
|------|-------|
| Drugs supported | 10 |
| Hand-coded interactions | 8 |
| FAERS target | up to 5,000 reports × 10 drugs = 50,000 |
| Real pipeline split / accuracy (README) | 40,000 / 10,000 → 63.1 % |
| Features | 11 |
| Classes | 4 (Low, Moderate, High, Critical) |
| Target formula | 0.2 serious + 0.2 hosp + 0.3 life-threat + 0.3 death; bins 0.1 / 0.3 / 0.6 |
| Severity score | 0.6·max + 0.4·mean over MedDRA weights (1–10) |
| Search | 972 combos, 10 sampled, 3-fold CV = 30 fits |
| Synthetic dataset | 12,000 rows → 9,600 / 2,400; accuracy 91.4 %; 10 % label noise |
| API endpoints | `/api/health`, `/api/drugs`, `/api/assess`, `/api/interactions` (+ `/`, `/{file}`, `/static`, `/docs`) |
| Ports | 8000 (local dev), 7860 (Docker/HF) |
| Model file | `models/drug_risk_model.pkl` |
