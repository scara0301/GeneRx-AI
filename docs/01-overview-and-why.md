# 01 — Overview: what the project is and *why* it is built this way

## 1. The problem

Prescribing safely means checking, for **every drug**, a long list of patient factors:

* **Kidney function (eGFR)** — drugs like Metformin and NSAIDs are dangerous when kidneys are failing.
* **Liver enzymes (ALT/AST)** — statins are contraindicated in active liver disease.
* **Comorbidities** — beta-blockers are dangerous in asthma/COPD, NSAIDs worsen heart failure.
* **Allergies** — penicillin allergy ⇒ no Amoxicillin.
* **Co-medications** — Warfarin + Ibuprofen is a classic, life-threatening bleeding interaction.
* **Age** — ≥ 65 needs lower starting doses, ≥ 80 needs high caution.

Humans forget; software doesn't. This is called **clinical decision support (CDS)**.

## 2. What GeneRx AI does

Input → Output:

```
Patient profile                      Drugs to check                 Output (per drug)
(age, sex, weight, conditions,  +    (Metformin, Warfarin, …)  ──►  suitability  (Suitable/Caution/Avoid/Contraindicated)
 labs, current meds, allergies)                                      risk level   (Low/Moderate/High/Critical)
                                                                     reasons, warnings, dose notes, monitoring
                                                                     common side-effects (+ probabilities)
                                                                     ML risk class + confidence + class probabilities
                                                          + once for the whole set: drug–drug interactions
```

Two user experiences:

| Mode | Audience | Style |
|------|----------|-------|
| **Clinician** | Doctor / pharmacist | Full form with labs, table of results, expandable detail cards, DDI alerts, stacked probability chart |
| **Patient** | Layperson | 6-step wizard with plain-English wording ("Kidney problems"), traffic-light verdicts ("Good Match", "Use With Caution", "Not Recommended", "Do Not Use") |

## 3. The core design idea — a **hybrid** system

```
              ┌────────────────────────────────┐
 Patient ───► │  Clinical Rules Engine         │──► Verdict (the DECISION)  ◄── explainable, auditable
              │  (clinical_engine.py)          │
              └────────────────────────────────┘
              ┌────────────────────────────────┐
 Patient ───► │  XGBoost risk model            │──► Risk class + probabilities (the SIGNAL) ◄── advisory only
              │  (ml_model.py)                 │
              └────────────────────────────────┘
```

**Why not pure ML?**
1. *Explainability* — a clinician must be able to read "eGFR 28 < 30 ⇒ contraindicated (lactic acidosis risk)". A black-box score is not acceptable for a contraindication.
2. *Safety* — hard contraindications are known medical facts; you don't want a probabilistic model to "average them away".
3. *Data* — FAERS contains adverse-event **reports**, not patient-level outcomes with lab values. It cannot teach a model "eGFR 28 + Metformin = bad". Rules encode that knowledge directly.

**Why not pure rules?**
Rules only know what a human wrote down. The ML model adds an empirical, data-derived signal (how serious are reported events for this drug, adjusted for age/polypharmacy) and gives *probabilities* that a UI can chart.

**Graceful degradation**: if the model file is missing the API still works — `/api/health` reports
`ml_model_loaded: false` and the UI shows "Rules Engine Only". (`ml_model.py:_load_model`)

## 4. Why this tech stack

| Choice | Why (the answer to give) |
|--------|--------------------------|
| **Python** | The ML ecosystem (pandas, scikit-learn, XGBoost) is Python; one language for data, model and API. |
| **FastAPI** | Async-capable, automatic request validation via Pydantic, auto-generated OpenAPI docs at `/docs`, very little boilerplate; much lighter than Django for a JSON API. Versus Flask: built-in validation + docs + type hints. |
| **Pydantic models** | `PatientProfile` gives type checking, defaults and 422 errors for free. |
| **XGBoost** | Best-in-class on tabular data; see [08](08-modelling-and-algorithm-choice.md). |
| **pandas / numpy** | Standard for cleaning, group-by aggregation (`groupby().agg`) and array ops. |
| **scikit-learn** | `LabelEncoder`, `train_test_split`, `RandomizedSearchCV`, metrics, fallback `GradientBoostingClassifier`. |
| **requests** | Simple HTTP client for the openFDA REST API and SIDER downloads. |
| **Vanilla JS + CSS (no React)** | One small SPA; no build step, no node toolchain; deployable as static files served by the same FastAPI process. Keeps the Docker image small and the project easy to run. |
| **Chart.js** | Light, CDN-loaded stacked-bar chart for the class probabilities. |
| **Lucide icons** | Tiny SVG icon set loaded from CDN. |
| **Docker + Hugging Face Spaces** | Free hosting for ML demos; `sdk: docker` in the README front-matter lets HF build the `Dockerfile`; exposes port 7860. |
| **openFDA FAERS** | Largest public source of *real-world* adverse-event reports; free JSON API, no key needed for modest use. |
| **SIDER 4.1** | Curated side-effect database mined from drug labels, with frequencies — complements the "report" view with the "label" view. |

## 5. Project history (from `git log` — useful for "walk me through how it evolved")

| Date | What happened |
|------|---------------|
| Apr 2025 | First version: apparently a **Streamlit** app (`app.py`; no longer in the repo) — commit messages like *"Delete streamlit directory"*, *"Update app.py"*. Probably the original gene-aware version (the leftover `data/drug_catalog.json` has `target_genes` such as CYP2C9, VKORC1 and tokenised `input_ids`). **(inferred from file contents & commit messages; I could not read those old commits)** |
| Mar 2026 | **Rebuild** ("rebuild" commit): FastAPI backend, FAERS/SIDER data pipeline, clinical rules engine, vanilla-JS SPA. |
| Mar 8 2026 | Deployment attempts: Vercel ("vercel", then *"hugging face commit cuz vercel is hell"*) → moved to **Hugging Face Spaces** with Docker. Fixed API routing for the HF iframe path (`basePath` logic in `app.js`). |
| Mar 15 2026 | Model-file problem on HF: tried committing the `.pkl`, then Git-LFS, then HF rejected binaries → **train the model during `docker build`** with `build_model.py` (synthetic data, no network). Then cleanup/"humanising" of comments. |

Why that matters in an interview: it shows you *iterated against real deployment constraints* — a good story.

## 6. What the project is **not** (say it up-front)

* Not a medical device and not clinically validated (disclaimer is in the README and footer).
* Not pharmacogenomic (despite the name) — "No genetic data required".
* Not a replacement for a validated interaction database like DrugBank/Lexicomp.
* ML output is **informational**; it does not modify the suitability verdict.
