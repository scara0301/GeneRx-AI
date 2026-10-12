# GeneRx AI — Interview Prep Docs (Start Here)

> Everything about the project from A to Z: what it is, why each choice was made, every file and
> function, the data, feature engineering, modelling, how to run it, and 200 likely interview questions with model answers.
>
> **Verified facts** in these docs come from reading every line of the code and running it
> (Python 3.13, XGBoost 3.4.1, scikit-learn 1.9.1). Anything I could *not* verify is marked **(unverified)**.
> The "why I did it" sections are the **best rationale that the code supports** — adjust the wording
> so it matches what you actually remember thinking. Interviewers can tell when an answer is memorised
> vs. owned.

## Reading order (if you only have one evening)

| # | File | Time | What you get |
|---|------|------|--------------|
| 1 | **this page** | 5 min | Pitch + the "truth sheet" (things to say before they find them) |
| 2 | [10-known-issues-and-defence.md](10-known-issues-and-defence.md) | 15 min | Every weakness in the project and how to talk about it |
| 3 | [01-overview-and-why.md](01-overview-and-why.md) | 10 min | Problem, users, why this design |
| 4 | [03-architecture.md](03-architecture.md) | 10 min | Diagrams you can redraw on a whiteboard |
| 5 | [07-data-and-feature-engineering.md](07-data-and-feature-engineering.md) | 20 min | Data sources, every feature, the target variable |
| 6 | [08-modelling-and-algorithm-choice.md](08-modelling-and-algorithm-choice.md) | 20 min | XGBoost vs the alternatives, metrics, tuning |
| 7 | [11-interview-qa.md](11-interview-qa.md) (+ 11b, 11c) | 60 min | The questions + model answers |
| — | the rest | reference | Function-by-function walkthroughs, call map, how to run, glossary |

## All files

| File | Purpose |
|------|---------|
| [01-overview-and-why.md](01-overview-and-why.md) | What the project is, the problem, why this stack, project history |
| [02-how-to-run.md](02-how-to-run.md) | Start the project (3 ways), the training pipelines, Docker/HF, troubleshooting |
| [03-architecture.md](03-architecture.md) | System, data, request-sequence and deployment diagrams (Mermaid + ASCII) |
| [04-file-and-function-map.md](04-file-and-function-map.md) | **Every file & function: what it calls, who calls it, input → output** |
| [05-backend-walkthrough.md](05-backend-walkthrough.md) | Backend code, function by function, with the *why* |
| [06-frontend-walkthrough.md](06-frontend-walkthrough.md) | Frontend code, function by function, with the *why* |
| [07-data-and-feature-engineering.md](07-data-and-feature-engineering.md) | FAERS, SIDER, cleaning, all 11 features, target construction |
| [08-modelling-and-algorithm-choice.md](08-modelling-and-algorithm-choice.md) | Why XGBoost, hyper-parameters, evaluation, experiments I ran |
| [09-clinical-rules-engine.md](09-clinical-rules-engine.md) | Every clinical rule, threshold, interaction and why rules + ML |
| [10-known-issues-and-defence.md](10-known-issues-and-defence.md) | Bugs/limitations found in the review + honest answers |
| [11-interview-qa.md](11-interview-qa.md) | Q&A part 1: project, architecture, data, feature engineering |
| [11b-interview-qa-modelling.md](11b-interview-qa-modelling.md) | Q&A part 2: modelling, evaluation, ML theory |
| [11c-interview-qa-engineering.md](11c-interview-qa-engineering.md) | Q&A part 3: backend, frontend, deployment, ethics, behavioural, curveballs |
| [12-tech-stack-and-glossary.md](12-tech-stack-and-glossary.md) | Every library & why; medical + ML glossary |

---

## The 30-second pitch

> "GeneRx AI is a clinical decision-support web app. You enter a patient's age, sex, weight, conditions,
> lab values, current medications and allergies, pick the drugs you're considering, and it returns a
> per-drug verdict — **Suitable / Caution / Avoid / Contraindicated** — with the reasons, dose notes,
> monitoring advice and any drug–drug interactions. It combines two things: a **deterministic clinical
> rules engine** that makes the actual safety decision and is fully explainable, and an **XGBoost model**
> trained on FDA adverse-event data (FAERS + SIDER) that adds a population-level risk signal with
> probabilities. Backend is FastAPI, frontend is vanilla JS with a clinician mode and a plain-language
> patient wizard, and it's containerised and deployed on Hugging Face Spaces."

## The 2-minute pitch (add this when they say "tell me more")

1. **Problem** — prescribers must juggle kidney function, liver enzymes, allergies, comorbidities and
   co-medications for every prescription. Errors here cause preventable adverse drug events.
2. **Design decision** — in healthcare, *explainability beats raw accuracy*. So the verdict comes from
   explicit, auditable rules (e.g. eGFR < 30 ⇒ Metformin contraindicated). ML is a *secondary* signal
   and never overrides a rule.
3. **Data science** — I pulled 50,000 real adverse-event reports from openFDA (FAERS) for 10 common
   drugs, joined them with SIDER side-effect profiles, engineered 11 features (incl. a custom
   MedDRA-term severity score), built a 4-class severity target from seriousness flags, and tuned
   XGBoost with `RandomizedSearchCV`. Test accuracy on the 4-class problem was ~63% (chance is 25% for four balanced classes; the real majority-class baseline depends on the class mix — check it).
4. **Engineering** — FastAPI + Pydantic validation, a singleton model loader with graceful
   degradation to *rules-only mode*, CORS, a static-file mount so one process serves UI + API, Docker
   image that trains the model at build time, deployed on Hugging Face Spaces.
5. **Honesty** — it's an educational tool, not a medical device; I know its limits (see below).

---

## ⚠️ The Truth Sheet — say these *before* the interviewer discovers them

These are real findings from my line-by-line review and from running the code. Owning them makes you
look senior; being caught out on them does the opposite. Full detail in
[10-known-issues-and-defence.md](10-known-issues-and-defence.md).

1. **The deployed model is trained on synthetic data.** The `Dockerfile` runs `backend/build_model.py`,
   which *generates* 12,000 rows from hand-written rules (10 drug "profiles" + age/co-med/reaction
   modifiers + 10% label noise). Its 91.4% accuracy is just the model re-learning those rules (the noise
   ceiling is ≈ 90%). The **real FAERS+SIDER pipeline** (`fetch_* → build_dataset → train_model`, ~63% on
   4 classes) exists and is what the README describes, but it is the *local/research* path — it is not
   what runs in production. Reason: the HF Spaces deploy couldn't ship a binary `.pkl` or fetch data at
   build time (see commit history: *"train ML model at Docker build time to bypass HF binary file restriction"*).
2. **Train/serve skew.** At inference `server.py` only knows 4 things about the model's inputs
   (age, sex, weight, # current meds). The other 6 features are hard-coded defaults in
   `ml_model.predict` (reaction_count=0, weighted_reaction_score=1.0, is_primary_suspect=1, and the three
   SIDER features = 0). I proved this matters: with the shipped model, Warfarin (designed as "High")
   predicts **Moderate** with serving defaults but **High** with its real SIDER values.
3. **The ML label ≠ patient suitability.** The target is the *seriousness of a reported adverse event*,
   not "is this drug safe for this patient". FAERS has no denominator (no exposure counts) and is
   voluntary-report data with known biases. That's why ML never changes the verdict (e.g. Metformin at
   eGFR 28 → rules say **Contraindicated** while ML says **Low Risk 97%** — they answer different
   questions).
4. **Unknown drug ⇒ crash.** `evaluate_drug` indexes `DRUG_CATALOG[drug_name]` directly, so an
   unlisted drug raises `KeyError` → HTTP 500 (the UI only offers the 10 known drugs, so users can't
   trigger it, but a direct API call can).
5. **Only 10 drugs, 8 hand-coded interaction pairs.** Rules are illustrative, not a validated
   formulary. Not a medical device, not clinically validated.
6. **Name vs. content.** "GeneRx" suggests pharmacogenomics, but v2 deliberately uses *no genetic data*
   (stated in `clinical_engine.py`'s header). `data/drug_catalog.json` (CYP2C9/VKORC1/tokenised ids) is a
   leftover from the earlier gene-based version and is unused.
7. A few small bugs (frontend sends ALT as AST; FAERS week/day age-unit mapping; inconsistent max-severity
   lookup; duplicated allergy reasons). Listed with fixes in file 10.

## One-line answers to the 6 questions you will 100% get

| Question | One-liner |
|----------|-----------|
| Why XGBoost? | Tabular, mixed numeric/categorical, non-linear interactions, small data, fast, regularised; matched/beat RF & sklearn-GBM in my comparison (91.4% vs 90.7%/91.3%) at 0.7 s vs 20 s fit, and far beat logistic regression (56%). |
| Why rules *and* ML? | Rules = safe, explainable, auditable decision. ML = extra data-driven signal with probabilities. In healthcare you never let a black box overrule a contraindication. |
| What's your target variable? | A 0–3 severity class built from FAERS flags: `0.2·serious + 0.2·hospitalization + 0.3·life-threatening + 0.3·death`, binned at 0.1/0.3/0.6. |
| Your accuracy is 63% — is that good? | For a noisy 4-class problem built from voluntary reports with mostly drug-level features, it's above chance (25% for 4 balanced classes) but not clinically deployable — which is why ML is advisory only. Always quote the majority-class baseline next to it. |
| What was the hardest part? | Getting honest labels from FAERS (no denominator, reporting bias) and deploying to HF Spaces (binary-file restriction → train at Docker build time). |
| What would you improve? | Real patient-level data (MIMIC/EHR), proper drug features (ChEMBL/DrugBank), calibrated probabilities, SHAP explanations, a validated interaction DB (DrugBank/RxNorm), unit tests + CI. |
