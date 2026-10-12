# 02 — How to start the project

All commands are run from the **repository root** (`GeneRx-AI/`). This matters:
`server.py` mounts `frontend/` with a *relative* path (`StaticFiles(directory="frontend")`), so starting
the server from another directory breaks the UI.

## 0. Prerequisites

* Python 3.10+ (Docker uses 3.10-slim; I tested the code on 3.13 with XGBoost 3.4.1, scikit-learn 1.9.1)
* `pip`
* Internet **only** for the real data pipeline (openFDA + SIDER) and for the browser to load the CDN
  scripts (Lucide, Chart.js, Google Fonts).

```bash
pip install -r requirements.txt
# = fastapi uvicorn pandas numpy scikit-learn xgboost requests   (unpinned)
```

## 1. Fastest path — what production does (synthetic model, no downloads)

```bash
python backend/build_model.py                 # writes models/drug_risk_model.pkl  (~2 seconds)
python -m uvicorn backend.server:app --port 8000
```
Open **http://localhost:8000** — the same process serves the UI *and* the API.
Interactive API docs: **http://localhost:8000/docs**

Expected console output:
```
Training samples : 9600
Test samples     : 2400
Test Accuracy: 0.9137          ← verified
[ML] Model loaded (accuracy: 0.91375)
```

> If you skip the first command the app still runs in **rules-only mode**
> (console: `[ML] No model found … running in rules-only mode`; UI badge: "Rules Engine Only").

## 2. The "real data" pipeline (what the README describes — FAERS + SIDER)

```bash
python -m backend.fetch_faers      # openFDA → data/faers/faers_events.csv   (≈ 50,000 rows, several minutes: rate-limited)
python -m backend.fetch_sider      # SIDER   → data/sider/side_effects.csv + side_effect_frequencies.csv
python -m backend.build_dataset    # merge + feature engineering → data/training_data.csv, data/drug_risk_profiles.csv
python -m backend.train_model      # RandomizedSearchCV XGBoost → models/drug_risk_model.pkl
python -m uvicorn backend.server:app --port 8000
```
Notes
* `data/` is in `.gitignore` ("regenerated via pipeline — don't commit raw data").
* Both `train_model.py` and `build_model.py` write the **same file name** (`models/drug_risk_model.pkl`)
  with the **same artifact dictionary shape**, so the server can load either. Whichever ran last wins.
* I did **not** run this download pipeline while writing these docs (it needs several minutes of
  rate-limited calls to api.fda.gov), so the README's *63.1 % accuracy / 40,000 training rows* figures are
  **unverified by me** — run it yourself before the interview if you want to quote fresh numbers.

## 3. Two-server dev mode (README's version; frontend and API on different ports)

```bash
python -m uvicorn backend.server:app --port 8000          # API
python -m http.server 3000 --directory frontend            # static UI
```
Open **http://localhost:3000**. `app.js` picks the API URL like this:

```js
const API_BASE = hostname is localhost/127.0.0.1 ? 'http://localhost:8000' : basePath;
```
This is why **CORS is enabled** in `server.py` (`allow_origins=["*"]`): the page origin (`:3000`) differs
from the API origin (`:8000`).

> ⚠️ Gotcha: if you run `python backend/server.py` (default port **7860**) and browse to
> `localhost:7860`, `app.js` will still call `localhost:8000` and show "API Offline". Use port 8000, or
> use the two-server mode.

## 4. Windows one-click

`run_app.bat` does: pip install → `fetch_faers` → `fetch_sider` → `build_dataset` → `train_model` →
opens two terminals (uvicorn on 8000 with `--reload`, and `npx serve` on 3000 for the frontend).
(Downside: it re-downloads everything on every run, and the `pip install` line omits `numpy` — it is pulled in transitively by pandas.)

## 5. Docker (identical to Hugging Face Spaces)

```bash
docker build -t generx-ai .
docker run -p 7860:7860 generx-ai
# open http://localhost:7860
```
What the `Dockerfile` does, line by line, and *why*:

| Line | Why |
|------|-----|
| `FROM python:3.10-slim` | Small official Python image. |
| `RUN useradd -m -u 1000 user` + `USER user` | **Hugging Face Spaces requires a non-root user with UID 1000.** |
| `WORKDIR $HOME/app` | Relative paths (`frontend/`) resolve correctly. |
| `COPY requirements.txt` → `pip install` → `COPY . .` | Layer caching: dependencies are only re-installed when `requirements.txt` changes. |
| `RUN python backend/build_model.py` | **Train the model at image build time.** HF rejected committing a binary `.pkl`; `build_model.py` needs no data download so it works offline. |
| `EXPOSE 7860` | Default port HF routes to. |
| `CMD uvicorn backend.server:app --host 0.0.0.0 --port 7860` | `0.0.0.0` so the container is reachable from outside. |

On Spaces the README front-matter (`sdk: docker`, `colorFrom`, …) tells HF how to build/show the Space.
The app is shown inside an iframe at a path like `/embed/<user>/<space>`, which is why `app.js` computes
`basePath = window.location.pathname` instead of assuming `/`.

## 6. Quick API smoke tests (curl)

```bash
curl localhost:8000/api/health
# {"status":"ok","ml_model_loaded":true,"drugs_available":10}

curl localhost:8000/api/drugs

curl -X POST localhost:8000/api/assess -H 'Content-Type: application/json' -d '{
  "patient": {"name":"Test","age":72,"sex":"M","weight_kg":80,
              "conditions":["Diabetes","CKD (Chronic Kidney Disease)"],"egfr":28,
              "current_meds":["Ibuprofen"],"allergies":["penicillin"]},
  "drugs": ["Metformin","Warfarin","Amoxicillin"]}'

curl -X POST localhost:8000/api/interactions -H 'Content-Type: application/json' \
     -d '{"drugs":["Warfarin","Ibuprofen"]}'
```
Verified results for the assess call above:

| Drug | Rules verdict | Rules risk | ML label (confidence) |
|------|---------------|------------|-----------------------|
| Metformin | Contraindicated (eGFR 28 < 30) | Critical | Low Risk (97.2 %) |
| Warfarin | Suitable (no indication/contraindication rule fires) | Low | Moderate (58.4 %) |
| Amoxicillin | Contraindicated (penicillin allergy) | Critical | Low Risk (94.1 %) |

Interactions found: Metformin+Ibuprofen (MODERATE), Warfarin+Amoxicillin (MODERATE), Warfarin+Ibuprofen (**CRITICAL**).

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| UI badge "API Offline" | wrong API port / server not running | run uvicorn on 8000 (see §3 gotcha) |
| UI badge "Rules Engine Only" | no `models/drug_risk_model.pkl` | `python backend/build_model.py` |
| `ModuleNotFoundError: clinical_engine` | running from a weird cwd | run from repo root; `server.py` inserts its own folder into `sys.path` |
| 404/blank UI | started uvicorn outside repo root | `cd` to repo root |
| `Training data not found` | skipped `build_dataset` | run the pipeline steps in order |
| `No results for X` from fetch_faers | openFDA 404 for that query | normal; script moves on |
| HTTP 500 on `/api/assess` | drug name not in catalog (`KeyError`) | send one of the 10 catalog names (case-sensitive!) |
| XGBoost warning about `use_label_encoder` | parameter is deprecated in new XGBoost | harmless; remove the argument to silence |
| Chart doesn't render | CDN blocked / offline | needs internet for Chart.js + Lucide |
