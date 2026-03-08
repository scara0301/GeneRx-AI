# GeneRx AI

Clinical drug safety assessment powered by machine learning and FDA adverse event data.

## What It Does

GeneRx AI evaluates medication safety for individual patients by combining:

- **ML Risk Model** — XGBoost classifier trained on 50,000 real FDA adverse event reports (FAERS) and SIDER side-effect data
- **Clinical Rules Engine** — Evidence-based drug interaction checking, contraindication detection, and dose guidance
- **Modern Web Interface** — Dark-themed SPA with separate Clinician and Patient modes

## Architecture

```
backend/
  server.py           → FastAPI REST API (port 8000)
  clinical_engine.py  → Rule-based drug assessment
  ml_model.py         → XGBoost model wrapper
  fetch_faers.py      → FDA adverse event downloader
  fetch_sider.py      → SIDER side-effect downloader
  build_dataset.py    → Feature engineering pipeline
  train_model.py      → Model training with hyperparameter tuning
  utils.py            → Clinical severity scoring
frontend/
  index.html          → Single-page application
  style.css           → Dark glassmorphism theme
  app.js              → Vanilla JS client logic
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Install dependencies
pip install fastapi uvicorn pandas numpy scikit-learn xgboost requests

# Fetch data and train model
python -m backend.fetch_faers
python -m backend.fetch_sider
python -m backend.build_dataset
python -m backend.train_model

# Start servers
python -m uvicorn backend.server:app --port 8000
python -m http.server 3000 --directory frontend
```

Open **http://localhost:3000** in your browser.

Or use the batch script:
```bash
run_app.bat
```

## Data Sources

| Source | Description | Records |
|--------|------------|---------|
| [openFDA FAERS](https://open.fda.gov/apis/drug/drugadverseevent/) | FDA Adverse Event Reporting System | 50,000 events |
| [SIDER 4.1](http://sideeffects.embl.de/) | Drug side-effect database (EMBL) | 1,934 pairs |

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost (RandomizedSearchCV tuned) |
| Training samples | 40,000 |
| Test accuracy | 63.1% |
| Risk categories | Low, Moderate, High, Critical |

## Tech Stack

- **Backend**: FastAPI, XGBoost, scikit-learn, pandas
- **Frontend**: Vanilla JS, CSS (dark theme, glassmorphism), Chart.js
- **Data**: openFDA API, SIDER 4.1

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute medical advice. Always consult a qualified healthcare professional before making medication decisions.

## License

MIT
