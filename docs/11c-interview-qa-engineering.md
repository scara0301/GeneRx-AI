# 11c — Interview Q&A, part 3: backend, frontend, deployment, ethics, behavioural, curveballs

← [Part 1](11-interview-qa.md) · [Part 2](11b-interview-qa-modelling.md)

---

## H. Backend / API

**Q131. Why FastAPI over Flask/Django?**
Type-hint-driven validation (Pydantic), automatic OpenAPI/Swagger at `/docs`, async support, high performance on Starlette/uvicorn, minimal boilerplate. Flask needs extra libraries for validation/docs; Django is heavyweight for a 4-endpoint JSON API with no DB/ORM.

**Q132. What does Pydantic give you here?**
Parsing + validation + defaults + docs. A request with `age: "abc"` returns 422 automatically. Weakness: I didn't add range constraints (`Field(ge=0, le=120)`).

**Q133. Sync `def` vs `async def` in your routes?**
API routes are plain `def`, which FastAPI runs in a thread pool — appropriate for CPU-bound, non-awaiting code (the model call). The static-file routes are `async def` but do no awaiting; harmless.

**Q134. Is the endpoint idempotent / safe to retry?**
Yes — it's a pure function of the request (POST only because of the body size/structure). No side effects.

**Q135. What status codes do you return?**
200 on success, 422 on validation errors (automatic), 500 on unhandled errors (e.g. unknown drug). I'd convert unknown drug to 422/404 explicitly.

**Q136. Input validation concerns?**
Free-text conditions/allergies must match lower-cased strings; no range checks; unknown drug crashes. Fix with enums/`Literal` for drugs, controlled vocab for conditions, `Field` constraints.

**Q137. Security review of the API?**
No auth/rate limit; CORS `*`; no PHI persisted; static catch-all route returns only files from `frontend/` (single-segment path param, so no `../` traversal); pickle loaded from a trusted local path only. Production: auth (OAuth2/JWT), HTTPS, rate limiting, request size limits, audit logging, dependency scanning.

**Q138. How would you test it?**
`pytest` + FastAPI `TestClient` (I used it to verify behaviour): unit tests for each rule branch and the interaction lookup; API tests for 200/422/500 cases; golden-file tests for responses; property tests (e.g. Metformin with eGFR<30 is always Contraindicated); model smoke test (artifact loads, probabilities sum to 1).

**Q139. Why `sys.path.insert` in `server.py`?**
So flat imports work regardless of how the server is started (package or script). Cleaner: relative imports and always run as a package.

**Q140. How does the server find the model file?**
`MODEL_DIR = <ml_model.py dir>/../models` — path relative to the file, not the cwd, so it works from any working directory (unlike the static files, which use a cwd-relative `"frontend"`).

**Q141. What happens on the first request?**
`get_predictor()` constructs `DrugRiskPredictor`, which unpickles the model (tens of ms). Subsequent calls reuse it. Better: load at startup.

**Q142. Why does `/api/assess` also pass `current_meds` into the interaction check?**
Interactions between a *new* drug and what the patient already takes are the clinically important ones.

**Q143. Complexity of interaction checking?**
O(n²) over n drugs with O(1) dict lookups; n is tiny. At scale, index interactions by drug (adjacency list).

**Q144. Why `allow_credentials=True` with `*`?**
Probably copy-paste default; browsers disallow wildcard origins with credentials, and the app doesn't use cookies. I'd set explicit origins and drop credentials.

**Q145. How would you add authentication and an audit log?**
OAuth2 password/JWT or SSO via FastAPI dependencies; a middleware that logs `{user, timestamp, rules_version, model_version, input hash, output}` to an append-only store, taking care not to log PHI in plaintext.

---

## I. Frontend

**Q146. Why vanilla JS?**
Single small page; no build tooling; served by the same container; easy to audit. React would be justified if the UI grew (routing, shared state, many components).

**Q147. How does the UI choose the API URL?**
If hostname is localhost/127.0.0.1 → `http://localhost:8000`; otherwise the current pathname (to work under Hugging Face's `/embed/user/space` iframe path). Simplest robust alternative: always call relative URLs.

**Q148. How are results rendered?**
Template-literal HTML → `innerHTML`. Simple but requires care with untrusted text (XSS).

**Q149. Walk through the patient wizard.**
`patientWizardStep` 1–6 drives `renderPatientWizard()`; each Next/Back calls `wizardSaveStep()` to persist the inputs into `patientData`, step 5 validates at least one drug, step 6 triggers `runPatientAssessment()` then `renderPatientResults()`. Plain-language maps (`PLAIN_CONDITIONS`, `suitMap`) translate medical terms.

**Q150. What does the chart show?**
Stacked bars per drug of the four class probabilities (%), via Chart.js; old chart destroyed before redraw.

**Q151. How do you handle API failure in the UI?**
Status badge "API Offline", fallback drug list, and error cards ("Connection Error" / "Could not reach the server").

**Q152. Accessibility?**
Labels bound to inputs, text labels with colours (not colour-only), dark theme contrast; missing: ARIA live regions, keyboard handling for custom cards (`div onclick`).

**Q153. Any frontend bug you know?**
`ast` is populated from the ALT field; `currentMode` implicit global; user-entered name inserted unescaped into `innerHTML`; patient mode hard-codes ALT/AST/INR and no allergies.

**Q154. How is the UI themed?**
CSS custom properties in `:root`, dark glassmorphism (`backdrop-filter: blur(16px)`), Inter font, responsive breakpoints at 900/600 px.

---

## J. Deployment & DevOps

**Q155. How is it deployed?**
Docker on Hugging Face Spaces (`sdk: docker`, port 7860). The image installs requirements, runs `build_model.py` to bake the model in, then `uvicorn backend.server:app`.

**Q156. Why Hugging Face and not Vercel?**
Vercel is geared to static sites and serverless functions: bundle-size limits make heavy scientific-Python dependencies (xgboost/pandas/scikit-learn) awkward, there's no long-lived process, and serving a pickled model plus static files from one app is clumsy. HF Spaces runs a full container, is free for demos and is ML-native. (Your commit message says "hugging face commit cuz vercel is hell" — **replace my generic reasons with what actually went wrong for you**; the git log shows several "vercel"/"deploy"/"fix api routing" commits on 8 Mar 2026.)

**Q157. Why train the model during `docker build`?**
HF rejected binary files in the repo (tried commit → Git LFS → still blocked). `build_model.py` needs no data download so it runs offline at build, making the image self-contained and reproducible.

**Q158. Downsides of that?**
The deployed model is synthetic; each rebuild retrains (deterministic seeds so identical); the README overstates what's deployed. Better: CI trains the real model and publishes it to the HF Hub; the container downloads a pinned version.

**Q159. Why a non-root user in the Dockerfile?**
HF Spaces requires UID 1000; also a general security best practice.

**Q160. Why copy `requirements.txt` first?**
Docker layer caching: dependencies re-install only when requirements change.

**Q161. Why `--host 0.0.0.0`?**
Bind all interfaces so the container port is reachable from outside the container.

**Q162. Why does the container listen on 7860?**
It's the default port for Docker Spaces on Hugging Face (configurable with `app_port` in the README front-matter; this project doesn't set it, so the default applies). The `EXPOSE`, `CMD` and `server.py` fallback all agree on 7860.

**Q163. How would you add CI/CD?**
GitHub Actions: lint (`ruff`), tests (`pytest`), build Docker image, run a smoke test (`/api/health`), push to HF Space on `main`.

**Q164. How would you do observability?**
Structured logs (request id, latency), Prometheus metrics (`prometheus-fastapi-instrumentator`), health/readiness endpoints (`/api/health` exists), error tracking (Sentry).

**Q165. What's in `.gitignore` and why?**
`data/` (regenerated by the pipeline; large), `__pycache__`, venvs, IDE files. `models/` deliberately not ignored (originally so the pkl could be committed for HF); the pkl is now untracked because the Docker build creates it.

**Q166. What is `.gitattributes` doing?**
`*.pkl filter=lfs` — leftover from the Git-LFS attempt; harmless but stale.

**Q167. Windows `run_app.bat` — why?**
One-click local setup for non-technical users on Windows: installs deps, runs the full data pipeline, launches API (port 8000, `--reload`) and static frontend (port 3000 via `npx serve`).

---

## K. Ethics, safety, regulation

**Q168. What are the ethical risks?**
Over-reliance (automation bias), false reassurance from the ML "Low Risk", patients self-medicating on a patient-mode verdict, biased data (FAERS skew), lack of validation.

**Q169. How did you mitigate them?**
Disclaimer in footer and README; patient mode wording ("discuss with your doctor"); rules-first verdicts with explicit reasons; ML labelled advisory; no PHI stored.

**Q170. Would this need regulatory approval?**
A tool that gives patient-specific treatment recommendations is likely *Software as a Medical Device* (FDA SaMD / EU MDR). CDS exemptions depend on whether a clinician can independently review the basis of the recommendation — my reasons list supports that, but this is a demo and not intended for clinical use.

**Q171. Privacy (HIPAA/GDPR)?**
Stateless, nothing stored; but if deployed publicly, inputs transit the network — need TLS, no logging of identifiers, a privacy policy. Real deployments need BAAs/DPAs and access controls.

**Q172. Alert fatigue?**
Too many low-value warnings cause clinicians to ignore alerts. My severity tiers (CRITICAL…LOW) and "key concern" summarisation address it partly; real systems tune thresholds with clinicians.

**Q173. Explainability vs accuracy?**
In this domain the explanation is part of the product. Hence rules + per-reason text; SHAP would explain the ML part.

---

## L. Behavioural & reflection

**Q174. What are you proudest of?**
Taking it from raw API data to a deployed, usable app, and making the deliberate decision to keep ML advisory and be transparent about limitations.

**Q175. Biggest mistake?**
Not designing serving features first (train/serve skew), and letting the README claim real-data training while the deployed model is synthetic. I now document the difference clearly.

**Q176. How did you debug the deployment issues?**
Iterated through Vercel → HF; read build logs; isolated the cause (binary files rejected); changed approach (build-time training) rather than fighting the platform; fixed API routing for the HF iframe path.

**Q177. What did you learn?**
Real data is messy and labels are the hard part; evaluation needs baselines; deployment constraints shape architecture; explainability matters in high-stakes domains.

**Q178. If you had a week more?**
(1) fix skew + persist SIDER stats; (2) tests + CI; (3) SHAP explanations; (4) validate and expand the interaction table from DDInter/DrugBank; (5) real-model deployment via HF Hub.

**Q179. How would you explain it to a non-technical person?**
"You tell it about the patient and the medicines you're considering; it flags the ones that could be dangerous for that person and explains why — like a pharmacist's checklist that never forgets."

**Q180. How do you know it works?**
Honest answer: the rules were verified by reading against textbook guidance and by running scenarios (e.g. eGFR 28 → Metformin contraindicated, penicillin allergy → Amoxicillin contraindicated, Warfarin+Ibuprofen → CRITICAL); the ML model was evaluated on a held-out split. It has *not* been clinically validated or covered by automated tests — the next step.

---

## M. Curveball / trap questions

**Q181. "Your ML says Low Risk 97 % but the rules say Contraindicated. Which is right?"**
The rules, for that question. The ML wasn't given kidney function and predicts reported-event seriousness, so it can't see the contraindication. Shows why ML is advisory. Improvement: include labs in a patient-level model and surface the conflict explicitly.

**Q182. "91 % is great — why did the README say 63 %?"**
Different models: 91 % is the synthetic Docker model, 63 % is the real FAERS+SIDER model. Own it (see [10](10-known-issues-and-defence.md) #1).

**Q183. "Isn't training on data you generated circular?"**
Yes. It's a placeholder for deployment; its score validates nothing clinically. The real pipeline exists for that purpose.

**Q184. "Can this model predict whether *this patient* will have an adverse event?"**
No. It estimates the typical seriousness of a reported event for a drug given a few attributes. A patient-level prediction needs patient-level outcome data.

**Q185. "Why accuracy and not AUC?"**
Multi-class tuning with a simple, aligned metric; but I'd prefer macro-F1 / log-loss, and report per-class recall. Acknowledge the limitation.

**Q186. "What if a drug isn't in your catalog?"**
ML returns "not in training data"; the rules engine currently raises `KeyError` (HTTP 500) — I'd validate and return a clear 4xx, and design a generic fallback ("no rules available for this drug").

**Q187. "How do you know your thresholds are right?"**
They follow common guidance (e.g. metformin eGFR 30/45, statin ALT >3×ULN, HbA1c 6.5 %) but I haven't had them clinically reviewed; production use needs guideline citations, pharmacist/physician sign-off, and tests.

**Q188. "Would you trust this with your own medication?"**
No — and the disclaimer says so. It's a learning/demo project; I'd trust a validated, regulated system with clinician oversight.

**Q189. "Walk me through what happens when I click Run Assessment."** 
Button → `runAssessment` → `collectPatientProfile` → `POST /api/assess` → Pydantic validation → for each drug `evaluate_drug` (rules) and `predict` (XGBoost) → `check_drug_interactions` → JSON → `renderResults` draws table, DDI banner, accordion cards and the Chart.js stacked bars. (Diagram: [03](03-architecture.md) §3.)

**Q190. "What's the time complexity?"**
Per request O(d·(R + M) + n²) — d drugs, R rule branch cost (constant), M model inference (trees × depth × classes, tiny), n² for interaction pairs. Microseconds-to-milliseconds.

**Q191. "Can the pickle be tampered with?"**
If an attacker can write the file they can run code on load. Mitigate with file permissions, signed artifacts/hashes, or non-pickle formats.

**Q192. "What if two requests arrive simultaneously?"**
The engine uses only local variables and the model object is read-only after load, so there's no shared mutable state in my code. FastAPI runs sync routes in a thread pool; XGBoost inference is generally treated as safe for concurrent reads, but check the docs for your installed version if pressed (I didn't load-test it).

**Q193. "Why 4 risk classes and not a continuous score?"**
Maps directly to the UI's tiers and the rules' risk levels; clinicians think in tiers. Regression on `risk_score` is a valid alternative (and handles ordinality).

**Q194. "What would break if openFDA changed its schema?"**
`fetch_drug_events` uses `.get()` defaults, so missing fields become defaults/NaN rather than exceptions, but silently degraded data is possible → add schema validation and row-count sanity checks in the pipeline.

**Q195. "Explain `get_reaction_severity('nausea')` returning 1.0."**
Single term: weight 1.0 for both max and mean ⇒ 0.6·1 + 0.4·1 = 1.0 ✅. And `["Cardiac arrest","nausea"]` ⇒ max 10, mean 5.5 ⇒ 0.6·10 + 0.4·5.5 = 8.2 ✅.

**Q196. "Why does Paracetamol map to 'acetaminophen'?"**
The UI/catalog use the INN (paracetamol); openFDA/FAERS generic names use the USAN (acetaminophen). Aliasing happens in `predict`; the training data are stored as acetaminophen.

**Q197. "How would you add a new drug?"**
Add to `DRUG_CATALOG` + rule branch + interactions + allergy map; add to `TARGET_DRUGS` in both fetchers (and `DRUG_PROFILES` for the synthetic path); re-run the pipeline and retrain; add UI fallback entry; write tests.

**Q198. "How would you add pharmacogenomics?"**
New optional patient fields (e.g., CYP2C9, VKORC1 genotypes); rule layer from CPIC guidelines (warfarin dosing algorithm, clopidogrel CYP2C19 poor metabolizers → alternative); keep ML advisory. That would finally match the "GeneRx" name.

**Q199. "What if the interviewer asks for the confusion matrix and you don't have it?"**
Say you didn't save it in the repo, and that `train_model.py` prints a per-class classification report you can regenerate in a minute (`python -m backend.train_model`), then describe what you'd look for (adjacent-class confusion, Critical recall).

**Q200. "Anything else you want to add?"**
Close with the roadmap in one breath: "Fix the serving skew and publish the real model, add tests/CI, replace my hand-written interactions with a curated database, move to patient-level data, and add SHAP explanations."
