# 08 — Modelling: why XGBoost, how it was trained, how it was evaluated

## 1. The ML task, precisely

* **Type:** supervised, **multi-class classification** (4 classes: 0 Low Risk, 1 Moderate, 2 High Risk, 3 Critical).
* **Unit of prediction:** one adverse-event *report* (drug + patient demographics + report characteristics).
* **Input:** 11 numeric features (see [07](07-data-and-feature-engineering.md)).
* **Output:** class + probability vector → UI shows label, confidence (= max probability) and a stacked probability bar.
* **Role in the product:** advisory signal; the rules engine decides suitability.

## 2. Why XGBoost? (the answer, in layers)

**Short answer:** "It's the strongest general-purpose algorithm for small-to-medium *tabular* data with mixed numeric/ordinal features and non-linear interactions, it's fast, regularised, and handles missing values — and I verified that it matches/beats the alternatives."

**Long answer — what the data looks like:**
* ~50k rows × 11 columns, all numeric, tabular ⇒ tree ensembles are the default best choice (deep learning rarely wins on this kind of data and needs far more data/tuning).
* **Non-linear interactions** are the whole point: e.g. risk jumps when *age is high AND many co-meds AND many reactions*. Trees model thresholds & interactions natively; logistic regression can't without manual interaction terms.
* **Heterogeneous scales** (age 0–100, weight 40–150, frequency 0–1, counts 0–8) ⇒ trees need no scaling/normalisation.
* **Label-encoded categorical** (`drug_encoded`) ⇒ trees split on it sensibly; linear models would wrongly treat the integer as an ordered magnitude.
* **Missing / zero-filled SIDER values** ⇒ XGBoost has built-in sparsity-aware split finding.
* **Probabilities needed** for the UI chart ⇒ `multi:softprob` gives a probability per class.

**Experiment I ran (same synthetic data, same split, seed 42; features as in production):**

| Model | Accuracy | Macro-F1 | Fit time |
|-------|---------:|---------:|---------:|
| Majority-class baseline | 0.287 | – | – |
| Logistic Regression (scaled) | 0.560 | 0.556 | 0.1 s |
| Decision Tree (depth 5) | 0.723 | 0.727 | <0.1 s |
| Random Forest (200 trees) | 0.907 | 0.907 | 1.1 s |
| sklearn GradientBoosting (200, depth 5) | 0.913 | 0.914 | 20.0 s |
| **XGBoost (shipped params)** | **0.914** | **0.915** | **0.7 s** |

Take-aways you can say out loud:
1. Linear model fails (56 %) ⇒ the problem is genuinely **non-linear/interaction-driven**.
2. Single tree 72 % ⇒ ensembles help a lot.
3. RF, sklearn GBM and XGBoost are **within noise of each other** (≈ 91 %) — honest: XGBoost is not magically better here. The differences are in **speed** (XGBoost ~28× faster than sklearn's GBM), **built-in regularisation**, and ecosystem (`RandomizedSearchCV`, `feature_importances_`, SHAP support).
4. Caveat: this comparison is on **synthetic** data whose rule is additive thresholds — tree-friendly by construction. On real FAERS, run the same bake-off.

### Alternatives and why not (rehearse these)

| Alternative | Why not (here) |
|-------------|----------------|
| **Logistic / softmax regression** | Linear; can't capture interactions (56 % vs 91 %); would need one-hot + scaling. Good baseline & more interpretable, though. |
| **Single decision tree** | Interpretable but high variance/low accuracy (72 %). |
| **Random Forest** | Strong, robust, little tuning; ≈ equal here. XGBoost generally gets a bit more from boosting on harder data and gives better-calibrated softprob via logloss optimisation; RF models are larger and slower to predict. A fair second choice. |
| **sklearn GradientBoosting** | Same idea, but ~28× slower in my test; it's the *fallback* in the code if XGBoost isn't installed (`try/except ImportError`). |
| **LightGBM / CatBoost** | Equally valid; LightGBM faster on huge data; CatBoost handles categoricals natively (would suit `drug_name` better). XGBoost chosen for familiarity/ubiquity; at 50k rows speed differences are irrelevant. |
| **SVM** | Needs scaling, O(n²–n³) training on 50k rows, no native probabilities. |
| **k-NN** | Scaling-sensitive, slow inference, poor with mixed discrete features. |
| **Neural network / TabNet** | Overkill for 11 features; needs more data, tuning, and is less explainable. |
| **Naive Bayes** | Independence assumption is false here (age/weight/conditions correlated). |
| **Ordinal regression** | Respects class order; a valid improvement (see §7) but less standard in tooling. |

## 3. How XGBoost works (be able to explain in 60 seconds)

* **Gradient boosting:** build trees *sequentially*; each new tree is fitted to the **gradient of the loss** (≈ residual errors) of the current ensemble, and added with a small weight (**learning_rate**). The final score is the sum of all tree outputs.
* **Multi-class:** objective `multi:softprob`; per boosting round it grows **one tree per class** (K=4); the K summed scores go through **softmax** to give probabilities; loss = **multi-class log-loss** (`eval_metric="mlogloss"`).
* **What makes it "extreme":** second-order (gradient + Hessian) Taylor approximation of the loss, explicit **L1/L2 regularisation** on leaf weights and a penalty `gamma` per split, **shrinkage**, **row subsampling** and **column subsampling**, histogram/approximate split finding, parallelism, sparsity-aware handling of missing values.
* **Bias–variance:** boosting reduces **bias** (many weak learners fix residual errors); depth, learning rate, subsampling and `min_child_weight` control **variance/overfitting**.

## 4. Hyper-parameters and what each does

| Parameter | Meaning | Real-pipeline search space (`train_model.py`) | Shipped fixed value (`build_model.py`) |
|-----------|---------|-----------------------------------------------|---------------------------------------|
| `n_estimators` | number of boosting rounds (trees per class) | 100, 200, 300 | 200 |
| `max_depth` | depth of each tree = interaction order allowed | 3, 5, 7, 9 | 5 |
| `learning_rate` (eta) | shrinkage per tree; lower = more robust, needs more trees | 0.01, 0.05, 0.1, 0.2 | 0.1 |
| `subsample` | fraction of rows per tree (stochastic GB, reduces variance) | 0.7, 0.8, 0.9 | 0.8 |
| `colsample_bytree` | fraction of features per tree | 0.7, 0.8, 0.9 | 0.8 |
| `min_child_weight` | min Hessian sum in a leaf; larger = more conservative | 1, 3, 5 | default (1) |
| `random_state` | reproducibility | 42 | 42 |
| `eval_metric` | `mlogloss` | – | – |
| `use_label_encoder=False` | legacy flag; **deprecated/removed** in recent XGBoost (prints a warning) | – | – |

### `RandomizedSearchCV(n_iter=10, cv=3, scoring="accuracy", n_jobs=-1, random_state=42)`
* Search space size = 3·4·4·3·3·3 = **972** combinations; random search samples **10**; with 3-fold CV that's **30 fits** (+1 refit on the full training set with the best params).
* **Why random over grid search?** (Bergstra & Bengio 2012) with several hyper-parameters, only a few matter; random search explores more distinct values per parameter for the same budget; grid search would need 972×3 = 2,916 fits.
* **Why 10 iterations?** Time budget; honest: it's a light search, results could improve with more iterations / Bayesian optimisation (Optuna).
* **Why `cv=3`?** Keeps the search fast; 5 is more standard.
* **Why scoring=accuracy?** Simple and aligned with the headline metric; with imbalanced classes `f1_macro` / `neg_log_loss` would be better aligned with the goal (calibrated probabilities).
* The best params are printed (`random_search.best_params_`) but not stored in the artifact. (I did not run this stage, so I can't quote the winning values — run `python -m backend.train_model` and note them.)

## 5. Train / validation / test discipline

* `train_test_split(test_size=0.2, random_state=42, stratify=y)` – **stratify** keeps class proportions equal in both parts (important for rare Critical class); fixed seed ⇒ reproducible.
* Hyper-parameter tuning uses **CV inside the training set only**; the 20 % test set is touched once at the end ⇒ no tuning leakage.
* Possible improvement: group-aware split. Rows for the same drug share drug-level features; a random split can put near-identical rows in both sets (mild optimistic bias). Time-based or report-ID-based splits are cleaner.

## 6. Evaluation

### What was reported
* Real pipeline: **63.1 % accuracy** (README; 10,000 test rows), plus `classification_report` (precision/recall/F1 per class).
* Deployed synthetic model: **91.4 %** (verified, 2,400 test rows).

### Metric definitions (rehearse)
* **Accuracy** = correct / total. Misleading with imbalance.
* **Precision (class c)** = TP / (TP + FP) – "when I say Critical, how often am I right?"
* **Recall (class c)** = TP / (TP + FN) – "of all true Critical, how many did I catch?" – *in drug safety recall on High/Critical matters most.*
* **F1** = harmonic mean; **macro-F1** averages classes equally; **weighted-F1** weights by support.
* **Confusion matrix** shows which classes are confused (adjacent ordinal classes should be the common errors).
* **Log-loss** evaluates probability quality; **calibration** (reliability curves, `CalibratedClassifierCV`) checks "70 % means 70 %". XGBoost probabilities are usually decent but not guaranteed calibrated; the UI's "confidence %" is just max-softmax, **not** a calibrated probability of clinical harm.
* **Baselines:** chance for 4 balanced classes = 25 %; majority-class baseline depends on the class mix (28.7 % on the synthetic data; unknown for real FAERS – check `value_counts` printed by `build_dataset.py`).

### Interpreting 63 %
"Meaningfully above baseline but far from clinical-grade. Expected given noisy voluntary-report labels and features that are mostly drug-level. That's why the model is advisory and why I don't let it override a rule."

### Overfitting checks I'd do
Compare train vs test accuracy; learning curves; early stopping with a validation set (`early_stopping_rounds`); feature importance sanity; permutation importance / SHAP.

## 7. Known modelling issues and how to improve (be proactive)

| Issue | Impact | Improvement |
|-------|--------|-------------|
| Train/serve skew (6 of 11 features defaulted at inference) | predictions don't reflect real inputs; Warfarin flips High→Moderate | store per-drug SIDER stats and look them up at predict time; drop event-level features from the *serving* model (train a "pre-event" model using only drug + patient features) |
| Label = report seriousness, no denominator | not an incidence/risk estimate | use disproportionality methods (PRR, ROR, IC) on FAERS; or EHR cohorts |
| `LabelEncoder` for drug imposes arbitrary order | trees cope, but encoding is unnatural | one-hot, target encoding with CV, CatBoost, or XGBoost `enable_categorical=True` |
| Ordinal target treated as nominal | ignores class order | ordinal classification / regression on `risk_score`; cost-sensitive loss |
| Hand-chosen target weights & severity weights | arbitrary | sensitivity analysis; clinician-validated scale |
| Accuracy as the tuning metric | ignores class balance | `f1_macro`, `neg_log_loss`; class weights / `sample_weight` |
| Small search (10 iterations) | likely sub-optimal | Optuna, more iterations, early stopping |
| Pickle persistence | version fragility, code-exec risk | `model.save_model()` JSON/UBJ + separate JSON for encoder, or ONNX / skops |
| No calibration, no uncertainty | "confidence" overstated | `CalibratedClassifierCV`, conformal prediction |
| No explainability | black box | SHAP `TreeExplainer` per-prediction contributions in the UI |
| Only 10 drugs | encoder fails on others (graceful) | scale the pipeline to the full FAERS drug list |
| No automated tests/CI/MLflow | reproducibility | pytest for rules & API, GitHub Actions, MLflow/W&B experiment tracking |

## 8. Model artifact (what's inside `drug_risk_model.pkl`)

```python
{
  "model":         XGBClassifier,         # fitted booster(s)
  "label_encoder": LabelEncoder,          # 10 drug names -> 0..9 (classes_ order = build order in the synthetic path, alphabetical in the real path)
  "feature_cols":  [... 11 names ...],    # documentation of column order
  "risk_labels":   ["Low Risk","Moderate","High Risk","Critical"],
  "accuracy":      0.91375                # test accuracy (printed at load: "[ML] Model loaded (accuracy: …)")
}
```
Subtle point: in the synthetic path the `LabelEncoder` is built by assigning `le.classes_` manually (not via `.fit`), in the same order used to produce the integer ids in the training rows (`DRUG_NAMES` order, **not** alphabetical). It works because `transform` maps by lookup; I verified predictions are consistent. In the real path `fit_transform` sorts alphabetically. Both are self-consistent within their own artifact — never mix a label encoder from one pipeline with the model from another.

## 9. Inference behaviour (verified)
* Latency: microseconds-to-milliseconds per drug (single row `predict` + `predict_proba`); the pickle loads once (singleton).
* Unknown drug ⇒ `available:false`; the UI shows "N/A" and hides the chart if none are available.
* Probability output example: Metformin, 72-year-old male, 80 kg, 1 current med ⇒ Low Risk 97.2 %, Moderate 2.7 %.

## 10. Rules vs ML — verified disagreement (great demo story)

| Patient | Drug | Rules | ML |
|---------|------|-------|----|
| 72 M, eGFR 28, diabetic, on Ibuprofen | Metformin | **Contraindicated / Critical** (eGFR < 30) | Low Risk (97.2 %) |
| same, penicillin allergy | Amoxicillin | **Contraindicated** (allergy) | Low Risk (94.1 %) |
| same | Warfarin | Suitable / Low (no rule fires) | Moderate (58.4 %) |

Explain: ML predicts *how serious a reported adverse event would typically be for this drug in a person like this*; it knows nothing about kidney function. The rules know the lab values. This is **why the product is hybrid** and why ML can't overrule rules. An interviewer may call this a flaw ("your ML contradicts your rules") — answer: "Correct, and by design they answer different questions; the next step is to feed lab values into the model or to explicitly reconcile the two signals in the UI."
