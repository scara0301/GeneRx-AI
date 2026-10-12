# 11b — Interview Q&A, part 2: modelling, evaluation, ML theory

← [Part 1](11-interview-qa.md) · [Part 3](11c-interview-qa-engineering.md) →

Numbers marked ✅ were reproduced by running the code on the synthetic pipeline; the 63.1 % figure comes
from the README (real pipeline, not re-run by me).

---

## E. Algorithm choice

**Q71. Why XGBoost?**
Tabular data with non-linear interactions and mixed scales; no scaling needed; regularised gradient boosting is usually top-tier on such data; fast; native `predict_proba`; sparsity-aware. ✅ In my bake-off: LogReg 56 %, Decision Tree 72 %, RF 90.7 %, sklearn-GBM 91.3 %, XGBoost 91.4 % (0.7 s vs 20 s for sklearn-GBM).

**Q72. Why not a neural network?**
11 features, tens of thousands of rows: trees match or beat NNs on small tabular data, train in seconds, need no scaling/embedding design, and are easier to explain (feature importance/SHAP). NNs would add tuning cost and opacity with no expected gain.

**Q73. Why not Random Forest?**
It's a perfectly good choice and scored within 0.5 pt of XGBoost in my test ✅. I chose boosting for typically better accuracy/probability quality on harder data, built-in regularisation and faster training than sklearn's GBM; RF is a sensible baseline/fallback.

**Q74. Why not logistic regression?**
It's my baseline and it did poorly (56 % ✅) because risk depends on interactions/thresholds (e.g. age>75 AND ≥5 drugs). I'd still use LR to sanity-check and for interpretability.

**Q75. Bagging vs boosting?**
Bagging (RF) trains independent trees on bootstrap samples and averages → reduces variance. Boosting trains trees sequentially on the previous errors → reduces bias (and with regularisation controls variance).

**Q76. Explain gradient boosting.**
Start with a constant prediction; repeatedly fit a small tree to the negative gradient of the loss w.r.t. current predictions (the "pseudo-residuals"), add it scaled by the learning rate. Final prediction = sum of trees.

**Q77. What's different in XGBoost?**
Second-order optimisation (uses gradient *and* Hessian), explicit L1/L2 regularisation of leaf weights + split penalty γ, shrinkage, row/column subsampling, efficient histogram split finding, sparsity-aware missing value handling, parallel/cached implementation.

**Q78. How does XGBoost do multi-class?**
`multi:softprob`: each round builds one tree per class; class scores pass through softmax; loss is multi-class log-loss (`mlogloss`); `predict` = argmax, `predict_proba` = the softmax vector.

**Q79. What's the loss and how is it related to the probabilities you show?**
Cross-entropy/log-loss; minimising it encourages calibrated-ish probabilities, but boosted trees are not guaranteed calibrated — hence "confidence" in the UI is just the max softmax, not a clinical probability.

**Q80. Which hyper-parameters matter most?**
Learning rate × n_estimators (trade-off), max_depth (interaction order/complexity), subsample & colsample (variance), min_child_weight / gamma / lambda (regularisation).

**Q81. How did you tune?**
`RandomizedSearchCV` over 972 combos (3·4·4·3·3·3), 10 sampled, 3-fold CV, accuracy scoring ⇒ 30 fits + refit. Random search beats grid for the same budget (Bergstra & Bengio). The deployed (synthetic) model uses fixed sensible defaults: 200 trees, depth 5, lr 0.1, subsample/colsample 0.8.

**Q82. Why only 10 iterations?**
Time/compute budget for a student-scale project; it's a light search — more iterations, Bayesian optimisation (Optuna) and early stopping would be next.

**Q83. How do you prevent overfitting in XGBoost?**
Lower depth, smaller learning rate with more trees, subsampling, `min_child_weight`, L2/γ regularisation, early stopping on a validation set, and CV. Evidence of overfit: large gap between train and validation metrics.

**Q84. What does `colsample_bytree=0.8` do?**
Each tree sees a random 80 % of features → decorrelates trees and reduces variance, like RF's feature subsampling.

**Q85. What's `min_child_weight`?**
Minimum sum of Hessian weights required in a child; higher values prevent splits on tiny/uncertain groups → more conservative.

**Q86. Did you use early stopping?**
No. Fixed `n_estimators` chosen by search. Early stopping with a held-out set is an easy improvement.

**Q87. Fallback model?**
If XGBoost isn't installed (`ImportError`) both training scripts fall back to sklearn `GradientBoostingClassifier(n_estimators=200, max_depth=5)`; the artifact format is unchanged so the server doesn't care.

**Q88. Why is `use_label_encoder=False` in your code?**
It silenced an old warning in XGBoost 1.x; in current versions the argument is deprecated/ignored (prints a warning). I'd remove it.

---

## F. Training & evaluation

**Q89. How did you split data?**
80/20 `train_test_split`, `random_state=42`, `stratify=y`; tuning via 3-fold CV inside the training 80 %; test set used once.

**Q90. Why stratify?**
Preserves class proportions in both splits — essential when a class (Critical) is rarer.

**Q91. What accuracy did you get and is it good?**
Real FAERS pipeline: 63.1 % (README) on 4 classes. Deployed synthetic model: 91.4 % ✅. 63 % is above baseline but modest; it reflects noisy labels and weak, mostly drug-level features. 91 % is circular (rediscovering the generator).

**Q92. What's the baseline?**
Majority-class ✅ 28.7 % on the synthetic data; chance for 4 balanced classes is 25 %. For real FAERS read `value_counts` from `build_dataset.py` — I'd always quote the baseline next to accuracy.

**Q93. Why is accuracy a poor metric here?**
Class imbalance, unequal costs (missing a Critical case is worse than over-warning), ordinal classes. Better: macro-F1, per-class recall for High/Critical, confusion matrix, log-loss, calibration, and cost-weighted evaluation.

**Q94. Explain precision, recall, F1.**
Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean. For safety, recall on High/Critical is prioritised.

**Q95. What does the classification report show?**
Per-class precision/recall/F1/support. In `train_model.py` it uses `labels=present_labels` to avoid errors if a class is absent from the test set.

**Q96. Did you use a confusion matrix?**
Not in code. I'd add it — for an ordinal target most errors should be between adjacent classes; errors between Low and Critical would be alarming.

**Q97. Is your 63 % overfit?**
Check train vs test accuracy; with CV-tuned moderate-depth trees and a held-out test set it's unlikely to be heavily overfit. But the random split can leak drug-level near-duplicates, so the test estimate may be slightly optimistic.

**Q98. How would you improve accuracy?**
Better features (indication, report-level SOC codes, time), patient-level data, remove drug-level redundancy, class weights, ordinal modelling, more tuning, ensembling (stack XGB + LightGBM + CatBoost), and reduce label noise via dedupe/cleaning.

**Q99. What's cross-validation and why 3 folds?**
Split training data into k folds, train on k−1, validate on 1, rotate; average scores for a stable estimate. k=3 for speed; 5 or 10 is more common.

**Q100. What's data leakage and did you have it?**
Information from outside the training context leaking into features/labels (e.g., using the target or future info). I avoided it for the seriousness flags; residual concerns: report-level features not available at serving time and drug-level aggregates (not used). Another: tuning on the test set — avoided.

**Q101. Are the predicted probabilities calibrated?**
Not checked. I'd plot reliability curves and apply `CalibratedClassifierCV` (isotonic/Platt) on a validation set; compute Brier score / ECE.

**Q102. How do you interpret the model?**
`feature_importances_` (gain), permutation importance, partial dependence, SHAP values for per-prediction explanations. SHAP would fit the UI ("why High Risk?").

**Q103. What does the "confidence" in the UI mean?**
`max(predict_proba)·100` — the model's top class probability, not a clinical certainty. Present it as "model confidence", not "probability of harm".

**Q104. What's `score` in the output?**
`category / 3` → a normalised 0–1 severity index (0, 0.33, 0.67, 1). Convenience for UIs; it's not a calibrated risk.

**Q105. What happens for an unseen drug?**
`LabelEncoder.transform` raises `ValueError` → caught → `available:false` and the label "'X' not in training data". The rules engine still answers (if the drug is in the catalog).

**Q106. Why does the model give Warfarin "Moderate" when you designed it as "High"?** ✅
Train/serve skew: at inference the SIDER features are 0 (and event features are defaults), values the model rarely/never saw for Warfarin. With the true SIDER values (count 5, mean .16, max .35) it predicts High. Fix: look up per-drug SIDER stats at predict time.

**Q107. What is train/serve skew and how do you detect it?**
A mismatch between feature distributions/definitions at training and serving. Detect with shared feature code, logging serving features and comparing distributions (PSI, KS test), and unit tests that run the same row through training and serving paths. Prevent with a feature store or a single feature function.

**Q108. How would you monitor the model in production?**
Log inputs/predictions (no PHI), drift metrics on features and predictions, performance against delayed ground truth, alert thresholds, shadow/canary releases before swapping models.

**Q109. How would you retrain?**
Scheduled pipeline (Airflow/GitHub Actions) fetching new FAERS quarters, running data validation (Great Expectations), training, comparing against the champion on a fixed benchmark, then promoting through a registry.

**Q110. How do you make the pipeline reproducible?**
Seeds, pinned dependencies, data versioning (DVC/hashes), saved hyper-parameters in the artifact (`best_params_` currently only printed), environment captured by the Dockerfile.

---

## G. ML theory quick-fire

**Q111. Bias–variance trade-off?** Bias = error from overly simple assumptions (underfit); variance = sensitivity to training data (overfit). Boosting lowers bias; subsampling/regularisation/shallow trees lower variance.

**Q112. L1 vs L2 regularisation?** L1 (lasso) adds |w| penalty → sparse weights; L2 (ridge) adds w² → shrinks weights smoothly. XGBoost: `alpha` (L1) and `lambda` (L2) on leaf weights.

**Q113. Why do trees not need feature scaling?** Splits depend only on ordering of values.

**Q114. How do decision trees choose splits?** Maximise impurity reduction (Gini/entropy) or, in XGBoost, a gain based on gradient/Hessian sums and regularisation: Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ.

**Q115. What is Gini impurity?** 1 − Σ pₖ²; 0 when a node is pure.

**Q116. What's one-vs-rest vs softmax multi-class?** OvR trains K binary models; softmax models all classes jointly with probabilities summing to 1 (XGBoost `multi:softprob`).

**Q117. Imbalanced data techniques?** Class weights/`sample_weight`, resampling (SMOTE, undersampling), threshold moving, appropriate metrics (PR-AUC, macro-F1), cost-sensitive learning.

**Q118. ROC-AUC for multi-class?** One-vs-rest or one-vs-one macro averages (`roc_auc_score(..., multi_class="ovr")`).

**Q119. What is a PR curve and when to prefer it?** Precision-recall; preferred when positives are rare because ROC can look good with many true negatives.

**Q120. Pickle vs joblib vs native save?** Pickle works for arbitrary Python objects (insecure, version-fragile); joblib is optimised for numpy arrays; native `Booster.save_model` (JSON/UBJ) is portable across XGBoost versions and languages; ONNX for cross-framework deployment.

**Q121. What is MLOps in this project's terms?** Today: scripted pipeline, Docker, fixed seeds. Missing: experiment tracking, model registry, CI/CD, data validation, monitoring.

**Q122. What's the curse of dimensionality — relevant here?** Not really: 11 features, ~50k rows.

**Q123. Would more data help?** More *diverse, patient-level* data would; more FAERS rows of the same drug-level features would saturate quickly because features are mostly constant per drug.

**Q124. How would you handle a 4th-class label noise?** Clean/dedupe reports, use noise-robust losses, label smoothing, or relabel by clinicians on a sample.

**Q125. Why `eval_metric="mlogloss"`?** The appropriate multi-class log-loss for XGBoost's softprob objective; used for monitoring/early stopping (not accuracy).

**Q126. How is `predict_proba` computed from tree outputs?** Sum each class's trees' leaf values (plus base score), then softmax across the 4 class scores.

**Q127. What is SHAP?** Shapley-value-based attributions that decompose a prediction into additive per-feature contributions; `shap.TreeExplainer` is exact and fast for XGBoost.

**Q128. Fairness considerations?** Sex/age/weight are inputs; FAERS under-represents some groups and over-represents US reports; performance should be evaluated per subgroup; the model must not be used to deny treatment.

**Q129. Could the ML be harmful?** If users over-trust a 97 % "Low Risk" for a drug the rules flag as contraindicated. Mitigation: ML is advisory, labelled, and verdict comes from rules; show ML only as a secondary "population signal"; add explicit disagreement warning.

**Q130. How would you evaluate the *whole system* (rules + ML), not just the model?** Build a gold-standard test set of patient–drug cases labelled by clinicians; measure sensitivity/specificity of "avoid/contraindicated" alerts; alert-fatigue rate; user studies on decision quality; compare with existing tools (e.g., Lexicomp, Medscape interaction checker).
