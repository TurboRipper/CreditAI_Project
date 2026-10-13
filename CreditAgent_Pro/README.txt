CREDITAGENT PRO — JAVASCRIPT UI + FLASK + LANGGRAPH
================================================
A complete local research upgrade. No Docker, API key, GPU or JavaScript build step.
Use Python 3.11 or 3.12 (tested on 3.12). Dependency versions are pinned. Internet is needed only to install Python packages.
The visual design uses lavender, cream, rounded panels and generous spacing,
inspired by https://www.lemonsqueezy.com/; no copied logos or commercial assets.

1. FILES AND SETUP
------------------
Keep your old project as a backup. Create a new folder called CreditAgent_Pro.
Rename each downloadable text file and place it in this structure:

CreditAgent_Pro/app.py                  <- CreditAgent_01_app_py.txt
CreditAgent_Pro/credit_engine.py        <- CreditAgent_02_credit_engine_py.txt
CreditAgent_Pro/templates/index.html    <- CreditAgent_03_index_html.txt
CreditAgent_Pro/static/app.js           <- CreditAgent_04_app_js.txt
CreditAgent_Pro/static/style.css        <- CreditAgent_05_style_css.txt
CreditAgent_Pro/requirements.txt        <- CreditAgent_06_requirements.txt
CreditAgent_Pro/README.txt              <- CreditAgent_07_README.txt
CreditAgent_Pro/german.data             <- your existing original dataset

Enable File Explorer > View > Show > File name extensions.
An app.py.txt file will not work. Save text as UTF-8.

OPTIONAL EASIER INSTALLATION:
Rename CreditAgent_Installer_py.txt to install_creditagent.py and run:
  python install_creditagent.py
It creates all seven application files in a new CreditAgent_Pro folder beside
itself. It will refuse to overwrite existing files. Then copy german.data into
that folder. All files remain editable; the installer installs no packages.

From PowerShell, inside CreditAgent_Pro:
  py -3.12 -m venv credit_env
  .\credit_env\Scripts\python.exe -m pip install -r requirements.txt
  .\credit_env\Scripts\python.exe app.py

If Python 3.11 is installed instead, replace -3.12 with -3.11.
You do not need to activate the environment or change PowerShell policies.
Open Chrome: http://127.0.0.1:5000
Keep the terminal open. The UI appears while three models train in a background
thread. Wait for "Models ready". Missing data or startup failures appear in UI.
Startup trains again; assessments and reviews persist in runtime/credit_agent.db.
To reopen tomorrow, just run .\credit_env\Scripts\python.exe app.py again.
Stop with Ctrl+C. After edits, restart and press Ctrl+F5 in Chrome.

If port 5000 is already in use:
  $env:PORT="5002"
  .\credit_env\Scripts\python.exe app.py
Then open http://127.0.0.1:5002.

Do not combine only one new file with old files: replace the complete set.
The upgraded app stores data in runtime/, separate from the older root database.
Old records are not automatically migrated; preserve your previous folder.

2. INCLUDED FEATURES
--------------------
- New assessment: 17 validated inputs, six LangGraph workflow nodes, probabilities,
  cost-sensitive risk label, model version, ID, time and recommendation.
- SHAP: genuine local contributions from the selected fitted estimator using 80
  training reference rows. One-hot values aggregate to the 17 original fields.
  Top eight shown in UI; all 17 included in PDF. If SHAP fails, display unavailable
  rather than substitute rule-based explanations.
- What-if: change amount, duration, savings or installment rate. Original stays
  unchanged. Simulations create no records and do not affect monitoring.
- Human review: name, action, reason and explicit agreement with the AI risk label.
  Approve/decline require a human selection. Every review is appended with time;
  prior reviews remain visible in history and reports.
- Application history: latest 100 records, local search, reopen, review and PDF.
- Fairness: held-out group accuracy, recall, false-positive and false-negative
  rates, maximum group gaps, sample counts and small-group flags.
- Monitoring: saved assessment count, pending first review, higher-risk share,
  average assigned-class probability, average latency and reviewer disagreement.
- Drift: population stability index (PSI), training baseline vs latest 200 saved
  applications of the current model. Requires at least 30. No per-person fairness
  or drift verdict. What-if scenarios are excluded.
- Batch CSV: download exact template, fill up to 100 rows, upload, inspect per-row
  errors, export results, reopen successful applications in history.
- PDF: all inputs, risk probabilities, threshold, model version, all SHAP values,
  agent trace and full reviewer history. Long reviews wrap and paginate.
- MLflow: local SQLite tracking of training metrics, model comparisons, threshold,
  dataset hash, version, assessment probabilities and latency. Monitoring shows
  actual tracking status and any failure. No separate MLflow service is required.

3. MLFLOW UI (OPTIONAL)
----------------------
The app writes MLflow experiments even when its dashboard is not open.
In another PowerShell window, in CreditAgent_Pro:
  .\credit_env\Scripts\mlflow.exe ui --backend-store-uri sqlite:///runtime/mlflow.db --port 5001
Open http://127.0.0.1:5001
If tracking says unavailable, read the message in Monitoring, check installation,
then restart. Failed logs are not silently presented as successful.

4. INPUT / PROCESS / OUTPUT
--------------------------
Input: checking account, duration, history, purpose, credit amount, savings,
employment duration, installment rate, guarantor, residence duration, property,
other installment plans, housing, existing credits, job, dependents, telephone.
Numeric form ranges describe this historical demo domain, not general lending.
Categorical options use the German Credit codes (A11, A12, etc.).

Process: JavaScript form -> Flask JSON API -> validate -> preprocess -> predict
-> cost analysis -> SHAP explanation -> recommendation -> human review.
SQLite stores real assessments and human reviews. MLflow records experiments.
Fairness auditing runs offline on held-out data; monitoring examines saved cases.
These are deterministic LangGraph workflow agents, not autonomous LLM agents.

Output: higher/lower risk, bad/good probability, assigned-class probability,
SHAP feature contributions, cost threshold, recommendation, trace, version,
assessment ID, human review record and downloadable report.

5. RESEARCH METHODS AND HONEST INTERPRETATION
--------------------------------------------
German Credit: historical 1,000-row dataset, original 21-column german.data.
Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Amounts are historical Deutsche Marks (DM), not rupees. Simply relabeling the
currency would make the model input incorrect.

Splits: 60% training, 20% validation, 20% held-out testing, stratified, seed 42.
Three-fold training cross-validation selects model/hyperparameters by ROC-AUC.
Candidates: Logistic Regression, Random Forest and Gradient Boosting.
Compact CPU search: four configurations per candidate. Test data never chooses
model, hyperparameters or threshold. Validation chooses a threshold between
0.01 and 0.99 minimizing false positives + 5 * false negatives; ties use F1.
The 5:1 cost is a research assumption, not an actual monetary lending loss.
The model is deliberately not selected or tuned to inflate displayed accuracy.
CV model metrics use each classifier's default decision rule; displayed held-out
metrics use the cost-selected threshold. They therefore answer different questions.

SHAP linear model output is in log-odds, NOT percentage points. Tree SHAP output
here is in probability points. Units are visible. Contributions use transformed
features and independent background masking, which can break correlations;
explanations describe model behavior rather than causal relationships.
Reference: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
MLflow: https://mlflow.org/docs/latest/ml/tracking/

Assigned-class probability can be below 50% when the cost-sensitive threshold
is below 0.5. It is not a calibrated guarantee. Calibration is not implemented.

Protected attributes are excluded from prediction but used in offline evaluation.
This does not eliminate proxy bias. Recorded sex is inferred from historical
combined personal-status codes, not self-identified gender. Groups under 30
cases or under 10 examples in either outcome class are flagged. Zero-denominator
rates are N/A. Group gaps have sampling uncertainty; no significance test or
fairness certification is claimed. This dataset cannot establish fairness or
performance on current Indian applicants.

Drift alert cutoffs: PSI >= 0.10 Watch, >= 0.25 Review. Heuristics only. Monitoring
has no true repayment outcomes and therefore cannot calculate live accuracy or
prove concept drift. Human agreement is not a substitute for repayment outcomes.

This is a localhost academic prototype, not a production credit decision system.
Reviewer names are self-entered and not authenticated. SQLite logs are append-only
through the UI but not cryptographically tamper-proof. No production authentication,
role authorization, encryption-at-rest or automated loan approval is implemented.

6. DEMONSTRATION CHECKLIST
--------------------------
1. Wait for Models ready and run one assessment.
2. Inspect actual probabilities and SHAP contributions; download the PDF.
3. Compare a changed amount in What-if. History count must stay unchanged.
4. Enter reviewer name, reason and action; save. Reopen from history.
5. Review group metrics and small-sample flags in Fairness.
6. Download the batch template; one blank numeric field must produce a row error.
7. Upload 30 valid research test rows to see population drift (repeated profiles
   artificially change the distribution; do not treat these as real applications).
8. Monitoring shows MLflow status; optionally open its UI on port 5001.
9. Restart: saved assessments and review history remain.

TESTING NOTES
-------------
Validated against UCI's original dataset. Backend checks cover SHAP additivity,
missing/invalid/fractional/NaN/infinite inputs, what-if count isolation, invalid
review requests, multiple review events, PDF output, per-row CSV errors, drift
activation and cross-origin POST rejection. Dataset-specific metrics vary with
package versions. No scores are hard-coded into the interface.

Full Chromium visual testing could not run in the build environment because its
sandbox blocked browser startup. Review the layout in your local Chrome. Backend
and rendered PDF checks were completed; JavaScript syntax passed.
JavaScript DOM tests also passed for 17 inputs, numeric serialization, SHAP/result
rendering, what-if, review submission, history reopen and monitoring. These DOM
tests use recorded API fixtures and do not replace a full browser layout check.
