
# Candidate-Fit Scoring

An end-to-end MLDLC (Machine Learning Development Life Cycle) build: a candidate-job fit prediction model for senior AI Product Manager hiring, built solo via spec-based vibe coding with Claude Code.

## Problem

A workforce solutions company's European enterprise client needs faster, more consistent shortlisting of senior (10+ years) candidates for AI Product Manager roles. Recruiters currently review candidate pools manually — slow and inconsistent at volume.

This project is a fast, vibe-coded prototype demonstrating the full ML methodology end to end, intended to validate the approach before any larger production investment — not a production system itself.

## What's in this repo

```
candidate-fit-scoring/
├── src/
│   ├── pipeline.py              # Data loading, filtering, EDA
│   ├── generate_candidates.py   # Synthetic candidate generation + rule-based labeling
│   ├── check_requirement7.py    # JD Requirement 7 verification
│   └── train_model.py           # Model training and evaluation
├── data/
│   ├── filtered_jobs.csv        # 2,553 filtered European AI/Data/PM job postings
│   ├── labeled_candidates.csv   # 250 synthetic candidates with computed Fit_Label
│   └── sample_upload*.csv       # Demo upload files (30/150/250/600 candidates)
├── models/
│   ├── logistic_regression.pkl  # Recommended model
│   └── decision_tree.pkl        # Alternative model
├── reports/
│   ├── eda_findings.md
│   ├── labeling_summary.md
│   └── model_evaluation.md
├── reference/
│   └── jd_ai_product_manager.md # The 7-requirement scoring rubric
├── app.py                       # Flask deployment
├── templates/ , static/         # Dashboard UI
└── requirements.txt
```

## Methodology (MLDLC)

1. **Frame the problem** — candidate-job fit prediction for AI Product Manager, European client
2. **Gather data** — Kaggle job-postings dataset (CC0, 1.6M+ rows), filtered to 2,553 relevant rows
3. **Preprocess** — chunked loading, PII/discriminatory field exclusion, country/role filtering
4. **EDA** — pandas-based fallback analysis (ydata-profiling was incompatible with the environment's Python version)
5. **Feature engineering** — JD authored as a 7-point rubric; 250 synthetic candidates generated and labeled against it
6. **Model training & evaluation** — Logistic Regression vs. Decision Tree, stratified 80/20 split
7. **Deployment** — Flask dashboard with batch upload, per-candidate explanations, human-in-the-loop review
8. **Testing** — held-out evaluation, verified failure-mode handling (missing model file → explicit error, never a fabricated score)
9. **Optimize** — ongoing

## Key results

| Model | ROC-AUC | Strong Fit Recall | Macro F1 |
|---|---|---|---|
| **Logistic Regression (recommended)** | 0.941 | 0.77 | 0.84 |
| Decision Tree (depth=5) | 0.872 | 0.85 | 0.73 |

Logistic Regression was chosen for its stronger overall performance; the trade-off against the Decision Tree's marginally better explainability is documented in `reports/model_evaluation.md`.

## Responsible AI design

- **Human-in-the-loop by construction** — the tool recommends; a recruiter always makes the final call. No candidate is ever auto-rejected.
- **Discriminatory field exclusion** — a gender-based hiring-preference field present in the source data was dropped before any processing touched it.
- **Explainability-first model choice** — Logistic Regression's coefficient-based reasoning is directly inspectable, consistent with EU AI Act expectations for high-risk hiring systems.
- **No black-box fallback** — if the model file fails to load, the app shows a clear error state rather than fabricating a prediction.

## Honest limitations

- All data is proxy/synthetic — no real company or client data was used or available for this exercise.
- The candidate pool is synthetic with a rule-derived label, not real hiring outcomes — this is a methodology demonstration, not a model trained on production data.
- Sample size (250 candidates, ~64 in the smallest class) supports the classical models used here; a production deployment would need substantially more real, labeled data.
- Formal beta testing with real recruiters has not been conducted — held-out evaluation and failure-mode testing are complete; real user validation is a stated next step.

## Running it

```bash
cd candidate-fit-scoring
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`, click **Upload & Score Candidates**, and select any file from `data/sample_upload*.csv`.

## Full documentation

A complete project summary — including the cost/value business case, positioning against existing tools, and detailed methodology write-up — is available separately as `final-project-summary.docx`. The full set of prompts used to build this with Claude Code is in `candidate-fit-scoring-prompts.pdf`.
