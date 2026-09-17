"""
FitScore - deployment dashboard for the pre-trained candidate fit model.

A recruiter uploads a CSV of candidates; this app loads the already-trained
models/logistic_regression.pkl (no retraining, no rule engine, no LLM calls),
runs .predict()/.predict_proba() per candidate, and shows each prediction
alongside a per-candidate breakdown derived from the model's own learned
coefficients. Results and Shortlist/Pass decisions persist in SQLite so a
reload shows the same last-scored batch rather than an empty page.

If the model file fails to load, every scoring path fails loudly (500 /
error banner) instead of silently fabricating a prediction.
"""

import io
import json
import os
import pickle
import re
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, g, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logistic_regression.pkl")
DB_PATH = os.path.join(PROJECT_ROOT, "fitscore.db")
SAMPLE_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "sample_upload.csv")

REQUIRED_COLUMNS = ["Experience", "Qualifications", "Skills", "Prior Job Title"]

# ColumnTransformer prefixes (see src/train_model.py build_pipeline) mapped to
# the 4 human-readable feature names used throughout the UI.
PREFIX_TO_FEATURE = {
    "experience": "Experience",
    "qualifications": "Qualifications",
    "prior_title": "Prior Job Title",
    "skills": "Skills",
}
FEATURE_ORDER = ["Experience", "Qualifications", "Skills", "Prior Job Title"]

BADGE_CLASS = {
    "Strong Fit": "badge-green",
    "Needs Review": "badge-amber",
    "Likely Not a Fit": "badge-red",
}

app = Flask(__name__)

_model_pipeline = None
_model_load_error = None
try:
    with open(MODEL_PATH, "rb") as f:
        _model_pipeline = pickle.load(f)
except Exception as e:
    _model_load_error = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uploaded_at TEXT NOT NULL,
            filename TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            experience TEXT,
            qualifications TEXT,
            skills TEXT,
            prior_job_title TEXT,
            predicted_label TEXT NOT NULL,
            prob_strong_fit REAL,
            prob_needs_review REAL,
            prob_not_a_fit REAL,
            breakdown_json TEXT NOT NULL,
            decision TEXT NOT NULL DEFAULT 'Pending',
            FOREIGN KEY (batch_id) REFERENCES batches (id)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Scoring + coefficient-based explanation
# ---------------------------------------------------------------------------


def parse_experience_years(experience_str):
    match = re.match(r"^\s*(\d+)", str(experience_str))
    if not match:
        raise ValueError(f"Could not parse a number of years from Experience value: {experience_str!r}")
    return int(match.group(1))


def score_candidates(raw_df):
    """
    Run the pre-trained pipeline on raw candidate rows and, for every row,
    decompose the predicted class's logit into a contribution per raw
    feature using the model's own coef_ matrix - not a templated or generic
    explanation, a real per-row weighted sum grouped by feature.
    """
    df = raw_df.copy()
    df["Experience_Years"] = df["Experience"].apply(parse_experience_years)
    X = df[["Experience_Years", "Qualifications", "Skills", "Prior Job Title"]]

    preprocessor = _model_pipeline.named_steps["preprocess"]
    clf = _model_pipeline.named_steps["clf"]

    encoded = preprocessor.transform(X)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    feature_names = preprocessor.get_feature_names_out()
    feature_groups = [PREFIX_TO_FEATURE[name.split("__")[0]] for name in feature_names]

    probabilities = clf.predict_proba(encoded)
    class_list = list(clf.classes_)
    predictions = clf.predict(encoded)

    results = []
    for row_idx in range(len(df)):
        predicted_label = predictions[row_idx]
        class_idx = class_list.index(predicted_label)
        coefs_for_class = clf.coef_[class_idx]
        row_contributions = coefs_for_class * encoded[row_idx]

        breakdown = {feature: 0.0 for feature in FEATURE_ORDER}
        for group, contribution in zip(feature_groups, row_contributions):
            breakdown[group] += float(contribution)

        top_feature = max(breakdown, key=lambda f: abs(breakdown[f]))
        top_value = breakdown[top_feature]

        probs = {cls: float(p) for cls, p in zip(class_list, probabilities[row_idx])}

        results.append(
            {
                "predicted_label": predicted_label,
                "probabilities": probs,
                "breakdown": breakdown,
                "top_feature": top_feature,
                "top_direction": "positive" if top_value >= 0 else "negative",
                "top_value": top_value,
            }
        )
    return results


def top_factor_sentence(result):
    feature = result["top_feature"]
    label = result["predicted_label"]
    if result["top_direction"] == "positive":
        return f"{feature} is the strongest factor supporting this “{label}” prediction."
    return f"{feature} is the strongest factor working against this “{label}” prediction — other factors outweighed it."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def dashboard():
    if _model_load_error:
        return render_template("dashboard.html", model_error=_model_load_error, last_run=None,
                                stats=None, candidates=[])

    db = get_db()
    batch = db.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 1").fetchone()

    if batch is None:
        return render_template("dashboard.html", model_error=None, last_run=None, stats=None, candidates=[])

    rows = db.execute(
        "SELECT * FROM candidates WHERE batch_id = ? ORDER BY id ASC", (batch["id"],)
    ).fetchall()

    candidates = []
    counts = {"Strong Fit": 0, "Needs Review": 0, "Likely Not a Fit": 0}
    for row in rows:
        breakdown = json.loads(row["breakdown_json"])
        counts[row["predicted_label"]] = counts.get(row["predicted_label"], 0) + 1
        candidates.append(
            {
                "row_id": row["id"],
                "candidate_id": row["candidate_id"],
                "experience": row["experience"],
                "qualifications": row["qualifications"],
                "skills": row["skills"],
                "prior_job_title": row["prior_job_title"],
                "predicted_label": row["predicted_label"],
                "badge_class": BADGE_CLASS.get(row["predicted_label"], "badge-plain"),
                "probabilities": {
                    "Strong Fit": row["prob_strong_fit"],
                    "Needs Review": row["prob_needs_review"],
                    "Likely Not a Fit": row["prob_not_a_fit"],
                },
                "breakdown": breakdown["breakdown"],
                "top_factor_sentence": breakdown["top_factor_sentence"],
                "decision": row["decision"],
            }
        )

    stats = {
        "total": len(candidates),
        "strong_fit": counts.get("Strong Fit", 0),
        "needs_review": counts.get("Needs Review", 0),
        "not_a_fit": counts.get("Likely Not a Fit", 0),
    }

    return render_template(
        "dashboard.html",
        model_error=None,
        last_run=batch["uploaded_at"],
        stats=stats,
        candidates=candidates,
        feature_order=FEATURE_ORDER,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if _model_load_error:
        return jsonify({"error": f"Model failed to load, cannot score candidates: {_model_load_error}"}), 503

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        raw_df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8", errors="replace")))
    except Exception as e:
        return jsonify({"error": f"Could not read that file as a CSV: {e}"}), 400

    missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        return jsonify({"error": f"CSV is missing required column(s): {', '.join(missing)}"}), 400

    if "Candidate ID" in raw_df.columns:
        candidate_ids = raw_df["Candidate ID"].astype(str).tolist()
    else:
        candidate_ids = [f"CAND-{i + 1:03d}" for i in range(len(raw_df))]

    try:
        results = score_candidates(raw_df)
    except Exception as e:
        return jsonify({"error": f"Scoring failed: {type(e).__name__}: {e}"}), 400

    db = get_db()
    db.execute("DELETE FROM candidates")
    db.execute("DELETE FROM batches")
    cursor = db.execute(
        "INSERT INTO batches (uploaded_at, filename) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file.filename),
    )
    batch_id = cursor.lastrowid

    for cand_id, (_, row), result in zip(candidate_ids, raw_df.iterrows(), results):
        payload = {
            "breakdown": result["breakdown"],
            "top_factor_sentence": top_factor_sentence(result),
        }
        db.execute(
            """
            INSERT INTO candidates (
                batch_id, candidate_id, experience, qualifications, skills, prior_job_title,
                predicted_label, prob_strong_fit, prob_needs_review, prob_not_a_fit,
                breakdown_json, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pending')
            """,
            (
                batch_id,
                cand_id,
                row["Experience"],
                row["Qualifications"],
                row["Skills"],
                row["Prior Job Title"],
                result["predicted_label"],
                result["probabilities"].get("Strong Fit"),
                result["probabilities"].get("Needs Review"),
                result["probabilities"].get("Likely Not a Fit"),
                json.dumps(payload),
            ),
        )
    db.commit()

    return jsonify({"ok": True, "scored": len(raw_df)})


@app.route("/decision/<int:candidate_row_id>", methods=["POST"])
def set_decision(candidate_row_id):
    data = request.get_json(silent=True) or {}
    decision = data.get("decision")
    if decision not in ("Shortlist", "Pass", "Pending"):
        return jsonify({"error": "decision must be Shortlist, Pass, or Pending"}), 400

    db = get_db()
    db.execute("UPDATE candidates SET decision = ? WHERE id = ?", (decision, candidate_row_id))
    db.commit()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True)
