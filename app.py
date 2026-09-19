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
import sys
from datetime import datetime

import pandas as pd
from flask import Flask, g, jsonify, render_template, request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "logistic_regression.pkl")
# Overridable so tests can point at an isolated throwaway database instead of
# the real one the app uses at runtime.
DB_PATH = os.environ.get("FITSCORE_DB_PATH", os.path.join(PROJECT_ROOT, "fitscore.db"))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from generate_candidates import (  # noqa: E402
    AI_MET, AI_PARTIAL, CORE_MET, CORE_PARTIAL, CROSS_MET, CROSS_PARTIAL,
    FILLER_SKILLS, TECH_MET, TECH_PARTIAL,
)

# Canonical display casing for skill tokens, reused from the same vocabulary
# generate_candidates.py used to build the training data (CountVectorizer
# itself lowercases everything, so this is needed to show "SQL" instead of
# "sql" when a specific skill is named as the top factor - see score_candidates).
SKILL_DISPLAY_NAMES = {
    s.lower(): s
    for s in (
        CORE_MET + CORE_PARTIAL + AI_MET + AI_PARTIAL + CROSS_MET + CROSS_PARTIAL
        + TECH_MET + TECH_PARTIAL + FILLER_SKILLS
    )
}

REQUIRED_COLUMNS = ["Experience", "Qualifications", "Skills", "Prior Job Title"]

# Realistic bounds for a candidate's years of experience. Values outside this
# range are clamped before scoring and flagged for the recruiter rather than
# fed to the model as-is or silently dropped.
EXPERIENCE_MIN_YEARS = 0
EXPERIENCE_MAX_YEARS = 50

# ColumnTransformer prefixes (see src/train_model.py build_pipeline) mapped to
# the 4 human-readable feature names used throughout the UI.
PREFIX_TO_FEATURE = {
    "experience": "Experience",
    "qualifications": "Qualifications",
    "prior_title": "Prior Job Title",
    "skills": "Skills",
}
BASELINE_ROW = "Baseline (intercept)"
FEATURE_ORDER = [BASELINE_ROW, "Experience", "Qualifications", "Skills", "Prior Job Title"]

BADGE_CLASS = {
    "Strong Fit": "badge-green",
    "Needs Review": "badge-amber",
    "Likely Not a Fit": "badge-red",
}

app = Flask(__name__)

_model_pipeline = None
_model_load_error = None
_known_qualifications = set()
_known_prior_titles = set()
try:
    with open(MODEL_PATH, "rb") as f:
        _model_pipeline = pickle.load(f)
    _preprocess = _model_pipeline.named_steps["preprocess"]
    _known_qualifications = set(_preprocess.named_transformers_["qualifications"].categories_[0])
    _known_prior_titles = set(_preprocess.named_transformers_["prior_title"].categories_[0])
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


def _column_display_label(feature_name):
    """
    Human-readable label for a single encoded column. For Skills - the only
    multi-valued field - this names the specific skill token (e.g. "Team
    Leadership") rather than the generic "Skills" category, since dozens of
    individual skill columns share that one category.
    """
    prefix, raw_name = feature_name.split("__", 1)
    if prefix == "skills":
        token = raw_name
        return SKILL_DISPLAY_NAMES.get(token, token.title())
    return PREFIX_TO_FEATURE[prefix]


def score_candidates(raw_df):
    """
    Run the pre-trained pipeline on raw candidate rows and, for every row,
    decompose the predicted class's logit into a contribution per raw
    feature using the model's own coef_ matrix - not a templated or generic
    explanation, a real per-row weighted sum grouped by feature.

    Also validates two things per row rather than silently absorbing them:
    - Experience outside a realistic 0-50 year range is clamped for scoring
      and flagged.
    - Qualifications / Prior Job Title values the encoder never saw during
      training (which OneHotEncoder(handle_unknown="ignore") would otherwise
      turn into a silent all-zero vector) are flagged explicitly.
    """
    df = raw_df.copy()

    raw_years = df["Experience"].apply(parse_experience_years)
    clamped_years = raw_years.clip(lower=EXPERIENCE_MIN_YEARS, upper=EXPERIENCE_MAX_YEARS)
    df["Experience_Years"] = clamped_years

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
        row_encoded = encoded[row_idx]
        row_contributions = coefs_for_class * row_encoded

        # Table view: net contribution per category (a legitimate "how much
        # did all of a candidate's skills move the needle" aggregate).
        breakdown = {feature: 0.0 for feature in FEATURE_ORDER}
        breakdown[BASELINE_ROW] = float(clf.intercept_[class_idx])
        for group, contribution in zip(feature_groups, row_contributions):
            breakdown[group] += float(contribution)

        # Headline "strongest factor": compare individual encoded columns
        # directly (not the group sums above), and only among columns that
        # are actually active for this candidate - otherwise a category with
        # many one-hot/skill columns (Skills has ~35) wins purely on column
        # count rather than real per-column importance.
        active = [
            (feature_names[i], row_contributions[i])
            for i in range(len(feature_names))
            if row_encoded[i] != 0
        ]
        supporting = [t for t in active if t[1] > 0]
        if supporting:
            top_name, top_value = max(supporting, key=lambda t: t[1])
            supports_prediction = True
        else:
            top_name, top_value = min(active, key=lambda t: t[1])
            supports_prediction = False
        top_label = _column_display_label(top_name)

        probs = {cls: float(p) for cls, p in zip(class_list, probabilities[row_idx])}

        validation_flags = []
        if raw_years.iloc[row_idx] != clamped_years.iloc[row_idx]:
            validation_flags.append(
                f"Experience value ({raw_years.iloc[row_idx]} years) is outside the realistic "
                f"{EXPERIENCE_MIN_YEARS}-{EXPERIENCE_MAX_YEARS} year range; clamped to "
                f"{clamped_years.iloc[row_idx]} years for scoring."
            )
        if df["Qualifications"].iloc[row_idx] not in _known_qualifications:
            validation_flags.append(
                f"Qualifications value “{df['Qualifications'].iloc[row_idx]}” was not seen "
                f"during training — the model treats it as no signal, not as a match or a gap."
            )
        if df["Prior Job Title"].iloc[row_idx] not in _known_prior_titles:
            validation_flags.append(
                f"Prior Job Title value “{df['Prior Job Title'].iloc[row_idx]}” was not seen "
                f"during training — the model treats it as no signal, not as a match or a gap."
            )

        results.append(
            {
                "predicted_label": predicted_label,
                "probabilities": probs,
                "breakdown": breakdown,
                "top_label": top_label,
                "supports_prediction": supports_prediction,
                "top_value": float(top_value),
                "validation_flags": validation_flags,
            }
        )
    return results


def top_factor_sentence(result):
    label = result["predicted_label"]
    if result["supports_prediction"]:
        return f"{result['top_label']} is the strongest factor supporting this “{label}” prediction."
    return (
        f"No factor supports this “{label}” prediction — "
        f"the strongest concern is {result['top_label']}."
    )


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
                "validation_flags": breakdown.get("validation_flags", []),
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
    # Each upload is its own batch (see the batches/candidates FK schema in
    # init_db) - we never delete prior batches here, so earlier Shortlist/Pass
    # decisions on past uploads are preserved. The dashboard route already
    # only ever reads the single latest batch, so this doesn't change what's
    # displayed - it only stops destroying history that isn't displayed.
    cursor = db.execute(
        "INSERT INTO batches (uploaded_at, filename) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file.filename),
    )
    batch_id = cursor.lastrowid

    for cand_id, (_, row), result in zip(candidate_ids, raw_df.iterrows(), results):
        payload = {
            "breakdown": result["breakdown"],
            "top_factor_sentence": top_factor_sentence(result),
            "validation_flags": result["validation_flags"],
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
    cursor = db.execute("UPDATE candidates SET decision = ? WHERE id = ?", (decision, candidate_row_id))
    db.commit()
    if cursor.rowcount == 0:
        return jsonify({"error": f"No candidate with id {candidate_row_id}."}), 404
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False)
