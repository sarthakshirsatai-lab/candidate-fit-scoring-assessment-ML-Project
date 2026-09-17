"""
MLDLC Step 6 - Model Training, Evaluation, Selection.

Trains a Logistic Regression and a Decision Tree classifier to predict
Fit_Label from data/labeled_candidates.csv using only the 4 raw candidate
attributes that actually exist on that file (Experience, Qualifications,
Skills, Prior Job Title) - never the Req1..Req7/Total_Score columns, which
are deterministic components of Fit_Label itself and would leak the label
straight into the features.

Note: the original task brief named 6 features (Job Title, Role, Skills,
Job Description, Experience, Qualifications) - those are column names from
the upstream *job postings* file (data/filtered_jobs.csv), not from the
*candidates* file this script trains on. That file has no Job Title, Role,
or Job Description columns. Confirmed with the user before building this;
see reports/model_evaluation.md for the writeup.
"""

import os
import pickle
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

LABELED_CSV_PATH = os.path.join(DATA_DIR, "labeled_candidates.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "model_evaluation.md")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
DT_MODEL_PATH = os.path.join(MODELS_DIR, "decision_tree.pkl")

RAW_FEATURE_COLUMNS = ["Experience_Years", "Qualifications", "Skills", "Prior Job Title"]
LEAKAGE_COLUMNS = [
    "Req1_Experience", "Req2_Education", "Req3_CoreSkill", "Req4_AIML",
    "Req5_CrossFunctional", "Req6_Technical", "Req7_TitleAndAI", "Total_Score",
]
CLASS_ORDER = ["Likely Not a Fit", "Needs Review", "Strong Fit"]

AUC_POOR_THRESHOLD = 0.65
MINORITY_RECALL_POOR_THRESHOLD = 0.20

# Matches each comma-separated segment of the Skills cell, trimmed of
# surrounding whitespace (the capturing group becomes the token). Deliberately
# a regex rather than a custom tokenizer function: a lambda/def callable would
# get pickled by reference to this module, breaking `pickle.load()` for any
# code that doesn't also import train_model under the exact same module path.
SKILLS_TOKEN_PATTERN = r"\s*([^,]+?)\s*(?:,|$)"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(str(msg))

    log("# Model Evaluation - Fit_Label Classification (MLDLC Step 6)\n")

    df = inspect_data(log)
    X, y = build_features_and_target(df, log)
    X_train, X_test, y_train, y_test = split_data(X, y, log)

    lr_pipeline = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    dt_pipeline = build_pipeline(
        DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=42
        )
    )

    log("\n## Models trained\n")
    log(
        "- **Logistic Regression** (`max_iter=1000`, `class_weight=\"balanced\"`): "
        "linear baseline; `class_weight=\"balanced\"` re-weights the loss for the "
        "thinner classes given the 25.6/45.2/29.2 split on only 250 rows."
    )
    log(
        "- **Decision Tree** (`max_depth=5`, `min_samples_leaf=5`, "
        "`class_weight=\"balanced\"`): depth/leaf caps are deliberate - with only "
        "200 training rows an unconstrained tree overfits and produces a tree too "
        "large to actually read, which would defeat the reason for choosing a tree "
        "(explainability) in the first place."
    )

    lr_pipeline.fit(X_train, y_train)
    dt_pipeline.fit(X_train, y_train)

    lr_metrics = evaluate_model(lr_pipeline, "Logistic Regression", X_test, y_test, log)
    dt_metrics = evaluate_model(dt_pipeline, "Decision Tree", X_test, y_test, log)

    recommend(lr_metrics, dt_metrics, log)

    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr_pipeline, f)
    with open(DT_MODEL_PATH, "wb") as f:
        pickle.dump(dt_pipeline, f)

    log("\n## Saved artifacts\n")
    log(f"- `{os.path.relpath(LR_MODEL_PATH, PROJECT_ROOT)}`")
    log(f"- `{os.path.relpath(DT_MODEL_PATH, PROJECT_ROOT)}`")
    log(
        "\nBoth pipelines are self-contained (raw columns in, prediction out) - "
        "preprocessing is baked into the pickled object, so no manual re-encoding "
        "is needed at load time."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved model evaluation report to {REPORT_PATH}")


def inspect_data(log):
    log("## 1. Data inspection\n")
    df = pd.read_csv(LABELED_CSV_PATH)
    log(f"Loaded `{os.path.relpath(LABELED_CSV_PATH, PROJECT_ROOT)}`: {len(df)} rows, "
        f"{len(df.columns)} columns.")
    log(f"\nColumns: {df.columns.tolist()}")
    log(f"\nDtypes:\n```\n{df.dtypes.to_string()}\n```")

    exp_format_ok = df["Experience"].str.match(r"^\d+\s+Years$").all()
    log(
        f"\n`Experience` format check - every value matches `'<int> Years'`: "
        f"{exp_format_ok}. Example values: {df['Experience'].head(3).tolist()}. "
        f"This is text, not numeric -> needs parsing before use."
    )

    log(
        f"\n`Qualifications` - {df['Qualifications'].nunique()} unique values, "
        f"already short categorical strings (e.g. {df['Qualifications'].unique()[:3].tolist()}) "
        f"-> no parsing needed, just encoding."
    )
    log(
        f"\n`Prior Job Title` - {df['Prior Job Title'].nunique()} unique values "
        f"(e.g. {df['Prior Job Title'].unique()[:3].tolist()}) -> no parsing needed, just encoding."
    )
    log(
        f"\n`Skills` - free-form comma-separated string per row (e.g. "
        f"\"{df['Skills'].iloc[0]}\") -> multi-label field, needs tokenizing, not "
        f"single-category encoding."
    )

    log(
        f"\nExcluding leakage columns before modeling: {LEAKAGE_COLUMNS}. These are "
        f"the Met/Partial/Gap rubric outcomes and their sum - each is a deterministic "
        f"function of the same raw fields used to compute Fit_Label, so including them "
        f"would let the model 'cheat' instead of learning the pattern itself."
    )

    return df


def build_features_and_target(df, log):
    df = df.copy()
    df["Experience_Years"] = df["Experience"].str.extract(r"^(\d+)").astype(int)

    X = df[RAW_FEATURE_COLUMNS]
    y = df["Fit_Label"]

    log("\n## 2. Feature encoding\n")
    log("| Column | Type | Encoding | Why |")
    log("|---|---|---|---|")
    log(
        "| `Experience_Years` (parsed from `Experience`) | numeric | `StandardScaler` "
        "| Helps Logistic Regression's gradient-based fit and its L2 regularization "
        "treat this feature fairly against the one-hot columns; a no-op for the "
        "Decision Tree since axis-aligned splits are scale-invariant, so one shared "
        "preprocessing pipeline works for both models. |"
    )
    log(
        "| `Qualifications` | categorical, ~20 distinct degree strings | "
        "`OneHotEncoder(handle_unknown=\"ignore\")` | Nominal category, no ordinal "
        "relationship between degree names (a `PhD` isn't \"more\" than a `B.Tech` "
        "in a linearly-encodable sense here). |"
    )
    log(
        "| `Prior Job Title` | categorical, ~16 distinct titles | "
        "`OneHotEncoder(handle_unknown=\"ignore\")` | Same reasoning as Qualifications - "
        "nominal, low cardinality. |"
    )
    log(
        "| `Skills` | free text, comma-separated multi-label field | "
        "`CountVectorizer` (custom comma tokenizer, `binary=True`) | Each candidate "
        "has *multiple* skills, so this isn't a single category - one-hot doesn't "
        "fit. `binary=True` bag-of-skills (presence/absence) is used instead of "
        "TF-IDF because the vocabulary is small and fixed and every skill token is "
        "equally diagnostic; there's no long-document \"common word\" problem here "
        "that IDF weighting would fix. |"
    )

    return X, y


def split_data(X, y, log):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    log("\n## 3. Train/test split\n")
    log("80/20 split, stratified on `Fit_Label` (`random_state=42`).\n")
    log("| Class | Full (n=250) | Train (n={}) | Test (n={}) |".format(len(X_train), len(X_test)))
    log("|---|---|---|---|")
    full_counts = y.value_counts()
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    for cls in CLASS_ORDER:
        full_pct = round(100 * full_counts.get(cls, 0) / len(y), 1)
        train_pct = round(100 * train_counts.get(cls, 0) / len(y_train), 1)
        test_pct = round(100 * test_counts.get(cls, 0) / len(y_test), 1)
        log(
            f"| {cls} | {full_counts.get(cls, 0)} ({full_pct}%) | "
            f"{train_counts.get(cls, 0)} ({train_pct}%) | "
            f"{test_counts.get(cls, 0)} ({test_pct}%) |"
        )

    return X_train, X_test, y_train, y_test


def build_pipeline(classifier):
    preprocessor = ColumnTransformer(
        transformers=[
            ("experience", StandardScaler(), ["Experience_Years"]),
            ("qualifications", OneHotEncoder(handle_unknown="ignore"), ["Qualifications"]),
            ("prior_title", OneHotEncoder(handle_unknown="ignore"), ["Prior Job Title"]),
            (
                "skills",
                CountVectorizer(token_pattern=SKILLS_TOKEN_PATTERN, binary=True),
                "Skills",
            ),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])


def evaluate_model(pipeline, name, X_test, y_test, log):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    report_dict = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, zero_division=0
    )

    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes)

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    log(f"\n## 4. Evaluation - {name}\n")
    log("Precision / recall / F1 (held-out 20%, n={}):\n```\n{}\n```".format(len(y_test), report_text))
    log(f"ROC-AUC (multi-class, one-vs-rest, macro-averaged): **{auc:.3f}**")

    log("\nConfusion matrix (rows = actual, columns = predicted):\n")
    header = "| Actual \\ Predicted | " + " | ".join(CLASS_ORDER) + " |"
    sep = "|---|" + "---|" * len(CLASS_ORDER)
    log(header)
    log(sep)
    for i, actual_cls in enumerate(CLASS_ORDER):
        row = " | ".join(str(cm[i][j]) for j in range(len(CLASS_ORDER)))
        log(f"| {actual_cls} | {row} |")

    return {
        "name": name,
        "auc": auc,
        "report": report_dict,
        "strong_fit_recall": report_dict["Strong Fit"]["recall"],
        "strong_fit_precision": report_dict["Strong Fit"]["precision"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
    }


def recommend(lr_metrics, dt_metrics, log):
    log("\n## 5. Recommendation\n")

    def is_poor(m):
        return (
            m["auc"] < AUC_POOR_THRESHOLD
            or m["strong_fit_recall"] < MINORITY_RECALL_POOR_THRESHOLD
        )

    lr_poor = is_poor(lr_metrics)
    dt_poor = is_poor(dt_metrics)

    log(
        f"Logistic Regression: ROC-AUC={lr_metrics['auc']:.3f}, "
        f"Strong Fit recall={lr_metrics['strong_fit_recall']:.3f}, "
        f"macro F1={lr_metrics['macro_f1']:.3f}."
    )
    log(
        f"Decision Tree: ROC-AUC={dt_metrics['auc']:.3f}, "
        f"Strong Fit recall={dt_metrics['strong_fit_recall']:.3f}, "
        f"macro F1={dt_metrics['macro_f1']:.3f}."
    )

    if lr_poor and dt_poor:
        log(
            "\n**Neither model is reliable enough to recommend as-is.** Both fall "
            f"below the ROC-AUC~{AUC_POOR_THRESHOLD} bar and/or fail to reliably "
            "catch the minority \"Strong Fit\" class (recall "
            f"< {MINORITY_RECALL_POOR_THRESHOLD:.0%}) on the held-out set. On 250 "
            "rows with a 3-way split, this is a real possibility, not just a modeling "
            "mistake - there may simply not be enough Strong Fit examples for either "
            "a linear or a single-tree model to separate it cleanly from Needs Review. "
            "**Next step: try a Random Forest.** Averaging many trees typically "
            "reduces the variance a single small Decision Tree suffers from on a "
            "dataset this size, and should help minority-class recall. The tradeoff: "
            "a Random Forest is no longer a single readable decision path - "
            "explainability moves from \"trace this exact branch\" to aggregate "
            "`feature_importances_` or per-prediction tools (e.g. SHAP), which is a "
            "weaker (but still workable) story for the EU AI Act explainability framing."
        )
        return

    better = "Decision Tree" if dt_metrics["auc"] >= lr_metrics["auc"] else "Logistic Regression"
    auc_gap = abs(lr_metrics["auc"] - dt_metrics["auc"])

    if auc_gap <= 0.05 and not (lr_poor or dt_poor):
        log(
            "\n**Recommendation: Decision Tree.** The two models are close enough "
            f"in ROC-AUC (gap={auc_gap:.3f}) and both clear the reliability bar, so "
            "the tie-break goes to explainability: a depth-5 tree gives a directly "
            "traceable if/else decision path per candidate (\"Experience >= 8 AND "
            "Qualifications is CS/Business AND ...\"), which is far easier to justify "
            "to a candidate or auditor under an EU AI Act framing than reasoning "
            "through Logistic Regression's coefficients across one-hot-encoded "
            "categories and a bag-of-skills vector."
        )
    else:
        log(
            f"\n**Recommendation: {better}.** It leads on ROC-AUC by {auc_gap:.3f}, "
            "which is large enough on a 50-row test set to matter more than the "
            "explainability edge of the alternative. "
            + (
                "Note the tradeoff taken: Logistic Regression's coefficients are "
                "less directly traceable per-decision than the Decision Tree's "
                "explicit branches, though they're still inspectable (sign and "
                "magnitude per encoded feature)."
                if better == "Logistic Regression"
                else "Note the tradeoff taken: choosing the tree over Logistic "
                "Regression here costs a small amount of the linear model's more "
                "gradual, coefficient-based confidence behavior in exchange for "
                "better held-out separation."
            )
        )


if __name__ == "__main__":
    main()
