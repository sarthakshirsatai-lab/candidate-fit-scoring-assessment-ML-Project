"""
Candidate-fit-scoring data foundation pipeline.

Reads the raw job_descriptions.csv in memory-safe chunks, drops
discriminatory/PII-shaped columns, filters to European AI/Data/Product
Manager-type postings, saves the result, and profiles it (ydata-profiling
if available on this Python version, otherwise a pandas-only fallback
report at the same output path).
"""

import os
import re

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The raw Kaggle job_descriptions.csv archive is not part of this repo (it's
# a large external download). Override its location via the
# RAW_JOB_DESCRIPTIONS_CSV env var if it lives elsewhere on your machine;
# otherwise it defaults to a sibling "archive" folder next to the project,
# so this runs on any machine without editing the file.
RAW_CSV_PATH = os.environ.get(
    "RAW_JOB_DESCRIPTIONS_CSV",
    os.path.join(os.path.dirname(PROJECT_ROOT), "archive", "job_descriptions.csv"),
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

FILTERED_CSV_PATH = os.path.join(DATA_DIR, "filtered_jobs.csv")
RUN_LOG_PATH = os.path.join(REPORTS_DIR, "pipeline_run_log.txt")
SUMMARY_STATS_PATH = os.path.join(REPORTS_DIR, "eda_summary_stats.txt")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_report.html")

CHUNK_SIZE = 100_000

DROP_COLUMNS = ["Preference", "Contact Person", "Contact"]

TARGET_COUNTRIES = {
    "Spain", "Portugal", "France", "Germany", "Italy", "Netherlands",
    "Belgium", "Ireland", "Poland", "Sweden", "United Kingdom",
}

# Case-insensitive, partial match. Deliberately wide (preliminary filter,
# not the final JD title requirement) but bounded: \s+ between multi-word
# tokens stops "Production Manager" matching "product manager"; \b...\b
# around bare ai/ml stops them matching inside words like "detail" or
# "retail"; "data entry" is deliberately absent.
KEYWORD_PATTERN = (
    r"\b(product\s+manager|product\s+owner|product\s+management"
    r"|data\s+scientist|data\s+analyst|data\s+engineer|data\s+science"
    r"|machine\s+learning|artificial\s+intelligence|ai|ml)\b"
)

NUMERIC_CANDIDATE_COLUMNS = ["Job Id", "latitude", "longitude", "Company Size"]

TOP_VALUE_COLUMNS = [
    "Country", "Work Type", "Job Portal", "Qualifications", "Job Title", "Role",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(RUN_LOG_PATH, "w", encoding="utf-8") as logfile:
        def log(msg=""):
            print(msg)
            logfile.write(str(msg) + "\n")

        final_df = run_chunked_filter(log)

        log()
        log(f"Final filtered columns ({len(final_df.columns)}): {final_df.columns.tolist()}")
        log()
        log("First 5 rows of final filtered dataframe:")
        log(final_df.head(5).to_string())

        final_df.to_csv(FILTERED_CSV_PATH, index=False)
        log()
        log(f"Saved filtered dataset to {FILTERED_CSV_PATH} ({len(final_df)} rows).")

        run_eda(final_df, log)

    print(f"\nDone. Run log written to {RUN_LOG_PATH}")


def run_chunked_filter(log):
    rows_total = 0
    rows_after_country = 0
    rows_final = 0
    kept_chunks = []

    reader = pd.read_csv(
        RAW_CSV_PATH,
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
    )

    for chunk in reader:
        rows_total += len(chunk)

        chunk = chunk.drop(columns=DROP_COLUMNS, errors="ignore")

        country_mask = chunk["Country"].isin(TARGET_COUNTRIES)
        rows_after_country += int(country_mask.sum())

        title_match = chunk["Job Title"].str.contains(
            KEYWORD_PATTERN, case=False, regex=True, na=False
        )
        role_match = chunk["Role"].str.contains(
            KEYWORD_PATTERN, case=False, regex=True, na=False
        )
        keyword_mask = title_match | role_match

        kept = chunk[country_mask & keyword_mask]
        rows_final += len(kept)
        if len(kept):
            kept_chunks.append(kept)

    final_df = (
        pd.concat(kept_chunks, ignore_index=True)
        if kept_chunks
        else pd.DataFrame(columns=chunk.columns)
    )

    log("=== Row counts by stage ===")
    log(f"Total rows processed:        {rows_total}")
    log(f"After country filter:        {rows_after_country}")
    log(f"Final (country + role/title): {rows_final}")

    log()
    log("=== Top 20 Job Title values (final filtered data) ===")
    log(final_df["Job Title"].value_counts().head(20).to_string())

    log()
    log("=== Top 20 Role values (final filtered data) ===")
    log(final_df["Role"].value_counts().head(20).to_string())

    return final_df


def run_eda(final_df, log):
    stats_sections = []

    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            final_df,
            title="Candidate Fit Scoring - Filtered Jobs EDA",
            minimal=True,
        )
        profile.to_file(EDA_REPORT_PATH)
        profiling_note = "ydata-profiling ran successfully."
        log()
        log(profiling_note)
        stats_sections.append(profiling_note)
        stats_sections.append(describe_section(final_df))
        stats_sections.append(missingness_section(final_df))
        stats_sections.append(correlation_section(final_df))
        stats_sections.append(top_values_section(final_df))
    except Exception as e:
        profiling_note = (
            f"ydata-profiling was not usable (reason: {type(e).__name__}: {e}). "
            f"This Python version is outside ydata-profiling's supported range, "
            f"so this is expected. Falling back to a custom lightweight EDA report."
        )
        log()
        log(profiling_note)
        write_fallback_html(final_df, profiling_note)
        stats_sections.append(profiling_note)
        stats_sections.append(describe_section(final_df))
        stats_sections.append(missingness_section(final_df))
        stats_sections.append(correlation_section(final_df))
        stats_sections.append(top_values_section(final_df))

    with open(SUMMARY_STATS_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(stats_sections))
    log(f"\nWrote EDA summary stats to {SUMMARY_STATS_PATH}")


def numeric_frame(df):
    numeric_df = pd.DataFrame()
    for col in NUMERIC_CANDIDATE_COLUMNS:
        if col in df.columns:
            numeric_df[col] = pd.to_numeric(df[col], errors="coerce")
    return numeric_df


def describe_section(df):
    return "=== describe(include='all') ===\n" + df.describe(include="all").to_string()


def missingness_section(df):
    missing = pd.DataFrame({
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
    })
    return "=== Missingness by column ===\n" + missing.to_string()


def correlation_section(df):
    numeric_df = numeric_frame(df)
    note = (
        "Note: Salary Range and Experience are stored as text ranges "
        "(e.g. \"$59K-$99K\", \"5 to 15 Years\") and are not included here; "
        "parsing them into numeric bounds is a worthwhile follow-up but is "
        "out of scope for this pipeline run.\n\n"
    )
    return "=== Correlation matrix (numeric columns) ===\n" + note + numeric_df.corr().to_string()


def top_values_section(df):
    parts = ["=== Top 15 values per key categorical column ==="]
    for col in TOP_VALUE_COLUMNS:
        if col in df.columns:
            parts.append(f"\n--- {col} ---")
            parts.append(df[col].value_counts().head(15).to_string())
    return "\n".join(parts)


def write_fallback_html(df, profiling_note):
    numeric_df = numeric_frame(df)
    missing = pd.DataFrame({
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().mean() * 100).round(2),
    })

    top_values_html = ""
    for col in TOP_VALUE_COLUMNS:
        if col in df.columns:
            top_values_html += f"<h3>{col}</h3>" + df[col].value_counts().head(15).to_frame().to_html()

    html = f"""<html>
<head><title>Candidate Fit Scoring - Filtered Jobs EDA (Fallback)</title></head>
<body>
<div style="background:#fff3cd;border:1px solid #ffcc00;padding:12px;margin-bottom:16px;">
<strong>ydata-profiling was not used for this report.</strong><br>{profiling_note}
</div>
<h2>describe(include='all')</h2>
{df.describe(include='all').to_html()}
<h2>Missingness by column</h2>
{missing.to_html()}
<h2>Correlation matrix (numeric columns)</h2>
<p>Salary Range and Experience are text ranges, not included; worth parsing into numeric bounds later.</p>
{numeric_df.corr().to_html()}
<h2>Top 15 values per key categorical column</h2>
{top_values_html}
</body>
</html>"""

    with open(EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
