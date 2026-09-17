"""One-off check: how many filtered postings satisfy the finalized Requirement 7."""

import os
import re

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTERED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "filtered_jobs.csv")

TITLE_PATTERN = r"product\s+manager"
AI_SIGNAL_PATTERN = r"\b(machine\s+learning|artificial\s+intelligence|data\s+science|ai|ml)\b"

df = pd.read_csv(FILTERED_CSV_PATH, dtype=str)

title_match = (
    df["Job Title"].str.contains(TITLE_PATTERN, case=False, regex=True, na=False)
    | df["Role"].str.contains(TITLE_PATTERN, case=False, regex=True, na=False)
)

ai_signal_match = (
    df["skills"].str.contains(AI_SIGNAL_PATTERN, case=False, regex=True, na=False)
    | df["Job Description"].str.contains(AI_SIGNAL_PATTERN, case=False, regex=True, na=False)
)

req7_met = title_match & ai_signal_match

print(f"Rows with Product Manager-type title: {title_match.sum()}")
print(f"Rows with AI/ML signal in skills or job description: {ai_signal_match.sum()}")
print(f"Rows satisfying Requirement 7 (both): {req7_met.sum()}")
