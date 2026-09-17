# EDA Findings — Filtered EU AI/Data/PM Postings (n=2,553)

*ydata-profiling isn't installable on this Python version (its package
metadata excludes 3.14+), confirmed by a failed `pip install`. This is a
manual interpretation of the pandas-only fallback report
(`eda_report.html`, `eda_summary_stats.txt`), not templated boilerplate.*

## Strongest apparent relationships

- **Job Title/Role → skills, Job Description, Responsibilities, Benefits is
  a near 1:1 template, not free text.** Only 12 unique values exist across
  each of these fields over 2,553 rows — every posting with the same title
  reuses identical boilerplate. This is the single biggest structural
  pattern in the data, more informative than any numeric correlation.
- **latitude ↔ longitude correlate at 0.54** — expected and trivial (they're
  just two encodings of the same country), not a modeling insight.
- Everything else numeric (`Job Id`, `Company Size`) correlates at ~0 with
  everything else — no real relationships there.

## Data quality issues & remediation

- **Templated text fields**: `skills`/`Job Description`/`Responsibilities`/
  `Benefits` carry no information beyond `Job Title`+`Role` combined (only
  12 distinct combos). Don't treat them as independent NLP features — use
  them only as a rule-based source for keyword matching (as Requirement 7
  already does).
- **Suspiciously uniform distributions**: `Qualifications` (10 degree types,
  each 226–294 rows) and `Country` (each 242–271 rows) are close to
  uniform-random — consistent with synthetic generation, not real hiring
  patterns. Don't let a future model over-weight degree type or country as
  if it reflects genuine market signal.
- **`Company Size` is noise**: 2,525 unique values across 2,553 rows,
  ~0 correlation with everything — effectively a random number per row.
  Exclude it from scoring features.
- **`Experience` and `Salary Range` are free-text ranges** (e.g. "5 to 15
  Years", "$59K-$99K"), not numeric — must be parsed into a numeric
  floor/midpoint before any threshold check (e.g. Requirement 1's "8+
  years").
- **8 missing `Company Profile` values** (0.31%) — negligible, safe to leave
  null or drop those rows.

## Features worth carrying forward (5–8)

1. **Job Title** — primary category; only 6 distinct values here, directly
   feeds the Requirement 7 title check.
2. **Role** — finer-grained than Job Title (12 values), needed to catch
   "Technical Product Manager" vs. plain "Product Manager".
3. **skills** — despite being templated per title, it's the direct source
   for Requirements 3, 4, and 6 (core/AI/technical keyword matches).
4. **Job Description** — secondary keyword source for Requirement 4/7 when
   `skills` alone doesn't mention an AI/ML term.
5. **Experience** — once parsed to a numeric floor, this is the only field
   that can satisfy Requirement 1's "8+ years" threshold.
6. **Qualifications** — required for Requirement 2's degree-field check,
   though (per above) treat it as a rule-compliance field, not a strong
   real-world predictor in this synthetic sample.

**Not carried forward**: `Company Size` and `Salary Range` (noise/high-
cardinality, no signal), `Country` (needed only to scope the filter, not to
score fit).
