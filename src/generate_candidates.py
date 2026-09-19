"""
Generate synthetic candidates and score them against the 7 finalized
AI Product Manager JD requirements (Met/Partial/Gap per requirement),
then derive an overall fit_label.

Generation targets ~25% Strong Fit / ~45% Needs Review / ~30% Likely Not a
Fit by constructing profiles with archetype-weighted attributes; the actual
fit_label is always computed live from the generated field values via the
same regex rules a real scoring pipeline would use, never assigned directly.
"""

import os
import random
import re

import pandas as pd

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "labeled_candidates.csv")
SUMMARY_PATH = os.path.join(REPORTS_DIR, "labeling_summary.md")

# ---------------------------------------------------------------------------
# Field pools
# ---------------------------------------------------------------------------

MET_QUALS = [
    "B.Tech Computer Science", "M.Tech Computer Science", "MBA",
    "M.Sc Data Science", "B.Sc Computer Science", "M.Eng Software Engineering",
    "B.Eng Industrial Engineering", "MBA Business Administration",
]
PARTIAL_QUALS = [
    "B.Sc Mathematics", "M.Sc Statistics", "B.Com Economics",
    "Diploma in Information Technology", "B.Sc Physics", "M.A. Economics",
]
GAP_QUALS = [
    "BA Fine Arts", "B.A. History", "BA Sociology",
    "Bachelor of Hospitality Management", "B.A. Literature",
    "Diploma in Culinary Arts",
]

CORE_MET = ["Product Management", "Product Strategy"]
CORE_PARTIAL = ["Project Management", "Product Design"]

AI_MET = ["Machine Learning", "Artificial Intelligence", "Data Science", "AI", "ML"]
AI_PARTIAL = ["Data Analytics", "Business Intelligence", "Statistics"]

CROSS_MET = ["Stakeholder Management", "Team Leadership"]
CROSS_PARTIAL = ["Communication", "Collaboration"]

TECH_MET = ["Python", "SQL", "Data Analysis"]
TECH_PARTIAL = ["Excel", "Tableau", "Power BI"]

FILLER_SKILLS = [
    "Agile", "Scrum", "Roadmapping", "Market Research", "A/B Testing",
    "JIRA", "Confluence", "Customer Research", "UX Research", "Negotiation",
    "Public Speaking", "Salesforce", "Google Analytics",
]

PM_TITLES = [
    "Product Manager", "Senior Product Manager", "Technical Product Manager",
    "Associate Product Manager", "Group Product Manager",
]
NON_PM_TITLES = [
    "Business Analyst", "Data Analyst", "Data Scientist", "Software Engineer",
    "Marketing Manager", "Operations Manager", "Customer Success Manager",
    "Sales Executive", "Project Coordinator", "UX Designer", "Data Engineer",
]

ARCHETYPES = {
    "Strong Fit (intended)": {
        "count": 48,
        "weights": {"Met": 0.80, "Partial": 0.15, "Gap": 0.05},
        "title_pm_prob": 0.90,
    },
    "Needs Review (intended)": {
        "count": 150,
        "weights": {"Met": 0.40, "Partial": 0.40, "Gap": 0.20},
        "title_pm_prob": 0.40,
    },
    "Likely Not a Fit (intended)": {
        "count": 52,
        "weights": {"Met": 0.03, "Partial": 0.22, "Gap": 0.75},
        "title_pm_prob": 0.05,
    },
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def weighted_tier(weights):
    tiers = list(weights.keys())
    probs = list(weights.values())
    return random.choices(tiers, weights=probs, k=1)[0]


def build_candidate(archetype_name, weights, title_pm_prob):
    exp_tier = weighted_tier(weights)
    qual_tier = weighted_tier(weights)
    core_tier = weighted_tier(weights)
    ai_tier = weighted_tier(weights)
    cross_tier = weighted_tier(weights)
    tech_tier = weighted_tier(weights)
    title_is_pm = random.random() < title_pm_prob

    years = {
        "Met": random.randint(8, 15),
        "Partial": random.randint(5, 7),
        "Gap": random.randint(0, 4),
    }[exp_tier]

    qualification = random.choice(
        {"Met": MET_QUALS, "Partial": PARTIAL_QUALS, "Gap": GAP_QUALS}[qual_tier]
    )

    core_token = {"Met": random.choice(CORE_MET), "Partial": random.choice(CORE_PARTIAL), "Gap": None}[core_tier]
    ai_token = {"Met": random.choice(AI_MET), "Partial": random.choice(AI_PARTIAL), "Gap": None}[ai_tier]
    cross_token = {"Met": random.choice(CROSS_MET), "Partial": random.choice(CROSS_PARTIAL), "Gap": None}[cross_tier]
    tech_token = {"Met": random.choice(TECH_MET), "Partial": random.choice(TECH_PARTIAL), "Gap": None}[tech_tier]

    fillers = random.sample(FILLER_SKILLS, k=random.randint(2, 4))
    skills_list = [t for t in [core_token, ai_token, cross_token, tech_token] if t] + fillers
    random.shuffle(skills_list)
    skills = ", ".join(skills_list)

    prior_title = random.choice(PM_TITLES if title_is_pm else NON_PM_TITLES)

    return {
        "Experience": f"{years} Years",
        "Qualifications": qualification,
        "Skills": skills,
        "Prior Job Title": prior_title,
        "_intended_archetype": archetype_name,
    }


# ---------------------------------------------------------------------------
# Scoring rubric (Met=2, Partial=1, Gap=0) — mirrors reference/jd_ai_product_manager.md
# ---------------------------------------------------------------------------

QUAL_MET_KEYWORDS = ["computer science", "data science", "engineer", "business", "mba", "software"]
QUAL_PARTIAL_KEYWORDS = ["mathematics", "statistics", "economics", "information technology", "physics"]

CORE_MET_PATTERN = r"\b(product management|product strategy)\b"
CORE_PARTIAL_PATTERN = r"\b(project management|product design)\b"

AI_MET_PATTERN = r"\b(machine learning|artificial intelligence|data science|ai|ml)\b"
AI_PARTIAL_PATTERN = r"\b(data analytics|business intelligence|statistics)\b"

CROSS_MET_PATTERN = r"\b(stakeholder management|team leadership)\b"
CROSS_PARTIAL_PATTERN = r"\b(communication|collaboration)\b"

TECH_MET_PATTERN = r"\b(python|sql|data analysis)\b"
TECH_PARTIAL_PATTERN = r"\b(excel|tableau|power bi)\b"

TITLE_PM_PATTERN = r"product manager"


def score_experience(experience_str):
    years = int(re.match(r"(\d+)", experience_str).group(1))
    if years >= 8:
        return "Met"
    if years >= 5:
        return "Partial"
    return "Gap"


def score_qualifications(qual_str):
    q = qual_str.lower()
    if any(k in q for k in QUAL_MET_KEYWORDS):
        return "Met"
    if any(k in q for k in QUAL_PARTIAL_KEYWORDS):
        return "Partial"
    return "Gap"


def _tier_from_patterns(text, met_pattern, partial_pattern):
    if re.search(met_pattern, text, re.IGNORECASE):
        return "Met"
    if re.search(partial_pattern, text, re.IGNORECASE):
        return "Partial"
    return "Gap"


def score_core_skill(skills):
    return _tier_from_patterns(skills, CORE_MET_PATTERN, CORE_PARTIAL_PATTERN)


def score_ai_ml(skills):
    return _tier_from_patterns(skills, AI_MET_PATTERN, AI_PARTIAL_PATTERN)


def score_cross_functional(skills):
    return _tier_from_patterns(skills, CROSS_MET_PATTERN, CROSS_PARTIAL_PATTERN)


def score_technical(skills):
    return _tier_from_patterns(skills, TECH_MET_PATTERN, TECH_PARTIAL_PATTERN)


def score_requirement7(prior_title, skills):
    title_is_pm = bool(re.search(TITLE_PM_PATTERN, prior_title, re.IGNORECASE))
    ai_signal = bool(re.search(AI_MET_PATTERN, skills, re.IGNORECASE))
    if title_is_pm and ai_signal:
        return "Met"
    if title_is_pm or ai_signal:
        return "Partial"
    return "Gap"


TIER_POINTS = {"Met": 2, "Partial": 1, "Gap": 0}


def fit_label_from_score(total_score):
    if total_score >= 11:
        return "Strong Fit"
    if total_score >= 6:
        return "Needs Review"
    return "Likely Not a Fit"


def score_candidate(candidate):
    req1 = score_experience(candidate["Experience"])
    req2 = score_qualifications(candidate["Qualifications"])
    req3 = score_core_skill(candidate["Skills"])
    req4 = score_ai_ml(candidate["Skills"])
    req5 = score_cross_functional(candidate["Skills"])
    req6 = score_technical(candidate["Skills"])
    req7 = score_requirement7(candidate["Prior Job Title"], candidate["Skills"])

    reqs = [req1, req2, req3, req4, req5, req6, req7]
    total_score = sum(TIER_POINTS[r] for r in reqs)
    fit_label = fit_label_from_score(total_score)

    return {
        **{k: v for k, v in candidate.items() if not k.startswith("_")},
        "Req1_Experience": req1,
        "Req2_Education": req2,
        "Req3_CoreSkill": req3,
        "Req4_AIML": req4,
        "Req5_CrossFunctional": req5,
        "Req6_Technical": req6,
        "Req7_TitleAndAI": req7,
        "Total_Score": total_score,
        "Fit_Label": fit_label,
        "_intended_archetype": candidate["_intended_archetype"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    candidates = []
    for archetype_name, cfg in ARCHETYPES.items():
        for _ in range(cfg["count"]):
            candidates.append(build_candidate(archetype_name, cfg["weights"], cfg["title_pm_prob"]))

    random.shuffle(candidates)
    scored = [score_candidate(c) for c in candidates]

    df = pd.DataFrame(scored)
    intended = df["_intended_archetype"]
    output_df = df.drop(columns=["_intended_archetype"])
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)

    balance = output_df["Fit_Label"].value_counts()
    print("=== Actual fit_label class balance ===")
    print(balance.to_string())
    print(f"\nTotal candidates: {len(output_df)}")

    crosstab = pd.crosstab(intended, output_df["Fit_Label"])
    print("\n=== Intended archetype vs actual computed label ===")
    print(crosstab.to_string())

    write_summary(output_df, balance, crosstab)
    print(f"\nSaved labeled candidates to {OUTPUT_CSV_PATH}")
    print(f"Saved labeling summary to {SUMMARY_PATH}")


def write_summary(df, balance, crosstab):
    total = len(df)
    lines = []
    lines.append("# Labeling Summary — Synthetic Candidate Scoring\n")
    lines.append(
        "250 synthetic candidates generated (Experience, Qualifications, Skills, "
        "Prior Job Title), scored against all 7 requirements in "
        "`reference/jd_ai_product_manager.md` (Met=2, Partial=1, Gap=0), summed "
        "to a Total_Score (max 14), and labeled by threshold: Strong Fit ≥11, "
        "Needs Review 6-10, Likely Not a Fit ≤5.\n"
    )

    lines.append("## Class balance (actual, computed)\n")
    for label, count in balance.items():
        pct = round(100 * count / total, 1)
        lines.append(f"- **{label}**: {count} ({pct}%)")
    lines.append("")

    lines.append(
        "## Generation vs. actual label\n"
        "Candidates were generated toward ~25/45/30 targets, but labels are "
        "always computed live from the fields, not assigned — actual counts "
        "differ slightly from intent due to randomized attribute variance:\n"
    )
    lines.append("```\n" + crosstab.to_string() + "\n```\n")

    lines.append("## Example candidates per tier\n")
    for label in ["Strong Fit", "Needs Review", "Likely Not a Fit"]:
        subset = df[df["Fit_Label"] == label].head(3)
        lines.append(f"### {label}\n")
        for _, row in subset.iterrows():
            lines.append(
                f"- **{row['Prior Job Title']}**, {row['Experience']}, "
                f"{row['Qualifications']}. Skills: {row['Skills']}. "
                f"Score: {row['Total_Score']}/14 "
                f"(Exp={row['Req1_Experience']}, Edu={row['Req2_Education']}, "
                f"Core={row['Req3_CoreSkill']}, AI/ML={row['Req4_AIML']}, "
                f"Cross-fn={row['Req5_CrossFunctional']}, Tech={row['Req6_Technical']}, "
                f"Req7={row['Req7_TitleAndAI']})."
            )
        lines.append("")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
