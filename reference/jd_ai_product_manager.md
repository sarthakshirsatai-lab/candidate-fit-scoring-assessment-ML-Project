# Job Description Reference — AI Product Manager (final)

This is the fixed scoring reference the candidate-fit model checks candidates
against. All 7 requirements are now locked.

The filtered dataset (2,553 EU AI/Data/PM postings) confirmed the dataset
predates "AI Product Manager" as a literal title. The closest real titles are
"Product Manager" (342 postings) and "Technical Product Manager" (180
postings as a Role value) — neither carries an explicit AI/ML signal in the
title itself. Requirement 7 is therefore defined as a compound check: a
Product Manager-type title alone is not sufficient — it must co-occur with an
explicit AI/ML signal elsewhere on the posting/candidate profile.

1. 8+ years of relevant experience
2. Bachelor's/Master's/MBA in a relevant field (CS, Data Science, Engineering, Business)
3. Core skill present: "Product Management" or "Product Strategy"
4. AI/ML familiarity present: "Machine Learning," "AI," or "Data Science"
5. Cross-functional skill present: "Stakeholder Management" or "Team Leadership"
6. Technical fluency present: "Python," "SQL," or "Data Analysis"
7. **Job Title or Role contains "Product Manager" (covers "Technical Product
   Manager") AND Skills or Job Description contains an AI/ML signal term**
   ("Machine Learning," "AI," "Artificial Intelligence," "Data Science," or
   "ML"). Title match alone does not satisfy this requirement — both
   conditions must hold.

Applied to `data/filtered_jobs.csv` (2,553 postings), Requirement 7 is met by
**0 postings** — no Product Manager / Technical Product Manager posting in
this filtered set has an AI/ML term in its skills or job description text.
This confirms the dataset has no literal "AI Product Manager"-equivalent
posting; the requirement stands as the model's scoring bar for candidates
going forward, not as a filter that any current posting happens to satisfy.
