# Labeling Summary — Synthetic Candidate Scoring

250 synthetic candidates generated (Experience, Qualifications, Skills, Prior Job Title), scored against all 7 requirements in `reference/jd_ai_product_manager.md` (Met=2, Partial=1, Gap=0), summed to a Total_Score (max 14), and labeled by threshold: Strong Fit ≥11, Needs Review 6-10, Likely Not a Fit ≤5.

## Class balance (actual, computed)

- **Needs Review**: 113 (45.2%)
- **Likely Not a Fit**: 73 (29.2%)
- **Strong Fit**: 64 (25.6%)

## Generation vs. actual label
Candidates were generated toward ~25/45/30 targets, but labels are always computed live from the fields, not assigned — actual counts differ slightly from intent due to randomized attribute variance:

```
Fit_Label                    Likely Not a Fit  Needs Review  Strong Fit
_intended_archetype                                                    
Likely Not a Fit (intended)                51             1           0
Needs Review (intended)                    22           107          21
Strong Fit (intended)                       0             5          43
```

## Example candidates per tier

### Strong Fit

- **Customer Success Manager**, 15 Years, B.Sc Mathematics. Skills: Team Leadership, Data Science, A/B Testing, SQL, Customer Research, Public Speaking, Project Management. Score: 11/14 (Exp=Met, Edu=Partial, Core=Partial, AI/ML=Met, Cross-fn=Met, Tech=Met, Req7=Partial).
- **Group Product Manager**, 11 Years, M.Sc Statistics. Skills: Product Strategy, SQL, JIRA, Data Analytics, Stakeholder Management, Roadmapping. Score: 11/14 (Exp=Met, Edu=Partial, Core=Met, AI/ML=Partial, Cross-fn=Met, Tech=Met, Req7=Partial).
- **Senior Product Manager**, 9 Years, B.Tech Computer Science. Skills: Confluence, Stakeholder Management, Customer Research, Data Analysis, Artificial Intelligence, Product Management, Market Research, UX Research. Score: 14/14 (Exp=Met, Edu=Met, Core=Met, AI/ML=Met, Cross-fn=Met, Tech=Met, Req7=Met).

### Needs Review

- **Data Analyst**, 14 Years, B.Sc Physics. Skills: Communication, Market Research, Python, Negotiation, Roadmapping, Salesforce. Score: 6/14 (Exp=Met, Edu=Partial, Core=Gap, AI/ML=Gap, Cross-fn=Partial, Tech=Met, Req7=Gap).
- **Group Product Manager**, 0 Years, B.Tech Computer Science. Skills: Market Research, Business Intelligence, JIRA, Product Strategy, Roadmapping, Stakeholder Management, Power BI, Customer Research. Score: 9/14 (Exp=Gap, Edu=Met, Core=Met, AI/ML=Partial, Cross-fn=Met, Tech=Partial, Req7=Partial).
- **Software Engineer**, 10 Years, B.A. History. Skills: SQL, Business Intelligence, A/B Testing, Stakeholder Management, Customer Research. Score: 7/14 (Exp=Met, Edu=Gap, Core=Gap, AI/ML=Partial, Cross-fn=Met, Tech=Met, Req7=Gap).

### Likely Not a Fit

- **Sales Executive**, 5 Years, Bachelor of Hospitality Management. Skills: A/B Testing, Customer Research, Business Intelligence, Roadmapping. Score: 2/14 (Exp=Partial, Edu=Gap, Core=Gap, AI/ML=Partial, Cross-fn=Gap, Tech=Gap, Req7=Gap).
- **Business Analyst**, 7 Years, BA Fine Arts. Skills: Product Design, Market Research, Business Intelligence, Roadmapping. Score: 3/14 (Exp=Partial, Edu=Gap, Core=Partial, AI/ML=Partial, Cross-fn=Gap, Tech=Gap, Req7=Gap).
- **Sales Executive**, 3 Years, BA Fine Arts. Skills: Scrum, Collaboration, Agile, Statistics. Score: 2/14 (Exp=Gap, Edu=Gap, Core=Gap, AI/ML=Partial, Cross-fn=Partial, Tech=Gap, Req7=Gap).
