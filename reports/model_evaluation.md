# Model Evaluation - Fit_Label Classification (MLDLC Step 6)

## 1. Data inspection

Loaded `data\labeled_candidates.csv`: 250 rows, 13 columns.

Columns: ['Experience', 'Qualifications', 'Skills', 'Prior Job Title', 'Req1_Experience', 'Req2_Education', 'Req3_CoreSkill', 'Req4_AIML', 'Req5_CrossFunctional', 'Req6_Technical', 'Req7_TitleAndAI', 'Total_Score', 'Fit_Label']

Dtypes:
```
Experience                str
Qualifications            str
Skills                    str
Prior Job Title           str
Req1_Experience           str
Req2_Education            str
Req3_CoreSkill            str
Req4_AIML                 str
Req5_CrossFunctional      str
Req6_Technical            str
Req7_TitleAndAI           str
Total_Score             int64
Fit_Label                 str
```

`Experience` format check - every value matches `'<int> Years'`: True. Example values: ['14 Years', '0 Years', '10 Years']. This is text, not numeric -> needs parsing before use.

`Qualifications` - 20 unique values, already short categorical strings (e.g. ['B.Sc Physics', 'B.Tech Computer Science', 'B.A. History']) -> no parsing needed, just encoding.

`Prior Job Title` - 16 unique values (e.g. ['Data Analyst', 'Group Product Manager', 'Software Engineer']) -> no parsing needed, just encoding.

`Skills` - free-form comma-separated string per row (e.g. "Communication, Market Research, Python, Negotiation, Roadmapping, Salesforce") -> multi-label field, needs tokenizing, not single-category encoding.

Excluding leakage columns before modeling: ['Req1_Experience', 'Req2_Education', 'Req3_CoreSkill', 'Req4_AIML', 'Req5_CrossFunctional', 'Req6_Technical', 'Req7_TitleAndAI', 'Total_Score']. These are the Met/Partial/Gap rubric outcomes and their sum - each is a deterministic function of the same raw fields used to compute Fit_Label, so including them would let the model 'cheat' instead of learning the pattern itself.

## 2. Feature encoding

| Column | Type | Encoding | Why |
|---|---|---|---|
| `Experience_Years` (parsed from `Experience`) | numeric | `StandardScaler` | Helps Logistic Regression's gradient-based fit and its L2 regularization treat this feature fairly against the one-hot columns; a no-op for the Decision Tree since axis-aligned splits are scale-invariant, so one shared preprocessing pipeline works for both models. |
| `Qualifications` | categorical, ~20 distinct degree strings | `OneHotEncoder(handle_unknown="ignore")` | Nominal category, no ordinal relationship between degree names (a `PhD` isn't "more" than a `B.Tech` in a linearly-encodable sense here). |
| `Prior Job Title` | categorical, ~16 distinct titles | `OneHotEncoder(handle_unknown="ignore")` | Same reasoning as Qualifications - nominal, low cardinality. |
| `Skills` | free text, comma-separated multi-label field | `CountVectorizer` (custom comma tokenizer, `binary=True`) | Each candidate has *multiple* skills, so this isn't a single category - one-hot doesn't fit. `binary=True` bag-of-skills (presence/absence) is used instead of TF-IDF because the vocabulary is small and fixed and every skill token is equally diagnostic; there's no long-document "common word" problem here that IDF weighting would fix. |

## 3. Train/test split

80/20 split, stratified on `Fit_Label` (`random_state=42`).

| Class | Full (n=250) | Train (n=200) | Test (n=50) |
|---|---|---|---|
| Likely Not a Fit | 73 (29.2%) | 58 (29.0%) | 15 (30.0%) |
| Needs Review | 113 (45.2%) | 91 (45.5%) | 22 (44.0%) |
| Strong Fit | 64 (25.6%) | 51 (25.5%) | 13 (26.0%) |

## Models trained

- **Logistic Regression** (`max_iter=1000`, `class_weight="balanced"`): linear baseline; `class_weight="balanced"` re-weights the loss for the thinner classes given the 25.6/45.2/29.2 split on only 250 rows.
- **Decision Tree** (`max_depth=5`, `min_samples_leaf=5`, `class_weight="balanced"`): depth/leaf caps are deliberate - with only 200 training rows an unconstrained tree overfits and produces a tree too large to actually read, which would defeat the reason for choosing a tree (explainability) in the first place.

## 4. Evaluation - Logistic Regression

Precision / recall / F1 (held-out 20%, n=50):
```
                  precision    recall  f1-score   support

Likely Not a Fit       0.94      1.00      0.97        15
    Needs Review       0.85      0.77      0.81        22
      Strong Fit       0.71      0.77      0.74        13

        accuracy                           0.84        50
       macro avg       0.83      0.85      0.84        50
    weighted avg       0.84      0.84      0.84        50

```
ROC-AUC (multi-class, one-vs-rest, macro-averaged): **0.941**

Confusion matrix (rows = actual, columns = predicted):

| Actual \ Predicted | Likely Not a Fit | Needs Review | Strong Fit |
|---|---|---|---|
| Likely Not a Fit | 15 | 0 | 0 |
| Needs Review | 1 | 17 | 4 |
| Strong Fit | 0 | 3 | 10 |

## 4. Evaluation - Decision Tree

Precision / recall / F1 (held-out 20%, n=50):
```
                  precision    recall  f1-score   support

Likely Not a Fit       0.93      0.87      0.90        15
    Needs Review       0.75      0.55      0.63        22
      Strong Fit       0.55      0.85      0.67        13

        accuracy                           0.72        50
       macro avg       0.74      0.75      0.73        50
    weighted avg       0.75      0.72      0.72        50

```
ROC-AUC (multi-class, one-vs-rest, macro-averaged): **0.872**

Confusion matrix (rows = actual, columns = predicted):

| Actual \ Predicted | Likely Not a Fit | Needs Review | Strong Fit |
|---|---|---|---|
| Likely Not a Fit | 13 | 2 | 0 |
| Needs Review | 1 | 12 | 9 |
| Strong Fit | 0 | 2 | 11 |

## 5. Recommendation

Logistic Regression: ROC-AUC=0.941, Strong Fit recall=0.769, macro F1=0.839.
Decision Tree: ROC-AUC=0.872, Strong Fit recall=0.846, macro F1=0.732.

**Recommendation: Logistic Regression.** It leads on ROC-AUC by 0.069, which is large enough on a 50-row test set to matter more than the explainability edge of the alternative. Note the tradeoff taken: Logistic Regression's coefficients are less directly traceable per-decision than the Decision Tree's explicit branches, though they're still inspectable (sign and magnitude per encoded feature).

## Saved artifacts

- `models\logistic_regression.pkl`
- `models\decision_tree.pkl`

Both pipelines are self-contained (raw columns in, prediction out) - preprocessing is baked into the pickled object, so no manual re-encoding is needed at load time.
