# FitScore — Candidate Fit Scoring Dashboard

A Flask dashboard where a recruiter uploads a CSV of candidates, a pre-trained
logistic regression model scores each one's fit for an AI Product Manager
role, and every prediction ships with a plain-language explanation derived
from the model's own coefficients (no black-box score, no auto-rejection).

## Project layout

```
app.py                  Flask app (the deployed dashboard)
templates/, static/     Dashboard HTML/CSS/JS
models/                 Pre-trained model (logistic_regression.pkl)
src/                    Data pipeline + model training scripts
data/                   Filtered postings, labeled training data, sample uploads
reports/                EDA / labeling / model evaluation write-ups
tests/                  Automated tests (see Testing below)
```

## Setup

**Windows (PowerShell / cmd):**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the app

```
python app.py
```

Then open `http://127.0.0.1:5000/`.

> **Note (Mac):** port 5000 is often already used by AirPlay Receiver on
> macOS. If the app fails to bind to port 5000, either disable AirPlay
> Receiver (System Settings → General → AirDrop & Handoff) or run on a
> fallback port instead, e.g. `flask --app app run --port 5001`, and open
> `http://127.0.0.1:5001/`.

## Demo data

Upload any of the sample CSVs in `data/` to see the dashboard populated:

- `sample_upload_small.csv` — 30 candidates (3 pages)
- `sample_upload_medium.csv` — 150 candidates (15 pages)
- `sample_upload_large.csv` — 600 candidates (60 pages, for stress-testing pagination/filtering)

Each CSV must contain the columns `Experience`, `Qualifications`, `Skills`,
`Prior Job Title` (an optional `Candidate ID` column is used for display if
present).

## Testing

```
python -m unittest discover -s tests -v
```

Covers model loading, known-input predictions (including the input
validation and explanation logic), and the `/decision/<id>` endpoint for
both valid and invalid candidate ids.

## License

MIT — see [LICENSE](LICENSE).
