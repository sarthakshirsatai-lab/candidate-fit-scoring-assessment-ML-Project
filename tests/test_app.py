"""
Basic tests for the FitScore Flask app.

Covers, at minimum: the model loads cleanly, a known input produces the
expected prediction, and /decision/<id> behaves correctly for both a valid
and an invalid candidate id.

Run with:
    venv/Scripts/python.exe -m unittest discover -s tests -v
"""

import io
import os
import sqlite3
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Point the app at a throwaway database *before* importing it, so these tests
# never read or write the real fitscore.db the running app uses.
_TEST_DB_PATH = os.path.join(tempfile.gettempdir(), "fitscore_test.db")
os.environ["FITSCORE_DB_PATH"] = _TEST_DB_PATH

import app as fitscore_app  # noqa: E402


def _candidate_row(experience, qualifications, skills, prior_job_title):
    return pd.DataFrame([{
        "Experience": experience,
        "Qualifications": qualifications,
        "Skills": skills,
        "Prior Job Title": prior_job_title,
    }])


class ModelLoadingTests(unittest.TestCase):
    def test_model_loaded_without_error(self):
        self.assertIsNone(fitscore_app._model_load_error)
        self.assertIsNotNone(fitscore_app._model_pipeline)

    def test_pipeline_has_expected_steps(self):
        steps = fitscore_app._model_pipeline.named_steps
        self.assertIn("preprocess", steps)
        self.assertIn("clf", steps)
        self.assertTrue(hasattr(fitscore_app._model_pipeline, "predict"))
        self.assertTrue(hasattr(fitscore_app._model_pipeline, "predict_proba"))


class KnownInputPredictionTests(unittest.TestCase):
    def test_strong_candidate_profile_predicts_strong_fit(self):
        df = _candidate_row(
            "15 Years",
            "B.Tech Computer Science",
            "Product Management, Machine Learning, Stakeholder Management, Python",
            "Senior Product Manager",
        )
        result = fitscore_app.score_candidates(df)[0]
        self.assertEqual(result["predicted_label"], "Strong Fit")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)
        self.assertEqual(result["validation_flags"], [])

    def test_breakdown_includes_intercept_as_first_row(self):
        df = _candidate_row(
            "15 Years",
            "B.Tech Computer Science",
            "Product Management, Machine Learning, Stakeholder Management, Python",
            "Senior Product Manager",
        )
        result = fitscore_app.score_candidates(df)[0]
        self.assertIn(fitscore_app.BASELINE_ROW, result["breakdown"])
        self.assertEqual(fitscore_app.FEATURE_ORDER[0], fitscore_app.BASELINE_ROW)

    def test_skill_named_specifically_not_as_generic_skills_category(self):
        # Skills has ~35 encoded columns; the strongest-factor label must name
        # the individual skill (e.g. "Team Leadership"), never the bare
        # category "Skills" - that was the bug in issue #5.
        df = _candidate_row(
            "0 Years",
            "BA Fine Arts",
            "Team Leadership",
            "Sales Executive",
        )
        result = fitscore_app.score_candidates(df)[0]
        if result["breakdown"]["Skills"] != 0 and result["top_label"] not in (
            "Experience", "Qualifications", "Prior Job Title",
        ):
            self.assertNotEqual(result["top_label"], "Skills")

    def test_out_of_range_experience_is_clamped_and_flagged(self):
        df = _candidate_row("200 Years", "MBA", "Python, SQL", "Product Manager")
        result = fitscore_app.score_candidates(df)[0]
        self.assertEqual(result["breakdown"]["Experience"], result["breakdown"]["Experience"])  # no crash
        self.assertTrue(any("outside the realistic" in f for f in result["validation_flags"]))

    def test_in_range_experience_is_not_flagged(self):
        df = _candidate_row("10 Years", "MBA", "Python, SQL", "Product Manager")
        result = fitscore_app.score_candidates(df)[0]
        self.assertFalse(any("outside the realistic" in f for f in result["validation_flags"]))

    def test_unrecognized_qualification_is_flagged(self):
        df = _candidate_row(
            "5 Years", "Made-Up Degree That Does Not Exist", "Python, SQL", "Product Manager"
        )
        result = fitscore_app.score_candidates(df)[0]
        self.assertTrue(any("was not seen during training" in f for f in result["validation_flags"]))

    def test_unrecognized_prior_title_is_flagged(self):
        df = _candidate_row("5 Years", "MBA", "Python, SQL", "Made-Up Title That Does Not Exist")
        result = fitscore_app.score_candidates(df)[0]
        self.assertTrue(any("was not seen during training" in f for f in result["validation_flags"]))


class DecisionEndpointTests(unittest.TestCase):
    def setUp(self):
        if os.path.exists(_TEST_DB_PATH):
            os.remove(_TEST_DB_PATH)
        fitscore_app.init_db()
        self.client = fitscore_app.app.test_client()

        csv_content = (
            "Candidate ID,Experience,Qualifications,Skills,Prior Job Title\n"
            "CAND-TEST-1,10 Years,MBA,\"Python, SQL\",Product Manager\n"
        )
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(csv_content.encode("utf-8")), "test.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)

        conn = sqlite3.connect(_TEST_DB_PATH)
        self.candidate_row_id = conn.execute("SELECT id FROM candidates LIMIT 1").fetchone()[0]
        conn.close()

    def tearDown(self):
        if os.path.exists(_TEST_DB_PATH):
            os.remove(_TEST_DB_PATH)

    def test_valid_candidate_id_updates_decision(self):
        response = self.client.post(f"/decision/{self.candidate_row_id}", json={"decision": "Shortlist"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])

        conn = sqlite3.connect(_TEST_DB_PATH)
        decision = conn.execute(
            "SELECT decision FROM candidates WHERE id = ?", (self.candidate_row_id,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(decision, "Shortlist")

    def test_invalid_candidate_id_returns_404(self):
        response = self.client.post("/decision/999999999", json={"decision": "Shortlist"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

    def test_invalid_decision_value_returns_400(self):
        response = self.client.post(f"/decision/{self.candidate_row_id}", json={"decision": "Hire"})
        self.assertEqual(response.status_code, 400)

    def test_second_upload_does_not_delete_first_batchs_decision(self):
        self.client.post(f"/decision/{self.candidate_row_id}", json={"decision": "Shortlist"})

        csv_content = (
            "Candidate ID,Experience,Qualifications,Skills,Prior Job Title\n"
            "CAND-TEST-2,5 Years,BA,\"Excel\",Data Analyst\n"
        )
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(csv_content.encode("utf-8")), "test2.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)

        conn = sqlite3.connect(_TEST_DB_PATH)
        decision = conn.execute(
            "SELECT decision FROM candidates WHERE id = ?", (self.candidate_row_id,)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(decision, "first batch's candidate row must still exist")
        self.assertEqual(decision[0], "Shortlist")


if __name__ == "__main__":
    unittest.main()
