"""API-level tests against the real trained artifacts. Run from the repo root:
    python -m unittest tests.test_end_to_end
Requires: merged graph (Module 1), TransE + DDI checkpoints (Module 2)."""

import unittest

from fastapi.testclient import TestClient

from app.backend.main import app

WILSON = "Disease::CUSTOM:wilsons_disease"
SCA3 = "Disease::CUSTOM:sca3"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()  # runs lifespan (loads models)

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_health(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["ddi_model_loaded"])

    def test_custom_rare_diseases_listed_first(self):
        diseases = self.client.get("/api/diseases?limit=10").json()
        self.assertTrue(all(d["custom"] for d in diseases[:4]))
        self.assertEqual(len([d for d in diseases if d["custom"]]), 4)

    def test_repurposing_excludes_known_treatments_and_explains(self):
        r = self.client.get(f"/api/repurposing?disease_id={WILSON}&top_k=5").json()
        names = {c["compound_name"] for c in r["candidates"]}
        self.assertFalse(names & {"Penicillamine", "Trientine", "Zinc acetate"})
        self.assertEqual(len(r["candidates"]), 5)
        for c in r["candidates"]:
            self.assertEqual(c["has_evidence_path"], bool(c["evidence_paths"]))

    def test_sparse_disease_gets_low_confidence_note(self):
        r = self.client.get(f"/api/repurposing?disease_id={SCA3}").json()
        self.assertTrue(any("low-confidence" in n for n in r["notes"]))

    def test_unknown_disease_404(self):
        self.assertEqual(self.client.get("/api/repurposing?disease_id=Disease::nope").status_code, 404)

    def test_neglect_weight_shifts_picks_toward_untreated_diseases(self):
        low = self.client.get("/api/equity?lam=0").json()["neglect_aware_ranking"]
        high = self.client.get("/api/equity?lam=0.5").json()["neglect_aware_ranking"]
        self.assertGreater(high["share_to_untreated_diseases"], low["share_to_untreated_diseases"])

    def test_evidence_candidates_are_disease_specific(self):
        r = self.client.get(f"/api/repurposing?disease_id={WILSON}").json()
        names = {c["compound_name"] for c in r["evidence_candidates"]}
        self.assertTrue(names & {"Carboplatin", "Cisplatin", "Oxaliplatin"})  # bind ATP7B
        for c in r["evidence_candidates"]:
            self.assertTrue(c["evidence_paths"])

    def test_disease_without_graph_links_has_no_evidence_candidates(self):
        r = self.client.get("/api/repurposing?disease_id=Disease::CUSTOM:ashermans_syndrome").json()
        self.assertEqual(r["evidence_candidates"], [])

    def _two_covered_drugs(self):
        covered = [c["id"] for c in self.client.get("/api/compounds?limit=500").json() if c["ddi_covered"]]
        self.assertGreaterEqual(len(covered), 2)
        return covered[0], covered[1]

    def test_ddi_covered_pair_has_score(self):
        a, b = self._two_covered_drugs()
        r = self.client.get(f"/api/ddi?compound_1={a}&compound_2={b}").json()
        self.assertTrue(r["covered"])
        self.assertTrue(0.0 <= r["interaction_score"] <= 1.0)

    def test_ddi_uncovered_drug_not_guessed(self):
        a, _ = self._two_covered_drugs()
        r = self.client.get(f"/api/ddi?compound_1={a}&compound_2=Compound::CUSTOM:trientine").json()
        self.assertFalse(r["covered"])
        self.assertIsNone(r["interaction_score"])

    def test_patient_safety_flow(self):
        a, b = self._two_covered_drugs()
        r = self.client.post("/api/patient-safety", json={"candidate": a, "current_medications": [b]}).json()
        self.assertIn(r["overall"], {"review_required", "caution", "no_flags"})
        self.assertEqual(len(r["checks"]), 1)
        self.assertIn("clinician", r["disclaimer"])

    def test_patient_safety_unassessable_is_incomplete_not_clear(self):
        a, _ = self._two_covered_drugs()
        r = self.client.post("/api/patient-safety",
                             json={"candidate": a, "current_medications": ["Compound::CUSTOM:trientine"]}).json()
        self.assertEqual(r["overall"], "incomplete")

    def test_patient_safety_unknown_compound_404(self):
        r = self.client.post("/api/patient-safety", json={"candidate": "Compound::nope", "current_medications": []})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
