import unittest

import networkx as nx
import numpy as np

from explanation.explanation_path import ExplanationEngine
from explanation.neglect_ranking import disease_neglect, equity_audit, rerank, top_pairs


def toy_graph() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n, kind in [("Compound::A", "Compound"), ("Compound::B", "Compound"),
                    ("Disease::X", "Disease"), ("Disease::Y", "Disease"),
                    ("Gene::1", "Gene"), ("Gene::2", "Gene")]:
        g.add_node(n, kind=kind, name=n.split("::")[1])
    g.add_edge("Compound::A", "Gene::1", metaedge="CbG")
    g.add_edge("Disease::X", "Gene::1", metaedge="DaG")
    g.add_edge("Compound::B", "Compound::A", metaedge="CrC")
    g.add_edge("Compound::A", "Disease::Y", metaedge="CtD")
    return g


class ExplanationTests(unittest.TestCase):
    def test_binding_path_found_with_readable_text(self):
        paths = ExplanationEngine(toy_graph()).explain("Compound::A", "Disease::X")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["type"], "binds-associated-gene")
        self.assertIn("A binds 1, a gene associated with X", paths[0]["text"])

    def test_no_path_returns_empty_not_error(self):
        self.assertEqual(ExplanationEngine(toy_graph()).explain("Compound::B", "Disease::X"), [])

    def test_resemblance_to_known_treatment(self):
        paths = ExplanationEngine(toy_graph()).explain("Compound::B", "Disease::Y")
        self.assertEqual([p["type"] for p in paths], ["similar-to-treatment"])

    def test_hub_gene_scores_lower_than_specific_gene(self):
        g = toy_graph()
        g.add_edge("Compound::A", "Gene::2", metaedge="CbG")
        g.add_edge("Disease::X", "Gene::2", metaedge="DaG")
        for i in range(50):  # make Gene::1 a hub
            g.add_node(f"Disease::H{i}", kind="Disease", name=f"H{i}")
            g.add_edge(f"Disease::H{i}", "Gene::1", metaedge="DaG")
        paths = ExplanationEngine(g).explain("Compound::A", "Disease::X")
        self.assertEqual(paths[0]["nodes"][1], "Gene::2")


class RankingTests(unittest.TestCase):
    def test_per_disease_normalization_removes_scale_differences(self):
        scores = np.array([[-1.0, -2.0, -3.0], [-10.0, -20.0, -30.0]])
        final = rerank(scores, np.zeros(2), np.zeros_like(scores, dtype=bool), lam=0.0)
        np.testing.assert_allclose(final[0], final[1], atol=1e-6)

    def test_known_treatments_excluded(self):
        scores = np.array([[0.0, -1.0]])
        known = np.array([[True, False]])
        final = rerank(scores, np.zeros(1), known, lam=0.0)
        self.assertEqual(top_pairs(final, 1)[0][1], 1)

    def test_neglect_bonus_lifts_untreated_low_degree_disease(self):
        g = toy_graph()
        diseases = ["Disease::X", "Disease::Y"]  # X untreated, Y treated by A
        neglect, _, untreated = disease_neglect(g, diseases)
        self.assertEqual(list(untreated), [1.0, 0.0])
        self.assertGreater(neglect[0], neglect[1])
        scores = np.array([[-1.0, -2.0], [-1.0, -2.0]])
        none_known = np.zeros_like(scores, dtype=bool)
        self.assertEqual(top_pairs(rerank(scores, neglect, none_known, lam=5.0), 1)[0][0], 0)

    def test_equity_audit_shares(self):
        degrees = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
        untreated = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        audit = equity_audit([(4, 0, 0.0), (0, 0, 0.0)], degrees, untreated)
        self.assertEqual(audit["share_to_best_connected_20pct"], 0.5)
        self.assertEqual(audit["share_to_untreated_diseases"], 0.5)
        self.assertEqual(audit["distinct_diseases"], 2)


if __name__ == "__main__":
    unittest.main()
