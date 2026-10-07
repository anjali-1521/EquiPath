"""Drug repurposing candidate prediction for EquiPath.

Scores every compound for a disease with the supervised treatment head
(models/prediction/treats_head.py), which is trained on TransE embeddings and
beat TransE's raw CtD distance on held-out treatments (see that module for the
numbers). If no head has been trained it falls back to the raw TransE distance
score(c, CtD, d) = -||e_c + e_CtD - e_d||. Compounds already known to treat
the disease are excluded and the top-K remaining are returned.

This is the module's real payoff for the project's rare-disease framing:
for the 3 diseases with no approved treatment (SCA3, Asherman's, MRKH), the
candidates below are genuine model output, not a demo of something already
known - they are exactly the kind of testable hypothesis a repurposing
pipeline exists to generate, and should be read as such (untested
computational leads), not as medical advice.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from models.embeddings.graph_utils import build_vocab, load_merged_graph

ROOT = Path(__file__).resolve().parents[2]
TRANSE_DIR = ROOT / "models" / "checkpoints" / "transe"

CTD_RELATION = "CtD"  # Compound - treats - Disease

CUSTOM_DISEASES = {
    "Disease::CUSTOM:wilsons_disease": "Wilson's disease",
    "Disease::CUSTOM:sca3": "SCA3 / Machado-Joseph disease",
    "Disease::CUSTOM:ashermans_syndrome": "Asherman's syndrome",
    "Disease::CUSTOM:mrkh_syndrome": "MRKH syndrome",
}


def rank_repurposing_candidates(disease_id: str, graph, entity_to_id, relation_to_id, top_k: int = 10):
    """Top-K untreated-by-record compounds for a disease, plus the scoring method used."""
    from models.prediction.treats_head import full_score_matrix

    if disease_id not in entity_to_id:
        raise KeyError(f"Disease not in graph: {disease_id}")
    known_treaters = {
        u for u, v, data in graph.in_edges(disease_id, data=True) if data.get("metaedge") == CTD_RELATION
    }
    compounds = [n for n, d in graph.nodes(data=True) if d.get("kind") == "Compound" and n not in known_treaters]
    scores, method = full_score_matrix(graph, entity_to_id, relation_to_id, [disease_id], compounds)
    order = np.argsort(-scores[0])[:top_k]
    return [{"rank": r, "compound_id": compounds[i], "compound_name": graph.nodes[compounds[i]].get("name"),
             "score": float(scores[0][i])} for r, i in enumerate(order, start=1)], method


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank drug repurposing candidates for a disease")
    parser.add_argument(
        "--disease", default="all",
        help="Disease node ID (e.g. Disease::CUSTOM:sca3), or 'all' for the 4 custom rare diseases",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print("Loading graph, vocabulary, and TransE embeddings...")
    graph = load_merged_graph()
    entity_to_id, id_to_entity, relation_to_id, id_to_relation = build_vocab(graph)
    targets = list(CUSTOM_DISEASES) if args.disease == "all" else [args.disease]

    for disease_id in targets:
        name = CUSTOM_DISEASES.get(disease_id, graph.nodes.get(disease_id, {}).get("name", disease_id))
        print(f"\n=== Repurposing candidates for {name} [{disease_id}] ===")
        results, method = rank_repurposing_candidates(disease_id, graph, entity_to_id, relation_to_id, args.top_k)
        print(f"  (scoring: {method})")
        for r in results:
            print(f"  #{r['rank']:>2}  {r['compound_name']:<40} score={r['score']:.4f}  [{r['compound_id']}]")


if __name__ == "__main__":
    main()
