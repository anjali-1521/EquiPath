"""Drug repurposing candidate prediction for EquiPath.

Uses the trained TransE embeddings (models/checkpoints/transe/), not
GraphSAGE: TransE gives every relation - including CtD, "Compound treats
Disease" - its own translation vector, so we can directly score "does
translating this compound's embedding by the treats-vector land near this
disease?" for every compound in the graph. GraphSAGE's embeddings have no
such per-relation decoder (see graphsage.py's docstring), so they aren't a
good fit for this specific, relation-specific question.

For a target disease d, every compound c is scored by the TransE distance
score(c, CtD, d) = -||e_c + e_CtD - e_d|| (higher = closer = more likely to
treat), compounds already known to treat d are excluded, and the top-K
remaining compounds are returned as repurposing candidates.

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


def rank_repurposing_candidates(
    disease_id: str,
    graph,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    entity_embeddings: np.ndarray,
    relation_embeddings: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    if disease_id not in entity_to_id:
        raise KeyError(f"Disease not in graph: {disease_id}")
    if CTD_RELATION not in relation_to_id:
        raise KeyError(f"Relation {CTD_RELATION} not in graph")

    disease_vec = entity_embeddings[entity_to_id[disease_id]]
    ctd_vec = relation_embeddings[relation_to_id[CTD_RELATION]]

    known_treaters = {
        u for u, v, data in graph.in_edges(disease_id, data=True) if data.get("metaedge") == CTD_RELATION
    }

    compound_ids = [n for n, d in graph.nodes(data=True) if d.get("kind") == "Compound"]
    candidates = [c for c in compound_ids if c not in known_treaters]

    compound_indices = np.array([entity_to_id[c] for c in candidates])
    compound_vecs = entity_embeddings[compound_indices]  # (num_candidates, dim)

    scores = -np.linalg.norm(compound_vecs + ctd_vec - disease_vec, axis=1)

    order = np.argsort(-scores)[:top_k]
    results = []
    for rank, idx in enumerate(order, start=1):
        compound_id = candidates[idx]
        results.append({
            "rank": rank,
            "compound_id": compound_id,
            "compound_name": graph.nodes[compound_id].get("name", compound_id),
            "score": float(scores[idx]),
        })
    return results


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
    entity_embeddings = np.load(TRANSE_DIR / "entity_embeddings.npy")
    relation_embeddings = np.load(TRANSE_DIR / "relation_embeddings.npy")

    targets = list(CUSTOM_DISEASES) if args.disease == "all" else [args.disease]

    for disease_id in targets:
        name = CUSTOM_DISEASES.get(disease_id, graph.nodes.get(disease_id, {}).get("name", disease_id))
        print(f"\n=== Repurposing candidates for {name} [{disease_id}] ===")
        results = rank_repurposing_candidates(
            disease_id, graph, entity_to_id, relation_to_id,
            entity_embeddings, relation_embeddings, top_k=args.top_k,
        )
        for r in results:
            print(f"  #{r['rank']:>2}  {r['compound_name']:<40} score={r['score']:.4f}  [{r['compound_id']}]")


if __name__ == "__main__":
    main()
