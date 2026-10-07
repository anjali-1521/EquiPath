"""Equity-aware re-ranking of repurposing predictions (Module 3).

Problem: a global ranking of (compound, disease) pairs by raw model score can
concentrate attention on diseases that are already well-studied, and a
well-connected disease gets a better-positioned embedding than a rare disease
with a handful of edges. Whether that actually happens depends on the scoring
model, so this module measures it instead of assuming it (see equity_audit):
with the supervised head used here the raw ranking is NOT skewed toward the
best-connected diseases, but it IS dominated by a few generally popular drugs
(gemcitabine, corticosteroids) that head almost every disease's list, which is
a different kind of bias the audit does not capture.

Two corrections, both deliberately simple and inspectable:

1. Per-disease normalization. Each disease's scores over all compounds are
   z-scored, so "unusually good for THIS disease" is comparable across
   diseases regardless of how well-connected the disease is.
2. Neglect bonus. final = z + lam * neglect, where
       neglect = 0.5 * (1 - degree_percentile) + 0.5 * (1 if no known treatment else 0)
   i.e. half "little is known about it in the graph", half "nothing treats
   it". lam is a tunable dial: 0 = pure model ranking, larger = push
   neglected diseases up. It is a prioritization policy, not a prediction of
   efficacy, and the audit below reports exactly how far it moves things.
   It overcorrects quickly: at lam >= 1 every top-100 pick goes to an
   untreated disease, so the default is a gentle 0.25.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRANSE_DIR = ROOT / "models" / "checkpoints" / "transe"


def score_matrix(entity_emb, relation_emb, entity_to_id, relation_to_id, diseases, compounds):
    """(num_diseases x num_compounds) TransE CtD scores; higher = more likely to treat."""
    d = entity_emb[[entity_to_id[x] for x in diseases]]
    c = entity_emb[[entity_to_id[x] for x in compounds]]
    ctd = relation_emb[relation_to_id["CtD"]]
    # ||c + ctd - d|| for every (d, c) pair, via broadcasting
    diff = c[None, :, :] + ctd[None, None, :] - d[:, None, :]
    return -np.linalg.norm(diff, axis=2)


def disease_neglect(graph, diseases) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (neglect scores in [0,1], degrees, untreated flags) aligned to `diseases`."""
    degrees = np.array([graph.degree(d) for d in diseases], dtype=float)
    untreated = np.array(
        [0.0 if any(e.get("metaedge") == "CtD" for _, _, e in graph.in_edges(d, data=True)) else 1.0
         for d in diseases]
    )
    log_deg = np.log1p(degrees)
    # percentile rank of connectivity, 0 = least connected
    pct = log_deg.argsort().argsort() / max(len(diseases) - 1, 1)
    neglect = 0.5 * (1.0 - pct) + 0.5 * untreated
    return neglect, degrees, untreated


def rerank(scores: np.ndarray, neglect: np.ndarray, known: np.ndarray, lam: float) -> np.ndarray:
    """Per-disease z-score + neglect bonus; known-treated pairs get -inf."""
    z = (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-9)
    final = z + lam * neglect[:, None]
    final[known] = -np.inf
    return final


def top_pairs(matrix: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    flat = np.argsort(-matrix, axis=None)[:k]
    rows, cols = np.unravel_index(flat, matrix.shape)
    return [(int(r), int(c), float(matrix[r, c])) for r, c in zip(rows, cols)]


def equity_audit(pairs, degrees, untreated) -> dict:
    """Where does a top-k list send its attention? Fractions of picks going to
    the best-connected fifth of diseases, to untreated diseases, and how many
    distinct diseases appear at all."""
    idx = np.array([r for r, _, _ in pairs])
    top_fifth_cut = np.quantile(degrees, 0.8)
    return {
        "k": len(pairs),
        "share_to_best_connected_20pct": float(np.mean(degrees[idx] >= top_fifth_cut)),
        "share_to_untreated_diseases": float(np.mean(untreated[idx] == 1.0)),
        "distinct_diseases": int(len(set(idx.tolist()))),
    }


def build_context(graph, entity_to_id, relation_to_id, entity_emb, relation_emb):
    diseases = sorted(n for n, d in graph.nodes(data=True) if d.get("kind") == "Disease")
    compounds = sorted(n for n, d in graph.nodes(data=True) if d.get("kind") == "Compound")
    from models.prediction.treats_head import full_score_matrix

    scores, _ = full_score_matrix(graph, entity_to_id, relation_to_id, diseases, compounds)
    c_index = {c: i for i, c in enumerate(compounds)}
    known = np.zeros_like(scores, dtype=bool)
    for i, d in enumerate(diseases):
        for u, _, e in graph.in_edges(d, data=True):
            if e.get("metaedge") == "CtD":
                known[i, c_index[u]] = True
    neglect, degrees, untreated = disease_neglect(graph, diseases)
    return diseases, compounds, scores, known, neglect, degrees, untreated


def main() -> None:
    from models.embeddings.graph_utils import build_vocab, load_merged_graph
    from models.prediction.treats_head import full_score_matrix

    parser = argparse.ArgumentParser(description="Neglect-aware repurposing ranking + equity audit")
    parser.add_argument("--lam", type=float, default=0.25, help="neglect bonus weight")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    graph = load_merged_graph()
    entity_to_id, _, relation_to_id, _ = build_vocab(graph)
    E = np.load(TRANSE_DIR / "entity_embeddings.npy")
    R = np.load(TRANSE_DIR / "relation_embeddings.npy")
    diseases, compounds, scores, known, neglect, degrees, untreated = build_context(
        graph, entity_to_id, relation_to_id, E, R
    )
    print(f"Scoring method: {full_score_matrix(graph, entity_to_id, relation_to_id, diseases[:1], compounds)[1]}")

    raw = scores.copy()
    raw[known] = -np.inf
    fair = rerank(scores, neglect, known, args.lam)

    for name, mat in [("raw global score ranking", raw), (f"neglect-aware (lam={args.lam})", fair)]:
        audit = equity_audit(top_pairs(mat, args.top_k), degrees, untreated)
        print(f"\n[{name}] top-{audit['k']}")
        print(f"  share to best-connected 20% of diseases: {audit['share_to_best_connected_20pct']:.0%}")
        print(f"  share to diseases with no known treatment: {audit['share_to_untreated_diseases']:.0%}")
        print(f"  distinct diseases represented: {audit['distinct_diseases']}")

    print(f"\nTop 10 neglect-aware pairs (known treatments excluded):")
    for r, c, v in top_pairs(fair, 10):
        print(f"  {graph.nodes[compounds[c]]['name']:<28} -> {graph.nodes[diseases[r]]['name']:<40} {v:.2f}")


if __name__ == "__main__":
    main()
