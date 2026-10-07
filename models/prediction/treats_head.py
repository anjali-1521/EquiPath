"""Supervised "does compound C treat disease D?" head on top of the embeddings.

Why this exists: TransE's raw distance score for the CtD relation turned out to
be weak for repurposing - even a disease's own *training* treatments often
ranked in the lower half of all compounds. "Treats" is a many-to-many relation
(one drug treats many diseases and vice versa), which is TransE's known weak
spot. So instead of reading treatment likelihood straight off the translation
score, we train a small classifier on pair features built from the embeddings:

    [e_c, e_d, e_c * e_d]    (e_c * e_d lets it learn which embedding
                              dimensions of a drug and a disease "match")

using TransE embeddings, GraphSAGE embeddings, or both concatenated.
GraphSAGE matters here: it builds each node's embedding by aggregating its
neighbours, so a sparsely-connected disease still inherits signal from the
genes and anatomy it links to - exactly the situation the rare-disease nodes
are in.

Evaluation uses only treatment edges that were held out of TransE/GraphSAGE
training (models/checkpoints/split_test_triples.npy), so neither the
embeddings nor this head have seen them. For each held-out (compound,
disease) pair we rank the true compound against all compounds not already
known to treat that disease (filtered ranking over ~1.5k candidates - much
harder than the sampled-negative protocol used for the generic link metrics).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.embeddings.graph_utils import build_vocab, load_merged_graph
from models.evaluation.metrics import compute_rank_metrics

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "checkpoints"
OUT_DIR = CKPT / "treats"


def load_embeddings(variant: str) -> np.ndarray:
    parts = []
    if variant in ("transe", "both"):
        parts.append(np.load(CKPT / "transe" / "entity_embeddings.npy"))
    if variant in ("graphsage", "both"):
        parts.append(np.load(CKPT / "graphsage" / "entity_embeddings.npy"))
    if not parts:
        raise ValueError(f"unknown variant {variant!r}")
    # unit-normalise each source so neither dominates by scale
    parts = [p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9) for p in parts]
    return np.concatenate(parts, axis=1)


def pair_features(emb: np.ndarray, c_idx, d_idx) -> np.ndarray:
    c, d = emb[c_idx], emb[d_idx]
    return np.concatenate([c, d, c * d], axis=1)


def ctd_pairs(graph, entity_to_id):
    return [(entity_to_id[u], entity_to_id[v]) for u, v, e in graph.edges(data=True)
            if e["metaedge"] == "CtD"]


def make_training_set(pos, known, compound_ids, neg_per_pos, rng):
    c_idx, d_idx, y = [], [], []
    for c, d in pos:
        c_idx.append(c); d_idx.append(d); y.append(1)
        added = 0
        while added < neg_per_pos:
            cn = int(rng.choice(compound_ids))
            if (cn, d) not in known:
                c_idx.append(cn); d_idx.append(d); y.append(0); added += 1
    return np.array(c_idx), np.array(d_idx), np.array(y)


def evaluate(score_fn, test_pos, known_by_disease, compound_ids):
    ranks = []
    for c, d in test_pos:
        scores = score_fn(compound_ids, np.full(len(compound_ids), d))
        true_score = scores[list(compound_ids).index(c)]
        others = np.array([s for cid, s in zip(compound_ids, scores)
                           if cid != c and cid not in known_by_disease[d]])
        ranks.append(1 + int((others > true_score).sum()))
    return compute_rank_metrics(ranks)


def run(variant: str, neg_per_pos: int = 5, C: float = 1.0, seed: int = 42, save: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    graph = load_merged_graph()
    entity_to_id, _, relation_to_id, _ = build_vocab(graph)
    emb = load_embeddings(variant)

    compound_ids = np.array([entity_to_id[n] for n, d in graph.nodes(data=True) if d.get("kind") == "Compound"])
    all_pos = ctd_pairs(graph, entity_to_id)
    known = set(all_pos)
    known_by_disease: dict[int, set[int]] = {}
    for c, d in all_pos:
        known_by_disease.setdefault(d, set()).add(c)

    # held-out = CtD edges that were in the embedding models' test split
    test_triples = np.load(CKPT / "split_test_triples.npy")
    ctd_id = relation_to_id["CtD"]
    held_out = {(int(h), int(t)) for h, r, t in test_triples if r == ctd_id}
    train_pos = [p for p in all_pos if p not in held_out]
    test_pos = sorted(held_out)

    c, d, y = make_training_set(train_pos, known, compound_ids, neg_per_pos, rng)
    clf = LogisticRegression(max_iter=2000, C=C)
    clf.fit(pair_features(emb, c, d), y)

    def clf_score(cs, ds):
        return clf.decision_function(pair_features(emb, np.asarray(cs), np.asarray(ds)))

    metrics = evaluate(clf_score, test_pos, known_by_disease, compound_ids)

    # AUROC on held-out positives vs fresh negatives
    tc, td, ty = make_training_set(test_pos, known, compound_ids, 5, np.random.default_rng(seed + 1))
    metrics["auroc"] = float(roc_auc_score(ty, clf_score(tc, td)))
    metrics["num_train_positives"] = len(train_pos)
    metrics["num_candidates"] = int(len(compound_ids))

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "classifier.pkl", "wb") as f:
            pickle.dump({"clf": clf, "variant": variant}, f)
        with open(OUT_DIR / "metrics.json", "w") as f:
            json.dump({"variant": variant, "metrics": metrics}, f, indent=2)
    return metrics


def transe_baseline(seed: int = 42) -> dict:
    """The raw TransE distance score, on the identical held-out pairs, for comparison."""
    graph = load_merged_graph()
    entity_to_id, _, relation_to_id, _ = build_vocab(graph)
    E = np.load(CKPT / "transe" / "entity_embeddings.npy")
    R = np.load(CKPT / "transe" / "relation_embeddings.npy")
    ctd = R[relation_to_id["CtD"]]
    compound_ids = np.array([entity_to_id[n] for n, d in graph.nodes(data=True) if d.get("kind") == "Compound"])
    all_pos = ctd_pairs(graph, entity_to_id)
    known_by_disease: dict[int, set[int]] = {}
    for c, d in all_pos:
        known_by_disease.setdefault(d, set()).add(c)
    test_triples = np.load(CKPT / "split_test_triples.npy")
    held_out = sorted({(int(h), int(t)) for h, r, t in test_triples if r == relation_to_id["CtD"]})

    def score(cs, ds):
        return -np.linalg.norm(E[np.asarray(cs)] + ctd - E[np.asarray(ds)], axis=1)

    metrics = evaluate(score, held_out, known_by_disease, compound_ids)
    return metrics


def full_score_matrix(graph, entity_to_id, relation_to_id, diseases, compounds):
    """(diseases x compounds) treatment scores plus the method name used.

    Uses the trained supervised head when its checkpoint exists, otherwise the
    raw TransE CtD distance. The head is linear in [c, d, c*d], so the score
    for every compound against one disease is a single matrix-vector product.
    """
    c_idx = np.array([entity_to_id[x] for x in compounds])
    d_idx = np.array([entity_to_id[x] for x in diseases])
    path = OUT_DIR / "classifier.pkl"
    if not path.exists():
        E = np.load(CKPT / "transe" / "entity_embeddings.npy")
        R = np.load(CKPT / "transe" / "relation_embeddings.npy")
        ctd = R[relation_to_id["CtD"]]
        diff = E[c_idx][None, :, :] + ctd - E[d_idx][:, None, :]
        return -np.linalg.norm(diff, axis=2), "transe-distance"

    with open(path, "rb") as f:
        blob = pickle.load(f)
    emb = load_embeddings(blob["variant"])
    w = blob["clf"].coef_[0]
    b = float(blob["clf"].intercept_[0])
    dim = emb.shape[1]
    w_c, w_d, w_cd = w[:dim], w[dim:2 * dim], w[2 * dim:]
    C = emb[c_idx]
    out = np.empty((len(diseases), len(compounds)), dtype=np.float32)
    for i, di in enumerate(d_idx):
        d = emb[di]
        out[i] = C @ (w_c + w_cd * d) + d @ w_d + b
    return out, f"supervised-head({blob['variant']})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the supervised treatment head")
    parser.add_argument("--variant", choices=["transe", "graphsage", "both", "compare"], default="compare")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    def show(name, m):
        print(f"  {name:<22} hits@10={m['hits@10']:.3f}  mrr={m['mrr']:.3f}  mean_rank={m['mean_rank']:.0f}"
              + (f"  auroc={m['auroc']:.3f}" if "auroc" in m else ""))

    base = transe_baseline()
    print(f"Held-out treatment edges: {base['num_evaluated']} (ranked against ~1.5k compounds each)")
    show("TransE distance score", base)

    variants = ["transe", "graphsage", "both"] if args.variant == "compare" else [args.variant]
    results = {}
    for v in variants:
        if v != "transe" and not (CKPT / "graphsage" / "entity_embeddings.npy").exists():
            print(f"  (skipping {v}: no GraphSAGE embeddings)")
            continue
        results[v] = run(v, C=args.C)
        show(f"head on {v}", results[v])

    if args.save_best and results:
        best = max(results, key=lambda k: results[k]["mrr"])
        run(best, C=args.C, save=True)
        print(f"Saved head trained on '{best}' embeddings to {OUT_DIR}")


if __name__ == "__main__":
    main()
