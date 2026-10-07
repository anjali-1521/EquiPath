"""Drug-drug interaction (DDI) risk prediction for EquiPath.

Ground truth comes from models/data/build_ddi_labels.py: real interacting
drug pairs from the Decagon polypharmacy dataset (Zitnik et al. 2018),
restricted to the 325 drugs that both map to a DrugBank ID and exist as a
Compound node in Hetionet - see that script's docstring for why Hetionet
alone has no DDI edges to train against.

Given two compounds, this predicts interaction risk from a small classifier
trained on top of their TransE embeddings (concat, absolute difference, and
elementwise product - the standard feature construction for edge/pair
classification on top of node embeddings), rather than from GraphSAGE:
TransE is trained across the whole heterogeneous graph including gene and
pathway relations, so its compound embeddings already reflect shared
biological context (targets, pathways) that plausibly correlates with
interaction risk. This is a learned statistical association from real
interaction reports, not a mechanistic interaction model - treat scores as
a triage signal, not a clinical determination.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from models.embeddings.graph_utils import build_vocab, load_merged_graph

ROOT = Path(__file__).resolve().parents[2]
TRANSE_DIR = ROOT / "models" / "checkpoints" / "transe"
DDI_DIR = ROOT / "models" / "checkpoints" / "ddi"
LABELS_PATH = ROOT / "data" / "processed" / "ddi_labels.csv"
UNIVERSE_PATH = ROOT / "data" / "processed" / "ddi_drug_universe.csv"


def load_labels() -> tuple[list[tuple[str, str]], list[str]]:
    with open(LABELS_PATH) as f:
        reader = csv.DictReader(f)
        positive_pairs = [(row["compound_1"], row["compound_2"]) for row in reader]
    with open(UNIVERSE_PATH) as f:
        reader = csv.DictReader(f)
        universe = [row["compound_id"] for row in reader]
    return positive_pairs, universe


def sample_negative_pairs(
    positive_pairs: list[tuple[str, str]], universe: list[str], seed: int = 42
) -> list[tuple[str, str]]:
    """Sample as many random non-interacting pairs as there are positives."""
    rng = np.random.default_rng(seed)
    positive_set = {frozenset(p) for p in positive_pairs}
    negatives: set[frozenset] = set()
    target = len(positive_pairs)
    universe_arr = np.array(universe)

    while len(negatives) < target:
        i, j = rng.integers(0, len(universe_arr), size=2)
        if i == j:
            continue
        pair = frozenset((universe_arr[i], universe_arr[j]))
        if pair not in positive_set and pair not in negatives:
            negatives.add(pair)

    return [tuple(p) for p in negatives]


def pair_features(emb: np.ndarray, id1: int, id2: int) -> np.ndarray:
    e1, e2 = emb[id1], emb[id2]
    return np.concatenate([e1, e2, np.abs(e1 - e2), e1 * e2])


def train_ddi_classifier(test_size: float = 0.2, seed: int = 42) -> dict:
    print("Loading TransE embeddings and DDI labels...")
    graph = load_merged_graph()
    entity_to_id, _, _, _ = build_vocab(graph)
    entity_embeddings = np.load(TRANSE_DIR / "entity_embeddings.npy")

    positive_pairs, universe = load_labels()
    universe = [c for c in universe if c in entity_to_id]
    print(f"  {len(positive_pairs)} positive pairs, {len(universe)} drugs in usable universe")

    negative_pairs = sample_negative_pairs(positive_pairs, universe, seed=seed)
    print(f"  Sampled {len(negative_pairs)} negative pairs")

    pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    X = np.array([
        pair_features(entity_embeddings, entity_to_id[a], entity_to_id[b]) for a, b in pairs
    ])
    y = np.array(labels)

    X_train, X_test, y_train, y_test, pairs_train, pairs_test = train_test_split(
        X, y, pairs, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Training logistic regression on {len(X_train)} pairs...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "auprc": float(average_precision_score(y_test, y_prob)),
        "accuracy": float(clf.score(X_test, y_test)),
        "num_train": len(X_train),
        "num_test": len(X_test),
    }
    print("  Metrics:", json.dumps(metrics, indent=2))

    DDI_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(DDI_DIR / "classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(DDI_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved DDI classifier and metrics to {DDI_DIR}")

    return {"metrics": metrics, "classifier": clf, "entity_embeddings": entity_embeddings, "entity_to_id": entity_to_id}


def predict_interaction(
    compound_1: str, compound_2: str, clf, entity_embeddings: np.ndarray, entity_to_id: dict[str, int]
) -> float:
    """Return predicted interaction probability for a pair of Compound node IDs."""
    features = pair_features(entity_embeddings, entity_to_id[compound_1], entity_to_id[compound_2])
    return float(clf.predict_proba(features.reshape(1, -1))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and demo the EquiPath DDI risk classifier")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo-pairs", type=int, default=5, help="Number of example pairs to score")
    args = parser.parse_args()

    result = train_ddi_classifier(test_size=args.test_size, seed=args.seed)
    clf = result["classifier"]
    entity_embeddings = result["entity_embeddings"]
    entity_to_id = result["entity_to_id"]

    graph = load_merged_graph()
    positive_pairs, universe = load_labels()
    rng = np.random.default_rng(args.seed)

    print(f"\n=== Example predictions ({args.demo_pairs} known-interacting pairs) ===")
    sample_idx = rng.choice(len(positive_pairs), size=min(args.demo_pairs, len(positive_pairs)), replace=False)
    for i in sample_idx:
        c1, c2 = positive_pairs[i]
        prob = predict_interaction(c1, c2, clf, entity_embeddings, entity_to_id)
        n1 = graph.nodes[c1].get("name", c1)
        n2 = graph.nodes[c2].get("name", c2)
        print(f"  {n1} + {n2}: predicted interaction probability = {prob:.3f}  (known: interacts)")


if __name__ == "__main__":
    main()
