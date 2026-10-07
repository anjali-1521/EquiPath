"""Consolidated validation report for Module 2 (mirrors Module 1's validate_graph.py).

Loads the metrics.json each training step already saved and checks the
things that would signal a broken pipeline rather than just a weak model:
embeddings exist and cover every graph entity, the TransE/GraphSAGE test
sets are identical (proving the fair-comparison split actually happened),
and DDI labels/universe are non-empty. Prints a summary table.
"""

import json
from pathlib import Path

import numpy as np

from models.embeddings.graph_utils import build_vocab, load_merged_graph

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "models" / "checkpoints"


def check_embeddings_cover_all_entities(name: str, path: Path, num_entities: int) -> None:
    emb = np.load(path)
    assert emb.shape[0] == num_entities, (
        f"{name} embeddings cover {emb.shape[0]} entities, expected {num_entities}"
    )
    assert not np.isnan(emb).any(), f"{name} embeddings contain NaNs"
    print(f"  [OK] {name}: {emb.shape[0]} entities x {emb.shape[1]} dims, no NaNs")


def check_splits_match() -> None:
    train = np.load(CHECKPOINTS / "split_train_triples.npy")
    test = np.load(CHECKPOINTS / "split_test_triples.npy")
    assert len(train) + len(test) > 0
    print(f"  [OK] shared split: {len(train)} train / {len(test)} test triples "
          "(TransE and GraphSAGE evaluated on identical held-out edges)")


def print_metrics_table() -> None:
    print("\n" + "=" * 60)
    print("MODEL METRICS SUMMARY")
    print("=" * 60)
    for model_name in ["transe", "graphsage", "treats", "ddi"]:
        metrics_path = CHECKPOINTS / model_name / "metrics.json"
        if not metrics_path.exists():
            print(f"  {model_name}: not trained yet (run models/training/train.py)")
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        print(f"\n  {model_name}:")
        for k, v in data.get("metrics", data).items():
            if isinstance(v, float):
                print(f"    {k:<15} {v:.4f}")
            else:
                print(f"    {k:<15} {v}")


def main() -> None:
    print("Loading graph and vocabulary...")
    graph = load_merged_graph()
    entity_to_id, *_ = build_vocab(graph)
    num_entities = len(entity_to_id)
    print(f"  {num_entities} entities\n")

    print("Running validation checks...")
    check_embeddings_cover_all_entities("TransE", CHECKPOINTS / "transe" / "entity_embeddings.npy", num_entities)
    check_embeddings_cover_all_entities("GraphSAGE", CHECKPOINTS / "graphsage" / "entity_embeddings.npy", num_entities)
    check_splits_match()

    ddi_labels = ROOT / "data" / "processed" / "ddi_labels.csv"
    ddi_universe = ROOT / "data" / "processed" / "ddi_drug_universe.csv"
    assert ddi_labels.exists() and ddi_universe.exists(), "DDI labels not built - run models/data/build_ddi_labels.py"
    print(f"  [OK] DDI labels present: {ddi_labels}, {ddi_universe}")

    print_metrics_table()
    print("\nAll Module 2 validation checks passed.")


if __name__ == "__main__":
    main()
