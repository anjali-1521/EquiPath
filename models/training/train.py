"""Orchestrate Module 2 training end-to-end: TransE -> GraphSAGE -> DDI classifier.

Run individual steps directly (models/embeddings/transe.py, etc.) for more
control over hyperparameters; this script just runs the standard pipeline
with sensible defaults, in the order each step depends on the previous one:

  1. TransE     - needs data/processed/merged_graph.pkl (Module 1)
  2. GraphSAGE   - reuses TransE's train/test edge split for a fair comparison
  3. DDI classifier - needs TransE's compound embeddings
"""

import argparse

from models.embeddings.graphsage import train_graphsage
from models.embeddings.transe import train_transe
from models.prediction.ddi import train_ddi_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all of Module 2's models in sequence")
    parser.add_argument("--skip-transe", action="store_true")
    parser.add_argument("--skip-graphsage", action="store_true")
    parser.add_argument("--skip-ddi", action="store_true")
    parser.add_argument("--transe-epochs", type=int, default=60)
    parser.add_argument("--graphsage-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.skip_transe:
        print("\n" + "=" * 70 + "\nSTEP 1/3: TransE\n" + "=" * 70)
        train_transe(num_epochs=args.transe_epochs, seed=args.seed)

    if not args.skip_graphsage:
        print("\n" + "=" * 70 + "\nSTEP 2/3: GraphSAGE\n" + "=" * 70)
        train_graphsage(num_epochs=args.graphsage_epochs, seed=args.seed)

    if not args.skip_ddi:
        print("\n" + "=" * 70 + "\nSTEP 3/3: DDI classifier\n" + "=" * 70)
        train_ddi_classifier(seed=args.seed)

    print("\nModule 2 training pipeline complete.")


if __name__ == "__main__":
    main()
