"""Train TransE knowledge-graph embeddings on the merged EquiPath graph.

TransE scores a triple (h, r, t) by how close e_h + e_r is to e_t; it's the
natural fit for repurposing prediction (models/prediction/repurposing.py),
since it gives every relation type - including CtD, "treats" - its own
translation vector, letting us directly ask "which compounds, when
translated by the treats vector, land near this disease?".

Training uses PyKEEN (SLCWA / negative sampling), with our own entity and
relation ID mapping (models/embeddings/graph_utils.build_vocab) passed in
explicitly, so the resulting embedding matrix is indexed identically to
GraphSAGE's and can be used interchangeably downstream.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop

from models.embeddings.graph_utils import build_vocab, get_labeled_triples, load_merged_graph
from models.evaluation.metrics import evaluate_link_prediction_sampled

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "models" / "checkpoints" / "transe"


def train_transe(
    embedding_dim: int = 128,
    num_epochs: int = 60,
    batch_size: int = 8192,
    num_negs_per_pos: int = 10,
    test_fraction: float = 0.05,
    num_negatives_eval: int = 100,
    seed: int = 42,
) -> dict:
    print("Loading merged graph and building vocabulary...")
    graph = load_merged_graph()
    entity_to_id, id_to_entity, relation_to_id, id_to_relation = build_vocab(graph)
    print(f"  {len(entity_to_id)} entities, {len(relation_to_id)} relations")

    print("Building triples factory...")
    triples = get_labeled_triples(graph)
    triples_array = np.array(triples, dtype=str)
    full_tf = TriplesFactory.from_labeled_triples(
        triples_array, entity_to_id=entity_to_id, relation_to_id=relation_to_id
    )

    # split() guarantees every entity/relation keeps >=1 training triple, but
    # a node with very few edges (e.g. SCA3 has exactly 1) can still have that
    # edge land in the *test* split, leaving it with zero training signal.
    # The project's whole point is producing meaningful predictions for these
    # sparsely-connected rare-disease nodes, so their edges are always kept in
    # train - moved back out of the test split below if split() put them
    # there. This removes at most ~15 triples from the ~112k-triple test set,
    # a negligible effect on the aggregate Hits@k/MRR numbers.
    train_tf, test_tf = full_tf.split([1 - test_fraction, test_fraction], random_state=seed)

    custom_entity_ids = {
        entity_to_id[node] for node, data in graph.nodes(data=True) if data.get("source") == "custom"
    }
    test_mapped = test_tf.mapped_triples
    is_custom = torch.isin(test_mapped[:, 0], torch.tensor(list(custom_entity_ids))) | torch.isin(
        test_mapped[:, 2], torch.tensor(list(custom_entity_ids))
    )
    if is_custom.any():
        moved_to_train = test_mapped[is_custom]
        train_tf = train_tf.clone_and_exchange_triples(
            torch.cat([train_tf.mapped_triples, moved_to_train])
        )
        test_tf = test_tf.clone_and_exchange_triples(test_mapped[~is_custom])
        print(f"  Moved {int(is_custom.sum())} custom-node triples from test back to train")

    print(f"  {train_tf.num_triples} train triples, {test_tf.num_triples} test triples")

    print(f"Training TransE (dim={embedding_dim}, epochs={num_epochs}, batch_size={batch_size}, "
          f"num_negs_per_pos={num_negs_per_pos})...")
    model = TransE(triples_factory=train_tf, embedding_dim=embedding_dim, random_seed=seed)
    optimizer = torch.optim.Adam(params=model.get_grad_params(), lr=0.01)
    training_loop = SLCWATrainingLoop(
        model=model, triples_factory=train_tf, optimizer=optimizer,
        negative_sampler="basic", negative_sampler_kwargs=dict(num_negs_per_pos=num_negs_per_pos),
    )

    start = time.time()
    losses = training_loop.train(
        triples_factory=train_tf, num_epochs=num_epochs, batch_size=batch_size, use_tqdm=True
    )
    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.1f}s. Final loss: {losses[-1]:.4f}")

    entity_embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    relation_embeddings = model.relation_representations[0](indices=None).detach().cpu().numpy()

    print(f"Evaluating on {test_tf.num_triples} held-out triples "
          f"(sampled-negative ranking, {num_negatives_eval} negatives/triple)...")

    # filtered ranking: for a given (head, relation), all tails ever seen
    # (train OR test) should be excluded from the negative pool.
    known_positive_tails: dict[tuple[int, int], set[int]] = {}
    for h, r, t in full_tf.mapped_triples.tolist():
        known_positive_tails.setdefault((h, r), set()).add(t)
    # re-key by relation LABEL to match the shared metrics.py interface
    known_positive_tails_labeled = {
        (h, id_to_relation[r]): tails for (h, r), tails in known_positive_tails.items()
    }

    def score_fn(head_id: int, relation_label: str, tail_id: int) -> float:
        r = relation_to_id[relation_label]
        vec = entity_embeddings[head_id] + relation_embeddings[r] - entity_embeddings[tail_id]
        return -float(np.linalg.norm(vec))

    test_triples_labeled = [
        (int(h), id_to_relation[int(r)], int(t)) for h, r, t in test_tf.mapped_triples.tolist()
    ]

    metrics = evaluate_link_prediction_sampled(
        score_fn=score_fn,
        test_triples=test_triples_labeled,
        all_entity_ids=list(range(len(entity_to_id))),
        num_negatives=num_negatives_eval,
        known_positive_tails=known_positive_tails_labeled,
        seed=seed,
    )
    print("  Metrics:", json.dumps(metrics, indent=2))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CHECKPOINT_DIR / "entity_embeddings.npy", entity_embeddings)
    np.save(CHECKPOINT_DIR / "relation_embeddings.npy", relation_embeddings)

    # Persist the exact train/test triple split so graphsage.py can reuse the
    # same held-out edges - directly comparable metrics between the two models.
    np.save(CHECKPOINT_DIR.parent / "split_train_triples.npy", train_tf.mapped_triples.numpy())
    np.save(CHECKPOINT_DIR.parent / "split_test_triples.npy", test_tf.mapped_triples.numpy())

    with open(CHECKPOINT_DIR / "metrics.json", "w") as f:
        json.dump({"config": {
            "embedding_dim": embedding_dim, "num_epochs": num_epochs,
            "batch_size": batch_size, "num_negs_per_pos": num_negs_per_pos,
            "test_fraction": test_fraction,
        }, "metrics": metrics, "training_seconds": elapsed}, f, indent=2)

    print(f"Saved TransE embeddings and metrics to {CHECKPOINT_DIR}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TransE embeddings on the EquiPath graph")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-negs-per-pos", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--num-negatives-eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_transe(
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_negs_per_pos=args.num_negs_per_pos,
        test_fraction=args.test_fraction,
        num_negatives_eval=args.num_negatives_eval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
