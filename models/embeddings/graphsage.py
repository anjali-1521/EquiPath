"""Train GraphSAGE node embeddings on the merged EquiPath graph.

Unlike TransE, GraphSAGE learns embeddings by aggregating each node's local
neighborhood (2 layers of mean/mean-pool aggregation here, via PyG's
SAGEConv) rather than from a per-relation translation vector. It has no
notion of relation type - it's trained on one relation-agnostic "does an
edge exist" objective - so its embeddings capture broader structural
similarity (e.g. "these two genes sit in similar neighborhoods") rather
than TransE's relation-specific "X treats Y" signal. That makes it a
complementary, not competing, embedding for downstream prediction: good
for similarity-based signals (e.g. DDI, where interaction risk plausibly
correlates with neighborhood/target overlap), less suited than TransE to
directly answering "what treats this disease" (see repurposing.py, which
uses TransE's CtD relation vector for exactly that reason).

Node features are a one-hot encoding of each node's `kind` (Disease, Gene,
Compound, ...) - a deliberate simplification given we have no richer
per-node attributes available (e.g. gene sequence, compound structure);
documented rather than dressed up as more sophisticated than it is.

Reuses the exact train/test edge split TransE produced (models/checkpoints/
split_{train,test}_triples.npy) so the two models are evaluated on
identical held-out edges and their Hits@k/MRR numbers are directly
comparable. Run transe.py first.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

from models.embeddings.graph_utils import build_vocab, load_merged_graph
from models.evaluation.metrics import evaluate_link_prediction_sampled

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "models" / "checkpoints" / "graphsage"
SPLIT_DIR = ROOT / "models" / "checkpoints"

NODE_KINDS = [
    "Anatomy", "Biological Process", "Cellular Component", "Compound",
    "Disease", "Gene", "Molecular Function", "Pathway",
    "Pharmacologic Class", "Side Effect", "Symptom",
]


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


def build_node_features(graph, entity_to_id: dict[str, int]) -> torch.Tensor:
    kind_to_idx = {kind: i for i, kind in enumerate(NODE_KINDS)}
    x = np.zeros((len(entity_to_id), len(NODE_KINDS)), dtype=np.float32)
    for node, node_id in entity_to_id.items():
        kind = graph.nodes[node].get("kind", "Unknown")
        x[node_id, kind_to_idx[kind]] = 1.0
    return torch.from_numpy(x)


def train_graphsage(
    hidden_dim: int = 64,
    embedding_dim: int = 128,
    num_epochs: int = 100,
    lr: float = 0.01,
    num_negatives_eval: int = 100,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)

    train_path = SPLIT_DIR / "split_train_triples.npy"
    test_path = SPLIT_DIR / "split_test_triples.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "No train/test split found. Run models/embeddings/transe.py first - "
            "graphsage.py reuses its split so both models are evaluated on the "
            "same held-out edges."
        )

    print("Loading merged graph and vocabulary...")
    graph = load_merged_graph()
    entity_to_id, id_to_entity, relation_to_id, id_to_relation = build_vocab(graph)
    num_nodes = len(entity_to_id)

    train_triples = np.load(train_path)  # (n, 3): head_id, relation_id, tail_id
    test_triples = np.load(test_path)
    print(f"  {len(train_triples)} train edges, {len(test_triples)} test edges (from TransE's split)")

    x = build_node_features(graph, entity_to_id)

    # Bidirectional edges for message passing, built only from TRAIN edges so
    # no test-edge information leaks into the learned representations.
    src = torch.from_numpy(train_triples[:, 0]).long()
    dst = torch.from_numpy(train_triples[:, 2]).long()
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    model = GraphSAGEEncoder(in_channels=x.shape[1], hidden_channels=hidden_dim, out_channels=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_edge_index = torch.stack([src, dst], dim=0)

    print(f"Training GraphSAGE (hidden={hidden_dim}, out={embedding_dim}, epochs={num_epochs})...")
    start = time.time()
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        z = model(x, edge_index)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=num_nodes, num_neg_samples=pos_edge_index.size(1)
        )

        pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
            print(f"  epoch {epoch}/{num_epochs}  loss={loss.item():.4f}")

    elapsed = time.time() - start
    print(f"  Training done in {elapsed:.1f}s")

    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index).cpu().numpy()

    print(f"Evaluating on {len(test_triples)} held-out edges "
          f"(sampled-negative ranking, {num_negatives_eval} negatives/edge)...")

    known_positive_tails: dict[tuple[int, str], set[int]] = {}
    for h, r, t in np.concatenate([train_triples, test_triples]).tolist():
        known_positive_tails.setdefault((h, id_to_relation[r]), set()).add(t)

    def score_fn(head_id: int, relation_label: str, tail_id: int) -> float:
        # relation-agnostic: GraphSAGE has no per-relation decoder
        return float(np.dot(final_embeddings[head_id], final_embeddings[tail_id]))

    test_triples_labeled = [(int(h), id_to_relation[int(r)], int(t)) for h, r, t in test_triples.tolist()]

    metrics = evaluate_link_prediction_sampled(
        score_fn=score_fn,
        test_triples=test_triples_labeled,
        all_entity_ids=list(range(num_nodes)),
        num_negatives=num_negatives_eval,
        known_positive_tails=known_positive_tails,
        seed=seed,
    )
    print("  Metrics:", json.dumps(metrics, indent=2))

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CHECKPOINT_DIR / "entity_embeddings.npy", final_embeddings)
    torch.save(model.state_dict(), CHECKPOINT_DIR / "model_state.pt")
    with open(CHECKPOINT_DIR / "metrics.json", "w") as f:
        json.dump({"config": {
            "hidden_dim": hidden_dim, "embedding_dim": embedding_dim,
            "num_epochs": num_epochs, "lr": lr,
        }, "metrics": metrics, "training_seconds": elapsed}, f, indent=2)

    print(f"Saved GraphSAGE embeddings and metrics to {CHECKPOINT_DIR}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphSAGE embeddings on the EquiPath graph")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-negatives-eval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_graphsage(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_epochs=args.num_epochs,
        lr=args.lr,
        num_negatives_eval=args.num_negatives_eval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
