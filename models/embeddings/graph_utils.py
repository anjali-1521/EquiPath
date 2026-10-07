"""Shared graph-loading and vocabulary utilities for EquiPath's embedding models.

Both transe.py and graphsage.py need the *same* entity_id <-> node_id mapping
so their resulting embedding matrices are indexable the same way downstream
(models/prediction/repurposing.py, models/prediction/ddi.py). This module is
the single source of truth for that mapping - built once, cached to
models/checkpoints/vocab.json, and reused.
"""

import json
import pickle
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
MERGED_GRAPH_PATH = ROOT / "data" / "processed" / "merged_graph.pkl"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"


def load_merged_graph() -> nx.MultiDiGraph:
    if not MERGED_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"{MERGED_GRAPH_PATH} not found. Run data/scripts/merge_graph.py first (Module 1)."
        )
    with open(MERGED_GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def build_vocab(
    graph: nx.MultiDiGraph, use_cache: bool = True
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """Return (entity_to_id, id_to_entity, relation_to_id, id_to_relation).

    Node/relation -> id assignment is deterministic (sorted order), so
    re-running without a cache reproduces the exact same mapping.
    """
    if use_cache and VOCAB_PATH.exists():
        with open(VOCAB_PATH) as f:
            cached = json.load(f)
        entity_to_id = cached["entity_to_id"]
        relation_to_id = cached["relation_to_id"]
    else:
        entity_to_id = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        relations = sorted({data["metaedge"] for _, _, data in graph.edges(data=True)})
        relation_to_id = {rel: i for i, rel in enumerate(relations)}

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(VOCAB_PATH, "w") as f:
            json.dump({"entity_to_id": entity_to_id, "relation_to_id": relation_to_id}, f)

    id_to_entity = {i: node for node, i in entity_to_id.items()}
    id_to_relation = {i: rel for rel, i in relation_to_id.items()}
    return entity_to_id, id_to_entity, relation_to_id, id_to_relation


def get_labeled_triples(graph: nx.MultiDiGraph) -> list[tuple[str, str, str]]:
    """Raw (head_label, relation_label, tail_label) triples, as string node/edge labels."""
    return [(u, data["metaedge"], v) for u, v, data in graph.edges(data=True)]
