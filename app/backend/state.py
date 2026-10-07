"""Process-wide model state, loaded once at startup (the graph pickle alone is ~100 MB)."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from explanation.explanation_path import ExplanationEngine
from explanation.neglect_ranking import build_context
from models.embeddings.graph_utils import build_vocab, load_merged_graph

ROOT = Path(__file__).resolve().parents[2]
TRANSE_DIR = ROOT / "models" / "checkpoints" / "transe"
DDI_CLASSIFIER = ROOT / "models" / "checkpoints" / "ddi" / "classifier.pkl"
DDI_UNIVERSE = ROOT / "data" / "processed" / "ddi_drug_universe.csv"


@dataclass
class AppState:
    graph: object
    entity_to_id: dict
    relation_to_id: dict
    entity_emb: np.ndarray
    relation_emb: np.ndarray
    engine: ExplanationEngine
    diseases: list
    compounds: list
    scores: np.ndarray
    known: np.ndarray
    neglect: np.ndarray
    degrees: np.ndarray
    untreated: np.ndarray
    ddi_clf: object | None
    ddi_universe: set


def load_state() -> AppState:
    graph = load_merged_graph()
    entity_to_id, _, relation_to_id, _ = build_vocab(graph)
    entity_emb = np.load(TRANSE_DIR / "entity_embeddings.npy")
    relation_emb = np.load(TRANSE_DIR / "relation_embeddings.npy")
    diseases, compounds, scores, known, neglect, degrees, untreated = build_context(
        graph, entity_to_id, relation_to_id, entity_emb, relation_emb
    )

    clf = None
    if DDI_CLASSIFIER.exists():
        with open(DDI_CLASSIFIER, "rb") as f:
            clf = pickle.load(f)
    universe = set()
    if DDI_UNIVERSE.exists():
        universe = {line.strip() for line in open(DDI_UNIVERSE).read().splitlines()[1:] if line.strip()}

    return AppState(graph, entity_to_id, relation_to_id, entity_emb, relation_emb,
                    ExplanationEngine(graph), diseases, compounds, scores, known,
                    neglect, degrees, untreated, clf, universe)
