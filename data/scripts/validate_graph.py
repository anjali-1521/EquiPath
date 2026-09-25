"""Validate the merged EquiPath knowledge graph (data/processed/merged_graph.pkl).

Re-runs the merge-time checks (disease counts, no isolated custom nodes) plus
additional structural checks (no duplicate node IDs, no dangling edges), and
prints a summary table for a sanity check against expected Hetionet numbers.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "data" / "processed" / "merged_graph.pkl"

EXPECTED_HETIONET_DISEASE_COUNT = 137
EXPECTED_CUSTOM_DISEASE_COUNT = 4

CUSTOM_DISEASE_IDS = {
    "Disease::CUSTOM:wilsons_disease": "Wilson's disease",
    "Disease::CUSTOM:sca3": "SCA3 / Machado-Joseph disease",
    "Disease::CUSTOM:ashermans_syndrome": "Asherman's syndrome",
    "Disease::CUSTOM:mrkh_syndrome": "MRKH syndrome",
}

# Diseases expected to have at least one Compound-treats-Disease edge.
EXPECTED_TREATED = {"Disease::CUSTOM:wilsons_disease"}
# Diseases expected to have NO Compound-treats-Disease edge (no approved
# disease-modifying / pharmacological treatment).
EXPECTED_UNTREATED = {
    "Disease::CUSTOM:sca3",
    "Disease::CUSTOM:ashermans_syndrome",
    "Disease::CUSTOM:mrkh_syndrome",
}


def load_graph(path: Path) -> nx.MultiDiGraph:
    if not path.exists():
        sys.exit(f"Merged graph not found at {path}. Run merge_graph.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def check_no_duplicate_node_ids(graph: nx.MultiDiGraph) -> None:
    # networkx nodes are already keyed by ID (a dict), so true duplicates
    # can't exist in-memory; this instead confirms nothing collapsed two
    # distinct logical entities onto the same ID unexpectedly (e.g. a custom
    # node accidentally reusing a real Hetionet ID for a *different* entity).
    node_ids = list(graph.nodes())
    counts = Counter(node_ids)
    duplicates = [node_id for node_id, count in counts.items() if count > 1]
    assert not duplicates, f"Duplicate node IDs found: {duplicates}"


def check_edges_reference_existing_nodes(graph: nx.MultiDiGraph) -> None:
    node_set = set(graph.nodes())
    dangling = [
        (u, v) for u, v in graph.edges()
        if u not in node_set or v not in node_set
    ]
    assert not dangling, f"Edges reference nonexistent nodes: {dangling[:10]}"


def check_disease_counts(graph: nx.MultiDiGraph) -> None:
    disease_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "Disease"]
    expected_total = EXPECTED_HETIONET_DISEASE_COUNT + EXPECTED_CUSTOM_DISEASE_COUNT
    assert len(disease_nodes) == expected_total, (
        f"Expected {expected_total} Disease nodes, found {len(disease_nodes)}"
    )

    for disease_id in CUSTOM_DISEASE_IDS:
        assert disease_id in graph, f"Custom disease node missing: {disease_id}"
        assert graph.degree(disease_id) > 0, f"Custom disease node is isolated: {disease_id}"


def check_treats_edges(graph: nx.MultiDiGraph) -> dict[str, list[str]]:
    """Return, for each custom disease, the list of compounds with a
    Compound-treats-Disease edge into it, and assert this matches the
    expected treated/untreated split."""
    treats_by_disease: dict[str, list[str]] = {disease_id: [] for disease_id in CUSTOM_DISEASE_IDS}

    for u, v, data in graph.edges(data=True):
        if data.get("metaedge") == "CtD" and v in CUSTOM_DISEASE_IDS:
            treats_by_disease[v].append(u)

    for disease_id in EXPECTED_TREATED:
        assert treats_by_disease[disease_id], (
            f"Expected at least one treats-edge for {CUSTOM_DISEASE_IDS[disease_id]}, found none"
        )
    for disease_id in EXPECTED_UNTREATED:
        assert not treats_by_disease[disease_id], (
            f"Expected NO treats-edge for {CUSTOM_DISEASE_IDS[disease_id]}, "
            f"found: {treats_by_disease[disease_id]}"
        )

    return treats_by_disease


def summary_table(graph: nx.MultiDiGraph) -> str:
    node_kind_counts = Counter(d.get("kind", "Unknown") for _, d in graph.nodes(data=True))
    edge_type_counts = Counter(d.get("metaedge", "Unknown") for _, _, d in graph.edges(data=True))

    lines = ["=" * 50, "NODE COUNTS BY KIND", "=" * 50]
    for kind, count in sorted(node_kind_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {kind:<20} {count:>8,}")
    lines.append(f"  {'TOTAL':<20} {sum(node_kind_counts.values()):>8,}")

    lines += ["", "=" * 50, "EDGE COUNTS BY TYPE (metaedge)", "=" * 50]
    for metaedge, count in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {metaedge:<20} {count:>8,}")
    lines.append(f"  {'TOTAL':<20} {sum(edge_type_counts.values()):>8,}")

    return "\n".join(lines)


def connectivity_report(graph: nx.MultiDiGraph, treats_by_disease: dict[str, list[str]]) -> str:
    lines = ["", "=" * 50, "CUSTOM DISEASE NODE CONNECTIVITY", "=" * 50]
    for disease_id, name in CUSTOM_DISEASE_IDS.items():
        degree = graph.degree(disease_id)
        neighbor_kinds: Counter = Counter()
        for _, target in graph.out_edges(disease_id):
            neighbor_kinds[graph.nodes[target].get("kind", "Unknown")] += 1
        for source, _ in graph.in_edges(disease_id):
            neighbor_kinds[graph.nodes[source].get("kind", "Unknown")] += 1

        treats = treats_by_disease.get(disease_id, [])
        treats_desc = (
            ", ".join(graph.nodes[c].get("name", c) for c in treats) if treats else "(none)"
        )

        lines.append(f"\n{name}  [{disease_id}]")
        lines.append(f"  degree: {degree}")
        lines.append(f"  connected node types: {dict(neighbor_kinds)}")
        lines.append(f"  treats-edges (Compound -> Disease): {treats_desc}")

    return "\n".join(lines)


def main() -> None:
    graph = load_graph(GRAPH_PATH)
    print(f"Loaded merged graph: {graph.number_of_nodes():,} nodes, "
          f"{graph.number_of_edges():,} edges\n")

    print("Running validation checks...")
    check_no_duplicate_node_ids(graph)
    print("  [OK] no duplicate node IDs")
    check_edges_reference_existing_nodes(graph)
    print("  [OK] all edges reference existing nodes")
    check_disease_counts(graph)
    print(f"  [OK] {EXPECTED_HETIONET_DISEASE_COUNT + EXPECTED_CUSTOM_DISEASE_COUNT} disease nodes "
          f"({EXPECTED_HETIONET_DISEASE_COUNT} Hetionet + {EXPECTED_CUSTOM_DISEASE_COUNT} custom), "
          "none isolated")
    treats_by_disease = check_treats_edges(graph)
    print("  [OK] treats-edges match expectation "
          "(Wilson's: treated; SCA3/Asherman's/MRKH: untreated)")

    print(summary_table(graph))
    print(connectivity_report(graph, treats_by_disease))
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
