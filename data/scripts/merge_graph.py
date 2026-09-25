"""Merge Hetionet v1.0 with EquiPath's custom rare-disease nodes into one graph.

Loads data/raw/hetionet/{nodes.tsv,edges.sif}, adds the 4 custom disease
nodes (+ supporting nodes and edges) from data/custom_nodes/*.json, validates
the result, and pickles the merged graph to data/processed/merged_graph.pkl.
"""

import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "hetionet"
CUSTOM_DIR = ROOT / "data" / "custom_nodes"
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_HETIONET_DISEASE_COUNT = 137
EXPECTED_CUSTOM_DISEASE_COUNT = 4


def load_hetionet(nodes_path: Path, edges_path: Path) -> nx.MultiDiGraph:
    nodes_df = pd.read_csv(nodes_path, sep="\t")
    edges_df = pd.read_csv(edges_path, sep="\t")

    graph = nx.MultiDiGraph()
    for row in nodes_df.itertuples(index=False):
        graph.add_node(row.id, name=row.name, kind=row.kind, source="hetionet")

    for row in edges_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, metaedge=row.metaedge, source="hetionet")

    return graph


def load_custom_nodes(custom_dir: Path) -> list[dict]:
    records = []
    for path in sorted(custom_dir.glob("*.json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def merge_custom_nodes(graph: nx.MultiDiGraph, records: list[dict]) -> None:
    for record in records:
        node = record["node"]
        graph.add_node(
            node["id"],
            name=node["name"],
            kind=node["kind"],
            source=node.get("source", "custom"),
        )

        for supporting_node in record.get("supporting_nodes", []):
            # add_node is idempotent: nodes already present (reused Hetionet
            # IDs, e.g. an existing Compound or Anatomy node) are left as-is
            # if already in the graph, otherwise created fresh.
            if supporting_node["id"] not in graph:
                graph.add_node(
                    supporting_node["id"],
                    name=supporting_node["name"],
                    kind=supporting_node["kind"],
                    source=supporting_node.get("source", "custom"),
                )

        for edge in record.get("edges", []):
            graph.add_edge(
                edge["source"],
                edge["target"],
                metaedge=edge["metaedge"],
                source="custom",
            )


def validate(graph: nx.MultiDiGraph, records: list[dict]) -> None:
    disease_nodes = [n for n, d in graph.nodes(data=True) if d.get("kind") == "Disease"]
    expected_total = EXPECTED_HETIONET_DISEASE_COUNT + EXPECTED_CUSTOM_DISEASE_COUNT
    assert len(disease_nodes) == expected_total, (
        f"Expected {expected_total} Disease nodes "
        f"({EXPECTED_HETIONET_DISEASE_COUNT} Hetionet + {EXPECTED_CUSTOM_DISEASE_COUNT} custom), "
        f"found {len(disease_nodes)}"
    )

    custom_disease_ids = [r["node"]["id"] for r in records]
    assert len(custom_disease_ids) == EXPECTED_CUSTOM_DISEASE_COUNT, (
        f"Expected {EXPECTED_CUSTOM_DISEASE_COUNT} custom disease node files, "
        f"found {len(custom_disease_ids)}"
    )

    for disease_id in custom_disease_ids:
        assert disease_id in graph, f"Custom disease node missing from graph: {disease_id}"
        degree = graph.degree(disease_id)
        assert degree > 0, f"Custom disease node is isolated (0 edges): {disease_id}"


def connectivity_report(graph: nx.MultiDiGraph, records: list[dict]) -> str:
    lines = ["Connectivity report for custom disease nodes:", ""]
    for record in records:
        node_id = record["node"]["id"]
        name = record["node"]["name"]
        degree = graph.degree(node_id)

        neighbor_kinds: dict[str, int] = {}
        for _, target, data in graph.out_edges(node_id, data=True):
            kind = graph.nodes[target].get("kind", "Unknown")
            neighbor_kinds[kind] = neighbor_kinds.get(kind, 0) + 1
        for source, _, data in graph.in_edges(node_id, data=True):
            kind = graph.nodes[source].get("kind", "Unknown")
            neighbor_kinds[kind] = neighbor_kinds.get(kind, 0) + 1

        connections = ", ".join(f"{kind}: {count}" for kind, count in sorted(neighbor_kinds.items()))
        lines.append(f"  {name} ({node_id})")
        lines.append(f"    degree: {degree}")
        lines.append(f"    connected node types: {connections if connections else '(none)'}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    nodes_path = RAW_DIR / "nodes.tsv"
    edges_path = RAW_DIR / "edges.sif"

    if not nodes_path.exists() or not edges_path.exists():
        sys.exit(
            f"Missing Hetionet raw files. Expected:\n  {nodes_path}\n  {edges_path}\n"
            "Download them first (see README / docs)."
        )

    print("Loading Hetionet nodes and edges...")
    graph = load_hetionet(nodes_path, edges_path)
    print(f"  Loaded {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Loading custom rare-disease node definitions...")
    records = load_custom_nodes(CUSTOM_DIR)
    print(f"  Loaded {len(records)} custom disease files: "
          f"{[r['node']['name'] for r in records]}")

    print("Merging custom nodes into graph...")
    merge_custom_nodes(graph, records)
    print(f"  Graph now has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Validating...")
    validate(graph, records)
    print("  All validation checks passed.")

    print()
    print(connectivity_report(graph, records))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "merged_graph.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"Merged graph pickled to {out_path}")


if __name__ == "__main__":
    main()
