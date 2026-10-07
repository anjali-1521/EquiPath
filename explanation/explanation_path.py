"""Path-based explanations for repurposing predictions (Module 3).

A TransE score alone says "this compound is close to the disease in embedding
space" but not *why*. This module finds concrete, inspectable evidence paths
in the knowledge graph between a compound and a disease, using the
metapath approach from the Hetionet paper (Himmelstein et al. 2017):

  CbG-DaG      compound BINDS a gene the disease is ASSOCIATED with
  CdG-DuG      compound DOWNREGULATES a gene the disease UPREGULATES
  CuG-DdG      compound UPREGULATES a gene the disease DOWNREGULATES
  CbG-GiG-DaG  compound binds a gene that INTERACTS with a disease gene
  CrC-CtD      compound RESEMBLES a compound that already TREATS the disease

Each path is scored by a degree-weighted path count: the product of
(degree of each node on the path)^-0.4. Paths through hub nodes (genes that
touch thousands of edges) are down-weighted, since a hub being connected to
both ends is weak evidence, while a path through a specific, low-degree gene
is strong evidence. This is the same damping Hetionet uses.

Paths are *supporting evidence*, not proof of efficacy. A prediction with no
path is explicitly reported as having none - for sparsely-connected rare
diseases that is common, and is itself useful information.
"""

from collections import defaultdict

import networkx as nx

DAMPING = 0.4

# (label, [metaedge steps], human-readable template)
METAPATHS = [
    ("binds-associated-gene", ["CbG", "DaG~"],
     "{c} binds {g1}, a gene associated with {d}"),
    ("opposing-regulation-down-up", ["CdG", "DuG~"],
     "{c} downregulates {g1}, which {d} upregulates"),
    ("opposing-regulation-up-down", ["CuG", "DdG~"],
     "{c} upregulates {g1}, which {d} downregulates"),
    ("binds-interacting-gene", ["CbG", "GiG", "DaG~"],
     "{c} binds {g1}, which interacts with {g2}, a gene associated with {d}"),
    ("similar-to-treatment", ["CrC", "CtD"],
     "{c} resembles {c2}, which treats {d}"),
]


class ExplanationEngine:
    """Indexes the graph by metaedge once, then answers many path queries."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.fwd: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.rev: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for u, v, data in graph.edges(data=True):
            me = data["metaedge"]
            self.fwd[me][u].append(v)
            self.rev[me][v].append(u)
        # GiG and CrC are semantically symmetric; index both directions.
        for me in ("GiG", "CrC"):
            for u, vs in list(self.fwd[me].items()):
                for v in vs:
                    if u not in self.fwd[me][v]:
                        self.fwd[me][v].append(u)
        self._degree = dict(graph.degree())

    def _neighbors(self, node: str, step: str) -> list[str]:
        # a trailing "~" means "traverse this metaedge backwards"
        if step.endswith("~"):
            return self.rev[step[:-1]].get(node, [])
        return self.fwd[step].get(node, [])

    def _name(self, node: str) -> str:
        return self.graph.nodes[node].get("name", node)

    def explain(self, compound: str, disease: str, max_paths: int = 5) -> list[dict]:
        """Return up to `max_paths` evidence paths, best first. Empty list = no evidence path."""
        results = []
        for label, steps, template in METAPATHS:
            frontier = [([compound], 1.0)]
            for i, step in enumerate(steps):
                is_last = i == len(steps) - 1
                nxt = []
                for path, weight in frontier:
                    for nb in self._neighbors(path[-1], step):
                        if nb in path or (is_last and nb != disease):
                            continue
                        nxt.append((path + [nb], weight * self._degree[nb] ** -DAMPING))
                frontier = nxt
            for path, weight in frontier:
                results.append(self._format(label, template, path, weight, compound, disease))

        results.sort(key=lambda r: -r["score"])
        return results[:max_paths]

    def rank_by_evidence(self, disease: str, compounds: list[str], exclude: set[str] = frozenset(),
                         top_k: int = 10) -> list[dict]:
        """Compounds ranked by total path evidence to the disease (disease-specific by
        construction, unlike an embedding score that can drift toward generally popular drugs).
        Only compounds with at least one path appear, so the list can be short or empty."""
        ranked = []
        for c in compounds:
            if c in exclude:
                continue
            paths = self.explain(c, disease, max_paths=1000)
            if paths:
                ranked.append({"compound_id": c, "compound_name": self._name(c),
                               "evidence_score": float(sum(p["score"] for p in paths)),
                               "num_paths": len(paths),
                               "evidence_paths": [{"type": p["type"], "text": p["text"], "score": p["score"]}
                                                  for p in paths[:3]]})
        ranked.sort(key=lambda r: -r["evidence_score"])
        return ranked[:top_k]

    def _format(self, label, template, path, weight, compound, disease) -> dict:
        fields = {"c": self._name(compound), "d": self._name(disease)}
        genes = [n for n in path if n.startswith("Gene::")]
        for i, g in enumerate(genes, start=1):
            fields[f"g{i}"] = self._name(g)
        if "{c2}" in template:
            fields["c2"] = self._name(path[1])
        return {
            "type": label,
            "nodes": path,
            "names": [self._name(n) for n in path],
            "text": template.format(**fields),
            "score": float(weight),
        }
