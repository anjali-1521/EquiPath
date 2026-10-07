import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from explanation.neglect_ranking import equity_audit, rerank, top_pairs

router = APIRouter()

SPARSE_DEGREE = 10  # below this, the graph holds too little about a disease for embeddings to be trusted


def _state(request: Request):
    return request.app.state.ctx


@router.get("/diseases")
def list_diseases(request: Request, q: str = "", limit: int = Query(50, le=500)):
    s = _state(request)
    out = []
    for i, d in enumerate(s.diseases):
        name = s.graph.nodes[d].get("name", d)
        if q.lower() in name.lower():
            out.append({"id": d, "name": name, "degree": int(s.degrees[i]),
                        "untreated": bool(s.untreated[i]), "custom": d.startswith("Disease::CUSTOM")})
    out.sort(key=lambda x: (not x["custom"], x["name"]))
    return out[:limit]


@router.get("/compounds")
def list_compounds(request: Request, q: str = "", limit: int = Query(50, le=500)):
    s = _state(request)
    out = [{"id": c, "name": s.graph.nodes[c].get("name", c), "ddi_covered": c in s.ddi_universe}
           for c in s.compounds if q.lower() in s.graph.nodes[c].get("name", c).lower()]
    return out[:limit]


@router.get("/repurposing")
def repurposing(request: Request, disease_id: str, top_k: int = Query(10, le=50),
                lam: float = Query(0.25, ge=0, le=2), explain: bool = True):
    s = _state(request)
    if disease_id not in s.diseases:
        raise HTTPException(404, f"Unknown disease: {disease_id}")
    i = s.diseases.index(disease_id)

    raw_row = s.scores[i].copy()
    raw_row[s.known[i]] = -np.inf
    fair_row = rerank(s.scores, s.neglect, s.known, lam)[i]
    raw_rank = {int(j): r for r, j in enumerate(np.argsort(-raw_row), start=1)}

    candidates = []
    for j in np.argsort(-fair_row)[:top_k]:
        c = s.compounds[int(j)]
        paths = s.engine.explain(c, disease_id, max_paths=3) if explain else []
        candidates.append({
            "compound_id": c,
            "compound_name": s.graph.nodes[c].get("name", c),
            "model_score": float(raw_row[j]),
            "raw_rank": raw_rank[int(j)],
            "adjusted_score": float(fair_row[j]),
            "evidence_paths": [{"type": p["type"], "text": p["text"], "score": p["score"]} for p in paths],
            "has_evidence_path": bool(paths),
        })

    known_treaters = {s.compounds[j] for j in np.where(s.known[i])[0]}
    evidence_candidates = s.engine.rank_by_evidence(disease_id, s.compounds, exclude=known_treaters, top_k=top_k)

    # How generic is the model's list? Overlap with the drugs that score highest for *every* disease.
    popular = set(np.argsort(-s.scores.mean(axis=0))[:top_k].tolist())
    generic_overlap = len({int(j) for j in np.argsort(-fair_row)[:top_k]} & popular) / max(top_k, 1)

    degree = int(s.degrees[i])
    notes = ["Computational hypotheses for further study, not treatment recommendations."]
    if degree < SPARSE_DEGREE:
        notes.append(f"This disease has only {degree} edge(s) in the knowledge graph, so its embedding "
                     "is weakly informed and rankings should be treated as low-confidence.")
    if generic_overlap >= 0.5:
        notes.append(f"{generic_overlap:.0%} of the model's top candidates are drugs that rank highly for almost "
                     "every disease (a popularity prior), so this list says little that is specific to this disease. "
                     "Prefer the graph-evidence candidates.")
    return {
        "disease": {"id": disease_id, "name": s.graph.nodes[disease_id].get("name", disease_id),
                    "degree": degree, "untreated": bool(s.untreated[i]), "neglect": float(s.neglect[i])},
        "lam": lam, "candidates": candidates, "evidence_candidates": evidence_candidates,
        "generic_overlap": generic_overlap, "notes": notes,
    }


@router.get("/equity")
def equity(request: Request, lam: float = Query(0.25, ge=0, le=2), top_k: int = Query(100, le=1000)):
    s = _state(request)
    raw = s.scores.copy()
    raw[s.known] = -np.inf
    fair = rerank(s.scores, s.neglect, s.known, lam)
    return {
        "lam": lam,
        "raw_ranking": equity_audit(top_pairs(raw, top_k), s.degrees, s.untreated),
        "neglect_aware_ranking": equity_audit(top_pairs(fair, top_k), s.degrees, s.untreated),
    }
