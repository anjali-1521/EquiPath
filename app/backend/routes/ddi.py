from fastapi import APIRouter, HTTPException, Request

from models.prediction.ddi import predict_interaction

router = APIRouter()


def ddi_probability(s, c1: str, c2: str) -> float | None:
    """None when either compound is outside the DDI model's 325-drug training universe."""
    if s.ddi_clf is None:
        raise HTTPException(503, "DDI classifier not trained yet (run models/prediction/ddi.py)")
    if c1 not in s.ddi_universe or c2 not in s.ddi_universe:
        return None
    return predict_interaction(c1, c2, s.ddi_clf, s.entity_emb, s.entity_to_id)


@router.get("/ddi")
def ddi(request: Request, compound_1: str, compound_2: str):
    s = request.app.state.ctx
    for c in (compound_1, compound_2):
        if c not in s.graph:
            raise HTTPException(404, f"Unknown compound: {c}")
    p = ddi_probability(s, compound_1, compound_2)
    return {
        "compound_1": s.graph.nodes[compound_1].get("name"),
        "compound_2": s.graph.nodes[compound_2].get("name"),
        "covered": p is not None,
        "interaction_score": p,
        "note": ("Screening score from a classifier trained on reported polypharmacy side effects "
                 "(Decagon); not a calibrated risk or clinical determination."
                 if p is not None else
                 "One or both drugs are outside the 325 drugs this model was trained on - no estimate available."),
    }
