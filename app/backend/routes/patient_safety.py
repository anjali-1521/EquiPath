from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.backend.routes.ddi import ddi_probability

router = APIRouter()

HIGH, MODERATE = 0.8, 0.5


class SafetyRequest(BaseModel):
    candidate: str = Field(description="Compound node ID being considered, e.g. Compound::DB00945")
    current_medications: list[str] = Field(default_factory=list)


def _side_effects(graph, compound: str) -> set[str]:
    return {v for _, v, e in graph.out_edges(compound, data=True) if e.get("metaedge") == "CcSE"}


@router.post("/patient-safety")
def patient_safety(req: SafetyRequest, request: Request):
    s = request.app.state.ctx
    for c in [req.candidate, *req.current_medications]:
        if c not in s.graph:
            raise HTTPException(404, f"Unknown compound: {c}")

    cand_effects = _side_effects(s.graph, req.candidate)
    checks, flags = [], []
    for med in req.current_medications:
        p = ddi_probability(s, req.candidate, med)
        shared = cand_effects & _side_effects(s.graph, med)
        if p is None:
            level = "not_assessed"
        elif p >= HIGH:
            level = "high"
        elif p >= MODERATE:
            level = "moderate"
        else:
            level = "low"
        checks.append({"medication": s.graph.nodes[med].get("name", med), "medication_id": med,
                       "interaction_score": p, "level": level, "shared_side_effects": len(shared),
                       "example_shared_side_effects": sorted(s.graph.nodes[x].get("name", x) for x in shared)[:5]})
        flags.append(level)

    overall = ("review_required" if "high" in flags else "caution" if "moderate" in flags
               else "incomplete" if "not_assessed" in flags else "no_flags")
    return {
        "candidate": s.graph.nodes[req.candidate].get("name"),
        "overall": overall, "checks": checks,
        "disclaimer": ("Decision-support screening only. Scores come from a statistical model, "
                       "are not calibrated risks, and drugs outside the model's coverage are not assessed. "
                       "A clinician or pharmacist must review any real prescribing decision."),
    }
