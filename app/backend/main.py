"""EquiPath API. Run from the repo root:  uvicorn app.backend.main:app --port 8000"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.backend.routes import ddi, patient_safety, repurposing
from app.backend.state import load_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = load_state()
    yield


app = FastAPI(title="EquiPath", version="0.1.0", lifespan=lifespan)
app.include_router(repurposing.router, prefix="/api")
app.include_router(ddi.router, prefix="/api")
app.include_router(patient_safety.router, prefix="/api")


@app.get("/api/health")
def health():
    s = app.state.ctx
    return {"status": "ok", "nodes": s.graph.number_of_nodes(), "ddi_model_loaded": s.ddi_clf is not None}


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
