# EquiPath

**Explainable, Equity-Aware Drug Repurposing & Drug-Drug Interaction Prediction for Rare Diseases using Knowledge Graph Embeddings**

Capstone project — MIT World Peace University, Pune | Department of Computer Engineering & Technology

## What this is

EquiPath predicts new therapeutic uses for existing drugs (repurposing) and flags risky drug-drug combinations (DDI), using knowledge graph embeddings trained on a biomedical knowledge graph. Every prediction comes with a traceable explanation, and the system deliberately prioritizes rare, under-treated diseases rather than defaulting to whatever's already well-documented — a knowledge graph like Hetionet, by default, favors common diseases simply because they're better connected.

More context on the problem this solves, the literature it's built on, and the full technical writeup: see [`/docs`](./docs).

## Team & module ownership

| Module | Owner | Scope |
|---|---|---|
| Data & Knowledge Graph | Ruchira | Graph construction, custom rare-disease nodes |
| Embedding & Prediction | Anjali | TransE + GraphSAGE, repurposing + DDI prediction |
| Explanation & Neglect-Ranking | Husaina | Path-based explanation, equity-aware re-ranking |
| Application | Aaditi | Backend API, frontend, patient safety filtering |

Guide: Dr. Suja Panicker

## Status

All four modules are implemented, each on its own branch (merge order: `module-1-data-kg` -> `module-2-embeddings` -> `module-3-explanation` -> `module-4-app`). See [`docs/architecture.md`](./docs/architecture.md) for measured results and, importantly, the limitations: rare-disease model rankings are weak and are flagged as such in the API and UI.

## Quick start

```bash
git clone https://github.com/anjali-1521/EquiPath.git
cd EquiPath
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # install torch from the CPU index first if you want a smaller build
```

The data and trained checkpoints are not committed (they are large and reproducible). [`docs/setup.md`](./docs/setup.md) lists the commands to download Hetionet, build the graph, train the models, run the tests, and start the app (`uvicorn app.backend.main:app`, UI at http://localhost:8000).

## Tech stack

- **Data/Graph:** Hetionet, NetworkX
- **Embeddings:** PyKEEN (TransE), PyTorch Geometric or DGL (GraphSAGE)
- **Backend:** FastAPI
- **Frontend:** dependency-free static HTML/JS served by the API (no build step)
- **Deployment:** Docker, Render/Railway

## Repository structure

See [directory structure](#directory-structure) below, or run `tree -L 2` from the repo root.

## License

TBD — add before making the repo public, if applicable to your institution's project submission policy.
