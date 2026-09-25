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

🚧 Implementation in progress. See [Issues](../../issues) and [Project board](../../projects) for current phase.

## Quick start

```bash
git clone https://github.com/<org>/equipath.git
cd equipath
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt --break-system-packages
```

See [`/docs/setup.md`](./docs/setup.md) for full environment setup, including the knowledge graph download step.

## Tech stack

- **Data/Graph:** Hetionet, NetworkX
- **Embeddings:** PyKEEN (TransE), PyTorch Geometric or DGL (GraphSAGE)
- **Backend:** FastAPI
- **Frontend:** React
- **Deployment:** Docker, Render/Railway

## Repository structure

See [directory structure](#directory-structure) below, or run `tree -L 2` from the repo root.

## License

TBD — add before making the repo public, if applicable to your institution's project submission policy.
