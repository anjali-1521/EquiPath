# Setup

Python 3.12+ recommended (developed on 3.13). All commands run from the repo root.

```bash
python3 -m venv venv && source venv/bin/activate
pip install networkx pandas scikit-learn fastapi "uvicorn[standard]" httpx pydantic
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pykeen torch-geometric
```

## Module 1 - knowledge graph

Needs `git-lfs` (`brew install git-lfs && git lfs install`).

```bash
curl -L https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv -o data/raw/hetionet/nodes.tsv
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/hetio/hetionet.git /tmp/hetionet && (cd /tmp/hetionet && git lfs pull)
gunzip -c /tmp/hetionet/hetnet/tsv/hetionet-v1.0-edges.sif.gz > data/raw/hetionet/edges.sif
python3 data/scripts/merge_graph.py
python3 data/scripts/validate_graph.py
```

## Module 2 - embeddings and prediction

```bash
python3 models/data/build_ddi_labels.py        # Decagon DDI labels, mapped to Hetionet compounds
python3 -m models.embeddings.transe --num-epochs 30
python3 -m models.embeddings.graphsage --num-epochs 100   # reuses TransE's train/test split; rerun after TransE
python3 -m models.prediction.ddi
python3 -m models.prediction.repurposing
python3 -m models.training.validate
```

Training is CPU-friendly but sustained: use `OMP_NUM_THREADS=4 nice -n 15 ...` on a laptop.

## Module 3 and 4

```bash
python3 -m unittest explanation.tests.test_ranking tests.test_end_to_end
python3 -m explanation.neglect_ranking --lam 1.0
uvicorn app.backend.main:app --port 8000      # UI at http://localhost:8000
```
