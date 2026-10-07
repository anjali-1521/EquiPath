# Architecture and results

```
Hetionet + 4 custom rare diseases  --(Module 1)-->  merged_graph.pkl (47,039 nodes, 2.25M edges)
        |
        +--(Module 2)--> TransE embeddings --> supervised treatment head --> repurposing scores
        |                GraphSAGE embeddings                                   |
        |                Decagon DDI labels --> DDI classifier                  |
        |                                                                       v
        +--(Module 3)--> path explanations (metapaths) + neglect-aware re-ranking + equity audit
                                          |
                                          v
                        (Module 4) FastAPI  /api/*  +  static frontend at /
```

| Module | Code | What it does |
|---|---|---|
| 1 Data and KG | `data/` | Hetionet + 4 custom disease nodes merged, validated (incl. connectivity) |
| 2 Embeddings and prediction | `models/` | TransE, GraphSAGE, supervised treatment head, DDI classifier, shared metrics |
| 3 Explanation and neglect-ranking | `explanation/` | Metapath evidence paths, graph-evidence ranking, per-disease normalisation + neglect bonus, equity audit |
| 4 Application | `app/`, `tests/` | REST API (repurposing, DDI, patient-safety screen), no-build web UI with Researcher and Clinician views |

## Measured results

| Model | Evaluation | Result |
|---|---|---|
| TransE (dim 128, 30 epochs, 10 negatives) | 112,511 held-out triples, 100 sampled negatives each | Hits@10 0.80, MRR 0.49, mean rank 7.7 |
| GraphSAGE (2 layers, relation-agnostic) | same triples and protocol | Hits@10 0.33, MRR 0.12 |
| Raw TransE "treats" distance | 34 held-out treatment edges, ranked against ~1.5k compounds | Hits@10 0.03, mean rank 497 |
| Supervised treatment head (on TransE) | same 34 edges | Hits@10 0.12, mean rank 365, AUROC 0.76 |
| DDI classifier (on TransE) | 6,827 held-out drug pairs | AUROC 0.88, AUPRC 0.88 |

GraphSAGE scores lower than TransE because it has no per-relation decoder (it learns "an edge exists", not which kind), and adding it did not help the treatment head.

## Limitations

- **Rare-disease repurposing is weak.** The treatment head's top candidates for the four custom diseases are largely the same generally-popular drugs for every disease (a popularity prior). The API reports this explicitly (`generic_overlap`) and ships the graph-evidence ranking beside the model ranking. The held-out treatment evaluation covers only 34 edges, so the Hits@10 estimates are noisy.
- **Graph-evidence candidates exist only where the graph has links.** Wilson's, SCA3 and MRKH have gene links (e.g. platinum drugs bind ATP7B); Asherman's has no gene and yields none. Some evidence is weak by nature (e.g. amino acids "resembling" penicillamine by structural similarity).
- **DDI labels** come from Decagon (FAERS-derived), mapped to Hetionet via PubChem cross-references: 325 drugs, 17,067 interacting pairs. Drugs outside this set are reported as "not assessed". Negatives are random unobserved pairs, which may include undiscovered interactions; scores are screening signals, not calibrated risks.
- **Neglect weighting is a policy, not a prediction.** It overcorrects fast (weight >= 1 sends every top-100 pick to an untreated disease); default 0.25. With the supervised head, the raw ranking was not measurably skewed toward well-connected diseases.
- Evaluation uses sampled negatives for the generic link metrics (cheaper than full ranking, optimistic versus published full-ranking numbers); the treatment evaluation is full filtered ranking.
- The Docker setup is written but untested.
- The custom gene nodes originally duplicated real Hetionet genes and left SCA3 isolated; Module 1 now links them to the real nodes and validation checks connectivity.

None of this is medical advice. Repurposing output is a set of computational hypotheses; the patient-safety screen is decision support that a clinician must review.
