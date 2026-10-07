"""Shared link-prediction evaluation metrics for EquiPath's embedding models.

Used by both models/embeddings/transe.py and models/embeddings/graphsage.py
so TransE and GraphSAGE are scored under the same protocol and are directly
comparable: for each held-out positive (head, relation, tail) triple, rank
the true tail against a pool of sampled negative tails using the model's
scoring function, then aggregate ranks into Hits@k / Mean Rank / MRR.
GraphSAGE has no per-relation decoder (it only models generic edge
existence), so it simply ignores the relation argument its score function
receives - the shared (head, relation, tail) signature still applies to it.

A full filtered-ranking evaluation (scoring every held-out triple against
ALL ~47k entities) is the textbook-correct protocol but is expensive to
rerun repeatedly at this graph's scale during development; sampled-negative
ranking (scoring against a few hundred negatives) is the standard, much
cheaper approximation used across the KG-embedding literature and is what's
used here - documented rather than silently swapped in.
"""

from collections.abc import Callable, Sequence

import numpy as np

Triple = tuple[int, str, int]  # (head_id, relation_label, tail_id)


def compute_rank_metrics(ranks: Sequence[int]) -> dict[str, float]:
    """Aggregate a list of 1-indexed ranks (rank 1 = model got it exactly right)
    into standard link-prediction metrics."""
    ranks_arr = np.asarray(ranks, dtype=float)
    if ranks_arr.size == 0:
        raise ValueError("compute_rank_metrics received an empty ranks list")

    return {
        "hits@1": float(np.mean(ranks_arr <= 1)),
        "hits@3": float(np.mean(ranks_arr <= 3)),
        "hits@10": float(np.mean(ranks_arr <= 10)),
        "mean_rank": float(np.mean(ranks_arr)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "num_evaluated": int(ranks_arr.size),
    }


def evaluate_link_prediction_sampled(
    score_fn: Callable[[int, str, int], float],
    test_triples: Sequence[Triple],
    all_entity_ids: Sequence[int],
    num_negatives: int = 100,
    known_positive_tails: dict[tuple[int, str], set[int]] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Rank each test triple's true tail against `num_negatives` sampled
    negative tails, using `score_fn(head_id, relation_label, tail_id) ->
    higher is better`.

    known_positive_tails: optional {(head_id, relation): {tail_id, ...}} of
        ALL known true tails for that (head, relation) across train+test -
        if given, sampled negatives that happen to also be true positives
        are excluded ("filtered" ranking), so the model isn't penalized for
        correctly scoring an undisclosed true edge above the one being
        evaluated.
    """
    rng = np.random.default_rng(seed)
    all_entity_ids = np.asarray(all_entity_ids)
    ranks = []

    for head, relation, true_tail in test_triples:
        exclude = set()
        if known_positive_tails:
            exclude = known_positive_tails.get((head, relation), set())
        exclude = exclude | {true_tail}

        negatives = []
        # oversample then filter, so we still end up with num_negatives candidates
        candidates = rng.choice(all_entity_ids, size=num_negatives * 2, replace=True)
        for cand in candidates:
            cand = int(cand)
            if cand not in exclude:
                negatives.append(cand)
            if len(negatives) >= num_negatives:
                break

        candidate_tails = [true_tail] + negatives
        scores = [score_fn(head, relation, t) for t in candidate_tails]

        # rank = 1 + number of candidates strictly better-scoring than the true tail
        true_score = scores[0]
        rank = 1 + sum(1 for s in scores[1:] if s > true_score)
        ranks.append(rank)

    return compute_rank_metrics(ranks)
