"""Standard Information Retrieval evaluation metrics (Recall@K, MRR@K, nDCG@K)."""

import math
from typing import Sequence, Union


def _extract_doc_ids(ranked_results: Union[Sequence[str], Sequence[tuple[str, float]]]) -> list[str]:
    """Helper to extract doc_ids whether results are list of doc_id or list of (doc_id, score)."""
    if not ranked_results:
        return []
    first = ranked_results[0]
    if isinstance(first, (tuple, list)):
        return [item[0] for item in ranked_results]
    return list(ranked_results)


def compute_recall_at_k(
    qrels: dict[str, dict[str, int]],
    run_results: dict[str, Union[Sequence[str], Sequence[tuple[str, float]]]],
    k: int,
) -> float:
    """
    Compute Recall@K averaged across all queries.

    Recall@K = |Retrieved@K ∩ Relevant| / |Relevant|
    """
    if not qrels or k <= 0:
        return 0.0

    scores: list[float] = []

    for query_id, rel_dict in qrels.items():
        relevant_docs = {doc_id for doc_id, score in rel_dict.items() if score > 0}
        if not relevant_docs:
            continue

        retrieved_list = _extract_doc_ids(run_results.get(query_id, []))[:k]
        retrieved_set = set(retrieved_list)

        hits = len(retrieved_set & relevant_docs)
        scores.append(hits / len(relevant_docs))

    return sum(scores) / len(scores) if scores else 0.0


def compute_mrr_at_k(
    qrels: dict[str, dict[str, int]],
    run_results: dict[str, Union[Sequence[str], Sequence[tuple[str, float]]]],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR@K).

    Reciprocal Rank is 1 / rank of the first relevant document if within top-K, else 0.
    """
    if not qrels or k <= 0:
        return 0.0

    rr_scores: list[float] = []

    for query_id, rel_dict in qrels.items():
        relevant_docs = {doc_id for doc_id, score in rel_dict.items() if score > 0}
        if not relevant_docs:
            continue

        retrieved_list = _extract_doc_ids(run_results.get(query_id, []))[:k]
        rr = 0.0
        for rank_1b, doc_id in enumerate(retrieved_list, start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank_1b
                break

        rr_scores.append(rr)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


def compute_ndcg_at_k(
    qrels: dict[str, dict[str, int]],
    run_results: dict[str, Union[Sequence[str], Sequence[tuple[str, float]]]],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG@K).

    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    if not qrels or k <= 0:
        return 0.0

    ndcg_scores: list[float] = []

    for query_id, rel_dict in qrels.items():
        if not rel_dict:
            continue

        # Calculate DCG@K
        retrieved_list = _extract_doc_ids(run_results.get(query_id, []))[:k]
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_list, start=1):
            rel = rel_dict.get(doc_id, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        # Calculate Ideal DCG@K
        ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            if rel > 0:
                idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        if idcg > 0.0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def compute_all_metrics(
    qrels: dict[str, dict[str, int]],
    run_results: dict[str, Union[Sequence[str], Sequence[tuple[str, float]]]],
    cutoffs: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K, MRR@K, and nDCG@K across all requested cutoffs."""
    metrics: dict[str, float] = {}
    max_k = max(cutoffs) if cutoffs else 10

    for k in cutoffs:
        metrics[f"Recall@{k}"] = round(compute_recall_at_k(qrels, run_results, k), 4)

    for k in cutoffs:
        metrics[f"MRR@{k}"] = round(compute_mrr_at_k(qrels, run_results, k), 4)

    for k in cutoffs:
        metrics[f"nDCG@{k}"] = round(compute_ndcg_at_k(qrels, run_results, k), 4)

    return metrics
