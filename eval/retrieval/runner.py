"""Runs retrieval benchmark queries against the isolated Qdrant evaluation collection."""

import logging
import time
from typing import Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from eval.config import (
    EVAL_COLLECTION_NAME,
    EVAL_VECTOR_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
)
from eval.retrieval.indexer import get_eval_qdrant_client
from query import QueryEncoder, get_query_encoder

logger = logging.getLogger("amg-eval-runner")


class ViDoReRunner:
    """Executes queries against Qdrant and retrieves ranked candidate pages."""

    def __init__(
        self,
        collection_name: str = EVAL_COLLECTION_NAME,
        vector_name: str = EVAL_VECTOR_NAME,
        client: Optional[QdrantClient] = None,
        encoder: Optional[QueryEncoder] = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.client = client or get_eval_qdrant_client()
        self.encoder = encoder

    def run_queries(
        self,
        queries: dict[str, str],
        top_k: int = 10,
    ) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
        """
        Execute benchmark queries and return ranked top-K candidates.

        Parameters
        ----------
        queries:
            Mapping of query_id -> query string.
        top_k:
            Number of top results to retrieve.

        Returns
        -------
        run_results:
            Mapping of query_id -> list of (doc_id, score) ranked descending.
        stats:
            Execution statistics (total_time_sec, avg_latency_ms).
        """
        if self.encoder is None:
            self.encoder = get_query_encoder()

        run_results: dict[str, list[tuple[str, float]]] = {}
        latencies_ms: list[float] = []

        logger.info(
            "Executing %d queries against collection %s (top_k=%d)...",
            len(queries),
            self.collection_name,
            top_k,
        )

        t_start = time.perf_counter()

        for query_id, query_text in tqdm(queries.items(), desc="Evaluating Queries"):
            q_start = time.perf_counter()

            # 1. Encode query tokens into multivector [N_tokens, 128]
            query_embedding = self.encoder.encode(query_text)

            # 2. Search Qdrant using native MAX_SIM late interaction
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=self.vector_name,
                limit=top_k,
                with_payload=["doc_id", "dataset_name"],
                with_vectors=False,
            )

            q_elapsed_ms = (time.perf_counter() - q_start) * 1000.0
            latencies_ms.append(q_elapsed_ms)

            ranked: list[tuple[str, float]] = []
            for point in response.points:
                if point.payload and "doc_id" in point.payload:
                    ranked.append((str(point.payload["doc_id"]), float(point.score)))

            run_results[query_id] = ranked

        total_time = time.perf_counter() - t_start
        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

        stats = {
            "total_queries": len(queries),
            "total_time_sec": round(total_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
        }

        logger.info(
            "Completed %d queries in %.2fs (avg: %.2f ms/query)",
            len(queries),
            total_time,
            avg_latency,
        )

        return run_results, stats
