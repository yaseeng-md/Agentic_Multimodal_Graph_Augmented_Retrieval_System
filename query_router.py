"""
AMG Multimodal RAG — Internal Query Router

Decides the optimal internal retrieval strategy (graph, visual, or hybrid)
using the external Laya decision model with Redis-based caching.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)


class InternalRetrievalDecision(BaseModel):
    """Structured response returned by the internal query router."""

    strategy: Literal["graph", "visual", "hybrid"]
    probabilities: dict[str, float]
    confidence: float | None = None
    model: str
    latency_ms: float | None = None


class InternalQueryRouter:
    """
    Decides which internal retrieval mechanism should be used.

    Supported strategies:
        - graph
        - visual
        - hybrid
    """

    STRATEGIES = ["graph", "visual", "hybrid"]

    def __init__(self, timeout: int = 20):
        self.access_token = os.getenv("LAYA_ACCESS_TOKEN")
        self.endpoint = os.getenv("LAYA_MODEL_END_POINT")
        self.model = os.getenv("LAYA_MODEL")
        self.checkpoint = os.getenv("LAYA_CHECKPOINT")

        if not self.access_token:
            raise ValueError("LAYA_ACCESS_TOKEN is missing")
        if not self.endpoint:
            raise ValueError("LAYA_MODEL_END_POINT is missing")
        if not self.model:
            raise ValueError("LAYA_MODEL is missing")

        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def route(self, query: str) -> str:
        """
        Determine whether Graph, Visual, or Hybrid retrieval
        is required for the given query. Checks Redis cache first.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        clean_query = query.strip()

        # -------------------------------------------------------------------
        # 1. Redis Decision Cache Lookup
        # -------------------------------------------------------------------
        cache_key = None
        try:
            from redis_manager import (
                DEFAULT_ROUTER_DECISION_TTL,
                cache_get_json,
                cache_set_json,
                router_decision_cache_key,
            )

            cache_key = router_decision_cache_key(
                query=clean_query,
                model=self.model,
                checkpoint=self.checkpoint,
            )
            cached_strategy = cache_get_json(cache_key)
            if cached_strategy is not None and cached_strategy in self.STRATEGIES:
                logger.info(
                    "Internal query router cache hit for query: '%s' -> %s",
                    clean_query,
                    cached_strategy,
                )
                return cached_strategy

            logger.info(
                "Internal query router cache miss for query: '%s'. Invoking Laya model...",
                clean_query,
            )
        except Exception as redis_exc:
            logger.warning(
                "Redis failure during router decision cache lookup: %s. Continuing with direct Laya invocation.",
                redis_exc,
            )

        # -------------------------------------------------------------------
        # 2. Laya Model Invocation
        # -------------------------------------------------------------------
        payload: dict[str, Any] = {
            "model": self.model,
            "state": clean_query,
            "questions": {
                "retrieval_strategy": {
                    "type": "choice",
                    "instructions": (
                        "Determine which internal retrieval "
                        "strategy is required to answer the "
                        "user's query. Select the smallest "
                        "sufficient retrieval strategy."
                    ),
                    "criteria": {
                        "graph": (
                            "Use Graph Retrieval when answering "
                            "requires structured entities, "
                            "relationships, dependencies, "
                            "connections or multi-hop reasoning. "
                            "The required evidence is relational."
                        ),
                        "visual": (
                            "Use Visual Retrieval when answering "
                            "requires visual information from "
                            "PDF pages, diagrams, flowcharts, "
                            "tables, charts, screenshots or "
                            "spatial layouts."
                        ),
                        "hybrid": (
                            "Use Hybrid Retrieval only when "
                            "both graph relationships and "
                            "visual document evidence are "
                            "necessary to answer the query."
                        ),
                    },
                }
            },
        }

        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint

        try:
            response = self.session.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Laya API request timed out") from exc

        except requests.exceptions.HTTPError as exc:
            try:
                error_details = response.json()
            except ValueError:
                error_details = response.text

            raise RuntimeError(
                f"Laya API returned HTTP {response.status_code}: {error_details}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Failed to connect to Laya API: {exc}") from exc

        except ValueError as exc:
            raise RuntimeError("Laya returned an invalid JSON response") from exc

        # Extract the model's decision
        try:
            answer = result["answers"]["retrieval_strategy"]
            strategy = answer["choice"]
            probabilities = answer["probabilities"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Laya response structure: {result}"
            ) from exc

        # Validate the selected strategy
        if strategy not in self.STRATEGIES:
            raise RuntimeError(
                f"Laya returned an unsupported strategy: {strategy}"
            )

        final_strategy = self.decide_internal_router(
            InternalRetrievalDecision(
                strategy=strategy,
                probabilities=probabilities,
                confidence=answer.get("confidence"),
                model=result.get("model", self.model),
                latency_ms=result.get("latency_ms"),
            )
        )

        # -------------------------------------------------------------------
        # 3. Cache Valid Routing Decision in Redis
        # -------------------------------------------------------------------
        if cache_key is not None and final_strategy in self.STRATEGIES:
            try:
                from redis_manager import DEFAULT_ROUTER_DECISION_TTL, cache_set_json

                cache_set_json(
                    cache_key,
                    final_strategy,
                    ttl_seconds=DEFAULT_ROUTER_DECISION_TTL,
                )
                logger.info(
                    "Cached internal routing decision for query: '%s' -> %s (ttl=%ds)",
                    clean_query,
                    final_strategy,
                    DEFAULT_ROUTER_DECISION_TTL,
                )
            except Exception as cache_set_exc:
                logger.warning(
                    "Failed to cache internal routing decision in Redis: %s",
                    cache_set_exc,
                )

        return final_strategy

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "InternalQueryRouter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def decide_internal_router(
        self,
        decision: InternalRetrievalDecision,
    ) -> str:
        """
        Decide the final strategy using the model's chosen decision,
        falling back to hybrid only on high uncertainty or near-ties.
        """
        probs = decision.probabilities or {}
        graph_p = probs.get("graph", 0.0)
        visual_p = probs.get("visual", 0.0)
        hybrid_p = probs.get("hybrid", 0.0)

        # 1. Clear single-modality dominance
        if graph_p > 0.5 and graph_p > visual_p:
            return "graph"
        if visual_p > 0.5 and visual_p > graph_p:
            return "visual"

        # 2. Model's explicit choice when significantly ahead of alternative single modality
        if decision.strategy == "graph" and graph_p > (visual_p + 0.1):
            return "graph"
        if decision.strategy == "visual" and visual_p > (graph_p + 0.1):
            return "visual"

        # 3. Default to safe Hybrid when ambiguous or both modalities are needed
        return "hybrid"
