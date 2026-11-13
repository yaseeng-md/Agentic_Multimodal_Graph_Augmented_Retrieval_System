
import os
import requests

from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


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
            "Content-Type": "application/json"
        }

    def route(self, query: str) -> str:
        """
        Determine whether Graph, Visual or Hybrid retrieval
        is required for the given query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        payload = {
            "model": self.model,

            "state": query.strip(),

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
                        )
                    }
                }
            }
        }

        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint

        try:
            response = self.session.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Laya API request timed out"
            ) from exc

        except requests.exceptions.HTTPError as exc:
            try:
                error_details = response.json()
            except ValueError:
                error_details = response.text

            raise RuntimeError(
                f"Laya API returned HTTP {response.status_code}: "
                f"{error_details}"
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to connect to Laya API: {exc}"
            ) from exc

        except ValueError as exc:
            raise RuntimeError(
                "Laya returned an invalid JSON response"
            ) from exc

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

        result = self.decide_internal_router(InternalRetrievalDecision(
            strategy=strategy,
            probabilities=probabilities,
            confidence=answer.get("confidence"),
            model=result.get("model", self.model),
            latency_ms=result.get("latency_ms")
        ))

        return result
    

    def close(self):
        """Close the underlying HTTP session."""
        self.session.close()


    def decide_internal_router(
            self,
            decision: InternalRetrievalDecision
        ) -> str:

        # Default to Hybrid when confidence is unavailable or low
        if decision.confidence is None:
            return "hybrid"

        if decision.confidence <= 0.5:
            return "hybrid"

        # Select Graph when strongly indicated
        if decision.probabilities.get("graph", 0) > 0.6:
            return "graph"

        # Select Visual when strongly indicated
        elif decision.probabilities.get("visual", 0) > 0.6:
            return "visual"

        # Default to Hybrid
        return "hybrid"