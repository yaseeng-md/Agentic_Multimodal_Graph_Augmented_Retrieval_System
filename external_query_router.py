"""
AMG Multimodal RAG — External Query Router

Context-aware external routing and query rewriting using the shared locally loaded
and quantized generation LLM. Dispatches queries to Conversation Context,
Internal Knowledge Base, and/or External Web Search (Brave).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional
import os
import requests
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel


from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

PROMPT_SCHEMA_VERSION = "v1"




load_dotenv()



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pydantic Schemas
# ===========================================================================

class ExternalRouterInput(BaseModel):
    """Input payload provided to the External Query Router."""

    query: str
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    previous_retrieval_context: Optional[dict[str, Any]] = None
    previous_execution_metadata: Optional[dict[str, Any]] = None
    available_sources: list[str] = Field(
        default_factory=lambda: [
            "conversation_context",
            "internal_retrieval",
            "web_search",
            "historical_conversation_search",
        ]
    )
    internal_retrieval_available: bool = True
    web_search_available: bool = True
    current_date: str = Field(default_factory=utc_now)


class ExternalRouteDecision(BaseModel):
    """Structured decision returned by the External Query Router."""

    actions: list[
        Literal["conversation_search", "internal_retrieval", "web_search"]
    ] = Field(default_factory=list)

    use_conversation_context: bool = False
    contextualized_query: str
    internal_query: Optional[str] = None
    web_query: Optional[str] = None
    conversation_query: Optional[str] = None
    reuse_previous_evidence: bool = False
    additional_retrieval_required: bool = False
    direct_answer: bool = False
    clarification_required: bool = False
    clarification_question: Optional[str] = None
    reason: str


# ===========================================================================
# System Prompt & Few-Shot Demonstrations
# ===========================================================================

ROUTER_SYSTEM_PROMPT = """You are the External Query Router for the AMG Multimodal RAG system.
Your job is to analyze the user's query, resolve any conversational references using prior conversation turns and previous retrieval evidence, rewrite the query into standalone source-specific search queries, and select the optimal information sources.

AVAILABLE SOURCES:
1. "conversation_context": Already loaded recent dialog turns. Use to resolve pronouns ('it', 'they', 'that') or earlier topics.
2. "internal_retrieval": Indexed internal technical documents, PDFs, architecture diagrams, graphs, and system manuals.
3. "web_search": Brave Web Search for latest external developments, public benchmarks, recent news, or current industry approaches.
4. "conversation_search": Search across older past conversations (only if requested by user for historical topics not in recent history).

DECISION RULES:
- General knowledge/coding questions with no internal context or external freshness requirement (e.g. "Explain Python decorators") -> direct_answer = true, actions = []
- Internal system/PDF/architecture questions (e.g. "Explain our AMG ingestion pipeline") -> actions = ["internal_retrieval"], produce standalone internal_query
- Questions asking for visual/PDF diagrams -> actions = ["internal_retrieval"], produce internal_query specifying the visual requirement
- Follow-up questions referring to previous conversation -> use_conversation_context = true, rewrite into a standalone contextualized_query
- If previous retrieval evidence is already sufficient to answer (e.g. user asks to explain the same architecture again) -> reuse_previous_evidence = true, additional_retrieval_required = false, actions = []
- If follow-up requires more depth/details -> reuse_previous_evidence = true, additional_retrieval_required = true, actions = ["internal_retrieval"]
- Recent external information or industry comparisons -> actions = ["web_search"], produce standalone web_query
- Comparing internal system with external approaches -> actions = ["internal_retrieval", "web_search"], produce separate internal_query and web_query
- Ambiguous references with NO usable context (e.g. "Compare those two") -> clarification_required = true, provide clarification_question, actions = []

You must output ONLY a valid JSON object matching the requested schema with no commentary or markdown wrappers.
"""


def _extract_json_block(text: str) -> str:
    """Extract JSON from raw LLM output, stripping markdown blocks if present."""
    clean = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Find outermost curly braces
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return clean[start : end + 1].strip()
    return clean


# ===========================================================================
# Deterministic Fallback Heuristic
# ===========================================================================

def fallback_heuristic_routing(input_data: ExternalRouterInput) -> ExternalRouteDecision:
    """
    High-reliability deterministic routing heuristic used as fallback if local LLM
    fails structured parsing or in headless/offline testing environments.
    """
    q = input_data.query.strip()
    q_lower = q.lower()
    history = input_data.conversation_history
    prev_retrieval = input_data.previous_retrieval_context or {}
    has_prev_evidence = prev_retrieval.get("has_previous_evidence", False)

    # 1. Ambiguous reference check without context
    ambiguous_triggers = ["compare those two", "what about that", "compare them", "tell me about it"]
    if not history and any(t in q_lower for t in ambiguous_triggers):
        return ExternalRouteDecision(
            actions=[],
            use_conversation_context=False,
            contextualized_query=q,
            direct_answer=False,
            clarification_required=True,
            clarification_question="Could you please specify which two items or topics you would like to compare?",
            reason="Query contains ambiguous references with no prior conversation context.",
        )

    # 2. Pronoun / context resolution
    use_context = False
    contextualized_q = q
    last_user_query = ""
    last_answer = ""
    if history:
        last_turn = history[-1]
        last_user_query = (
            last_turn.get("query", "")
            if isinstance(last_turn, dict)
            else getattr(last_turn, "query", "")
        )
        last_answer = (
            last_turn.get("answer", "")
            if isinstance(last_turn, dict)
            else getattr(last_turn, "answer", "")
        )

    pronoun_pattern = re.compile(r"\b(it|its|they|their|that|this|the previous|the latter|the former)\b", re.IGNORECASE)
    if history and (pronoun_pattern.search(q_lower) or "again" in q_lower or "further" in q_lower):
        use_context = True
        contextualized_q = f"{q} (in the context of previous query: '{last_user_query}')"

    # 3. Evidence reuse vs additional retrieval
    reuse_evidence = False
    additional_retrieval = False
    if has_prev_evidence and history:
        if any(w in q_lower for w in ["again", "repeat", "summarize that", "as mentioned"]):
            reuse_evidence = True
            additional_retrieval = False
            return ExternalRouteDecision(
                actions=[],
                use_conversation_context=True,
                contextualized_query=contextualized_q,
                reuse_previous_evidence=True,
                additional_retrieval_required=False,
                direct_answer=False,
                clarification_required=False,
                reason="Reusing previous retrieval evidence from the conversation as the query repeats earlier topic.",
            )
        elif any(w in q_lower for w in ["bottleneck", "issue", "limit", "more detail", "how does it calculate"]):
            reuse_evidence = True
            additional_retrieval = True

    # 4. Web vs Internal vs General
    web_triggers = [
        "latest", "recent", "current developments", "news", "industry approaches",
        "state of the art", "sota", "benchmark", "market", "2026", "today"
    ]
    internal_triggers = [
        "amg", "colpali", "colqwen", "ingestion", "pipeline", "pdf", "page",
        "diagram", "architecture", "neo4j", "graph", "qdrant", "docling", "our",
        "internal", "document", "table", "flowchart"
    ]

    is_web = any(w in q_lower for w in web_triggers)
    is_internal = any(w in q_lower for w in internal_triggers) or additional_retrieval or use_context

    # Standalone general coding/concept check
    general_triggers = ["explain python", "how to write", "what is a decorator", "tutorial", "syntax"]
    if any(g in q_lower for g in general_triggers) and not is_internal and not is_web:
        return ExternalRouteDecision(
            actions=[],
            use_conversation_context=False,
            contextualized_query=q,
            direct_answer=True,
            reason="General programming or conceptual question answerable directly without document or web retrieval.",
        )

    actions: list[Literal["conversation_search", "internal_retrieval", "web_search"]] = []
    internal_query = None
    web_query = None

    if is_internal:
        actions.append("internal_retrieval")
        internal_query = contextualized_q

    if is_web:
        actions.append("web_search")
        web_query = q.replace("our", "").strip()

    if not actions:
        # Default safe path: internal retrieval
        actions.append("internal_retrieval")
        internal_query = contextualized_q

    return ExternalRouteDecision(
        actions=actions,
        use_conversation_context=use_context,
        contextualized_query=contextualized_q,
        internal_query=internal_query,
        web_query=web_query,
        conversation_query=None,
        reuse_previous_evidence=reuse_evidence,
        additional_retrieval_required=additional_retrieval,
        direct_answer=False,
        clarification_required=False,
        reason=(
            f"Heuristic routing determined actions {actions} based on keywords "
            f"(internal={is_internal}, web={is_web}, use_context={use_context})."
        ),
    )


# ===========================================================================
# External Query Router Class
# ===========================================================================

class ExternalQueryRouter:
    """
    Context-aware router utilizing the shared local LLM (Qwen) for query rewriting
    and source selection across internal documents and Brave Web Search.
    """

    def __init__(self, generator: Optional[Any] = None):
        self._generator = generator

    def _get_generator(self) -> Optional[Any]:
        if self._generator is not None:
            return self._generator
        try:
            from generate import get_generator
            return get_generator()
        except Exception:
            return None

    def build_prompt_messages(self, input_data: ExternalRouterInput) -> list[dict[str, Any]]:
        """Construct the prompt messages for the local LLM."""
        context_lines: list[str] = []

        if input_data.conversation_history:
            context_lines.append("RECENT CONVERSATION HISTORY:")
            for idx, turn in enumerate(input_data.conversation_history[-3:], start=1):
                q = turn.get("query", "") if isinstance(turn, dict) else getattr(turn, "query", "")
                a = turn.get("answer", "") if isinstance(turn, dict) else getattr(turn, "answer", "")
                context_lines.append(f"  [Turn {idx}] User: {q}")
                context_lines.append(f"  [Turn {idx}] Assistant: {a[:200]}...")

        if input_data.previous_retrieval_context:
            pr = input_data.previous_retrieval_context
            context_lines.append("PREVIOUS RETRIEVAL CONTEXT:")
            context_lines.append(f"  Has Evidence: {pr.get('has_previous_evidence', False)}")
            context_lines.append(f"  Result Count: {pr.get('result_count', 0)}")
            refs = pr.get("page_references", [])
            if refs:
                context_lines.append(f"  Pages Referenced: {refs[:4]}")

        context_lines.append(f"CURRENT DATE: {input_data.current_date}")
        context_lines.append(f"USER QUERY: {input_data.query}")

        user_content = "\n".join(context_lines)

        return [
            {"role": "system", "content": [{"type": "text", "text": ROUTER_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
        ]

    def route(
        self,
        input_data: ExternalRouterInput | dict[str, Any],
    ) -> tuple[ExternalRouteDecision, dict[str, Any]]:
        """
        Execute external routing decision. Returns (ExternalRouteDecision, routing_metadata).
        """
        if isinstance(input_data, dict):
            input_data = ExternalRouterInput(**input_data)

        if not input_data.query or not input_data.query.strip():
            raise ValueError("Query cannot be empty")

        generator = self._get_generator()
        start_time = time.perf_counter()

        # If generator is available and has structured text generation
        if generator is not None and hasattr(generator, "generate_structured_text"):
            messages = self.build_prompt_messages(input_data)
            max_retries = 1
            for attempt in range(max_retries + 1):
                try:
                    raw_text, duration_ms = generator.generate_structured_text(
                        messages=messages,
                        max_new_tokens=400,
                        temperature=0.0,
                    )
                    json_str = _extract_json_block(raw_text)
                    parsed_dict = json.loads(json_str)
                    decision = ExternalRouteDecision(**parsed_dict)

                    metadata = {
                        "router_model": getattr(generator, "model_id", "Qwen3-VL-local"),
                        "prompt_version": PROMPT_SCHEMA_VERSION,
                        "routing_latency_ms": duration_ms,
                        "validation_status": "valid_schema",
                    }
                    return decision, metadata

                except (json.JSONDecodeError, ValidationError) as parse_exc:
                    logger.warning(
                        "External routing LLM attempt %d failed schema validation: %s",
                        attempt,
                        parse_exc,
                    )
                    if attempt == max_retries:
                        break
                except Exception as exc:
                    logger.warning("External routing local LLM error: %s", exc)
                    break

        # Fallback to deterministic heuristic
        decision = fallback_heuristic_routing(input_data)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        metadata = {
            "router_model": "heuristic_fallback",
            "prompt_version": PROMPT_SCHEMA_VERSION,
            "routing_latency_ms": elapsed_ms,
            "validation_status": "fallback_applied",
        }
        return decision, metadata


_ROUTER_SINGLETON: Optional[ExternalQueryRouter] = None


def get_external_query_router() -> ExternalQueryRouter:
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        _ROUTER_SINGLETON = ExternalQueryRouter()
    return _ROUTER_SINGLETON


# Re-export InternalQueryRouter and InternalRetrievalDecision from query_router
from query_router import InternalQueryRouter, InternalRetrievalDecision

__all__ = [
    "ExternalRouterInput",
    "ExternalRouteDecision",
    "ExternalQueryRouter",
    "get_external_query_router",
    "fallback_heuristic_routing",
    "InternalQueryRouter",
    "InternalRetrievalDecision",
]
