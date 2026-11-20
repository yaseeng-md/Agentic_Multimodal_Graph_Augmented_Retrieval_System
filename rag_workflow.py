"""
AMG Multimodal RAG — LangGraph Orchestration Workflow

Coordinates the complete RAG lifecycle:
Context Loading -> External Query Routing -> Internal (Laya) Routing / Web Search ->
Evidence Aggregation -> Answer Generation -> Metadata & Turn Persistence.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

import conversation_manager
from conversation_manager import (
    get_conversation_store,
    new_conversation_id,
    validate_conversation_id,
)
from external_query_router import (
    ExternalQueryRouter,
    ExternalRouterInput,
    get_external_query_router,
    InternalQueryRouter
)
from web_search import get_brave_search_client

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    """Execution state passed between LangGraph nodes."""

    # User & Tenant Context
    user_id: str
    organization: str
    role: str
    original_query: str
    conversation_id: str

    # Retrieval Constraints
    document_ids: list[str]
    version_ids: list[str]
    recursive_search: bool
    top_k: int
    enable_rerank: bool
    rerank_threshold: float
    retrieval_top_k: int

    # Loaded Context
    conversation_history: list[dict[str, Any]]
    previous_retrieval_context: Optional[dict[str, Any]]

    # Routing Decisions & Metadata
    routing_decision: dict[str, Any]
    routing_metadata: dict[str, Any]

    # Source Results
    reused_sources: list[dict[str, Any]]
    retrieval_result: Optional[dict[str, Any]]
    retrieval_metadata: dict[str, Any]
    web_search_result: Optional[dict[str, Any]]
    web_metadata: dict[str, Any]
    web_evidence_text: str

    # Aggregated Evidence & Output
    aggregated_sources: list[dict[str, Any]]
    context_pages_for_generator: list[dict[str, Any]]
    answer: str
    generation_response: Optional[dict[str, Any]]
    error: Optional[str]


# ===========================================================================
# Node Implementations
# ===========================================================================

def node_load_context(state: RAGState) -> dict[str, Any]:
    """Load conversation history and previous retrieval evidence."""
    conv_store = conversation_manager.get_conversation_store()
    cid = state.get("conversation_id")
    user_id = state.get("user_id", "default_user")
    org = state.get("organization", "default_org")

    if not cid or not str(cid).strip():
        cid = new_conversation_id()

    conv, _ = conv_store.get_or_create_conversation(
        conversation_id=cid,
        user_id=user_id,
        organization=org,
    )
    resolved_cid = conv.conversation_id

    # Load turns and previous retrieval metadata
    raw_turns = conv_store.get_conversation_turns(resolved_cid, user_id=user_id, organization=org)
    history = [t.to_dict() for t in raw_turns]
    prev_retrieval = conv_store.get_last_retrieval_context(resolved_cid, user_id=user_id, organization=org)

    return {
        "conversation_id": resolved_cid,
        "conversation_history": history,
        "previous_retrieval_context": prev_retrieval,
    }


def node_route_query(state: RAGState) -> dict[str, Any]:
    """Execute External Query Router using shared local LLM."""
    router = get_external_query_router()
    router_input = ExternalRouterInput(
        query=state["original_query"],
        conversation_history=state.get("conversation_history", []),
        previous_retrieval_context=state.get("previous_retrieval_context"),
    )

    decision, metadata = router.route(router_input)
    dec_dict = decision.model_dump()

    # Enrich metadata for persistence
    metadata.update({
        "contextualized_query": dec_dict.get("contextualized_query"),
        "selected_actions": dec_dict.get("actions", []),
        "use_conversation_context": dec_dict.get("use_conversation_context", False),
        "internal_retrieval_selected": "internal_retrieval" in dec_dict.get("actions", []),
        "web_search_selected": "web_search" in dec_dict.get("actions", []),
        "internal_query": dec_dict.get("internal_query"),
        "web_query": dec_dict.get("web_query"),
        "reuse_previous_evidence": dec_dict.get("reuse_previous_evidence", False),
        "additional_retrieval_required": dec_dict.get("additional_retrieval_required", False),
        "clarification_required": dec_dict.get("clarification_required", False),
        "clarification_question": dec_dict.get("clarification_question"),
        "reason": dec_dict.get("reason", ""),
    })

    return {
        "routing_decision": dec_dict,
        "routing_metadata": metadata,
    }


def node_clarification(state: RAGState) -> dict[str, Any]:
    """Handle query clarification without executing search or generation."""
    dec = state.get("routing_decision", {})
    clarification_q = dec.get("clarification_question") or (
        "Could you please clarify your question or specify which item you are referring to?"
    )
    return {
        "answer": clarification_q,
        "aggregated_sources": [],
        "generation_response": {
            "answer": clarification_q,
            "sources": [],
            "generation": {"status": "clarification_requested"},
        },
    }


def node_retrieval_and_search(state: RAGState) -> dict[str, Any]:
    """Execute selected internal and/or external information sources."""
    dec = state.get("routing_decision", {})
    actions = dec.get("actions", [])
    reuse_evidence = dec.get("reuse_previous_evidence", False)
    additional_retrieval = dec.get("additional_retrieval_required", False)

    reused_sources: list[dict[str, Any]] = []
    retrieval_result: Optional[dict[str, Any]] = None
    retrieval_metadata: dict[str, Any] = {}
    web_search_result: Optional[dict[str, Any]] = None
    web_metadata: dict[str, Any] = {}
    web_evidence_text = ""

    # 1. Reuse previous evidence if flagged
    if reuse_evidence:
        prev_ctx = state.get("previous_retrieval_context") or {}
        reused_sources = prev_ctx.get("sources", [])

    # 2. Internal Retrieval
    needs_internal = "internal_retrieval" in actions
    if needs_internal and (not reuse_evidence or additional_retrieval):
        internal_q = dec.get("internal_query") or dec.get("contextualized_query") or state["original_query"]

        # Run Stage 2 Internal Router (Laya)
        try:
            with InternalQueryRouter() as internal_router:
                strategy = internal_router.route(internal_q)
        except Exception as exc:
            logger.warning("Internal router failed, falling back to hybrid: %s", exc)
            strategy = "hybrid"

        if strategy == "graph":
            enable_graph, enable_visual = True, False
        elif strategy == "visual":
            enable_graph, enable_visual = False, True
        else:
            enable_graph, enable_visual = True, True

        # Execute retrieve()
        try:
            from query import retrieve
            from redis_manager import cache_get_json, cache_set_json, retrieval_cache_key

            cache_key = retrieval_cache_key(
                organization=state.get("organization", "default_org"),
                user_id=state.get("user_id", "default_user"),
                role=state.get("role", "member"),
                query=internal_q,
                document_ids=state.get("document_ids", []),
                version_ids=state.get("version_ids", []),
                recursive_search=state.get("recursive_search", False),
                top_k=state.get("top_k", 5),
                enable_graph=enable_graph,
                enable_visual=enable_visual,
            )
            try:
                retrieval_result = cache_get_json(cache_key)
            except Exception:
                retrieval_result = None

            start_t = time.perf_counter()
            if retrieval_result is None:
                retrieval_result = retrieve(
                    query=internal_q,
                    document_ids=state.get("document_ids"),
                    version_ids=state.get("version_ids"),
                    recursive_search=state.get("recursive_search", False),
                    top_k=state.get("top_k", 5),
                    enable_rerank=state.get("enable_rerank", False),
                    rerank_threshold=state.get("rerank_threshold", 0.75),
                    retrieval_top_k=state.get("retrieval_top_k", 20),
                    enable_graph=enable_graph,
                    enable_visual=enable_visual,
                )
                try:
                    cache_set_json(cache_key, retrieval_result)
                except Exception:
                    pass

            r_latency = round((time.perf_counter() - start_t) * 1000, 2)
            results = retrieval_result.get("results", []) if retrieval_result else []
            retrieval_metadata = {
                "strategy": strategy,
                "enable_graph": enable_graph,
                "enable_visual": enable_visual,
                "retrieval_query": internal_q,
                "result_count": len(results),
                "status": "success",
                "latency_ms": r_latency,
                "document_ids": list({r.get("document_id") for r in results if r.get("document_id")}),
                "page_references": [
                    {"document_id": r.get("document_id"), "page_number": r.get("page_number")}
                    for r in results[:10]
                ],
                "reranking": retrieval_result.get("rerank", {}),
            }
        except Exception as exc:
            logger.error("Internal retrieval execution failed: %s", exc, exc_info=True)
            retrieval_metadata = {"status": "error", "error": str(exc), "strategy": strategy}

    # 3. Web Search (Brave)
    if "web_search" in actions:
        web_q = dec.get("web_query") or dec.get("contextualized_query") or state["original_query"]
        client = get_brave_search_client()
        web_resp = client.search(web_q, count=5)
        web_search_result = web_resp.to_dict()
        web_evidence_text = web_resp.format_evidence_text()
        web_metadata = {
            "search_query": web_q,
            "status": web_resp.status,
            "result_count": web_resp.result_count,
            "latency_ms": web_resp.latency_ms,
            "sources": [
                {"title": item.title, "url": item.url}
                for item in web_resp.results
            ],
            "error": web_resp.error,
        }

    return {
        "reused_sources": reused_sources,
        "retrieval_result": retrieval_result,
        "retrieval_metadata": retrieval_metadata,
        "web_search_result": web_search_result,
        "web_metadata": web_metadata,
        "web_evidence_text": web_evidence_text,
    }


def node_aggregate_evidence(state: RAGState) -> dict[str, Any]:
    """Combine reused document evidence, new retrieval evidence, and web search evidence."""
    reused = state.get("reused_sources", [])
    new_retrieval = state.get("retrieval_result", {}) or {}
    new_results = new_retrieval.get("results", []) if isinstance(new_retrieval, dict) else []
    web_res = state.get("web_search_result", {}) or {}
    web_items = web_res.get("results", []) if isinstance(web_res, dict) else []

    aggregated: list[dict[str, Any]] = []
    seen_pages: set[tuple[str, int]] = set()

    # Reused sources first
    for s in reused:
        doc_id = s.get("document_id")
        page_num = s.get("page_number")
        key = (doc_id, page_num)
        if key not in seen_pages:
            seen_pages.add(key)
            aggregated.append(s)

    # New internal retrieval sources
    for r in new_results:
        doc_id = r.get("document_id")
        page_num = r.get("page_number")
        key = (doc_id, page_num)
        if key not in seen_pages:
            seen_pages.add(key)
            aggregated.append(r)

    # Web sources
    for w in web_items:
        aggregated.append({
            "source_type": "web",
            "title": w.get("title"),
            "url": w.get("url"),
            "snippet": w.get("snippet") or w.get("description"),
        })

    return {"aggregated_sources": aggregated}


def node_generate_answer(state: RAGState) -> dict[str, Any]:
    """Execute Answer Generation using the local Qwen model with all combined evidence."""
    from generate import generate_answer

    dec = state.get("routing_decision", {})
    contextualized_q = dec.get("contextualized_query") or state["original_query"]
    direct_answer = dec.get("direct_answer", False)
    retrieval_res = state.get("retrieval_result") or {"results": state.get("aggregated_sources", [])}
    web_evidence = state.get("web_evidence_text", "")
    history = state.get("conversation_history", [])

    response_dict = generate_answer(
        query=contextualized_q,
        retrieval_result=retrieval_res,
        conversation_history=history,
        conversation_id=state.get("conversation_id"),
        web_evidence=web_evidence if web_evidence else None,
        allow_empty_pages=direct_answer or bool(web_evidence),
    )

    answer_text = response_dict.get("answer", "")
    return {
        "answer": answer_text,
        "generation_response": response_dict,
    }


def node_persist_turn(state: RAGState) -> dict[str, Any]:
    """Persist completed turn, sources, and complete routing & retrieval metadata."""
    conv_store = conversation_manager.get_conversation_store()
    cid = state["conversation_id"]
    user_id = state.get("user_id", "default_user")
    org = state.get("organization", "default_org")

    gen_resp = state.get("generation_response") or {}
    gen_meta = gen_resp.get("generation", {})
    sources = state.get("aggregated_sources", []) or gen_resp.get("sources", [])

    try:
        conv_store.add_turn(
            conversation_id=cid,
            user_id=user_id,
            organization=org,
            query=state["original_query"],
            answer=state.get("answer", ""),
            sources=sources,
            generation_metadata=gen_meta,
            routing_metadata=state.get("routing_metadata", {}),
            retrieval_metadata=state.get("retrieval_metadata", {}),
            web_metadata=state.get("web_metadata", {}),
        )
    except Exception as exc:
        logger.warning("Failed to persist conversation turn for %s: %s", cid, exc, exc_info=True)

    return {}


# ===========================================================================
# Conditional Edges
# ===========================================================================

def should_clarify(state: RAGState) -> str:
    """Branch to clarification node if clarification is required."""
    dec = state.get("routing_decision", {})
    if dec.get("clarification_required", False):
        return "clarification"
    return "execute"


# ===========================================================================
# Build & Compile Graph
# ===========================================================================

def build_rag_graph() -> Any:
    """Construct and compile the complete RAG StateGraph workflow."""
    workflow = StateGraph(RAGState)

    workflow.add_node("load_context", node_load_context)
    workflow.add_node("route_query", node_route_query)
    workflow.add_node("clarification", node_clarification)
    workflow.add_node("retrieval_and_search", node_retrieval_and_search)
    workflow.add_node("aggregate_evidence", node_aggregate_evidence)
    workflow.add_node("generate_answer", node_generate_answer)
    workflow.add_node("persist_turn", node_persist_turn)

    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "route_query")
    workflow.add_conditional_edges(
        "route_query",
        should_clarify,
        {
            "clarification": "clarification",
            "execute": "retrieval_and_search",
        },
    )
    workflow.add_edge("clarification", "persist_turn")
    workflow.add_edge("retrieval_and_search", "aggregate_evidence")
    workflow.add_edge("aggregate_evidence", "generate_answer")
    workflow.add_edge("generate_answer", "persist_turn")
    workflow.add_edge("persist_turn", END)

    return workflow.compile()


_COMPILED_RAG_GRAPH = None


def get_rag_graph() -> Any:
    global _COMPILED_RAG_GRAPH
    if _COMPILED_RAG_GRAPH is None:
        _COMPILED_RAG_GRAPH = build_rag_graph()
    return _COMPILED_RAG_GRAPH


def run_rag_pipeline(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Execute the compiled LangGraph pipeline synchronously."""
    graph = get_rag_graph()
    return graph.invoke(initial_state)
