"""
Unit and integration test suite for ExternalQueryRouter and LangGraph RAG Workflow.
Validates all 10 scenarios specified in the task contract:
  Scenario 1: Standalone general question -> Direct answer, no retrieval
  Scenario 2: Internal document question -> Internal retrieval, internal query generated
  Scenario 3: Visual question -> Internal retrieval with visual context
  Scenario 4: Follow-up question -> Conversation context used, query contextualized
  Scenario 5: Previous evidence reuse -> Reuses previous evidence, skips unnecessary retrieval
  Scenario 6: Follow-up requiring additional retrieval -> Context resolved + additional retrieval
  Scenario 7: Current external information -> Brave web search selected, search-optimized query
  Scenario 8: Internal + external comparison -> Internal retrieval + Brave search, dual queries
  Scenario 9: Ambiguous reference without context -> Clarification requested, no invented entities
  Scenario 10: Missing or invalid metadata -> Graceful handling, backward-compatible
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from conversation_manager import ConversationStore, ConversationTurn
from external_query_router import (
    ExternalQueryRouter,
    ExternalRouteDecision,
    ExternalRouterInput,
    fallback_heuristic_routing,
)
from web_search import BraveSearchClient, WebSearchResponse, WebSearchResultItem


# ===========================================================================
# Scenario 1: Standalone General Question
# ===========================================================================

def test_scenario_1_standalone_general_question():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Explain Python decorators and how to write one.",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert decision.direct_answer is True
    assert "internal_retrieval" not in decision.actions
    assert "web_search" not in decision.actions
    assert decision.contextualized_query == "Explain Python decorators and how to write one."
    assert decision.clarification_required is False


# ===========================================================================
# Scenario 2: Internal Document Question
# ===========================================================================

def test_scenario_2_internal_document_question():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Explain our AMG ingestion pipeline and chunking.",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert "internal_retrieval" in decision.actions
    assert decision.internal_query is not None
    assert "amg ingestion pipeline" in decision.internal_query.lower()
    assert decision.direct_answer is False


# ===========================================================================
# Scenario 3: Visual Question
# ===========================================================================

def test_scenario_3_visual_question():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Explain the architecture diagram in our PDF on page 4.",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert "internal_retrieval" in decision.actions
    assert decision.internal_query is not None
    assert "diagram" in decision.internal_query.lower() or "architecture" in decision.internal_query.lower()


# ===========================================================================
# Scenario 4: Follow-up Question with Contextualization
# ===========================================================================

def test_scenario_4_follow_up_question():
    router = ExternalQueryRouter()
    history = [
        {
            "turn_id": 1,
            "query": "Explain ColPali multi-vector retrieval.",
            "answer": "ColPali produces patch embeddings for late interaction scoring.",
        }
    ]
    inp = ExternalRouterInput(
        query="How does it calculate similarity?",
        conversation_history=history,
    )
    decision, meta = router.route(inp)
    assert decision.use_conversation_context is True
    assert "ColPali" in decision.contextualized_query or "previous query" in decision.contextualized_query
    assert "internal_retrieval" in decision.actions
    assert "conversation_search" not in decision.actions


# ===========================================================================
# Scenario 5: Previous Evidence Reuse
# ===========================================================================

def test_scenario_5_previous_evidence_reuse():
    router = ExternalQueryRouter()
    history = [
        {
            "turn_id": 1,
            "query": "Explain our AMG architecture.",
            "answer": "AMG combines ColPali visual embeddings with Neo4j knowledge graphs.",
        }
    ]
    prev_ctx = {
        "has_previous_evidence": True,
        "result_count": 3,
        "page_references": [{"document_id": "amg_arch", "page_number": 1}],
        "sources": [{"document_id": "amg_arch", "page_number": 1}],
    }
    inp = ExternalRouterInput(
        query="Explain the architecture again as mentioned.",
        conversation_history=history,
        previous_retrieval_context=prev_ctx,
    )
    decision, meta = router.route(inp)
    assert decision.reuse_previous_evidence is True
    assert decision.additional_retrieval_required is False
    assert "internal_retrieval" not in decision.actions


# ===========================================================================
# Scenario 6: Follow-up Requiring Additional Retrieval
# ===========================================================================

def test_scenario_6_follow_up_additional_retrieval():
    router = ExternalQueryRouter()
    history = [
        {
            "turn_id": 1,
            "query": "Explain our AMG architecture.",
            "answer": "AMG combines ColPali visual embeddings with Neo4j knowledge graphs.",
        }
    ]
    prev_ctx = {
        "has_previous_evidence": True,
        "result_count": 2,
        "sources": [{"document_id": "amg_arch", "page_number": 1}],
    }
    inp = ExternalRouterInput(
        query="What are its latency bottlenecks in detail?",
        conversation_history=history,
        previous_retrieval_context=prev_ctx,
    )
    decision, meta = router.route(inp)
    assert decision.use_conversation_context is True
    assert decision.reuse_previous_evidence is True
    assert decision.additional_retrieval_required is True
    assert "internal_retrieval" in decision.actions


# ===========================================================================
# Scenario 7: Current External Information (Web Search)
# ===========================================================================

def test_scenario_7_current_external_information():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="What are the latest developments and recent benchmarks in multimodal RAG in 2026?",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert "web_search" in decision.actions
    assert decision.web_query is not None
    assert "multimodal rag" in decision.web_query.lower()


# ===========================================================================
# Scenario 8: Internal + External Comparison
# ===========================================================================

def test_scenario_8_internal_external_comparison():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Compare our AMG architecture with the latest industry multimodal RAG approaches.",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert "internal_retrieval" in decision.actions
    assert "web_search" in decision.actions
    assert decision.internal_query is not None
    assert decision.web_query is not None


# ===========================================================================
# Scenario 9: Ambiguous Reference without Context
# ===========================================================================

def test_scenario_9_ambiguous_reference():
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Compare those two.",
        conversation_history=[],
    )
    decision, meta = router.route(inp)
    assert decision.clarification_required is True
    assert decision.clarification_question is not None
    assert len(decision.actions) == 0


# ===========================================================================
# Scenario 10: Missing or Invalid Metadata & Backward Compatibility
# ===========================================================================

def test_scenario_10_missing_or_invalid_metadata(tmp_path: Path):
    db_path = tmp_path / "legacy_conv.db"
    store = ConversationStore(db_path=db_path)

    # Historical turn added with minimal columns
    conv, _ = store.get_or_create_conversation(
        conversation_id="conv_legacy_001",
        user_id="user_test",
        organization="org_test",
    )
    turn = store.add_turn(
        conversation_id="conv_legacy_001",
        user_id="user_test",
        organization="org_test",
        query="What is RAG?",
        answer="Retrieval Augmented Generation.",
    )
    assert turn.routing_metadata == {}
    assert turn.retrieval_metadata == {}
    assert turn.web_metadata == {}

    # Read back legacy conversation safely
    loaded = store.get_conversation("conv_legacy_001")
    assert loaded is not None
    assert len(loaded.turns) == 1
    assert loaded.turns[0].routing_metadata == {}

    # Router handles None/missing retrieval context safely
    router = ExternalQueryRouter()
    inp = ExternalRouterInput(
        query="Explain our system.",
        conversation_history=[],
        previous_retrieval_context=None,
        previous_execution_metadata=None,
    )
    decision, meta = router.route(inp)
    assert decision is not None
    assert "internal_retrieval" in decision.actions


# ===========================================================================
# Brave Search Client Mock Test
# ===========================================================================

def test_brave_search_client_mock():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "ColPali: Efficient Document Retrieval",
                    "url": "https://arxiv.org/abs/2407.01449",
                    "description": "ColPali leverages Vision Language Models for document retrieval.",
                    "extra_snippets": ["ColPali uses ColBERT late interaction over patch vectors."],
                }
            ]
        }
    }

    client = BraveSearchClient(api_key="test_key")
    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.search("ColPali multimodal retrieval")
        assert res.status == "success"
        assert res.result_count == 1
        assert res.results[0].title == "ColPali: Efficient Document Retrieval"
        evidence_text = res.format_evidence_text()
        assert "[Web 1] Title: ColPali" in evidence_text
        assert "https://arxiv.org/abs/2407.01449" in evidence_text
    client.close()


# ===========================================================================
# LangGraph Workflow Execution Test
# ===========================================================================

def test_langgraph_workflow_clarification(tmp_path: Path, monkeypatch):
    from rag_workflow import run_rag_pipeline
    import conversation_manager

    db_path = tmp_path / "test_workflow.db"
    store = ConversationStore(db_path=db_path)
    monkeypatch.setattr(conversation_manager, "get_conversation_store", lambda: store)

    initial_state = {
        "user_id": "user_demo",
        "organization": "org_demo",
        "role": "admin",
        "original_query": "Compare those two.",
        "conversation_id": None,
    }

    final_state = run_rag_pipeline(initial_state)
    assert final_state.get("conversation_id") is not None
    assert "Could you please specify" in final_state.get("answer", "") or "clarify" in final_state.get("answer", "").lower()
    assert final_state.get("routing_metadata", {}).get("clarification_required") is True

    # Verify turn was persisted
    cid = final_state["conversation_id"]
    turns = store.get_conversation_turns(cid)
    assert len(turns) == 1
    assert turns[0].routing_metadata.get("clarification_required") is True


# ===========================================================================
# Explicit Tool Calling Exclusion Tests
# ===========================================================================

def test_tool_calling_explicitly_excluded():
    """Verify that Tool Calling is excluded from ExternalRouteDecision schema and actions."""
    from pydantic import ValidationError

    # 1. ExternalRouteDecision must NOT have tool calling fields
    decision = ExternalRouteDecision(
        actions=["internal_retrieval"],
        contextualized_query="Explain the system",
        reason="Internal query",
    )
    assert not hasattr(decision, "tool_calls")
    assert not hasattr(decision, "tools")

    # 2. Schema must reject tool_call or tool_calling in actions
    with pytest.raises(ValidationError):
        ExternalRouteDecision(
            actions=["tool_call"],  # type: ignore
            contextualized_query="Run calculator",
            reason="Tool invocation",
        )

    # 3. Permitted actions must strictly be conversation_search, internal_retrieval, web_search
    valid_decision = ExternalRouteDecision(
        actions=["conversation_search", "internal_retrieval", "web_search"],
        contextualized_query="Comprehensive research",
        reason="Multi-source query",
    )
    assert len(valid_decision.actions) == 3

