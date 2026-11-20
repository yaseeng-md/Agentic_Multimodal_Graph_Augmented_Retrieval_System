"""
Unit test suite for query_router.py covering initialization,
input validation, routing heuristics, error handling, and session lifecycle.
"""

import sys
from unittest.mock import MagicMock, patch

if "colpali_engine" not in sys.modules:
    sys.modules["colpali_engine"] = MagicMock()
    sys.modules["colpali_engine.models"] = MagicMock()

import pytest
import requests

from external_query_router import InternalQueryRouter, InternalRetrievalDecision


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("LAYA_ACCESS_TOKEN", "fake_test_token")
    monkeypatch.setenv("LAYA_MODEL_END_POINT", "https://api.laya.example.com/v1/decide")
    monkeypatch.setenv("LAYA_MODEL", "convai-innovations/laya")
    monkeypatch.setenv("LAYA_CHECKPOINT", "typed-decisions")


def test_init_missing_env_vars(monkeypatch):
    monkeypatch.delenv("LAYA_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("LAYA_MODEL_END_POINT", "https://example.com")
    monkeypatch.setenv("LAYA_MODEL", "test-model")

    with pytest.raises(ValueError, match="LAYA_ACCESS_TOKEN is missing"):
        InternalQueryRouter()

    monkeypatch.setenv("LAYA_ACCESS_TOKEN", "token")
    monkeypatch.delenv("LAYA_MODEL_END_POINT", raising=False)
    with pytest.raises(ValueError, match="LAYA_MODEL_END_POINT is missing"):
        InternalQueryRouter()

    monkeypatch.setenv("LAYA_MODEL_END_POINT", "https://example.com")
    monkeypatch.delenv("LAYA_MODEL", raising=False)
    with pytest.raises(ValueError, match="LAYA_MODEL is missing"):
        InternalQueryRouter()


def test_route_empty_query(mock_env):
    router = InternalQueryRouter()
    with pytest.raises(ValueError, match="Query cannot be empty"):
        router.route("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        router.route("   \n\t ")
    router.close()


def test_decide_internal_router_visual(mock_env):
    router = InternalQueryRouter()

    decision = InternalRetrievalDecision(
        strategy="visual",
        probabilities={"visual": 0.58, "graph": 0.22, "hybrid": 0.20},
        confidence=0.12,
        model="convai-innovations/laya",
    )
    assert router.decide_internal_router(decision) == "visual"
    router.close()


def test_decide_internal_router_graph(mock_env):
    router = InternalQueryRouter()

    decision = InternalRetrievalDecision(
        strategy="graph",
        probabilities={"visual": 0.20, "graph": 0.55, "hybrid": 0.25},
        confidence=0.08,
        model="convai-innovations/laya",
    )
    assert router.decide_internal_router(decision) == "graph"
    router.close()


def test_decide_internal_router_ambiguous_defaults_to_hybrid(mock_env):
    router = InternalQueryRouter()

    # Close probabilities with no strong leader
    decision = InternalRetrievalDecision(
        strategy="visual",
        probabilities={"visual": 0.36, "graph": 0.33, "hybrid": 0.31},
        confidence=0.01,
        model="convai-innovations/laya",
    )
    assert router.decide_internal_router(decision) == "hybrid"
    router.close()


def test_decide_internal_router_explicit_hybrid(mock_env):
    router = InternalQueryRouter()

    decision = InternalRetrievalDecision(
        strategy="hybrid",
        probabilities={"visual": 0.30, "graph": 0.30, "hybrid": 0.40},
        confidence=0.05,
        model="convai-innovations/laya",
    )
    assert router.decide_internal_router(decision) == "hybrid"
    router.close()


def test_route_successful_call(mock_env):
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "convai-innovations/laya",
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "visual",
                "probabilities": {"visual": 0.59, "graph": 0.21, "hybrid": 0.20},
                "confidence": 0.14,
            }
        },
        "latency_ms": 120.5,
    }

    with patch.object(router.session, "post", return_value=mock_response) as mock_post:
        result = router.route("Show me the diagram on page 3")
        assert result == "visual"
        assert mock_post.called
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["state"] == "Show me the diagram on page 3"
        assert sent_payload["model"] == "convai-innovations/laya"
        assert sent_payload["checkpoint"] == "typed-decisions"

    router.close()


def test_route_http_error(mock_env):
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": "Invalid token"}
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )

    with patch.object(router.session, "post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Laya API returned HTTP 401"):
            router.route("Any query")

    router.close()


def test_route_timeout(mock_env):
    router = InternalQueryRouter()

    with patch.object(
        router.session, "post", side_effect=requests.exceptions.Timeout("Timed out")
    ):
        with pytest.raises(RuntimeError, match="Laya API request timed out"):
            router.route("Any query")

    router.close()


def test_route_invalid_json(mock_env):
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("No JSON object could be decoded")

    with patch.object(router.session, "post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Laya returned an invalid JSON response"):
            router.route("Any query")

    router.close()


def test_route_unsupported_strategy(mock_env):
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answers": {
            "retrieval_strategy": {
                "choice": "unknown_strategy",
                "probabilities": {"unknown_strategy": 1.0},
            }
        }
    }

    with patch.object(router.session, "post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Laya returned an unsupported strategy"):
            router.route("Any query")

    router.close()


def test_context_manager(mock_env):
    with InternalQueryRouter() as router:
        assert router.session is not None


# ===========================================================================
# Redis Caching Tests for InternalQueryRouter
# ===========================================================================

def test_redis_cache_hit_avoids_laya_invocation(mock_env):
    """A cache hit in Redis must return the cached strategy immediately without calling Laya."""
    router = InternalQueryRouter()

    with patch("redis_manager.cache_get_json", return_value="graph") as mock_cache_get, \
         patch.object(router.session, "post") as mock_post:

        result = router.route("Show all entity relationships in the architecture")

        assert result == "graph"
        assert mock_cache_get.called
        assert not mock_post.called

    router.close()


def test_redis_cache_miss_invokes_laya_and_caches_decision(mock_env):
    """On a cache miss, Laya is invoked, response is validated, and result is cached in Redis."""
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "convai-innovations/laya",
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "visual",
                "probabilities": {"visual": 0.70, "graph": 0.15, "hybrid": 0.15},
                "confidence": 0.85,
            }
        },
        "latency_ms": 95.0,
    }

    with patch("redis_manager.cache_get_json", return_value=None), \
         patch("redis_manager.cache_set_json") as mock_cache_set, \
         patch.object(router.session, "post", return_value=mock_response) as mock_post:

        result = router.route("Show me the chart on page 4")

        assert result == "visual"
        assert mock_post.called
        assert mock_cache_set.called

        # Validate arguments to cache_set_json
        args, kwargs = mock_cache_set.call_args
        cache_key, cached_val = args[0], args[1]
        assert "router_decision" in cache_key
        assert cached_val == "visual"

    router.close()


def test_redis_failure_does_not_break_routing(mock_env):
    """If Redis fails (e.g. connection refused), router continues directly with Laya."""
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "convai-innovations/laya",
        "answers": {
            "retrieval_strategy": {
                "type": "choice",
                "choice": "hybrid",
                "probabilities": {"visual": 0.35, "graph": 0.35, "hybrid": 0.30},
                "confidence": 0.1,
            }
        },
    }

    with patch("redis_manager.cache_get_json", side_effect=Exception("Redis connection refused")), \
         patch("redis_manager.cache_set_json", side_effect=Exception("Redis connection refused")), \
         patch.object(router.session, "post", return_value=mock_response) as mock_post:

        result = router.route("Explain system pipeline and flow")

        assert result == "hybrid"
        assert mock_post.called

    router.close()


def test_invalid_decisions_are_not_cached(mock_env):
    """When Laya returns an HTTP error, invalid JSON, or unsupported strategy, nothing is cached."""
    router = InternalQueryRouter()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error", response=mock_response)

    with patch("redis_manager.cache_get_json", return_value=None), \
         patch("redis_manager.cache_set_json") as mock_cache_set, \
         patch.object(router.session, "post", return_value=mock_response):

        with pytest.raises(RuntimeError):
            router.route("Failing query")

        assert not mock_cache_set.called

    router.close()


# ===========================================================================
# Graph-to-Qdrant Fallback Tests (query.py retrieve)
# ===========================================================================

def test_graph_retrieval_with_results_does_not_trigger_fallback():
    """When Graph retrieval yields usable pages, fallback to Qdrant is NOT triggered."""
    from query import retrieve

    mock_graph_res = MagicMock()
    mock_graph_res.status = "success"
    mock_graph_res.matched_entities = ["AMG", "Pipeline"]
    mock_graph_res.claims = []
    mock_graph_res.source_pages = [
        {"document_id": "doc_arch_1", "page_number": 1, "score": 0.95},
        {"document_id": "doc_arch_1", "page_number": 2, "score": 0.88},
    ]
    mock_graph_res.duration_ms = 45.0
    mock_graph_res.error = None

    with patch("graph.retriever.GraphRetriever.retrieve", return_value=mock_graph_res), \
         patch("query.create_qdrant_client") as mock_qdrant:

        res = retrieve(
            query="Explain the AMG architecture entities",
            enable_graph=True,
            enable_visual=False,
            top_k=5,
        )

        assert res["retrieval_strategy"] == "graph"
        assert res["final_strategy"] == "graph"
        assert res["fallback"]["triggered"] is False
        assert res["result_count"] == 2
        assert len(res["results"]) == 2
        assert not mock_qdrant.called


def test_empty_graph_retrieval_triggers_qdrant_fallback():
    """When Graph retrieval produces 0 usable pages, visual retrieval (Qdrant) is triggered as fallback."""
    from query import retrieve

    mock_graph_res = MagicMock()
    mock_graph_res.status = "success"
    mock_graph_res.matched_entities = []
    mock_graph_res.claims = []
    mock_graph_res.source_pages = []  # Empty!
    mock_graph_res.duration_ms = 12.0
    mock_graph_res.error = None

    mock_visual_candidates = [
        {
            "rank": 1,
            "score": 0.91,
            "document_id": "doc_fallback_1",
            "version_id": "v1",
            "version_number": 1,
            "page_number": 3,
            "page_text": "Fallback visual page evidence",
            "page_image_path": "data/images/p3.png",
        }
    ]

    with patch("graph.retriever.GraphRetriever.retrieve", return_value=mock_graph_res), \
         patch("query.get_query_encoder") as mock_encoder, \
         patch("query.create_qdrant_client") as mock_client, \
         patch("query._check_collection"), \
         patch("query._build_qdrant_filter"), \
         patch("query._serialise_point", return_value=mock_visual_candidates[0]):

        mock_encoder.return_value.encode.return_value = [[0.1] * 128]
        mock_response = MagicMock()
        mock_response.points = [MagicMock()]
        mock_client.return_value.query_points.return_value = mock_response

        res = retrieve(
            query="What is the architecture?",
            enable_graph=True,
            enable_visual=False,
            top_k=5,
        )

        assert res["retrieval_strategy"] == "graph"
        assert res["final_strategy"] == "visual"
        assert res["fallback"]["triggered"] is True
        assert res["fallback"]["from_strategy"] == "graph"
        assert res["fallback"]["to_strategy"] == "visual"
        assert res["result_count"] == 1
        assert res["results"][0]["document_id"] == "doc_fallback_1"


def test_visual_and_hybrid_strategies_unaffected_by_fallback():
    """Visual-only and Hybrid strategies must operate normally without triggering fallback."""
    from query import retrieve

    mock_visual_candidate = {
        "rank": 1,
        "score": 0.88,
        "document_id": "doc_visual_1",
        "page_number": 1,
    }

    # 1. Visual-only test
    with patch("query.get_query_encoder") as mock_enc, \
         patch("query.create_qdrant_client") as mock_cli, \
         patch("query._check_collection"), \
         patch("query._build_qdrant_filter"), \
         patch("query._serialise_point", return_value=mock_visual_candidate):

        mock_enc.return_value.encode.return_value = [[0.1] * 128]
        mock_resp = MagicMock()
        mock_resp.points = [MagicMock()]
        mock_cli.return_value.query_points.return_value = mock_resp

        res_vis = retrieve(
            query="Show visual page",
            enable_graph=False,
            enable_visual=True,
            top_k=5,
        )
        assert res_vis["retrieval_strategy"] == "visual"
        assert res_vis["final_strategy"] == "visual"
        assert res_vis["fallback"]["triggered"] is False

