"""
Integration and API tests for conversation_id lifecycle, context preservation,
and conversation management endpoints.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

if "colpali_engine" not in sys.modules:
    sys.modules["colpali_engine"] = MagicMock()
    sys.modules["colpali_engine.models"] = MagicMock()

import pytest
from fastapi.testclient import TestClient

import api
from authorization import AuthContext, get_current_user
from conversation_manager import ConversationStore


@pytest.fixture
def test_app(tmp_path: Path):
    # Isolated conversations DB for test run
    db_file = tmp_path / "test_api_conv.db"
    store = ConversationStore(db_path=db_file)

    with patch("api.get_conversation_store", return_value=store), \
         patch("conversation_manager.get_conversation_store", return_value=store):
        yield store


def test_conversation_api_end_to_end(test_app: ConversationStore):
    client = TestClient(api.app)

    user1 = AuthContext(
        user_id="user_alice",
        name="Alice",
        email="alice@example.com",
        organization_id="org_wonderland",
        organization_name="Wonderland",
        role="engineer",
        status="active",
    )
    api.app.dependency_overrides[get_current_user] = lambda: user1

    mock_retrieval = {
        "query": "What is the capital of France?",
        "results": [
            {
                "rank": 1,
                "score": 0.92,
                "document_id": "doc1",
                "version_id": "v1",
                "version_number": 1,
                "filename": "geography.pdf",
                "page_number": 3,
                "page_hash": "hash123",
                "page_image_path": "data/images/geo_p3.png",
                "page_text": "Paris is the capital of France.",
            }
        ],
        "scope": {"mode": "current_only"},
        "result_count": 1,
    }

    # Turn 1: POST /answer without conversation_id
    with patch("query.retrieve", return_value=mock_retrieval), \
         patch("api.retrieve", return_value=mock_retrieval), \
         patch("api.cache_get_json", return_value=None), \
         patch("api.cache_set_json"), \
         patch("generate.generate_answer") as mock_gen, \
         patch("api.generate_answer", new=mock_gen):

        mock_gen.return_value = {
            "answer": "The capital of France is Paris [S1].",
            "sources": [{"source_id": "S1", "page_number": 3}],
            "generation": {"model_id": "test_model"},
        }

        payload = {
            "user_id": "user_alice",
            "organization": "Wonderland",
            "role": "engineer",
            "query": "What is the capital of France?",
        }
        res1 = client.post("/answer", json=payload)
        assert res1.status_code == 200, res1.text
        data1 = res1.json()
        assert "conversation_id" in data1
        cid = data1["conversation_id"]
        assert cid.startswith("conv_")
        assert data1["answer"] == "The capital of France is Paris [S1]."

    # Verify conversation store recorded Turn 1
    turns = test_app.get_conversation_turns(cid)
    assert len(turns) == 1
    assert turns[0].query == "What is the capital of France?"
    assert turns[0].answer == "The capital of France is Paris [S1]."

    # Turn 2: Follow-up question with conversation_id passed
    with patch("query.retrieve", return_value=mock_retrieval), \
         patch("api.retrieve", return_value=mock_retrieval), \
         patch("api.cache_get_json", return_value=None), \
         patch("api.cache_set_json"), \
         patch("generate.generate_answer") as mock_gen2, \
         patch("api.generate_answer", new=mock_gen2):

        mock_gen2.return_value = {
            "answer": "Its population is approximately 2.1 million.",
            "sources": [{"source_id": "S1", "page_number": 4}],
            "generation": {"model_id": "test_model"},
        }

        payload2 = {
            "user_id": "user_alice",
            "organization": "Wonderland",
            "role": "engineer",
            "query": "What is its population?",
            "conversation_id": cid,
        }
        res2 = client.post("/answer", json=payload2)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["conversation_id"] == cid

        # Verify that mock_gen2 received conversation_history!
        call_kwargs = mock_gen2.call_args[1]
        assert "conversation_history" in call_kwargs
        conv_hist = call_kwargs["conversation_history"]
        assert len(conv_hist) == 1
        first_turn_query = conv_hist[0].get("query") if isinstance(conv_hist[0], dict) else getattr(conv_hist[0], "query", None)
        assert first_turn_query == "What is the capital of France?"

    # Verify conversation store now has 2 turns
    conv_detail = test_app.get_conversation(cid)
    assert conv_detail is not None
    assert len(conv_detail.turns) == 2
    assert conv_detail.turns[0].query == "What is the capital of France?"
    assert conv_detail.turns[1].query == "What is its population?"

    # Turn 3: Streaming generation with conversation_id
    with patch("query.retrieve", return_value=mock_retrieval), \
         patch("api.retrieve", return_value=mock_retrieval), \
         patch("api.cache_get_json", return_value=None), \
         patch("api.cache_set_json"), \
         patch("generate.generate_answer_stream") as mock_stream, \
         patch("api.generate_answer_stream", new=mock_stream):

        def fake_stream(**kwargs):
            # Verify stream receives conversation_history
            assert "conversation_history" in kwargs
            assert len(kwargs["conversation_history"]) == 2
            yield {"type": "token", "token": "It "}
            yield {"type": "token", "token": "is famous for the Eiffel Tower."}
            yield {
                "type": "complete",
                "data": {
                    "answer": "It is famous for the Eiffel Tower.",
                    "sources": [{"source_id": "S1", "page_number": 5}],
                    "generation": {"model_id": "test_model"},
                    "latency_ms": 12.0,
                    "ttft_ms": 2.0,
                },
            }

        mock_stream.side_effect = fake_stream

        payload3 = {
            "user_id": "user_alice",
            "organization": "Wonderland",
            "role": "engineer",
            "query": "What landmarks is it famous for?",
            "conversation_id": cid,
        }
        res3 = client.post("/answer/stream", json=payload3)
        assert res3.status_code == 200
        lines = [line.strip() for line in res3.text.split("\n") if line.strip()]
        complete_line = [l for l in lines if l.startswith("data:") and "Eiffel Tower" in l][-1]
        complete_json = json.loads(complete_line[5:].strip())
        assert complete_json["conversation_id"] == cid
        assert complete_json["answer"] == "It is famous for the Eiffel Tower."

    # Turn 3 should now be stored
    turns = test_app.get_conversation_turns(cid)
    assert len(turns) == 3

    # Test GET /conversations
    res_list = client.get("/conversations")
    assert res_list.status_code == 200
    list_data = res_list.json()
    assert list_data["total"] == 1
    assert list_data["conversations"][0]["conversation_id"] == cid
    assert list_data["conversations"][0]["turn_count"] == 3

    # Test GET /conversations/{conversation_id}
    res_get = client.get(f"/conversations/{cid}")
    assert res_get.status_code == 200
    detail_data = res_get.json()
    assert detail_data["conversation_id"] == cid
    assert len(detail_data["turns"]) == 3

    # Test Cross-Tenant isolation: User Bob cannot view or delete Alice's conversation
    user_bob = AuthContext(
        user_id="user_bob",
        name="Bob",
        email="bob@example.com",
        organization_id="org_other",
        organization_name="OtherCorp",
        role="engineer",
        status="active",
    )
    api.app.dependency_overrides[get_current_user] = lambda: user_bob

    res_bob_get = client.get(f"/conversations/{cid}")
    assert res_bob_get.status_code == 404

    res_bob_del = client.delete(f"/conversations/{cid}")
    assert res_bob_del.status_code == 404

    # Switch back to Alice and delete
    api.app.dependency_overrides[get_current_user] = lambda: user1
    res_del = client.delete(f"/conversations/{cid}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # Confirm deletion
    res_after = client.get(f"/conversations/{cid}")
    assert res_after.status_code == 404

    # Verify legacy /generate aliases also work
    with patch("query.retrieve", return_value=mock_retrieval), \
         patch("generate.generate_answer") as mock_gen_alias:
        mock_gen_alias.return_value = {
            "answer": "Paris is in France.",
            "sources": [],
            "generation": {},
        }
        res_alias = client.post("/generate", json=payload)
        assert res_alias.status_code == 200

    # Verify decommissioned /query returns 404 Not Found
    res_query = client.post("/query", json=payload)
    assert res_query.status_code == 404
