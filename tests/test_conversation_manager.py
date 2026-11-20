"""
Unit tests for conversation_manager.py verifying persistence, turn logging,
tenant isolation, history formatting, and CRUD operations.
"""

from pathlib import Path
import pytest

from conversation_manager import (
    ConversationStore,
    ConversationTurn,
    format_conversation_history,
    new_conversation_id,
    validate_conversation_id,
)


def test_id_generation_and_validation():
    cid = new_conversation_id()
    assert cid.startswith("conv_")
    assert validate_conversation_id(cid) == cid

    with pytest.raises(ValueError):
        validate_conversation_id("")

    with pytest.raises(ValueError):
        validate_conversation_id("invalid spaces here!")


def test_conversation_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_conversations.db"
    store = ConversationStore(db_path=db_file)

    user_id = "user_123"
    org = "TestOrg"

    # 1. Create / get new conversation
    conv, created = store.get_or_create_conversation(
        conversation_id=None,
        user_id=user_id,
        organization=org,
    )
    assert created is True
    assert conv.conversation_id.startswith("conv_")
    assert conv.user_id == user_id
    assert conv.organization == org
    assert conv.turns == []

    cid = conv.conversation_id

    # 2. Add Turn 1
    t1 = store.add_turn(
        conversation_id=cid,
        user_id=user_id,
        organization=org,
        query="What is the total revenue in 2023?",
        answer="The total revenue in 2023 was $10M according to page 4.",
        sources=[{"source_id": "S1", "page_number": 4}],
        generation_metadata={"model": "Qwen3-VL"},
    )
    assert t1.turn_id == 1
    assert t1.query == "What is the total revenue in 2023?"
    assert len(t1.sources) == 1

    # Check conversation title updated from first question
    c_updated = store.get_conversation(cid, user_id=user_id, organization=org)
    assert c_updated is not None
    assert c_updated.title == "What is the total revenue in 2023?"
    assert len(c_updated.turns) == 1

    # 3. Add Turn 2 (Follow-up)
    t2 = store.add_turn(
        conversation_id=cid,
        user_id=user_id,
        organization=org,
        query="And what was the profit margin?",
        answer="The profit margin was 18%.",
        sources=[{"source_id": "S1", "page_number": 5}],
    )
    assert t2.turn_id == 2

    # 4. Fetch turns & history
    turns = store.get_conversation_turns(cid)
    assert len(turns) == 2
    assert turns[0].query == "What is the total revenue in 2023?"
    assert turns[1].query == "And what was the profit margin?"

    history_str = format_conversation_history(turns)
    assert "[Turn 1]" in history_str
    assert "User: What is the total revenue in 2023?" in history_str
    assert "Assistant: The total revenue in 2023 was $10M" in history_str
    assert "[Turn 2]" in history_str
    assert "User: And what was the profit margin?" in history_str

    # 5. List conversations
    conv_list = store.list_conversations(user_id=user_id, organization=org)
    assert len(conv_list) == 1
    assert conv_list[0]["conversation_id"] == cid
    assert conv_list[0]["turn_count"] == 2

    # 6. Tenant isolation: different user/org cannot access
    other_conv = store.get_conversation(cid, user_id="intruder", organization=org)
    assert other_conv is None

    other_org_conv = store.get_conversation(cid, user_id=user_id, organization="OtherOrg")
    assert other_org_conv is None

    # 7. Delete conversation
    assert store.delete_conversation(cid, user_id=user_id, organization=org) is True
    assert store.get_conversation(cid) is None
    assert store.get_conversation_turns(cid) == []
    assert store.list_conversations(user_id=user_id, organization=org) == []


def test_conversation_redis_caching_and_ttl(tmp_path: Path, monkeypatch):
    """
    Verify that conversations are cached in Redis with a 60-minute (3600s) TTL,
    refreshed on new turns, served from cache on hit, and evicted on deletion.
    """
    from unittest.mock import MagicMock
    import redis_manager

    fake_cache = {}
    mock_get = MagicMock(side_effect=lambda key: fake_cache.get(key))
    
    def fake_set(key, val, ttl_seconds=3600):
        fake_cache[key] = (val, ttl_seconds)
        return True

    mock_set = MagicMock(side_effect=fake_set)
    mock_del = MagicMock(side_effect=lambda key: fake_cache.pop(key, None))

    monkeypatch.setattr(redis_manager, "cache_get_json", mock_get)
    monkeypatch.setattr(redis_manager, "cache_set_json", mock_set)
    monkeypatch.setattr(redis_manager, "cache_delete", mock_del)

    db_file = tmp_path / "test_redis_conv.db"
    store = ConversationStore(db_path=db_file)

    # 1. Create conversation -> sets cache with 3600s (60min) TTL
    conv, _ = store.get_or_create_conversation(
        conversation_id="conv_test_redis",
        user_id="user_1",
        organization="org_1",
    )
    cache_key = redis_manager.conversation_cache_key("conv_test_redis")
    assert cache_key == "amg:conv:conv_test_redis"
    assert cache_key in fake_cache
    cached_val, ttl = fake_cache[cache_key]
    assert ttl == 3600  # 60 minutes
    assert cached_val["conversation_id"] == "conv_test_redis"

    # 2. Add turn -> cache renewed with 3600s TTL
    store.add_turn(
        conversation_id="conv_test_redis",
        user_id="user_1",
        organization="org_1",
        query="What is multimodal RAG?",
        answer="Multimodal RAG combines text and visual representations.",
    )
    cached_val, ttl = fake_cache[cache_key]
    assert ttl == 3600  # Renewed 60-minute TTL
    assert len(cached_val["turns"]) == 1
    assert cached_val["turns"][0]["query"] == "What is multimodal RAG?"

    # 3. Retrieve -> hits cache
    retrieved = store.get_conversation("conv_test_redis", user_id="user_1", organization="org_1")
    assert retrieved is not None
    assert retrieved.turns[0].answer == "Multimodal RAG combines text and visual representations."
    assert mock_get.called

    # 4. Delete -> removes from Redis cache
    store.delete_conversation("conv_test_redis", user_id="user_1", organization="org_1")
    assert cache_key not in fake_cache
    assert mock_del.called

