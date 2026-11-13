"""
AMG Multimodal RAG — Conversation History & Context Manager

Manages persistent multi-turn conversations, turns, and context assembly
for multi-turn Q&A and follow-up query resolution.

Storage:
    - Primary: SQLite database (data/conversations.db)
    - Fast Cache: Redis caching for active conversation turns (when Redis is available)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# Base storage paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DEFAULT_CONVERSATIONS_DB_PATH = DATA_DIR / "conversations.db"
CONVERSATIONS_DB_PATH = Path(
    os.getenv("CONVERSATIONS_DB_PATH", str(DEFAULT_CONVERSATIONS_DB_PATH))
)

CONVERSATION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")
DEFAULT_MAX_CONTEXT_TURNS = int(os.getenv("CONVERSATION_MAX_CONTEXT_TURNS", "5"))
DEFAULT_CONVERSATION_TTL_SECONDS = int(
    os.getenv("REDIS_CONVERSATION_TTL", "3600")
)  # 60 minutes (3600 seconds)


def utc_now() -> str:
    """Current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def new_conversation_id() -> str:
    """Generate a readable, timestamped, unique conversation ID."""
    return f"conv_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"


def validate_conversation_id(conversation_id: str) -> str:
    """Validate and clean conversation ID."""
    val = (conversation_id or "").strip()
    if not val:
        raise ValueError("conversation_id cannot be empty")
    if not CONVERSATION_ID_RE.match(val):
        raise ValueError(
            f"Invalid conversation_id '{val}'. Allowed characters: alphanumeric, _, -, ."
        )
    return val


# ===========================================================================
# Redis Caching Helpers (60-minute TTL)
# ===========================================================================

def _conv_cache_key(conversation_id: str) -> str:
    try:
        from redis_manager import conversation_cache_key
        return conversation_cache_key(conversation_id)
    except Exception:
        return f"amg:conv:{conversation_id.strip()}"


def _cache_get_conversation(conversation_id: str) -> Optional[dict[str, Any]]:
    """Fetch cached conversation JSON from Redis; returns None on miss or error."""
    try:
        from redis_manager import cache_get_json
        return cache_get_json(_conv_cache_key(conversation_id))
    except Exception:
        return None


def _cache_set_conversation(
    conversation_id: str,
    data: dict[str, Any],
    ttl_seconds: int = DEFAULT_CONVERSATION_TTL_SECONDS,
) -> None:
    """Cache conversation context in Redis with 60-minute TTL."""
    try:
        from redis_manager import cache_set_json
        cache_set_json(_conv_cache_key(conversation_id), data, ttl_seconds=ttl_seconds)
    except Exception:
        pass


def _cache_delete_conversation(conversation_id: str) -> None:
    """Invalidate conversation in Redis cache."""
    try:
        from redis_manager import cache_delete
        cache_delete(_conv_cache_key(conversation_id))
    except Exception:
        pass


def _record_from_dict(d: dict[str, Any]) -> ConversationRecord:
    """Reconstruct a ConversationRecord from a cached dictionary."""
    turns = [
        ConversationTurn(
            turn_id=t["turn_id"],
            conversation_id=t["conversation_id"],
            user_id=t["user_id"],
            organization=t["organization"],
            query=t["query"],
            answer=t["answer"],
            sources=t.get("sources", []),
            generation_metadata=t.get("generation_metadata", {}),
            created_at=t["created_at"],
        )
        for t in d.get("turns", [])
    ]
    return ConversationRecord(
        conversation_id=d["conversation_id"],
        user_id=d["user_id"],
        organization=d["organization"],
        title=d.get("title", "New Conversation"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        metadata=d.get("metadata", {}),
        turns=turns,
    )


@dataclass
class ConversationTurn:
    turn_id: int
    conversation_id: str
    user_id: str
    organization: str
    query: str
    answer: str
    sources: list[dict[str, Any]]
    generation_metadata: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "organization": self.organization,
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "generation_metadata": self.generation_metadata,
            "created_at": self.created_at,
        }


@dataclass
class ConversationRecord:
    conversation_id: str
    user_id: str
    organization: str
    title: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    turns: list[ConversationTurn]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "organization": self.organization,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "turn_count": len(self.turns),
            "turns": [turn.to_dict() for turn in self.turns],
        }


class ConversationStore:
    """
    SQLite-backed store managing persistent conversations and multi-turn dialogs.
    Thread-safe and multi-tenant isolated by (user_id, organization).
    """

    def __init__(self, db_path: Path = CONVERSATIONS_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    generation_metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_turns_cid
                ON conversation_turns (conversation_id, turn_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_user_org
                ON conversations (user_id, organization, updated_at DESC)
                """
            )

    def get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: str,
        organization: str,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[ConversationRecord, bool]:
        """
        Get an existing conversation or create a new one.
        Returns (ConversationRecord, created_flag).
        """
        now = utc_now()
        meta_json = json.dumps(metadata or {})

        if conversation_id:
            cid = validate_conversation_id(conversation_id)
            existing = self.get_conversation(cid, user_id=user_id, organization=organization)
            if existing:
                return existing, False
            # If specified conversation_id does not exist, initialize it
            conv_title = title or "New Conversation"
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id, user_id, organization, title, created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cid, user_id, organization, conv_title, now, now, meta_json),
                )
            created_conv = ConversationRecord(
                conversation_id=cid,
                user_id=user_id,
                organization=organization,
                title=conv_title,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
                turns=[],
            )
            _cache_set_conversation(cid, created_conv.to_dict())
            return created_conv, True

        # Generate a new conversation ID
        cid = new_conversation_id()
        conv_title = title or "New Conversation"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id, user_id, organization, title, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, user_id, organization, conv_title, now, now, meta_json),
            )

        created_conv = ConversationRecord(
            conversation_id=cid,
            user_id=user_id,
            organization=organization,
            title=conv_title,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            turns=[],
        )
        _cache_set_conversation(cid, created_conv.to_dict())
        return created_conv, True

    def add_turn(
        self,
        conversation_id: str,
        user_id: str,
        organization: str,
        query: str,
        answer: str,
        sources: Optional[list[dict[str, Any]]] = None,
        generation_metadata: Optional[dict[str, Any]] = None,
    ) -> ConversationTurn:
        """
        Add a Q&A interaction turn to the conversation.
        Updates the conversation's `updated_at` and sets a descriptive title on the first turn.
        """
        cid = validate_conversation_id(conversation_id)
        now = utc_now()
        sources_json = json.dumps(sources or [])
        gen_meta_json = json.dumps(generation_metadata or {})

        # Ensure conversation exists
        self.get_or_create_conversation(cid, user_id=user_id, organization=organization)

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_turns (
                    conversation_id, user_id, organization, query, answer, sources, generation_metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    user_id,
                    organization,
                    query.strip(),
                    answer.strip(),
                    sources_json,
                    gen_meta_json,
                    now,
                ),
            )
            turn_id = cursor.lastrowid

            # Update conversation timestamp & auto-title if still default
            row = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id=?",
                (cid,),
            ).fetchone()
            if row and row["title"] == "New Conversation":
                # Create a concise title from the first question
                clean_q = query.strip().split("\n")[0][:60].strip()
                if clean_q:
                    new_title = clean_q
                    conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at=?, title=?
                        WHERE conversation_id=?
                        """,
                        (now, new_title, cid),
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                        (now, cid),
                    )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                    (now, cid),
                )

        turn = ConversationTurn(
            turn_id=turn_id,
            conversation_id=cid,
            user_id=user_id,
            organization=organization,
            query=query.strip(),
            answer=answer.strip(),
            sources=sources or [],
            generation_metadata=generation_metadata or {},
            created_at=now,
        )

        # Refresh Redis fast cache with updated turn list and renewed 60-min TTL
        try:
            full_conv = self._load_from_sqlite(cid)
            if full_conv:
                _cache_set_conversation(cid, full_conv.to_dict())
        except Exception:
            pass

        return turn

    def get_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Optional[ConversationRecord]:
        """
        Retrieve a conversation and its turns.
        Checks Redis fast cache first (TTL 60 min); falls back to SQLite on cache miss.
        Enforces tenant and user isolation when provided.
        """
        cid = validate_conversation_id(conversation_id)

        # 1. Check Redis fast cache first
        cached_dict = _cache_get_conversation(cid)
        if cached_dict is not None and isinstance(cached_dict, dict):
            user_ok = user_id is None or cached_dict.get("user_id") == user_id
            org_ok = organization is None or cached_dict.get("organization") == organization
            if user_ok and org_ok:
                return _record_from_dict(cached_dict)

        # 2. Cache miss -> query SQLite
        record = self._load_from_sqlite(cid, user_id=user_id, organization=organization)
        if record:
            _cache_set_conversation(cid, record.to_dict())
        return record

    def _load_from_sqlite(
        self,
        cid: str,
        user_id: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> Optional[ConversationRecord]:
        with self._connect() as conn:
            query = "SELECT * FROM conversations WHERE conversation_id=?"
            params: list[Any] = [cid]
            if user_id is not None:
                query += " AND user_id=?"
                params.append(user_id)
            if organization is not None:
                query += " AND organization=?"
                params.append(organization)

            row = conn.execute(query, params).fetchone()
            if not row:
                return None

            turn_rows = conn.execute(
                """
                SELECT * FROM conversation_turns
                WHERE conversation_id=?
                ORDER BY turn_id ASC
                """,
                (cid,),
            ).fetchall()

            turns: list[ConversationTurn] = []
            for tr in turn_rows:
                turns.append(
                    ConversationTurn(
                        turn_id=tr["turn_id"],
                        conversation_id=tr["conversation_id"],
                        user_id=tr["user_id"],
                        organization=tr["organization"],
                        query=tr["query"],
                        answer=tr["answer"],
                        sources=json.loads(tr["sources"] or "[]"),
                        generation_metadata=json.loads(tr["generation_metadata"] or "{}"),
                        created_at=tr["created_at"],
                    )
                )

            metadata = json.loads(row["metadata"] or "{}")
            return ConversationRecord(
                conversation_id=row["conversation_id"],
                user_id=row["user_id"],
                organization=row["organization"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=metadata,
                turns=turns,
            )

    def get_conversation_turns(
        self,
        conversation_id: str,
        max_turns: Optional[int] = None,
        user_id: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> list[ConversationTurn]:
        """
        Retrieve the most recent turns for a conversation in chronological order.
        """
        conv = self.get_conversation(conversation_id, user_id=user_id, organization=organization)
        if not conv:
            return []
        if max_turns is not None and max_turns > 0:
            return conv.turns[-max_turns:]
        return conv.turns

    def list_conversations(
        self,
        user_id: str,
        organization: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List conversations for a given user and organization with turn counts and last activity.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                       COUNT(t.turn_id) AS turn_count,
                       MAX(t.created_at) AS last_message_at
                FROM conversations c
                LEFT JOIN conversation_turns t ON c.conversation_id = t.conversation_id
                WHERE c.user_id=? AND c.organization=?
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, organization, limit, offset),
            ).fetchall()

            result = []
            for r in rows:
                result.append(
                    {
                        "conversation_id": r["conversation_id"],
                        "user_id": r["user_id"],
                        "organization": r["organization"],
                        "title": r["title"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "turn_count": r["turn_count"],
                        "last_message_at": r["last_message_at"] or r["updated_at"],
                    }
                )
            return result

    def count_conversations(self, user_id: str, organization: str) -> int:
        """Count total conversations for a user."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM conversations
                WHERE user_id=? AND organization=?
                """,
                (user_id, organization),
            ).fetchone()
            return int(row["count"]) if row else 0

    def delete_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> bool:
        """
        Delete a conversation and all its associated turns.
        """
        cid = validate_conversation_id(conversation_id)
        with self._lock, self._connect() as conn:
            query = "SELECT conversation_id FROM conversations WHERE conversation_id=?"
            params: list[Any] = [cid]
            if user_id is not None:
                query += " AND user_id=?"
                params.append(user_id)
            if organization is not None:
                query += " AND organization=?"
                params.append(organization)

            row = conn.execute(query, params).fetchone()
            if not row:
                return False

            conn.execute("DELETE FROM conversations WHERE conversation_id=?", (cid,))
            _cache_delete_conversation(cid)
            return True


# ===========================================================================
# History Formatting Helpers for LLM Prompt Integration
# ===========================================================================

def format_conversation_history(
    turns: Sequence[ConversationTurn | dict[str, Any]],
    max_turns: int = DEFAULT_MAX_CONTEXT_TURNS,
) -> str:
    """
    Format prior conversation turns into a clean, structured context block
    for injection into vision-language model prompts.
    """
    if not turns:
        return ""

    selected_turns = turns[-max_turns:] if max_turns > 0 else turns
    lines: list[str] = []

    for idx, turn in enumerate(selected_turns, start=1):
        if isinstance(turn, ConversationTurn):
            query = turn.query.strip()
            answer = turn.answer.strip()
        elif isinstance(turn, dict):
            query = str(turn.get("query", "")).strip()
            answer = str(turn.get("answer", "")).strip()
        else:
            continue

        if not query and not answer:
            continue

        lines.append(f"[Turn {idx}]")
        lines.append(f"User: {query}")
        lines.append(f"Assistant: {answer}\n")

    return "\n".join(lines).strip()


# Singleton instance
_CONVERSATION_STORE: Optional[ConversationStore] = None
_STORE_LOCK = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Get or initialize the global conversation store."""
    global _CONVERSATION_STORE
    if _CONVERSATION_STORE is None:
        with _STORE_LOCK:
            if _CONVERSATION_STORE is None:
                _CONVERSATION_STORE = ConversationStore()
    return _CONVERSATION_STORE
