"""
Run from the same project directory as ingest.py:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from collections.abc import Iterator

from fastapi import FastAPI, Header, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from ingest import DATA_DIR, REGISTRY_DB_PATH, IngestJob, ingest_files
from registry import Registry
from docker_manager import start_qdrant, stop_qdrant
from query import retrieve
from generate import generate_answer, generate_answer_stream, GENERATOR_MODEL
from conversation_manager import (
    get_conversation_store,
    validate_conversation_id,
    new_conversation_id,
)
from authorization import AuthContext, authenticate_user, create_user, get_current_user
from redis_manager import (
    answer_cache_key,
    cache_get_json,
    cache_set_json,
    ensure_redis_running,
    get_job_state,
    health_check as redis_health_check,
    reserve_idempotency,
    retrieval_cache_key,
    set_job_state,
    update_idempotency_status,
)

logger = logging.getLogger("multimodal-rag-api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_JOB_DB_PATH = Path(
    os.getenv("API_JOB_DB_PATH", str(DATA_DIR / "ingestion_jobs.db"))
)
API_JOB_DATA_DIR = Path(
    os.getenv("API_JOB_DATA_DIR", str(DATA_DIR / "api_jobs"))
)
S3_DOWNLOAD_TIMEOUT = int(os.getenv("S3_DOWNLOAD_TIMEOUT", "120"))
S3_MAX_FILE_SIZE_MB = int(os.getenv("S3_MAX_FILE_SIZE_MB", "512"))
MAX_FILES_PER_REQUEST = int(os.getenv("API_MAX_FILES_PER_REQUEST", "20"))

# Optional comma-separated allow-list. Example:
# S3_ALLOWED_HOSTS=my-bucket.s3.amazonaws.com,s3.amazonaws.com
S3_ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.getenv("S3_ALLOWED_HOSTS", "").split(",")
    if host.strip()
}

GENERATE_JOB_DB_PATH = Path(
    os.getenv(
        "GENERATE_JOB_DB_PATH",
        str(DATA_DIR / "generate_jobs.db"),
    )
)

GENERATE_LOG_DIR = Path(
    os.getenv(
        "GENERATE_LOG_DIR",
        str(DATA_DIR / "generate"),
    )
)
GENERATE_LOG_DIR.mkdir(parents=True, exist_ok=True)

JOB_ID_RE = re.compile(r"^ingest_[a-zA-Z0-9_-]+$")
GENERATE_JOB_ID_RE = re.compile(r"^generate_[a-zA-Z0-9_-]+$")

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    return f"ingest_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid.uuid4().hex[:8]}"


def safe_filename(name: str) -> str:
    value = Path(name).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "document.pdf"


def get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            # Sequential execution prevents multiple simultaneous ColPali
            # models from competing for the same GPU memory.
            _executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ingestion",
            )
        return _executor



def new_generate_job_id() -> str:
    return (
        f"generate_"
        f"{datetime.now(timezone.utc):%Y%m%d%H%M%S}_"
        f"{uuid.uuid4().hex[:8]}"
    )



# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    user_id: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    role: str = Field(min_length=1)
    s3_links: list[str] = Field(default_factory=list)
    local_pdf_paths: list[str] = Field(default_factory=list)

    @field_validator("user_id", "organization", "role")
    @classmethod
    def strip_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("s3_links")
    @classmethod
    def validate_s3_links(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("s3_links cannot contain empty strings")
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"Unsupported S3 URL scheme: {value}")
            if not parsed.netloc:
                raise ValueError(f"Invalid S3 URL: {value}")
            cleaned.append(value)
        return cleaned

    @field_validator("local_pdf_paths")
    @classmethod
    def validate_local_paths(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("local_pdf_paths cannot contain empty strings")
            if not value.lower().endswith(".pdf"):
                raise ValueError(f"Expected a PDF local path: {value}")
            cleaned.append(value)
        return cleaned

    @model_validator(mode="after")
    def validate_sources(self) -> "IngestRequest":
        total = len(self.s3_links) + len(self.local_pdf_paths)
        if total == 0:
            raise ValueError(
                "At least one source is required in s3_links or local_pdf_paths"
            )
        if total > MAX_FILES_PER_REQUEST:
            raise ValueError(
                f"Too many files. Maximum is {MAX_FILES_PER_REQUEST} per request."
            )
        return self


class AcceptedResponse(BaseModel):
    job_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    role: str = Field(min_length=1)

    query: str = Field(min_length=1)

    document_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)

    recursive_search: bool = False

    top_k: int = Field(default=5, ge=1)

    enable_rerank: bool = False
    rerank_threshold: float = 0.75
    retrieval_top_k: int = 20
    enable_graph: bool | None = None
    conversation_id: str | None = None

    @field_validator("conversation_id")
    @classmethod
    def validate_conv_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > 128:
            raise ValueError("conversation_id must be at most 128 characters")
        if not re.match(r"^[a-zA-Z0-9_\-\.]{1,128}$", value):
            raise ValueError(
                "conversation_id contains invalid characters. Allowed: alphanumeric, _, -, ."
            )
        return value

    @field_validator("user_id", "organization", "role", "query")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("must not be empty")

        return value

    @field_validator("document_ids", "version_ids")
    @classmethod
    def clean_ids(cls, values: list[str]) -> list[str]:
        cleaned = []

        for value in values:
            value = value.strip()

            if not value:
                raise ValueError("IDs cannot contain empty strings")

            cleaned.append(value)

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(cleaned))


# Canonical alias for answering requests
AnswerRequest = QueryRequest


class GenerateJobStore:
    """SQLite-backed status/result store for answer-generation jobs."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generate_jobs (
                    generate_job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    role TEXT NOT NULL,
                    query TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    version_ids TEXT NOT NULL,
                    recursive_search INTEGER NOT NULL,
                    top_k INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    result TEXT,
                    error TEXT,
                    conversation_id TEXT
                )
                """
            )
            try:
                conn.execute(
                    "ALTER TABLE generate_jobs ADD COLUMN conversation_id TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def create_job(
        self,
        *,
        generate_job_id: str,
        request: QueryRequest,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO generate_jobs (
                    generate_job_id,
                    user_id,
                    organization,
                    role,
                    query,
                    document_ids,
                    version_ids,
                    recursive_search,
                    top_k,
                    status,
                    created_at,
                    conversation_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    generate_job_id,
                    request.user_id,
                    request.organization,
                    request.role,
                    request.query,
                    json.dumps(request.document_ids),
                    json.dumps(request.version_ids),
                    int(request.recursive_search),
                    request.top_k,
                    utc_now(),
                    getattr(request, "conversation_id", None),
                ),
            )

    def mark_running(self, generate_job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE generate_jobs
                SET status='running',
                    started_at=?
                WHERE generate_job_id=?
                """,
                (utc_now(), generate_job_id),
            )

    def mark_completed(
        self,
        generate_job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE generate_jobs
                SET status='completed',
                    finished_at=?,
                    result=?,
                    error=NULL
                WHERE generate_job_id=?
                """,
                (
                    utc_now(),
                    json.dumps(result, ensure_ascii=False),
                    generate_job_id,
                ),
            )

    def mark_failed(
        self,
        generate_job_id: str,
        error: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE generate_jobs
                SET status='failed',
                    finished_at=?,
                    error=?
                WHERE generate_job_id=?
                """,
                (
                    utc_now(),
                    error,
                    generate_job_id,
                ),
            )

    def get_job(
        self,
        generate_job_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM generate_jobs
                WHERE generate_job_id=?
                """,
                (generate_job_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "generate_job_id": row["generate_job_id"],
            "user_id": row["user_id"],
            "organization": row["organization"],
            "role": row["role"],
            "query": row["query"],
            "conversation_id": (
                row["conversation_id"] if "conversation_id" in row.keys() else None
            ),
            "document_ids": json.loads(row["document_ids"]),
            "version_ids": json.loads(row["version_ids"]),
            "recursive_search": bool(row["recursive_search"]),
            "top_k": row["top_k"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "result": (
                json.loads(row["result"])
                if row["result"]
                else None
            ),
            "error": row["error"],
        }

    def recover_running_jobs(self) -> None:
        """
        Mark queued/running generation jobs as failed after API restart.

        V1 uses an in-process executor, so these jobs are not automatically
        resumed.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE generate_jobs
                SET status='failed',
                    finished_at=?,
                    error=?
                WHERE status IN ('queued', 'running')
                """,
                (
                    utc_now(),
                    "API process restarted while generation job was running.",
                ),
            )


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

class JobStore:
    """Small SQLite-backed V1 job store.

    Every DB operation opens its own SQLite connection because API work runs in
    a background thread and request handlers may execute concurrently.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    organization TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    ingested_count INTEGER NOT NULL DEFAULT 0,
                    duplicate_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS job_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    filename TEXT,
                    status TEXT NOT NULL,
                    document_id TEXT,
                    version_id TEXT,
                    version_number INTEGER,
                    pages INTEGER,
                    new_embeddings INTEGER,
                    reused_embeddings INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );

                CREATE INDEX IF NOT EXISTS idx_job_files_job_id
                ON job_files(job_id);
                """
            )

    def create_job(
        self,
        *,
        job_id: str,
        user_id: str,
        organization: str,
        role: str,
        total_files: int,
        files: list[dict[str, str]],
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, user_id, organization, role, status,
                    created_at, total_files
                ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
                """,
                (job_id, user_id, organization, role, now, total_files),
            )
            conn.executemany(
                """
                INSERT INTO job_files (
                    job_id, source_type, source, filename, status, created_at
                ) VALUES (?, ?, ?, ?, 'queued', ?)
                """,
                [
                    (
                        job_id,
                        item["source_type"],
                        item["source"],
                        item.get("filename"),
                        now,
                    )
                    for item in files
                ],
            )

    def mark_running(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE job_id=?",
                (utc_now(), job_id),
            )

    def mark_file_started(self, file_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE job_files SET status='running', started_at=? WHERE id=?",
                (utc_now(), file_id),
            )

    def list_job_files(self, job_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM job_files WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()

    def update_file(
        self,
        file_id: int,
        *,
        status_value: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        result = result or {}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE job_files
                SET status=?,
                    filename=COALESCE(?, filename),
                    document_id=?,
                    version_id=?,
                    version_number=?,
                    pages=?,
                    new_embeddings=?,
                    reused_embeddings=?,
                    error=?,
                    finished_at=?
                WHERE id=?
                """,
                (
                    status_value,
                    result.get("filename"),
                    result.get("document_id"),
                    result.get("version_id"),
                    result.get("version_number"),
                    result.get("pages"),
                    result.get("new_embeddings"),
                    result.get("reused_embeddings"),
                    error,
                    utc_now(),
                    file_id,
                ),
            )

    def finish_job(
        self,
        job_id: str,
        *,
        status_value: str,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status != 'queued' AND status != 'running' THEN 1 ELSE 0 END) AS processed,
                    SUM(CASE WHEN status = 'ingested' THEN 1 ELSE 0 END) AS ingested,
                    SUM(CASE WHEN status = 'duplicate' THEN 1 ELSE 0 END) AS duplicate,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
                FROM job_files
                WHERE job_id=?
                """,
                (job_id,),
            ).fetchone()
            conn.execute(
                """
                UPDATE jobs
                SET status=?,
                    finished_at=?,
                    processed_files=?,
                    ingested_count=?,
                    duplicate_count=?,
                    failed_count=?,
                    error=?
                WHERE job_id=?
                """,
                (
                    status_value,
                    utc_now(),
                    int(counts["processed"] or 0),
                    int(counts["ingested"] or 0),
                    int(counts["duplicate"] or 0),
                    int(counts["failed"] or 0),
                    error,
                    job_id,
                ),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            files = conn.execute(
                "SELECT * FROM job_files WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()

        processed = sum(1 for file in files if file["status"] not in {"queued", "running"})
        ingested = sum(1 for file in files if file["status"] == "ingested")
        duplicate = sum(1 for file in files if file["status"] == "duplicate")
        failed = sum(1 for file in files if file["status"] == "failed")

        return {
            "job_id": row["job_id"],
            "user_id": row["user_id"],
            "organization": row["organization"],
            "role": row["role"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "total_files": row["total_files"],
            "processed_files": processed,
            "ingested_count": ingested,
            "duplicate_count": duplicate,
            "failed_count": failed,
            "error": row["error"],
            "results": [self._serialize_file(file) for file in files],
        }

    @staticmethod
    def _serialize_file(row: sqlite3.Row) -> dict[str, Any]:
        result = {
            "source_type": row["source_type"],
            "source": row["source"],
            "filename": row["filename"],
            "status": row["status"],
        }
        for key in (
            "document_id",
            "version_id",
            "version_number",
            "pages",
            "new_embeddings",
            "reused_embeddings",
            "error",
            "started_at",
            "finished_at",
        ):
            value = row[key]
            if value is not None:
                result[key] = value
        return result

    def recover_running_jobs(self) -> None:
        """Mark jobs interrupted by an API process restart as failed."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='failed',
                    finished_at=?,
                    error='API process restarted while this job was running.'
                WHERE status IN ('queued', 'running')
                """,
                (utc_now(),),
            )
            conn.execute(
                """
                UPDATE job_files
                SET status='failed',
                    finished_at=?,
                    error='API process restarted before this file completed.'
                WHERE status IN ('queued', 'running')
                """,
                (utc_now(),),
            )


job_store = JobStore(API_JOB_DB_PATH)
generate_job_store = GenerateJobStore(GENERATE_JOB_DB_PATH)


# ---------------------------------------------------------------------------
# Source handling
# ---------------------------------------------------------------------------


def validate_s3_host(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"Invalid S3 URL: {url}")

    if S3_ALLOWED_HOSTS:
        if host not in S3_ALLOWED_HOSTS:
            raise ValueError(f"S3 host is not allowed: {host}")
        return

    # Default V1 safety check: accept AWS S3-style hosts only.
    if not (host == "s3.amazonaws.com" or host.endswith(".amazonaws.com")):
        raise ValueError(
            f"S3 URL host is not an AWS S3-style host: {host}. "
            "Set S3_ALLOWED_HOSTS to explicitly allow your endpoint."
        )


def download_s3_pdf(url: str, destination: Path) -> Path:
    validate_s3_host(url)

    destination.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = S3_MAX_FILE_SIZE_MB * 1024 * 1024

    request = Request(
        url,
        headers={"User-Agent": "AMG-Ingestion-V1/1.0"},
    )

    try:
        with urlopen(request, timeout=S3_DOWNLOAD_TIMEOUT) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError(
                    f"Remote PDF exceeds the {S3_MAX_FILE_SIZE_MB} MB limit"
                )

            total = 0
            with destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"Remote PDF exceeds the {S3_MAX_FILE_SIZE_MB} MB limit"
                        )
                    output.write(chunk)

    except HTTPError as exc:
        raise RuntimeError(
            f"S3 download failed with HTTP {exc.code}: {url}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"S3 download failed: {url}: {exc.reason}") from exc

    if destination.stat().st_size == 0:
        raise ValueError(f"Downloaded PDF is empty: {url}")

    return destination


def prepare_sources(job_id: str, request: IngestRequest) -> list[dict[str, Any]]:
    job_dir = API_JOB_DATA_DIR / job_id / "inputs"
    job_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []

    # Keep source order identical to create_ingestion_job(): S3 first, then
    # local paths. This makes persisted job_file rows map 1:1 to these sources.
    for index, url in enumerate(request.s3_links, start=1):
        parsed = urlparse(url)
        raw_name = Path(parsed.path).name or f"document_{index}.pdf"
        filename = safe_filename(raw_name)
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        downloaded = job_dir / f"{index:03d}_{filename}"
        item = {
            "source_type": "s3",
            "source": url,
            "path": downloaded,
            "filename": filename,
        }
        try:
            download_s3_pdf(url, downloaded)
        except Exception as exc:
            item["error"] = str(exc)
        sources.append(item)

    for local_path in request.local_pdf_paths:
        path = Path(local_path).expanduser().resolve()
        item = {
            "source_type": "local",
            "source": str(path),
            "path": path,
            "filename": path.name,
        }
        try:
            if not path.exists():
                raise FileNotFoundError(f"Local PDF does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Local PDF path is not a file: {path}")
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Expected a PDF file: {path}")
        except Exception as exc:
            item["error"] = str(exc)
        sources.append(item)

    return sources


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------


def _run_job(job_id: str, request: IngestRequest) -> None:
    logger.info("Starting job %s", job_id)
    job_store.mark_running(job_id)

    job_dir = API_JOB_DATA_DIR / job_id

    try:
        sources = prepare_sources(job_id, request)
        file_rows = job_store.list_job_files(job_id)

        if len(sources) != len(file_rows):
            raise RuntimeError("Job source count does not match persisted file count")

        jobs: list[IngestJob] = []
        row_to_source: list[tuple[int, dict[str, Any]]] = []
        immediate_results: list[dict[str, Any]] = []

        for row, source in zip(file_rows, sources):
            file_id = int(row["id"])
            if source.get("error"):
                result = {
                    "status": "failed",
                    "filename": source.get("filename"),
                    "error": source["error"],
                }
                job_store.update_file(
                    file_id,
                    status_value="failed",
                    result=result,
                    error=result["error"],
                )
                immediate_results.append(result)
                continue

            jobs.append(IngestJob(pdf_path=source["path"]))
            row_to_source.append((file_id, source))

        # ingest_files() now isolates individual ingestion failures and keeps
        # the single shared ColPali model alive across the whole batch.
        results = ingest_files(jobs) if jobs else []

        if len(results) != len(row_to_source):
            raise RuntimeError(
                "Ingestion result count does not match the submitted file count"
            )

        all_results = list(immediate_results)
        for (file_id, _source), result in zip(row_to_source, results):
            result_status = result.get("status")
            if result_status not in {"ingested", "duplicate", "failed"}:
                result_status = "failed"
                result = {
                    **result,
                    "error": f"Unexpected ingestion status: {result_status}",
                }

            job_store.update_file(
                file_id,
                status_value=result_status,
                result=result,
                error=result.get("error"),
            )
            all_results.append(result)

        failed_count = sum(1 for result in all_results if result.get("status") == "failed")
        if failed_count == 0:
            final_status = "completed"
        elif failed_count == len(results):
            final_status = "failed"
        else:
            final_status = "completed_with_errors"

        job_store.finish_job(job_id, status_value=final_status)
        logger.info("Finished job %s with status=%s", job_id, final_status)

    except Exception as exc:
        logger.exception("Job failed: %s", job_id)
        job_store.finish_job(
            job_id,
            status_value="failed",
            error=str(exc),
        )
    finally:
        # shortcut: sync Redis job cache on state change; upgrade to pubsub/worker queue if scaled
        try:
            latest = job_store.get_job(job_id)
            if latest:
                set_job_state("ingest", job_id, latest)
        except Exception:
            pass
        # The existing ingestion pipeline has already copied the PDF into its
        # permanent PDF_DIR before returning success. Temporary downloads can
        # therefore be removed after the job finishes.
        shutil.rmtree(job_dir, ignore_errors=True)


def save_generate_log(
    generate_job_id: str,
    request: QueryRequest,
    *,
    status_value: str,
    created_at: str,
    started_at: str | None,
    finished_at: str | None,
    retrieval: dict[str, Any] | None,
    response: dict[str, Any] | None,
    error: str | None,
) -> None:
    """
    Persist the complete generation execution record.

    File:
        data/generate/<generate_job_id>.json

    Includes:
        - exact request input
        - retrieval configuration and ranked pages
        - generation metadata
        - final response
        - timing
        - error information
    """
    duration_ms = None

    if started_at and finished_at:
        try:
            started_dt = datetime.fromisoformat(started_at)
            finished_dt = datetime.fromisoformat(finished_at)
            duration_ms = round(
                (finished_dt - started_dt).total_seconds() * 1000,
                3,
            )
        except Exception:
            duration_ms = None

    generation_metadata = None

    if isinstance(response, dict):
        generation_metadata = response.get("generation")

    log_data = {
        "job": {
            "generate_job_id": generate_job_id,
            "status": status_value,
            "created_at": created_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
        },
        "request": {
            "user_id": request.user_id,
            "organization": request.organization,
            "role": request.role,
            "query": request.query,
            "document_ids": request.document_ids,
            "version_ids": request.version_ids,
            "recursive_search": request.recursive_search,
            "top_k": request.top_k,
        },
        "retrieval": retrieval,
        "routing": response.get("routing") if isinstance(response, dict) else None,
        "web_metadata": response.get("web_metadata") if isinstance(response, dict) else None,
        "execution": generation_metadata,
        "response": {
            "status": (
                "success"
                if status_value == "completed"
                else "failed"
            ),
            "result": response,
        },
        "error": error,
    }

    log_path = GENERATE_LOG_DIR / f"{generate_job_id}.json"
    temp_path = GENERATE_LOG_DIR / f".{generate_job_id}.tmp"

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            log_data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # Atomic replacement just like the existing query logging approach.
    temp_path.replace(log_path)

    logger.info(
        "Saved generation log: %s",
        log_path,
    )


def _run_generate_job(
    generate_job_id: str,
    request: QueryRequest,
) -> None:
    """
    Execute:
        retrieve() -> generate_answer()

    Retrieval is kept as a separate local stage so that, if generation fails,
    the retrieved evidence can still be written to the audit JSON.
    """
    logger.info(
        "Starting generation job %s",
        generate_job_id,
    )
    generate_job_store.mark_running(generate_job_id)

    job = generate_job_store.get_job(generate_job_id)

    created_at = (
        job["created_at"]
        if job
        else utc_now()
    )

    started_at = (
        job["started_at"]
        if job
        else utc_now()
    )

    retrieval_result: dict[str, Any] | None = None
    response: dict[str, Any] | None = None

    try:
        from rag_workflow import run_rag_pipeline

        pipeline_input: dict[str, Any] = {
            "user_id": request.user_id,
            "organization": request.organization,
            "role": request.role,
            "original_query": request.query,
            "conversation_id": getattr(request, "conversation_id", None),
            "document_ids": list(request.document_ids or []),
            "version_ids": list(request.version_ids or []),
            "recursive_search": request.recursive_search,
            "top_k": request.top_k,
            "enable_rerank": getattr(request, "enable_rerank", False),
            "rerank_threshold": getattr(request, "rerank_threshold", 0.75),
            "retrieval_top_k": getattr(request, "retrieval_top_k", 20),
        }

        final_state = run_rag_pipeline(pipeline_input)
        conversation_id = final_state.get("conversation_id")
        retrieval_result = final_state.get("retrieval_result")

        gen_resp = final_state.get("generation_response") or {}
        response = {
            "answer": final_state.get("answer", "") or gen_resp.get("answer", ""),
            "sources": final_state.get("aggregated_sources", []) or gen_resp.get("sources", []),
            "conversation_id": conversation_id,
            "generation": gen_resp.get("generation", {}),
            "routing": final_state.get("routing_metadata", {}),
            "retrieval_metadata": final_state.get("retrieval_metadata", {}),
            "web_metadata": final_state.get("web_metadata", {}),
        }

        generate_job_store.mark_completed(
            generate_job_id,
            response,
        )

        completed_job = generate_job_store.get_job(
            generate_job_id,
        )

        finished_at = (
            completed_job["finished_at"]
            if completed_job
            else utc_now()
        )

        try:
            if completed_job:
                set_job_state("generate", generate_job_id, completed_job)
        except Exception:
            pass

        save_generate_log(
            generate_job_id=generate_job_id,
            request=request,
            status_value="completed",
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            retrieval=retrieval_result,
            response=response,
            error=None,
        )

        logger.info(
            "Finished generation job %s",
            generate_job_id,
        )

    except Exception as exc:
        logger.exception(
            "Generation job failed: %s",
            generate_job_id,
        )

        generate_job_store.mark_failed(
            generate_job_id,
            str(exc),
        )

        failed_job = generate_job_store.get_job(
            generate_job_id,
        )

        finished_at = (
            failed_job["finished_at"]
            if failed_job
            else utc_now()
        )

        try:
            if failed_job:
                set_job_state("generate", generate_job_id, failed_job)
        except Exception:
            pass

        save_generate_log(
            generate_job_id=generate_job_id,
            request=request,
            status_value="failed",
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            retrieval=retrieval_result,
            response=response,
            error=str(exc),
        )


class AuthenticatedIngestRequest(IngestRequest):
    user_id: str
    organization: str
    role: str


class AuthenticatedQueryRequest(QueryRequest):
    user_id: str
    organization: str
    role: str


class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    organization: str = Field(min_length=1)
    role: str = Field(min_length=1)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)



def build_authenticated_ingest_request(
    request: IngestRequest,
    current_user: AuthContext,
) -> AuthenticatedIngestRequest:

    return AuthenticatedIngestRequest(
        user_id=current_user.user_id,
        organization=current_user.organization_name,
        role=current_user.role,
        s3_links=request.s3_links,
        local_pdf_paths=request.local_pdf_paths,
    )


def build_authenticated_query_request(
    request: QueryRequest,
    current_user: AuthContext,
) -> AuthenticatedQueryRequest:

    return AuthenticatedQueryRequest(
        user_id=current_user.user_id,
        organization=current_user.organization_name,
        role=current_user.role,
        query=request.query,
        document_ids=request.document_ids,
        version_ids=request.version_ids,
        recursive_search=request.recursive_search,
        top_k=request.top_k,
        enable_rerank=request.enable_rerank,
        rerank_threshold=request.rerank_threshold,
        retrieval_top_k=request.retrieval_top_k,
        enable_graph=request.enable_graph,
        conversation_id=request.conversation_id,
    )


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    global _executor

    # Make sure Docker + Qdrant are available before accepting ingestion work.
    start_qdrant()
    ensure_redis_running()
    job_store.recover_running_jobs()
    generate_job_store.recover_running_jobs()

    _executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ingestion",
    )

    logger.info("AMG ingestion API started")

    try:
        yield
    finally:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=False)

        _executor = None

        # Only stops Qdrant if this API process started it.
        stop_qdrant()

        logger.info("AMG ingestion API stopped")

app = FastAPI(
    title="AMG Multimodal RAG Ingestion API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "redis": redis_health_check(),
    }


@app.get("/me")
def get_me(
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:

    return current_user.as_dict()


@app.post("/create_user")
def register_user(
    request: CreateUserRequest,
) -> dict[str, Any]:

    try:
        result = create_user(
            name=request.name,
            email=request.email,
            password=request.password,
            organization=request.organization,
            role=request.role,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@app.post("/login")
def login(
    request: LoginRequest,
) -> dict[str, Any]:

    return authenticate_user(
        email=request.email,
        password=request.password,
    )



@app.get("/list_documents")
def list_documents(
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, list[dict[str, Any]]]:

    registry = Registry(REGISTRY_DB_PATH)

    try:
        rows = registry.list_document_versions()

        documents = [
            {
                "filename": row["filename"],
                "document_id": row["document_id"],
                "version_id": row["version_id"],
                "version_number": row["version_number"],
                "is_current": bool(
                    row["is_current"]
                ),
            }
            for row in rows
        ]

        return {
            "documents": documents
        }

    finally:
        registry.close()


@app.post(
    "/ingest_document",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_ingestion_job(
    request: IngestRequest,
    current_user: AuthContext = Depends(get_current_user),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> AcceptedResponse:

    authenticated_request = (
        build_authenticated_ingest_request(
            request,
            current_user,
        )
    )

    job_id = new_job_id()
    identity = f"{request.organization}:{request.user_id}"

    if idempotency_key:
        try:
            res = reserve_idempotency(
                identity=identity,
                endpoint="ingest_document",
                idempotency_key=idempotency_key,
                payload=request.model_dump(),
                job_id=job_id,
            )
            if res["state"] == "conflict":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Idempotency key reused with different request payload.",
                )
            if res["state"] == "existing":
                return AcceptedResponse(
                    job_id=res["record"]["job_id"],
                    status="accepted",
                    message="Ingestion job already accepted (idempotent).",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Idempotency reservation check failed: %s", exc)

    try:
        files = [
            {
                "source_type": "s3",
                "source": url,
                "filename": (
                    Path(urlparse(url).path).name
                    or None
                ),
            }
            for url in authenticated_request.s3_links
        ]

        files.extend(
            {
                "source_type": "local",
                "source": path,
                "filename": Path(path).name,
            }
            for path in authenticated_request.local_pdf_paths
        )

        job_store.create_job(
            job_id=job_id,
            user_id=authenticated_request.user_id,
            organization=authenticated_request.organization,
            role=authenticated_request.role,
            total_files=len(files),
            files=files,
        )

        if idempotency_key:
            try:
                update_idempotency_status(
                    identity=identity,
                    endpoint="ingest_document",
                    idempotency_key=idempotency_key,
                    job_id=job_id,
                    status="queued",
                )
            except Exception:
                pass

        executor = get_executor()

        executor.submit(
            _run_job,
            job_id,
            authenticated_request,
        )

    except Exception as exc:
        logger.exception(
            "Could not create ingestion job"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Could not create ingestion job: {exc}"
            ),
        ) from exc

    return AcceptedResponse(
        job_id=job_id,
        status="accepted",
        message="Ingestion job accepted.",
    )



@app.get("/ingest_document/{job_id}")
def get_ingestion_job(
    job_id: str,
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:

    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format",
        )

    # shortcut: check fast Redis job state cache first; fall back to SQLite durable store
    cached = None
    try:
        cached = get_job_state("ingest", job_id)
    except Exception:
        cached = None

    if cached is not None:
        if cached.get("user_id") != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job.",
            )
        return cached

    # If not found in cache, retrieve from the durable store
    job = job_store.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    if job.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this job.",
        )

    try:
        set_job_state("ingest", job_id, job)
    except Exception:
        pass

    return job


@app.post(
    "/answer",
    summary="Answer user query using multimodal RAG pipeline",
)
@app.post(
    "/generate",
    summary="Legacy alias for /answer",
    include_in_schema=False,
)
def create_generate_job(
    request: QueryRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:

    authenticated_request = build_authenticated_query_request(
        request,
        current_user,
    )

    generate_job_id = new_generate_job_id()

    # Use the normalized, authenticated request for identity and payload.
    # Ensure build_authenticated_query_request overwrites client-supplied
    # organization/user identity with trusted authentication data.
    identity = (
        f"{authenticated_request.organization}:"
        f"{authenticated_request.user_id}"
    )

    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()

        if not idempotency_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Idempotency-Key cannot be empty.",
            )

    def update_idempotency(status_value: str) -> None:
        """Update the idempotency record without masking the main result."""
        if not idempotency_key:
            return

        try:
            update_idempotency_status(
                identity=identity,
                endpoint="generate",
                idempotency_key=idempotency_key,
                job_id=generate_job_id,
                status=status_value,
            )
        except Exception:
            logger.exception(
                "Could not update idempotency status for job %s to %s",
                generate_job_id,
                status_value,
            )

    # ---------------------------------------------------------
    # 1. Reserve/check idempotency before creating a generation job
    # ---------------------------------------------------------
    if idempotency_key:
        try:
            reservation = reserve_idempotency(
                identity=identity,
                endpoint="generate",
                idempotency_key=idempotency_key,
                payload=authenticated_request.model_dump(),
                job_id=generate_job_id,
            )

        except Exception as exc:
            logger.exception(
                "Idempotency reservation failed for generate request"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Unable to verify the request's idempotency key. "
                    "Retry using the same Idempotency-Key."
                ),
            ) from exc

        reservation_state = reservation.get("state")

        if reservation_state == "conflict":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Idempotency key reused with a different "
                    "request payload."
                ),
            )

        if reservation_state == "existing":
            record = reservation.get("record") or {}
            existing_job_id = record.get("job_id")

            if not existing_job_id:
                logger.error(
                    "Existing idempotency record has no job ID: %s",
                    identity,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "The existing request could not be identified. "
                        "Retry using the same Idempotency-Key."
                    ),
                )

            # Prefer the latest job state, then fall back to the job store.
            existing_job = None

            try:
                existing_job = get_job_state(
                    "generate",
                    existing_job_id,
                )
            except Exception:
                logger.warning(
                    "Could not retrieve cached state for job %s",
                    existing_job_id,
                    exc_info=True,
                )

            if not existing_job:
                try:
                    existing_job = generate_job_store.get_job(
                        existing_job_id
                    )
                except Exception:
                    logger.warning(
                        "Could not retrieve stored state for job %s",
                        existing_job_id,
                        exc_info=True,
                    )

            if existing_job:
                existing_status = str(
                    existing_job.get("status", "")
                ).lower()

                # Completed: replay the original result.
                if (
                    existing_status == "completed"
                    and "result" in existing_job
                ):
                    return existing_job["result"]

                # Still running: never create a second generation job.
                if existing_status in {
                    "pending",
                    "queued",
                    "running",
                    "processing",
                    "in_progress",
                }:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=(
                            "This generation request is already "
                            "processing. Retry with the same "
                            "Idempotency-Key."
                        ),
                    )

                # Failed: do not silently execute the same operation again.
                if existing_status == "failed":
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=(
                            existing_job.get("error")
                            or "The original generation request failed."
                        ),
                    )

            # A reservation exists, but its job state cannot be verified.
            # Fail safely instead of risking duplicate model execution.
            logger.error(
                "Could not determine the state of existing job %s",
                existing_job_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "The existing request's status is unavailable. "
                    "Retry using the same Idempotency-Key."
                ),
            )

    # ---------------------------------------------------------
    # 2. Create and execute a new generation job
    # ---------------------------------------------------------
    try:
        generate_job_store.create_job(
            generate_job_id=generate_job_id,
            request=authenticated_request,
        )

        executor = get_executor()

        future = executor.submit(
            _run_generate_job,
            generate_job_id,
            authenticated_request,
        )

        # Intentionally wait because this endpoint returns the model result.
        future.result()

        job = generate_job_store.get_job(generate_job_id)

        if job is None:
            raise RuntimeError(
                f"Generation job disappeared: {generate_job_id}"
            )

        job_status = str(job.get("status", "")).lower()

        if job_status == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=job.get("error") or "Generation failed.",
            )

        if job_status != "completed":
            raise RuntimeError(
                f"Unexpected generation job status: {job_status}"
            )

        if "result" not in job:
            raise RuntimeError(
                f"Completed job has no result: {generate_job_id}"
            )

        # -----------------------------------------------------
        # 3. Record successful completion
        # -----------------------------------------------------
        update_idempotency("completed")

        # Return only the generated result.
        return job["result"]

    except HTTPException:
        update_idempotency("failed")
        raise

    except Exception as exc:
        update_idempotency("failed")

        logger.exception(
            "Could not complete generation request %s",
            generate_job_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation failed.",
        ) from exc


@app.post(
    "/answer/stream",
    summary="Stream answer tokens via Server-Sent Events (SSE)",
)
@app.post(
    "/generate/stream",
    summary="Legacy alias for /answer/stream",
    include_in_schema=False,
)
def create_generate_stream(
    request: QueryRequest,
    current_user: AuthContext = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream generated answer tokens incrementally via Server-Sent Events (SSE).

    Protocol:
        event: token
        data: {"token": "..."}

        event: complete
        data: {
            "answer": "...",
            "sources": [...],
            "generation": {...},
            "latency_ms": 1234.5,
            "ttft_ms": 450.2
        }

        event: error
        data: {"error": "..."}
    """
    authenticated_request = build_authenticated_query_request(
        request,
        current_user,
    )

    def event_stream_generator() -> Iterator[str]:
        generate_job_id = new_generate_job_id()
        created_at = utc_now()
        started_at = utc_now()

        generate_job_store.create_job(
            generate_job_id=generate_job_id,
            request=authenticated_request,
        )
        generate_job_store.mark_running(generate_job_id)

        retrieval_result: dict[str, Any] | None = None
        ans_cache_key: str | None = None

        try:
            # 1. Load conversation context & previous retrieval evidence
            conv_store = get_conversation_store()
            conv, _ = conv_store.get_or_create_conversation(
                conversation_id=getattr(
                    authenticated_request, "conversation_id", None
                ),
                user_id=authenticated_request.user_id,
                organization=authenticated_request.organization,
            )
            conversation_id = conv.conversation_id
            raw_history = conv_store.get_conversation_turns(
                conversation_id,
                user_id=authenticated_request.user_id,
                organization=authenticated_request.organization,
            )
            history = [t.to_dict() for t in raw_history]
            prev_retrieval = conv_store.get_last_retrieval_context(
                conversation_id,
                user_id=authenticated_request.user_id,
                organization=authenticated_request.organization,
            )

            # 2. External Query Routing (Qwen shared local model)
            from external_query_router import get_external_query_router, ExternalRouterInput
            router = get_external_query_router()
            router_input = ExternalRouterInput(
                query=authenticated_request.query,
                conversation_history=history,
                previous_retrieval_context=prev_retrieval,
            )
            decision, routing_metadata = router.route(router_input)

            # 3. Handle Clarification if needed
            if decision.clarification_required:
                clarification_text = decision.clarification_question or (
                    "Could you please clarify your question?"
                )
                yield f"event: token\ndata: {json.dumps({'token': clarification_text}, ensure_ascii=False)}\n\n"
                complete_payload = {
                    "answer": clarification_text,
                    "sources": [],
                    "conversation_id": conversation_id,
                    "generation": {"status": "clarification_requested"},
                    "routing": routing_metadata,
                }
                yield f"event: complete\ndata: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"
                try:
                    conv_store.add_turn(
                        conversation_id=conversation_id,
                        user_id=authenticated_request.user_id,
                        organization=authenticated_request.organization,
                        query=authenticated_request.query,
                        answer=clarification_text,
                        sources=[],
                        generation_metadata={"status": "clarification_requested"},
                        routing_metadata=routing_metadata,
                    )
                except Exception:
                    pass
                generate_job_store.mark_completed(generate_job_id, complete_payload)
                return

            # 4. Execute Retrieval & Web Search
            reused_sources = []
            if decision.reuse_previous_evidence and prev_retrieval:
                reused_sources = prev_retrieval.get("sources", [])

            retrieval_metadata: dict[str, Any] = {}
            retrieval_result = None
            if "internal_retrieval" in decision.actions and (
                not decision.reuse_previous_evidence or decision.additional_retrieval_required
            ):
                internal_q = decision.internal_query or decision.contextualized_query or authenticated_request.query
                try:
                    from external_query_router import InternalQueryRouter
                    with InternalQueryRouter() as internal_router:
                        strategy = internal_router.route(internal_q)
                except Exception:
                    strategy = "hybrid"

                enable_graph = strategy in ("graph", "hybrid")
                enable_visual = strategy in ("visual", "hybrid")

                cache_key = retrieval_cache_key(
                    organization=authenticated_request.organization,
                    user_id=authenticated_request.user_id,
                    role=authenticated_request.role,
                    query=internal_q,
                    document_ids=authenticated_request.document_ids,
                    version_ids=authenticated_request.version_ids,
                    recursive_search=authenticated_request.recursive_search,
                    top_k=authenticated_request.top_k,
                    enable_graph=enable_graph,
                    enable_visual=enable_visual,
                )
                try:
                    retrieval_result = cache_get_json(cache_key)
                except Exception:
                    retrieval_result = None

                if retrieval_result is None:
                    retrieval_result = retrieve(
                        query=internal_q,
                        document_ids=authenticated_request.document_ids,
                        version_ids=authenticated_request.version_ids,
                        recursive_search=authenticated_request.recursive_search,
                        top_k=authenticated_request.top_k,
                        enable_rerank=getattr(authenticated_request, "enable_rerank", False),
                        rerank_threshold=getattr(authenticated_request, "rerank_threshold", 0.75),
                        retrieval_top_k=getattr(authenticated_request, "retrieval_top_k", 20),
                        enable_graph=enable_graph,
                        enable_visual=enable_visual,
                    )
                    try:
                        cache_set_json(cache_key, retrieval_result)
                    except Exception:
                        pass

                retrieval_metadata = {
                    "strategy": strategy,
                    "enable_graph": enable_graph,
                    "enable_visual": enable_visual,
                    "query": internal_q,
                    "result_count": len(retrieval_result.get("results", [])) if retrieval_result else 0,
                }

            # Web Search (Brave)
            web_metadata: dict[str, Any] = {}
            web_evidence_text = ""
            if "web_search" in decision.actions:
                from web_search import get_brave_search_client
                web_q = decision.web_query or decision.contextualized_query or authenticated_request.query
                web_resp = get_brave_search_client().search(web_q, count=5)
                web_evidence_text = web_resp.format_evidence_text()
                web_metadata = {
                    "search_query": web_q,
                    "status": web_resp.status,
                    "result_count": web_resp.result_count,
                    "latency_ms": web_resp.latency_ms,
                    "sources": [{"title": i.title, "url": i.url} for i in web_resp.results],
                }

            # 5. Check Answer cache
            ans_cache_key = answer_cache_key(
                organization=authenticated_request.organization,
                user_id=authenticated_request.user_id,
                role=authenticated_request.role,
                query=decision.contextualized_query,
                retrieval_result=retrieval_result,
                model_id=GENERATOR_MODEL,
            )
            if history:
                ans_cache_key += f":conv:{conversation_id}:turns:{len(history)}"

            try:
                cached_answer = cache_get_json(ans_cache_key)
            except Exception:
                cached_answer = None

            if cached_answer is not None and isinstance(cached_answer, dict):
                cached_answer["conversation_id"] = conversation_id
                answer_text = cached_answer.get("answer", "")
                sources = cached_answer.get("sources", [])
                gen_meta = dict(cached_answer.get("generation", {}))
                gen_meta["cached"] = True

                token_payload = {"token": answer_text}
                yield f"event: token\ndata: {json.dumps(token_payload, ensure_ascii=False)}\n\n"

                complete_data = {
                    "answer": answer_text,
                    "sources": sources,
                    "conversation_id": conversation_id,
                    "generation": gen_meta,
                    "routing": routing_metadata,
                    "retrieval_metadata": retrieval_metadata,
                    "web_metadata": web_metadata,
                    "latency_ms": 1.0,
                    "ttft_ms": 1.0,
                    "cached": True,
                }
                yield f"event: complete\ndata: {json.dumps(complete_data, ensure_ascii=False)}\n\n"

                generate_job_store.mark_completed(generate_job_id, cached_answer)
                save_generate_log(
                    generate_job_id=generate_job_id,
                    request=authenticated_request,
                    status_value="completed",
                    created_at=created_at,
                    started_at=started_at,
                    finished_at=utc_now(),
                    retrieval=retrieval_result,
                    response=cached_answer,
                    error=None,
                )
                return

            # 6. Stream from model
            final_response = None
            rerank_metadata = (
                retrieval_result.get("rerank", {})
                if isinstance(retrieval_result, dict)
                else {}
            )

            # Build aggregated sources
            aggregated_sources = list(reused_sources)
            if retrieval_result and "results" in retrieval_result:
                aggregated_sources.extend(retrieval_result["results"])

            for event in generate_answer_stream(
                query=decision.contextualized_query,
                retrieval_result=retrieval_result or {"results": aggregated_sources},
                conversation_history=history,
                conversation_id=conversation_id,
                web_evidence=web_evidence_text if web_evidence_text else None,
                allow_empty_pages=decision.direct_answer or bool(web_evidence_text),
            ):
                event_type = event.get("type")
                if event_type == "token":
                    token_payload = {"token": event.get("token", "")}
                    yield f"event: token\ndata: {json.dumps(token_payload, ensure_ascii=False)}\n\n"
                elif event_type == "complete":
                    complete_payload = event.get("data", {})
                    if rerank_metadata and "generation" in complete_payload:
                        complete_payload["generation"]["rerank"] = rerank_metadata

                    complete_payload["conversation_id"] = conversation_id
                    complete_payload["routing"] = routing_metadata
                    complete_payload["retrieval_metadata"] = retrieval_metadata
                    complete_payload["web_metadata"] = web_metadata

                    final_response = {
                        "answer": complete_payload.get("answer", ""),
                        "sources": complete_payload.get("sources", []) or aggregated_sources,
                        "conversation_id": conversation_id,
                        "generation": complete_payload.get("generation", {}),
                        "routing": routing_metadata,
                        "retrieval_metadata": retrieval_metadata,
                        "web_metadata": web_metadata,
                    }
                    yield f"event: complete\ndata: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"

            if final_response:
                try:
                    conv_store.add_turn(
                        conversation_id=conversation_id,
                        user_id=authenticated_request.user_id,
                        organization=authenticated_request.organization,
                        query=authenticated_request.query,
                        answer=final_response.get("answer", ""),
                        sources=final_response.get("sources", []),
                        generation_metadata=final_response.get("generation", {}),
                        routing_metadata=routing_metadata,
                        retrieval_metadata=retrieval_metadata,
                        web_metadata=web_metadata,
                    )
                except Exception:
                    logger.warning(
                        "Failed to record turn for streaming conversation %s",
                        conversation_id,
                        exc_info=True,
                    )
                try:
                    if ans_cache_key:
                        cache_set_json(ans_cache_key, final_response)
                except Exception:
                    pass

                generate_job_store.mark_completed(
                    generate_job_id, final_response
                )
                completed_job = generate_job_store.get_job(generate_job_id)
                try:
                    if completed_job:
                        set_job_state(
                            "generate", generate_job_id, completed_job
                        )
                except Exception:
                    pass

                save_generate_log(
                    generate_job_id=generate_job_id,
                    request=authenticated_request,
                    status_value="completed",
                    created_at=created_at,
                    started_at=started_at,
                    finished_at=utc_now(),
                    retrieval=retrieval_result,
                    response=final_response,
                    error=None,
                )

        except Exception as exc:
            logger.exception(
                "Streaming generation failed for job %s", generate_job_id
            )
            generate_job_store.mark_failed(generate_job_id, str(exc))
            failed_job = generate_job_store.get_job(generate_job_id)
            try:
                if failed_job:
                    set_job_state("generate", generate_job_id, failed_job)
            except Exception:
                pass

            save_generate_log(
                generate_job_id=generate_job_id,
                request=authenticated_request,
                status_value="failed",
                created_at=created_at,
                started_at=started_at,
                finished_at=utc_now(),
                retrieval=retrieval_result,
                response=None,
                error=str(exc),
            )
            error_payload = {"error": str(exc)}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/answer/{generate_job_id}",
    summary="Fetch answer generation job by ID",
)
@app.get(
    "/generate/{generate_job_id}",
    summary="Legacy alias for /answer/{generate_job_id}",
    include_in_schema=False,
)
def get_generate_job(
    generate_job_id: str,
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Poll a generation job.

    While running:
        returns only status/job_id.

    When completed:
        returns ONLY the generated result.

    Complete request/retrieval/generation audit data is stored in:
        data/generate/<generate_job_id>.json
    """
    if not GENERATE_JOB_ID_RE.fullmatch(
        generate_job_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid generate_job_id format",
        )

    # shortcut: check fast Redis job state cache first; fall back to SQLite durable store
    cached = None
    try:
        cached = get_job_state("generate", generate_job_id)
    except Exception:
        cached = None

    if cached is not None:
        if cached.get("user_id") != current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job.",
            )
        if cached.get("status") in {"queued", "running"}:
            return {
                "job_id": generate_job_id,
                "status": cached["status"],
            }
        if cached.get("status") == "failed":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=cached.get("error") or "Generation job failed.",
            )
        if cached.get("status") == "completed" and "result" in cached:
            return cached["result"]

    job = generate_job_store.get_job(
        generate_job_id,
    )

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Generation job not found: "
                f"{generate_job_id}"
            ),
        )

    if job.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this job.",
        )

    try:
        set_job_state("generate", generate_job_id, job)
    except Exception:
        pass

    if job["status"] in {"queued", "running"}:
        return {
            "job_id": generate_job_id,
            "status": job["status"],
        }

    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=job["error"] or "Generation job failed.",
        )

    # Completed: ONLY return the actual generated result.
    return job["result"]


# ===========================================================================
# Conversation Management Endpoints
# ===========================================================================

@app.get("/conversations")
def list_user_conversations(
    limit: int = 50,
    offset: int = 0,
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:
    """
    List persistent multi-turn conversations for the authenticated user and organization.
    """
    store = get_conversation_store()
    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, offset)
    conversations = store.list_conversations(
        user_id=current_user.user_id,
        organization=current_user.organization_name,
        limit=bounded_limit,
        offset=bounded_offset,
    )
    total = store.count_conversations(
        user_id=current_user.user_id,
        organization=current_user.organization_name,
    )
    return {
        "conversations": conversations,
        "total": total,
        "limit": bounded_limit,
        "offset": bounded_offset,
    }


@app.get("/conversations/{conversation_id}")
def get_user_conversation(
    conversation_id: str,
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Retrieve full history, turns, queries, answers, and sources for a given conversation ID.
    Enforces tenant and user isolation.
    """
    store = get_conversation_store()
    try:
        cid = validate_conversation_id(conversation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    conv = store.get_conversation(
        cid,
        user_id=current_user.user_id,
        organization=current_user.organization_name,
    )
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return conv.to_dict()


@app.delete("/conversations/{conversation_id}")
def delete_user_conversation(
    conversation_id: str,
    current_user: AuthContext = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Delete a conversation and all its associated Q&A turns.
    Enforces tenant and user isolation.
    """
    store = get_conversation_store()
    try:
        cid = validate_conversation_id(conversation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    deleted = store.delete_conversation(
        cid,
        user_id=current_user.user_id,
        organization=current_user.organization_name,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {
        "status": "deleted",
        "conversation_id": cid,
        "message": f"Conversation '{cid}' deleted successfully.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        workers=1,
    )
