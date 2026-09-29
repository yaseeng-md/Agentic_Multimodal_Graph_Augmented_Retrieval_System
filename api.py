"""
AMG Multimodal RAG - V1 ingestion API
======================================

POST   /ingest_document
GET    /ingest_document/{job_id}

The API is intentionally thin: the existing ingest.py owns PDF ingestion,
ColPali, duplicate detection, page caching, Qdrant, and the registry.
This module owns the HTTP contract and asynchronous job tracking.

Run from the same project directory as ingest.py:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

Use ONE Uvicorn worker in V1 because the ingestion pipeline owns a GPU-backed
ColPali model and the background executor is process-local.
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
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from ingest import DATA_DIR, IngestJob, ingest_files

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

JOB_ID_RE = re.compile(r"^ingest_[a-zA-Z0-9_-]+$")

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
        # The existing ingestion pipeline has already copied the PDF into its
        # permanent PDF_DIR before returning success. Temporary downloads can
        # therefore be removed after the job finishes.
        shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _executor
    job_store.recover_running_jobs()
    _executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="ingestion",
    )
    logger.info("AMG ingestion API started")
    yield
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=False)
    _executor = None
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
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/ingest_document",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_ingestion_job(request: IngestRequest) -> AcceptedResponse:
    job_id = new_job_id()

    try:
        # Persist only the original sources here. Local path resolution and S3
        # download happen in the background so the POST returns immediately.
        files = [
            {
                "source_type": "s3",
                "source": url,
                "filename": Path(urlparse(url).path).name or None,
            }
            for url in request.s3_links
        ]
        files.extend(
            {
                "source_type": "local",
                "source": path,
                "filename": Path(path).name,
            }
            for path in request.local_pdf_paths
        )

        job_store.create_job(
            job_id=job_id,
            user_id=request.user_id,
            organization=request.organization,
            role=request.role,
            total_files=len(files),
            files=files,
        )

        executor = get_executor()
        executor.submit(_run_job, job_id, request)

    except Exception as exc:
        logger.exception("Could not create ingestion job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create ingestion job: {exc}",
        ) from exc

    return AcceptedResponse(
        job_id=job_id,
        status="accepted",
        message="Ingestion job accepted.",
    )


@app.get("/ingest_document/{job_id}")
def get_ingestion_job(job_id: str) -> dict[str, Any]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job_id format",
        )

    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return job


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        workers=1,
    )
