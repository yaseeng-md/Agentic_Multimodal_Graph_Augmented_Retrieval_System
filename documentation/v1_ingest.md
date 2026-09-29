# AMG Multimodal RAG — V1 Ingestion System

## Overview

AMG Multimodal RAG V1 is a **multimodal PDF ingestion system** built around:

- FastAPI for the HTTP API
- SQLite for ingestion job tracking
- SQLite for document/version/page registry
- PyMuPDF for PDF page rendering
- ColPali v1.3 merged for visual page embeddings
- BitsAndBytes 4-bit NF4 quantization
- Qdrant for multimodal vector storage
- Docker-managed Qdrant with persistent storage
- Background ingestion using a process-local `ThreadPoolExecutor`

The API layer is intentionally thin. The existing ingestion pipeline owns PDF processing, duplicate detection, page caching, ColPali embedding, Qdrant storage, and the document registry. The API layer owns the HTTP contract, job creation, polling, source preparation, and job tracking.

---

## Current Capabilities

### 1. Asynchronous PDF ingestion API

The system accepts an ingestion request and immediately returns a `job_id`.

The actual ingestion work happens in the background.

```text
Client
  |
  | POST /ingest_document
  v
FastAPI
  |
  +--> Persist job in SQLite
  |
  +--> Submit background job
  |
  +--> Return 202 + job_id
             |
             v
       Background worker
             |
             v
       PDF ingestion pipeline
```

V1 uses a **single background worker** so only one ingestion workload runs at a time. This prevents multiple ColPali instances from competing for GPU memory.

---

## 2. Input sources

A job can contain:

- Local PDF paths on the same server
- HTTP/HTTPS S3-style PDF links
- Both source types in the same request
- Multiple PDFs in a single request

The current default maximum is:

```text
20 files per request
```

This can be changed using:

```text
API_MAX_FILES_PER_REQUEST
```

---

## 3. Identity metadata

Every ingestion request requires:

```text
user_id
organization
role
```

These values are stored with the job.

### Current V1 behavior

These fields are currently **request metadata**, not an authenticated identity.

For example:

```json
{
  "user_id": "user_001",
  "organization": "terafac",
  "role": "researcher"
}
```

Future authentication should derive trusted identity information instead of relying on caller-supplied values for authorization.

---

## 4. Exact PDF duplicate detection

Before embedding, the pipeline calculates a SHA-256 hash for the PDF.

If the exact file hash already exists in the ingestion registry:

```text
PDF
 |
 +--> SHA-256
       |
       +--> already known
              |
              +--> status = duplicate
              +--> no PDF re-ingestion
              +--> no new ColPali embedding
```

The duplicate result contains the existing:

- `document_id`
- `version_id`
- `version_number`
- `filename`
- `file_hash`

---

## 5. Document and version management

The ingestion registry maintains:

```text
documents
    |
    +-- document_id
    +-- display_name
    +-- current_version_id
    +-- created_at
    +-- updated_at

versions
    |
    +-- version_id
    +-- document_id
    +-- version_number
    +-- file_hash
    +-- original_filename
    +-- stored_pdf_path
    +-- page_count
    +-- status
    +-- timestamps

pages
    |
    +-- version_id
    +-- page_number
    +-- page_hash
    +-- point_id
```

A normal new ingestion creates a new logical document.

The existing CLI also supports explicit version creation:

```bash
python ingest.py --update-doc DOC-123 updated.pdf
```

The current V1 HTTP contract does **not yet expose `update_document_id`**.

---

## 6. Page rendering and page hashing

Every PDF page is rendered to an image using PyMuPDF.

The rendered RGB pixels are SHA-256 hashed.

This gives the system a second level of reuse:

```text
PDF duplicate
    |
    +--> exact PDF already known
           -> skip entire file

New PDF/version
    |
    +--> render pages
    |
    +--> hash each rendered page
    |
    +--> search Qdrant for compatible page embedding
```

---

## 7. Page-level embedding reuse

A page embedding can be reused when the same rendered page content already exists with a compatible embedding signature.

The signature includes values such as:

- ColPali model ID
- model revision
- quantization settings
- compute dtype
- double quantization setting
- PDF rendering DPI

Therefore the system avoids recomputing embeddings when the page content and embedding configuration are compatible.

---

## 8. ColPali visual embeddings

The current ingestion implementation uses:

```text
vidore/colpali-v1.3-merged
```

The model is configured for:

```text
BitsAndBytes
4-bit
NF4
```

The compute dtype is configurable, with the current default:

```text
bfloat16
```

The ColPali model is loaded **once per ingestion batch** and reused for all files in that batch.

This is important for GPU memory usage.

---

## 9. Qdrant vector storage

Embeddings are stored in:

```text
QDRANT_COLLECTION
```

with the default:

```text
personal_collection
```

The Qdrant store uses a multivector representation with:

```text
MAX_SIM
```

and the vector name:

```text
page_embedding
```

Payload indexes are created for frequently filtered fields such as:

```text
document_id
version_id
file_hash
page_hash
embedding_signature
is_current
```

The page payload also contains metadata including:

```text
document_id
version_id
version_number
filename
page_number
page_hash
page_image_path
page dimensions
model information
embedding signature
quantization
embedding type
page text (when enabled)
is_current
```

---

## 10. Current-version handling

When a new version is successfully ingested:

1. The new version is marked as current.
2. Older versions remain stored.
3. Older versions are marked as historical using `is_current=false`.

This allows the system to retain document history while identifying the current version.

---

## 11. Partial failure handling

A multi-file ingestion job does not have to fail completely because one PDF fails.

Example:

```text
10 submitted PDFs

8 -> ingested
1 -> duplicate
1 -> failed
```

The job can finish with:

```text
completed_with_errors
```

The failed file retains its error message, while successful files are still committed.

This behavior is implemented in the ingestion batch loop and in the API job store.

---

## 12. S3 source handling

The API accepts HTTP/HTTPS URLs for remote PDFs.

Before download:

- The URL scheme is validated.
- The hostname is checked.
- By default, AWS/S3-style hosts are accepted.
- `S3_ALLOWED_HOSTS` can be used to restrict accepted hosts.

The download also has:

```text
S3_DOWNLOAD_TIMEOUT
S3_MAX_FILE_SIZE_MB
```

limits.

Downloaded PDFs are placed into a temporary per-job directory.

After ingestion completes, the temporary job directory is removed.

The ingestion pipeline separately stores successful PDFs in the permanent PDF storage configured by `PDF_DIR`.

---

## 13. Automatic Qdrant startup

When the FastAPI application starts:

```text
FastAPI startup
   |
   +--> start_qdrant()
          |
          +--> connect to Docker
          |
          +--> start Docker Desktop if configured
          |
          +--> pull Qdrant image if missing
          |
          +--> create/start Qdrant container
          |
          +--> wait for Qdrant REST API
   |
   +--> recover interrupted jobs
   |
   +--> start ingestion worker
```

The Qdrant container uses a persistent Docker volume.

The container is stopped during API shutdown **only when this API process started it**.

If Qdrant was already running independently, it is left running.

---

# API Contract

## Base URL

Default local URL:

```text
http://localhost:8000
```

---

## `POST /ingest_document`

Creates a new asynchronous ingestion job.

### Request

```json
{
  "user_id": "user_001",
  "organization": "terafac",
  "role": "researcher",
  "s3_links": [
    "https://my-bucket.s3.amazonaws.com/paper1.pdf"
  ],
  "local_pdf_paths": [
    "/mnt/papers/paper2.pdf"
  ]
}
```

### Request fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `user_id` | string | Yes | Caller-provided user identifier |
| `organization` | string | Yes | Organization identifier |
| `role` | string | Yes | Caller-provided role |
| `s3_links` | string[] | No | HTTP/HTTPS remote PDF links |
| `local_pdf_paths` | string[] | No | PDF paths visible to the server |

At least one entry across `s3_links` and `local_pdf_paths` is required.

The current maximum number of files is:

```text
20
```

### Success response

HTTP:

```text
202 Accepted
```

Body:

```json
{
  "job_id": "ingest_20260929183000_a1b2c3d4",
  "status": "accepted",
  "message": "Ingestion job accepted."
}
```

The request does **not** wait for PDF ingestion to finish.

---

## `GET /ingest_document/{job_id}`

Returns the current state of an ingestion job.

Example:

```json
{
  "job_id": "ingest_20260929183000_a1b2c3d4",
  "user_id": "user_001",
  "organization": "terafac",
  "role": "researcher",
  "status": "completed_with_errors",
  "created_at": "2026-09-29T18:30:00+00:00",
  "started_at": "2026-09-29T18:30:01+00:00",
  "finished_at": "2026-09-29T18:31:42+00:00",
  "total_files": 3,
  "processed_files": 3,
  "ingested_count": 1,
  "duplicate_count": 1,
  "failed_count": 1,
  "error": null,
  "results": [
    {
      "source_type": "local",
      "source": "/mnt/papers/paper1.pdf",
      "filename": "paper1.pdf",
      "status": "ingested",
      "document_id": "DOC-ABC123",
      "version_id": "VER-ABC123",
      "version_number": 1,
      "pages": 20,
      "new_embeddings": 18,
      "reused_embeddings": 2,
      "started_at": "...",
      "finished_at": "..."
    },
    {
      "source_type": "local",
      "source": "/mnt/papers/paper2.pdf",
      "filename": "paper2.pdf",
      "status": "duplicate",
      "document_id": "DOC-DEF456",
      "version_id": "VER-DEF456",
      "version_number": 1
    },
    {
      "source_type": "s3",
      "source": "https://...",
      "filename": "paper3.pdf",
      "status": "failed",
      "error": "..."
    }
  ]
}
```

### Job status values

```text
queued
running
completed
completed_with_errors
failed
```

### File status values

```text
queued
running
ingested
duplicate
failed
```

---

## `GET /health`

Current health endpoint:

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

This is currently a lightweight API health check. It does not yet expose detailed dependency health.

---

# Example Usage

## Start the API

Run from the same directory as `api.py` and `ingest.py`:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

Or:

```bash
python api.py
```

V1 intentionally uses:

```text
workers=1
```

because the ingestion executor and GPU-backed ColPali model are process-local.

---

## Submit local PDFs

```bash
curl -X POST http://localhost:8000/ingest_document \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "organization": "terafac",
    "role": "researcher",
    "s3_links": [],
    "local_pdf_paths": [
      "/mnt/papers/paper1.pdf",
      "/mnt/papers/paper2.pdf"
    ]
  }'
```

---

## Submit S3 PDFs

```bash
curl -X POST http://localhost:8000/ingest_document \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "organization": "terafac",
    "role": "researcher",
    "s3_links": [
      "https://my-bucket.s3.amazonaws.com/paper1.pdf"
    ],
    "local_pdf_paths": []
  }'
```

For private S3 objects, the current implementation expects a URL that is directly downloadable by the process, such as a pre-signed URL.

---

## Poll a job

```bash
curl http://localhost:8000/ingest_document/ingest_20260929183000_a1b2c3d4
```

---

# CLI Capability Still Available

The ingestion engine can also be used without the API.

### Ingest multiple PDFs

```bash
python ingest.py --pdf a.pdf b.pdf c.pdf
```

### Create a new document version

```bash
python ingest.py --update-doc DOC-123 updated.pdf
```

### Mix normal ingestion and document updates

```bash
python ingest.py \
  --pdf a.pdf b.pdf \
  --update-doc DOC-123 c.pdf
```

### Dummy ingestion test

```bash
python ingest.py --dummy-test
```

### Reset the ingestion registry

```bash
python ingest.py --reset-registry
```

Resetting the registry does **not** delete Qdrant data.

---

# Current Data / Storage Layout

The exact paths are configurable through environment variables.

The default logical layout is:

```text
data/
├── pdfs/
│   └── <document_id>/
│       └── v<version>_<filename>.pdf
│
├── pages/
│   └── <document_id>/
│       └── <version_id>/
│           ├── page_0001.png
│           ├── page_0002.png
│           └── ...
│
├── ingestion_registry.db
│
├── ingestion_jobs.db
│
└── api_jobs/
    └── <job_id>/
        └── inputs/
            └── downloaded remote PDFs
```

Temporary API job input directories are removed after the job finishes.

---

# Main Configuration

## Ingestion

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_COLLECTION` | `personal_collection` | Qdrant collection |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `QDRANT_API_KEY` | empty | Optional Qdrant API key |
| `COLPALI_MODEL` | `vidore/colpali-v1.3-merged` | ColPali model |
| `COLPALI_MODEL_REVISION` | `main` | Hugging Face revision |
| `QUANTIZATION` | `4bit` | Quantization mode |
| `BNB_4BIT_QUANT_TYPE` | `nf4` | BitsAndBytes quantization |
| `BNB_4BIT_COMPUTE_DTYPE` | `bfloat16` | Compute dtype |
| `BNB_4BIT_USE_DOUBLE_QUANT` | `true` | Double quantization |
| `COLPALI_BATCH_SIZE` | `1` | New-page embedding batch size |
| `QDRANT_UPSERT_BATCH_SIZE` | `8` | Qdrant upsert batch size |
| `PDF_DPI` | `150` | Page rendering DPI |
| `STORE_PAGE_TEXT` | `true` | Store extracted page text in payload |
| `DEVICE` | `auto` | Compute device |
| `REGISTRY_DB_PATH` | `./data/ingestion_registry.db` | Document registry |

## API

| Variable | Default | Purpose |
|---|---|---|
| `API_PORT` | `8000` | API port |
| `API_JOB_DB_PATH` | `./data/ingestion_jobs.db` | Job DB |
| `API_JOB_DATA_DIR` | `./data/api_jobs` | Temporary API job data |
| `API_MAX_FILES_PER_REQUEST` | `20` | Request file limit |
| `S3_DOWNLOAD_TIMEOUT` | `120` | Remote download timeout |
| `S3_MAX_FILE_SIZE_MB` | `512` | Remote file size limit |
| `S3_ALLOWED_HOSTS` | empty | Optional S3 hostname allow-list |

## Qdrant / Docker

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_CONTAINER_NAME` | `amg-qdrant` | Container name |
| `QDRANT_IMAGE` | `qdrant/qdrant:latest` | Qdrant image |
| `QDRANT_STORAGE_VOLUME` | `amg_qdrant_storage` | Persistent Docker volume |
| `QDRANT_HTTP_HOST` | `127.0.0.1` | Qdrant HTTP bind host |
| `QDRANT_HTTP_PORT` | `6333` | Qdrant HTTP port |
| `QDRANT_GRPC_HOST` | `127.0.0.1` | Qdrant gRPC bind host |
| `QDRANT_GRPC_HOST_PORT` | `6334` | Qdrant gRPC host port |
| `QDRANT_DOCKER_START_TIMEOUT` | `120` | Docker readiness timeout |
| `QDRANT_API_READY_TIMEOUT` | `60` | Qdrant readiness timeout |
| `QDRANT_START_DOCKER_DESKTOP` | `true` | Auto-start Docker Desktop when configured |

---

# Current Architecture

```text
                         ┌─────────────────────┐
                         │      Client         │
                         └──────────┬──────────┘
                                    │
                                    │ POST /ingest_document
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │   HTTP Contract     │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
             ┌───────────────┐             ┌───────────────┐
             │ Job SQLite DB │             │ Background     │
             │ ingestion_jobs│             │ Executor       │
             └───────────────┘             └───────┬───────┘
                                                    │
                                                    ▼
                                         ┌───────────────────┐
                                         │ Source Resolution │
                                         │ Local / S3        │
                                         └─────────┬─────────┘
                                                   │
                                                   ▼
                                         ┌───────────────────┐
                                         │   ingest_files()  │
                                         └─────────┬─────────┘
                                                   │
                           ┌───────────────────────┼───────────────────────┐
                           │                       │                       │
                           ▼                       ▼                       ▼
                    SHA-256 duplicate       Page rendering          ColPali NF4
                       detection             + page hash             embeddings
                           │                       │                       │
                           └───────────────────────┼───────────────────────┘
                                                   │
                                                   ▼
                                          ┌───────────────────┐
                                          │      Qdrant       │
                                          │   Multivectors    │
                                          └───────────────────┘

                                          ┌───────────────────┐
                                          │ Registry SQLite   │
                                          │ Docs / Versions   │
                                          │ Pages / Hashes    │
                                          └───────────────────┘
```

---

# Current V1 Limitations

The current system is intentionally focused on ingestion.

### 1. No retrieval/query API yet

The system can create and store multimodal embeddings, but there is currently no HTTP endpoint for:

```text
query -> Qdrant search -> relevant pages -> results
```

### 2. No authentication/authorization

The request accepts `user_id`, `organization`, and `role`, but the API does not yet authenticate the caller.

These fields should not be treated as trusted authorization claims.

### 3. API does not expose document update/versioning

The CLI supports:

```bash
--update-doc DOC_ID PDF
```

but the HTTP contract currently always creates normal ingestion jobs without an `update_document_id` field.

### 4. Single-worker execution

The API intentionally runs with one Uvicorn worker and one ingestion executor.

This protects the GPU but limits throughput.

### 5. In-memory process executor

The job metadata is persisted in SQLite, but the execution queue itself is process-local.

After an API process restart, queued/running jobs are marked failed instead of being automatically resumed.

### 6. Basic health endpoint

`/health` currently returns API health only.

It does not yet report:

- Qdrant health
- GPU availability
- model status
- worker status

### 7. No cancellation / retry endpoint

A running job cannot currently be:

```text
cancelled
retried
paused
resumed
```

through the API.

### 8. No progress at page-level

Polling currently reports file-level progress.

It does not expose detailed page-level embedding progress such as:

```text
page 34 / 120
```

---

# What Can Be Done Next

The recommended next development stages are:

## Phase 1 — Retrieval API

Add the actual RAG query layer.

Example:

```text
POST /query
```

Possible contract:

```json
{
  "user_id": "user_001",
  "organization": "terafac",
  "role": "researcher",
  "query": "What is the recommended welding procedure?",
  "document_ids": [],
  "top_k": 5
}
```

The query service would:

```text
Query
  ↓
ColPali / query embedding
  ↓
Qdrant
  ↓
Top-K relevant pages
  ↓
Metadata + page images
  ↓
RAG answer layer
```

This is the biggest missing piece before the system becomes an end-to-end RAG service.

---

## Phase 2 — Document management API

Add endpoints such as:

```text
GET    /documents
GET    /documents/{document_id}
GET    /documents/{document_id}/versions
POST   /documents/{document_id}/versions
```

This would expose the existing registry capabilities through HTTP.

---

## Phase 3 — Better identity and authorization

Introduce authentication and derive:

```text
user_id
organization
role
```

from the authenticated principal.

Then add authorization rules such as:

```text
organization -> accessible documents
role         -> allowed operations
user         -> ownership / permissions
```

---

## Phase 4 — Persistent job queue

Replace the process-local executor with a persistent queue.

Possible choices:

```text
Redis + Celery
Redis + RQ
RabbitMQ
AWS SQS
```

This would allow:

- multiple workers
- retries
- job leasing
- cancellation
- better failure recovery
- horizontal scaling

---

## Phase 5 — Better document ingestion lifecycle

Add:

```text
POST /documents/{id}/versions
POST /jobs/{id}/cancel
POST /jobs/{id}/retry
DELETE /documents/{id}
```

and define document lifecycle semantics.

---

## Phase 6 — Observability

Add:

- structured logs
- job duration
- pages processed
- embedding latency
- Qdrant latency
- GPU memory metrics
- failure classification
- ingestion throughput

This becomes especially important once ingestion volume grows.

---

## Phase 7 — Production storage

The current implementation uses local filesystem storage and SQLite.

For a multi-machine deployment, move:

```text
PDF files       -> object storage
page images     -> object storage
registry        -> PostgreSQL
job queue       -> persistent queue
Qdrant          -> managed / dedicated deployment
```

---

## Phase 8 — Multimodal RAG answer generation

Once retrieval is available, add the final generation layer:

```text
User question
      ↓
Multimodal query embedding
      ↓
Qdrant retrieval
      ↓
Relevant PDF pages
      ↓
Page images + text / metadata
      ↓
Vision-language / multimodal LLM
      ↓
Answer + citations
```

The final response should ideally include page-level citations such as:

```text
Document: welding_manual_v2.pdf
Page: 47
Document Version: VER-ABC123
```

---

# V1 Definition of Done

The current V1 ingestion service is considered complete when it can:

- Accept one or many PDF sources through HTTP.
- Accept both local paths and remote S3-style links.
- Return a `job_id` immediately.
- Persist job state in SQLite.
- Allow clients to poll job status.
- Start and manage Qdrant automatically.
- Process PDFs in the existing ColPali ingestion pipeline.
- Avoid re-ingesting exact PDF duplicates.
- Reuse compatible page embeddings.
- Store document/version/page metadata.
- Handle partial batch failures.
- Keep historical versions in the registry/Qdrant.
- Clean up temporary remote-download files.
- Run with one GPU-safe ingestion worker.

---

# Project Responsibility Split

```text
api.py
    HTTP contract
    request validation
    job creation
    job polling
    source preparation
    background execution
    API lifecycle

ingest.py
    PDF hashing
    PDF rendering
    page hashing
    ColPali embedding
    embedding reuse
    document/version ingestion
    Qdrant upsert orchestration

registry.py
    document registry
    version registry
    page registry

qdrant_store.py
    Qdrant collection
    payload indexes
    vector upserts
    cached vector lookup
    current-version payload management

docker_manager.py
    Docker availability
    Qdrant container creation/startup
    Qdrant readiness
    Qdrant shutdown

Qdrant
    persistent multimodal vector storage
```

---

# Summary

AMG V1 currently provides a **working asynchronous ingestion foundation** rather than the complete RAG application.

The current flow is:

```text
PDF sources
   ↓
FastAPI ingestion job
   ↓
Background processing
   ↓
SHA-256 duplicate detection
   ↓
PDF rendering
   ↓
Page hashing
   ↓
Page embedding reuse
   ↓
ColPali v1.3 NF4 embeddings
   ↓
Qdrant multivectors
   ↓
Document/version/page registry
```

The next major capability should be the **retrieval/query API**, because ingestion and vector storage are now in place but the system does not yet expose a user-facing search/RAG path.
