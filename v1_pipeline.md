# AMG Multimodal RAG V1 — API, Contracts, Flows & System Reference

> **Source of truth:** this document is derived from the uploaded V1 implementation files:
> `api.py`, `ingest.py`, `query.py`, `generate.py`, `qdrant_store.py`, `registry.py`, and `docker_manager.py`.
>
> The document describes the behavior implemented by those files, including endpoint contracts, asynchronous job handling, storage, model configuration, retrieval scope rules, generation behavior, versioning, duplicate detection, and Qdrant structure.

---

## 1. System Overview

AMG V1 is a multimodal PDF RAG system with four main stages:

```text
                    ┌─────────────────────────────┐
                    │          FastAPI            │
                    │          api.py             │
                    └─────────────┬───────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
       /ingest_document        /query             /generate
             │                    │                    │
             ▼                    ▼                    ▼
       Ingestion Pipeline    Retrieval Pipeline   Retrieval
             │                    │                    │
             ▼                    ▼                    ▼
       PDF → Pages →         ColPali query       ColPali query
       ColPali → Qdrant      embedding           embedding
             │                    │                    │
             │                    ▼                    ▼
             │               Qdrant MaxSim       Qdrant MaxSim
             │                    │                    │
             │                    ▼                    ▼
             │              ranked pages       ranked pages
             │                                         │
             │                                         ▼
             │                                  Qwen3-VL generation
             │                                         │
             ▼                                         ▼
        Registry + Qdrant                       final answer
```

### Primary responsibilities

| Module | Responsibility |
|---|---|
| `api.py` | FastAPI endpoints, request validation, job orchestration, polling, logging, lifecycle |
| `ingest.py` | PDF hashing, versioning, page rendering, page hashing, ColPali embeddings, Qdrant upsert |
| `query.py` | ColPali query encoding, retrieval scope resolution, Qdrant MaxSim search |
| `generate.py` | Retrieved-page context construction and Qwen3-VL answer generation |
| `qdrant_store.py` | Qdrant collection creation, payload indexes, vector reuse, current-version flags |
| `registry.py` | SQLite registry for documents, versions, pages, hashes and current version |
| `docker_manager.py` | Docker/Qdrant startup, readiness checks, persistent volume, shutdown behavior |

---

# 2. API Endpoint Inventory

Base URL:

```text
http://localhost:8000
```

## Endpoint table

| Method | Endpoint | Purpose | Execution |
|---|---|---|---|
| `GET` | `/health` | API health check | Immediate |
| `GET` | `/list_documents` | List documents and all known versions | Immediate |
| `POST` | `/ingest_document` | Submit one or more PDFs for ingestion | Async |
| `GET` | `/ingest_document/{job_id}` | Poll ingestion job | Immediate |
| `POST` | `/query` | Submit a retrieval query | Async |
| `GET` | `/query/{query_job_id}` | Poll retrieval job | Immediate |
| `POST` | `/generate` | Retrieve + generate answer | Synchronous HTTP response, internally job-backed |
| `GET` | `/generate/{generate_job_id}` | Poll an internally stored generation job | Immediate |

---

# 3. Common Identity Contract

The ingestion, query, and generation request contracts use these identity fields:

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer"
}
```

Rules:

- All three fields are required.
- They must be non-empty strings.
- Leading/trailing whitespace is stripped.
- They are persisted into the corresponding job store.

---

# 4. `GET /health`

## Purpose

Basic service liveness check.

## Request

```bash
curl http://localhost:8000/health
```

## Response

```json
{
  "status": "ok"
}
```

---

# 5. `GET /list_documents`

## Purpose

Returns every known document/version pair from the local registry.

This endpoint is useful for:

- discovering `document_id`
- discovering `version_id`
- determining the current version
- selecting a specific version for retrieval

## Request

```bash
curl http://localhost:8000/list_documents
```

## Response contract

```json
{
  "documents": [
    {
      "filename": "manual.pdf",
      "document_id": "DOC-ABC123...",
      "version_id": "VER-ABC123...",
      "version_number": 1,
      "is_current": true
    }
  ]
}
```

The endpoint returns one entry for every document/version pair.

---

# 6. `POST /ingest_document`

## Purpose

Accept local PDF paths and/or S3 URLs and create an asynchronous ingestion job.

The endpoint returns immediately after the job is persisted and submitted to the executor.

## Request contract

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "s3_links": [],
  "local_pdf_paths": [
    "/data/manual.pdf"
  ]
}
```

### Fields

| Field | Type | Required | Default | Rules |
|---|---|---:|---|---|
| `user_id` | string | yes | — | non-empty |
| `organization` | string | yes | — | non-empty |
| `role` | string | yes | — | non-empty |
| `s3_links` | `string[]` | no | `[]` | HTTP/HTTPS URLs |
| `local_pdf_paths` | `string[]` | no | `[]` | values must end in `.pdf` |

### Source rules

At least one source must be provided:

```text
len(s3_links) + len(local_pdf_paths) > 0
```

The API rejects requests above:

```text
API_MAX_FILES_PER_REQUEST
default = 20
```

So the default maximum is 20 files per ingestion request.

## S3 validation

By default, S3 URLs must use:

```text
s3.amazonaws.com
*.amazonaws.com
```

An explicit allow-list can be configured through:

```text
S3_ALLOWED_HOSTS
```

Remote PDF download also enforces:

```text
S3_DOWNLOAD_TIMEOUT = 120 seconds
S3_MAX_FILE_SIZE_MB = 512 MB
```

## Accepted response

HTTP status:

```text
202 Accepted
```

Response:

```json
{
  "job_id": "ingest_20261002113000_ab12cd34",
  "status": "accepted",
  "message": "Ingestion job accepted."
}
```

The ingestion result is obtained through:

```text
GET /ingest_document/{job_id}
```

---

# 7. Ingestion Job Lifecycle

```text
POST /ingest_document
        │
        ▼
create job in SQLite
        │
        ▼
status = queued
        │
        ▼
ThreadPoolExecutor
        │
        ▼
status = running
        │
        ├── S3 download if required
        │
        ├── validate PDF
        │
        ├── SHA-256 duplicate check
        │
        ├── create/reuse document + version
        │
        ├── render PDF pages
        │
        ├── calculate page hashes
        │
        ├── reuse compatible cached page embeddings
        │
        ├── generate new ColPali embeddings
        │
        ├── upsert page vectors to Qdrant
        │
        ├── mark version current
        │
        └── update registry
        │
        ▼
completed / failed
```

---

# 8. `GET /ingest_document/{job_id}`

## Purpose

Poll an ingestion job and retrieve per-file results.

## Request

```bash
curl http://localhost:8000/ingest_document/ingest_20261002113000_ab12cd34
```

## Response contract

```json
{
  "job_id": "ingest_20261002113000_ab12cd34",
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "status": "completed",
  "created_at": "2026-10-02T06:00:00+00:00",
  "started_at": "2026-10-02T06:00:01+00:00",
  "finished_at": "2026-10-02T06:01:25+00:00",
  "total_files": 2,
  "processed_files": 2,
  "ingested_count": 1,
  "duplicate_count": 1,
  "failed_count": 0,
  "error": null,
  "results": [
    {
      "source_type": "local",
      "source": "/data/manual.pdf",
      "filename": "manual.pdf",
      "status": "ingested",
      "document_id": "DOC-...",
      "version_id": "VER-...",
      "version_number": 1,
      "pages": 12,
      "new_embeddings": 12,
      "reused_embeddings": 0,
      "started_at": "...",
      "finished_at": "..."
    },
    {
      "source_type": "local",
      "source": "/data/manual.pdf",
      "filename": "manual.pdf",
      "status": "duplicate",
      "document_id": "DOC-...",
      "version_id": "VER-...",
      "version_number": 1
    }
  ]
}
```

## Job statuses

Implemented job states include:

```text
queued
running
completed
failed
```

Per-file ingestion statuses include:

```text
queued
running
ingested
duplicate
failed
```

### Important V1 behavior

A failed file does **not** abort the remaining files in the same ingestion batch.

The pipeline catches a file-level exception, records that file as `failed`, and continues processing the remaining files.

---

# 9. Ingestion Data Flow

## 9.1 Exact file duplicate detection

The whole PDF is hashed with SHA-256.

```text
PDF
 │
 ▼
SHA-256(file)
 │
 ▼
registry.versions.file_hash
```

If the hash already exists:

```text
status = duplicate
```

and ColPali processing is skipped for that file.

---

## 9.2 New document

A normal first upload creates:

```text
document_id
version_id
version_number = 1
```

Example:

```text
DOC-ABCD123456
VER-ABCD12345678
version_number = 1
```

---

## 9.3 Explicit new version

To create a new version, the ingestion job must explicitly contain:

```text
update_document_id
```

The next version number is:

```text
previous_max_version + 1
```

For example:

```text
DOC-123
 ├── VER-AAA      version 1
 ├── VER-BBB      version 2
 └── VER-CCC      version 3   <-- current
```

An exact whole-file hash duplicate is always treated as a duplicate even if the caller attempts to ingest it again.

---

# 10. Page-Level Incremental Reuse

A new version does not necessarily require re-embedding every page.

Each rendered page gets a content hash:

```text
page image pixels
      │
      ▼
SHA-256(page pixels)
      │
      ▼
page_hash
```

A cached vector is reusable when both are compatible:

```text
page_hash
+
embedding_signature
```

The embedding signature includes:

- ColPali model ID
- model revision
- quantization setting
- BitsAndBytes quantization type
- compute dtype
- double-quantization setting
- PDF rendering DPI

Therefore:

```text
same page content
+
same embedding configuration
=
reuse vector
```

---

# 11. PDF Rendering

Each PDF page is rendered with PyMuPDF.

Default:

```text
PDF_DPI = 150
```

Rendered files are written under:

```text
data/pages/<document_id>/<version_id>/
```

Page image names:

```text
page_0001.png
page_0002.png
...
```

Optional extracted page text is stored according to:

```text
STORE_PAGE_TEXT=true
```

Default:

```text
true
```

---

# 12. Ingestion Model

The ingestion pipeline uses:

```text
vidore/colpali-v1.3-merged
```

Default environment configuration:

```text
COLPALI_MODEL=vidore/colpali-v1.3-merged
COLPALI_MODEL_REVISION=main
QUANTIZATION=4bit
BNB_4BIT_QUANT_TYPE=nf4
BNB_4BIT_COMPUTE_DTYPE=bfloat16
BNB_4BIT_USE_DOUBLE_QUANT=true
```

The implementation is explicitly configured for 4-bit NF4.

---

# 13. Ingestion Batching and GPU Behavior

When multiple files are submitted in one request:

```text
PDF A ─┐
PDF B ─┼──► one shared ColPali model
PDF C ─┘
```

The model is loaded once and reused across the batch.

Files are processed sequentially.

The embedding batch size is configurable with:

```text
COLPALI_BATCH_SIZE
default = 1
```

Qdrant upsert batch size:

```text
QDRANT_UPSERT_BATCH_SIZE
default = 8
```

---

# 14. `POST /query`

## Purpose

Run retrieval only.

This endpoint does **not** generate an answer.

It creates an asynchronous query job and returns a query job ID.

## Request contract

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What is the welding tolerance?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

## Field contract

| Field | Type | Required | Default |
|---|---|---:|---|
| `user_id` | string | yes | — |
| `organization` | string | yes | — |
| `role` | string | yes | — |
| `query` | string | yes | — |
| `document_ids` | `string[]` | no | `[]` |
| `version_ids` | `string[]` | no | `[]` |
| `recursive_search` | boolean | no | `false` |
| `top_k` | integer | no | `5` |

`top_k` must be:

```text
>= 1
```

IDs are stripped and duplicate IDs are removed while preserving order.

## Accepted response

HTTP:

```text
202 Accepted
```

Response:

```json
{
  "job_id": "query_20261002113000_ab12cd34",
  "status": "accepted",
  "message": "Query job accepted."
}
```

---

# 15. Retrieval Scope Rules

This is one of the most important V1 contracts.

## Case A — No document IDs and no version IDs

Request:

```json
{
  "query": "What is the tolerance?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false
}
```

Scope:

```text
latest_versions
```

Meaning:

```text
all documents
+
current/latest version only
```

---

## Case B — `document_ids` supplied

```json
{
  "document_ids": ["DOC-123", "DOC-456"]
}
```

Scope:

```text
specified_documents
```

Searches only those documents' current versions.

---

## Case C — `version_ids` supplied

```json
{
  "version_ids": ["VER-123"]
}
```

Scope:

```text
specified_versions
```

Searches exactly the requested versions.

Explicit `version_ids` take precedence over `document_ids` unless `recursive_search=true`.

---

## Case D — `recursive_search=true`

```json
{
  "recursive_search": true
}
```

Scope:

```text
all_documents_all_versions
```

This ignores:

```text
document_ids
version_ids
```

and searches every document/version.

---

# 16. Query Retrieval Flow

```text
User query
   │
   ▼
QueryRequest validation
   │
   ▼
resolve_retrieval_scope()
   │
   ▼
ColPali query encoder
   │
   ▼
query multivector
   │
   ▼
Qdrant page_embedding
   │
   ▼
MaxSim multivector search
   │
   ▼
top-k ranked pages
   │
   ▼
JSON retrieval response
```

---

# 17. Retrieval Response Contract

The `query.py::retrieve()` function returns:

```json
{
  "query": "What is the tolerance?",
  "top_k": 5,
  "scope": {
    "mode": "latest_versions",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false
  },
  "result_count": 2,
  "results": [
    {
      "rank": 1,
      "score": 0.91,
      "point_id": "...",
      "document_id": "DOC-...",
      "version_id": "VER-...",
      "version_number": 2,
      "filename": "manual.pdf",
      "page_number": 12,
      "page_hash": "...",
      "page_image_path": "data/pages/...",
      "page_text": "...",
      "is_current": true,
      "payload": {}
    }
  ]
}
```

Each ranked result contains page/document/version identity plus the stored Qdrant payload.

---

# 18. `GET /query/{query_job_id}`

## Purpose

Poll an asynchronous retrieval job.

## Request

```bash
curl http://localhost:8000/query/query_20261002113000_ab12cd34
```

## Running response

```json
{
  "query_job_id": "query_20261002113000_ab12cd34",
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What is the welding tolerance?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5,
  "status": "running",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "result": null,
  "error": null
}
```

## Completed response

The same job structure contains:

```json
{
  "status": "completed",
  "result": {
    "query": "...",
    "top_k": 5,
    "scope": {},
    "result_count": 5,
    "results": []
  },
  "error": null
}
```

## Failed response state

```json
{
  "status": "failed",
  "error": "..."
}
```

---

# 19. Query Audit Logging

Every query is persisted to:

```text
data/query/<query_job_id>.json
```

The JSON audit record includes:

- query job metadata
- exact request
- retrieval scope
- result count
- execution/model metadata
- returned result
- timing
- error information

The file is written atomically:

```text
temporary file
    ↓
replace final JSON
```

This makes the query result useful for debugging, auditing and fault recovery.

---

# 20. `POST /generate`

## Purpose

Run the complete:

```text
retrieval → answer generation
```

pipeline.

Important:

> The HTTP contract intentionally matches `POST /query`.

So the request body is the same `QueryRequest`.

## Request

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "Explain the welding procedure.",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

## Execution model

Unlike `/query` and `/ingest_document`, the `/generate` HTTP request waits for the generation job to complete.

Internally:

```text
create generation job
        │
        ▼
submit to executor
        │
        ▼
wait for future.result()
        │
        ▼
retrieve()
        │
        ▼
generate_answer()
        │
        ▼
store result
        │
        ▼
return only generated response
```

The internal `generate_job_id` is still persisted for debugging and audit logging.

---

# 21. `POST /generate` Response Contract

Successful response:

```json
{
  "answer": "The welding tolerance is ...",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.91,
      "document_id": "DOC-...",
      "version_id": "VER-...",
      "version_number": 2,
      "filename": "manual.pdf",
      "page_number": 12,
      "page_hash": "...",
      "page_image_path": "data/pages/..."
    }
  ],
  "generation": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "prequantized": false,
    "quantization": "bitsandbytes_4bit_nf4",
    "compute_dtype": "bfloat16",
    "context_pages": 2,
    "images_used": 1,
    "min_pixels": 262144,
    "max_pixels": 524288,
    "max_new_tokens": 256,
    "duration_ms": 1234.56,
    "retrieval_scope": {},
    "retrieval_result_count": 5
  }
}
```

### Important response rule

`POST /generate` returns:

```text
GenerationResponse
```

not:

```text
{
  "job_id": ...
}
```

The job ID is internal and persisted in SQLite + JSON logs.

---

# 22. Generation Source Contract

Each generation source contains:

```text
source_id
rank
score
document_id
version_id
version_number
filename
page_number
page_hash
page_image_path
```

The generation prompt refers to retrieved pages as:

```text
[S1]
[S2]
[S3]
...
```

The generation instructions explicitly tell Qwen to use only the supplied retrieved sources.

---

# 23. Generation Context Construction

Retrieval may return up to `top_k` results, but generation applies its own context budget.

Defaults:

```text
GENERATOR_CONTEXT_TOP_K = 2
GENERATOR_MAX_IMAGES = 1
GENERATOR_INCLUDE_IMAGES = true
GENERATOR_MAX_PAGE_TEXT_CHARS = 5000
GENERATOR_MAX_TOTAL_CONTEXT_CHARS = 12000
```

Therefore, by default:

```text
retrieval top_k = 5
        ↓
generation context = at most 2 usable pages
```

A page is usable when it has either:

- extracted page text, or
- an available page image

The generator includes at most one image by default.

---

# 24. Qwen Generation Model

Default model:

```text
Qwen/Qwen3-VL-2B-Instruct
```

Configurable through:

```text
GENERATOR_MODEL
GENERATOR_MODEL_PATH
GENERATOR_MODEL_REVISION
```

Default revision:

```text
main
```

---

# 25. Qwen Quantization Configuration

Default:

```text
GENERATOR_USE_4BIT=true
GENERATOR_PREQUANTIZED=false
GENERATOR_BNB_4BIT_QUANT_TYPE=nf4
GENERATOR_BNB_4BIT_DOUBLE_QUANT=true
GENERATOR_BNB_4BIT_COMPUTE_DTYPE=bfloat16
```

Attention implementation:

```text
GENERATOR_ATTN_IMPLEMENTATION=sdpa
```

---

# 26. Generation Parameters

Defaults:

```text
GENERATOR_MAX_NEW_TOKENS=256
GENERATOR_TEMPERATURE=0.1
GENERATOR_TOP_P=0.9
```

The model uses sampling when:

```text
temperature > 0
```

Otherwise it uses deterministic generation.

---

# 27. GPU / Model Lifecycle

The generation pipeline is explicitly designed for a constrained GPU.

Default lifecycle:

```text
ColPali retrieval model
        │
        ▼
release ColPali
        │
        ▼
cleanup / empty CUDA cache
        │
        ▼
load Qwen3-VL
        │
        ▼
generate
        │
        ▼
release Qwen
        │
        ▼
cleanup / empty CUDA cache
```

Relevant settings:

```text
GENERATOR_RELEASE_RETRIEVER=true
GENERATOR_RELEASE_AFTER_GENERATION=true
GENERATOR_GC_ON_MODEL_SWITCH=true
GENERATOR_EMPTY_CUDA_CACHE=true
GENERATOR_SERIALIZE=true
```

---

# 28. `GET /generate/{generate_job_id}`

Although `/generate` waits synchronously, the generation job is persisted and can be inspected/polled.

## Running

```json
{
  "job_id": "generate_20261002113000_ab12cd34",
  "status": "running"
}
```

## Completed

The endpoint returns only the actual generation result:

```json
{
  "answer": "...",
  "sources": [],
  "generation": {}
}
```

## Failed

HTTP 500 is returned with the stored job error.

---

# 29. Generation Audit Logging

Every generation job is stored at:

```text
data/generate/<generate_job_id>.json
```

The log contains:

```text
job
request
retrieval
execution
response
error
```

The job section includes:

```text
generate_job_id
status
created_at
started_at
finished_at
duration_ms
```

The request section stores:

```text
user_id
organization
role
query
document_ids
version_ids
recursive_search
top_k
```

The retrieval section stores the retrieval result.

The execution section stores generation metadata.

The response section stores the final response.

---

# 30. Qdrant Architecture

Default collection:

```text
personal_collection
```

Default connection:

```text
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_PORT=6334
```

Vector name:

```text
page_embedding
```

---

# 31. Qdrant Vector Configuration

The collection is created with:

```text
distance = COSINE
multivector comparator = MAX_SIM
HNSW m = 0
```

Conceptually:

```text
PDF page
   ↓
ColPali
   ↓
[vector_1, vector_2, ..., vector_n]
   ↓
Qdrant multivector
   ↓
MaxSim query scoring
```

---

# 32. Qdrant Payload Indexes

The following payload fields are indexed:

```text
document_id
version_id
file_hash
page_hash
embedding_signature
is_current
```

`is_current` uses a boolean payload type.

The ID-like fields use keyword indexes.

---

# 33. Qdrant Page Payload

Ingested page points contain metadata including:

```json
{
  "document_id": "...",
  "version_id": "...",
  "version_number": 1,
  "file_hash": "...",
  "filename": "manual.pdf",
  "page_number": 1,
  "page_hash": "...",
  "page_image_path": "...",
  "page_width": 595.0,
  "page_height": 842.0,
  "model_id": "vidore/colpali-v1.3-merged",
  "model_revision": "main",
  "embedding_signature": "...",
  "quantization": "bitsandbytes_4bit_nf4",
  "embedding_type": "colpali_multivector",
  "embedding_shape": [ ... ],
  "is_current": true,
  "page_text": "..."
}
```

`page_text` is included when:

```text
STORE_PAGE_TEXT=true
```

---

# 34. Stable Qdrant Point IDs

Each page point ID is deterministic for:

```text
document_id
+
version_id
+
page_number
```

The implementation uses UUID5.

Therefore a specific document-version-page combination maps to a stable point ID.

---

# 35. Current Version Handling in Qdrant

When a new version is completed:

```text
new version
    ↓
mark is_current = true
    ↓
demote previous current version
    ↓
new version becomes retrieval default
```

Historical versions remain in Qdrant.

This is what makes the following two modes possible:

```text
normal query
    → current/latest versions

recursive_search=true
    → all versions
```

---

# 36. Registry Database

The registry is a SQLite database.

Default:

```text
data/ingestion_registry.db
```

The registry has three core tables.

## `documents`

```text
document_id
display_name
current_version_id
created_at
updated_at
```

## `versions`

```text
version_id
document_id
version_number
file_hash
original_filename
stored_pdf_path
page_count
status
created_at
completed_at
```

`file_hash` is unique.

## `pages`

```text
version_id
page_number
page_hash
point_id
```

Primary key:

```text
(version_id, page_number)
```

---

# 37. Registry State Model

At the logical document level:

```text
Document
   │
   ├── Version 1
   ├── Version 2
   └── Version N ← current
```

At the storage level:

```text
Registry (SQLite)
        │
        ├── document/version metadata
        │
        └── page → Qdrant point mapping

Qdrant
        │
        ├── current version vectors
        └── historical version vectors
```

---

# 38. SQLite Job Stores

There are separate databases for different operations.

## Ingestion jobs

Default:

```text
data/ingestion_jobs.db
```

Contains:

```text
jobs
job_files
```

## Query jobs

Default:

```text
data/query_jobs.db
```

Contains:

```text
query_jobs
```

## Generation jobs

Default:

```text
data/generate_jobs.db
```

Contains:

```text
generate_jobs
```

---

# 39. API Restart / Recovery Behavior

The API uses an in-process `ThreadPoolExecutor`.

Therefore queued/running work cannot be automatically resumed after an API process restart.

On startup:

```text
queued/running ingestion jobs
    → failed

queued/running query jobs
    → existing DB state remains according to its store behavior

queued/running generation jobs
    → failed
```

Generation recovery explicitly records:

```text
API process restarted while generation job was running.
```

Ingestion recovery explicitly marks both jobs and queued/running file records as failed.

---

# 40. Concurrency Model

The FastAPI process creates:

```python
ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="ingestion",
)
```

This is a critical V1 design decision.

It means GPU-heavy ingestion/query/generation work is serialized through one worker.

Example:

```text
User A → /ingest_document
User B → /ingest_document
User C → /query
User D → /generate

                 │
                 ▼

        ┌─────────────────┐
        │ single worker   │
        │ max_workers = 1 │
        └────────┬────────┘
                 │
        A → B → C → D
```

The API request handlers themselves can be concurrent, but the submitted heavy jobs execute one at a time.

This protects the GPU from multiple ColPali/Qwen instances competing for VRAM.

---

# 41. Generation Serialization

In addition to the executor-level serialization, `generate.py` has its own generation lock controlled by:

```text
GENERATOR_SERIALIZE=true
```

This provides an additional guard around Qwen generation.

---

# 42. FastAPI Lifecycle

Startup sequence:

```text
FastAPI starts
    │
    ▼
start_qdrant()
    │
    ├── connect to Docker
    ├── start Docker Desktop if configured
    ├── find/pull Qdrant image
    ├── create/start Qdrant container
    └── wait for REST API
    │
    ▼
recover interrupted jobs
    │
    ▼
create single-worker executor
    │
    ▼
accept requests
```

Shutdown sequence:

```text
FastAPI shutdown
    │
    ▼
shutdown executor
    │
    ▼
stop_qdrant()
    │
    ▼
stop Qdrant only if this API process started it
```

---

# 43. Qdrant Docker Configuration

Defaults:

```text
QDRANT_CONTAINER_NAME=amg-qdrant
QDRANT_IMAGE=qdrant/qdrant:latest

QDRANT_URL=http://localhost:6333

QDRANT_HTTP_HOST=127.0.0.1
QDRANT_HTTP_PORT=6333

QDRANT_GRPC_HOST=127.0.0.1
QDRANT_GRPC_HOST_PORT=6334

QDRANT_STORAGE_VOLUME=amg_qdrant_storage

QDRANT_DOCKER_START_TIMEOUT=120
QDRANT_API_READY_TIMEOUT=60

QDRANT_START_DOCKER_DESKTOP=true
```

Persistent storage is mounted to:

```text
/qdrant/storage
```

using the named Docker volume.

---

# 44. Main Environment Configuration

## API

```text
API_JOB_DB_PATH
API_JOB_DATA_DIR
API_MAX_FILES_PER_REQUEST
API_PORT
S3_DOWNLOAD_TIMEOUT
S3_MAX_FILE_SIZE_MB
S3_ALLOWED_HOSTS
QUERY_JOB_DB_PATH
QUERY_LOG_DIR
GENERATE_JOB_DB_PATH
GENERATE_LOG_DIR
```

## Qdrant

```text
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
QDRANT_VECTOR_NAME
QDRANT_UPSERT_BATCH_SIZE
QDRANT_CONTAINER_NAME
QDRANT_IMAGE
QDRANT_HTTP_HOST
QDRANT_HTTP_PORT
QDRANT_GRPC_HOST
QDRANT_GRPC_HOST_PORT
QDRANT_STORAGE_VOLUME
QDRANT_START_DOCKER_DESKTOP
QDRANT_DOCKER_START_TIMEOUT
QDRANT_API_READY_TIMEOUT
```

## Ingestion / ColPali

```text
COLPALI_MODEL
COLPALI_MODEL_REVISION
HF_LOCAL_FILES_ONLY
QUANTIZATION
BNB_4BIT_QUANT_TYPE
BNB_4BIT_COMPUTE_DTYPE
BNB_4BIT_USE_DOUBLE_QUANT
SAVE_QUANTIZED_MODEL
QUANTIZED_MODEL_DIR
COLPALI_BATCH_SIZE
PDF_DPI
STORE_PAGE_TEXT
DEVICE
DATA_DIR
PDF_DIR
PAGE_IMAGE_DIR
DUMMY_DIR
REGISTRY_DB_PATH
LOG_LEVEL
```

## Generation / Qwen

```text
GENERATOR_MODEL
GENERATOR_MODEL_PATH
GENERATOR_MODEL_REVISION
GENERATOR_USE_4BIT
GENERATOR_PREQUANTIZED
GENERATOR_BNB_4BIT_QUANT_TYPE
GENERATOR_BNB_4BIT_DOUBLE_QUANT
GENERATOR_BNB_4BIT_COMPUTE_DTYPE
GENERATOR_ATTN_IMPLEMENTATION

GENERATOR_RELEASE_RETRIEVER
GENERATOR_RELEASE_AFTER_GENERATION
GENERATOR_GC_ON_MODEL_SWITCH
GENERATOR_EMPTY_CUDA_CACHE
GENERATOR_SERIALIZE

GENERATOR_CONTEXT_TOP_K
GENERATOR_MAX_IMAGES
GENERATOR_INCLUDE_IMAGES
GENERATOR_MAX_PAGE_TEXT_CHARS
GENERATOR_MAX_TOTAL_CONTEXT_CHARS
GENERATOR_MIN_PIXELS
GENERATOR_MAX_PIXELS

GENERATOR_MAX_NEW_TOKENS
GENERATOR_TEMPERATURE
GENERATOR_TOP_P
```

---

# 45. Default Storage Layout

A representative V1 runtime layout is:

```text
data/
├── pdfs/
│
├── pages/
│   └── <document_id>/
│       └── <version_id>/
│           ├── page_0001.png
│           ├── page_0002.png
│           └── ...
│
├── ingestion_registry.db
├── ingestion_jobs.db
├── query_jobs.db
├── generate_jobs.db
│
├── query/
│   ├── query_....json
│   └── ...
│
└── generate/
    ├── generate_....json
    └── ...
```

The API additionally uses:

```text
data/api_jobs/
```

for API-side ingestion inputs such as downloaded S3 files.

---

# 46. End-to-End Ingestion Example

```bash
curl -X POST http://localhost:8000/ingest_document \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "s3_links": [],
    "local_pdf_paths": [
      "/home/user/docs/manual.pdf",
      "/home/user/docs/safety.pdf"
    ]
  }'
```

Response:

```json
{
  "job_id": "ingest_20261002113000_ab12cd34",
  "status": "accepted",
  "message": "Ingestion job accepted."
}
```

Poll:

```bash
curl http://localhost:8000/ingest_document/ingest_20261002113000_ab12cd34
```

---

# 47. End-to-End Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What is the welding tolerance?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 5
  }'
```

Accepted:

```json
{
  "job_id": "query_20261002113000_ab12cd34",
  "status": "accepted",
  "message": "Query job accepted."
}
```

Poll:

```bash
curl http://localhost:8000/query/query_20261002113000_ab12cd34
```

---

# 48. End-to-End Generate Example

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "Explain the welding tolerance from the documents.",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 5
  }'
```

Successful response:

```json
{
  "answer": "...",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.92,
      "document_id": "DOC-...",
      "version_id": "VER-...",
      "version_number": 1,
      "filename": "manual.pdf",
      "page_number": 8,
      "page_hash": "...",
      "page_image_path": "..."
    }
  ],
  "generation": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "quantization": "bitsandbytes_4bit_nf4",
    "compute_dtype": "bfloat16",
    "context_pages": 2,
    "images_used": 1,
    "max_new_tokens": 256
  }
}
```

---

# 49. Error Handling Summary

## Validation errors

Typical HTTP status:

```text
400
```

Examples:

- blank identity fields
- empty query
- invalid local file extension
- invalid S3 URL
- too many files
- invalid job ID format

## Missing job

```text
404
```

Example:

```json
{
  "detail": "Job not found: ..."
}
```

## Runtime execution failure

Typical:

```text
500
```

The exact error is also written to the corresponding job store and audit log.

---

# 50. Important Architectural Contracts

These should be treated as V1 invariants.

### Ingestion

```text
exact PDF hash duplicate
    → never re-ingest

explicit update_document_id
    → create next version

same page_hash + same embedding_signature
    → reuse existing embedding
```

### Retrieval

```text
no IDs + recursive=false
    → current/latest versions

document_ids
    → current versions of specified documents

version_ids
    → exact versions

recursive=true
    → all documents + all versions
```

### Generation

```text
retrieve()
    →
release retriever
    →
Qwen3-VL
    →
GenerationResponse
```

### Storage

```text
SQLite
    → operational metadata / jobs / registry

Qdrant
    → page multivectors + page metadata

JSON logs
    → full query/generation audit/debug records
```

### GPU

```text
one global worker
+
retriever release
+
generator release
+
CUDA cache cleanup
```

---

# 51. Module Dependency Map

```text
api.py
 ├── ingest.py
 │    ├── registry.py
 │    └── qdrant_store.py
 │
 ├── query.py
 │    └── Qdrant
 │
 ├── generate.py
 │    └── query.py
 │
 └── docker_manager.py
      └── Docker / Qdrant

registry.py
 └── SQLite

qdrant_store.py
 └── Qdrant

ingest.py
 ├── PyMuPDF
 ├── PIL
 ├── ColPali
 ├── BitsAndBytes
 └── Qdrant

query.py
 ├── ColPali
 ├── BitsAndBytes
 └── Qdrant

generate.py
 ├── Qwen3-VL
 ├── Transformers
 ├── BitsAndBytes
 └── query.py
```

---

# 52. Recommended Request/Response Mental Model

## Ingestion

```text
POST
  ↓
job_id
  ↓
poll
  ↓
file-level results
  ↓
document_id + version_id
```

## Retrieval

```text
POST
  ↓
query_job_id
  ↓
poll
  ↓
ranked pages
```

## Generation

```text
POST
  ↓
retrieve
  ↓
generate
  ↓
final GenerationResponse
```

---

# 53. V1 Operational Notes

### Heavy work is serialized

Even though FastAPI can receive multiple HTTP requests, the current executor has:

```text
max_workers = 1
```

Therefore multiple heavy requests do not execute concurrently.

### `/generate` occupies the request while waiting

Unlike `/query` and `/ingest_document`, the generation route waits on:

```text
future.result()
```

before sending the response.

### Historical vectors remain available

Older versions are not deleted from Qdrant when a new version becomes current.

Instead:

```text
is_current = false
```

is used to exclude them from normal latest-version retrieval.

### JSON logs are part of the fault/debugging design

Query and generation logs are persisted independently from the API response.

---

# 54. Current V1 Contract Summary

```text
                   ┌──────────────────────┐
                   │       /health        │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │   /list_documents   │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │ /ingest_document    │
                   │ POST → job_id        │
                   │ GET  → job status    │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │       /query         │
                   │ POST → job_id        │
                   │ GET  → result        │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │      /generate       │
                   │ POST → final answer  │
                   │ GET  → stored job    │
                   └──────────────────────┘
```

The central data model is:

```text
Document
   │
   ├── Version 1
   │      ├── Page 1 → Qdrant point
   │      ├── Page 2 → Qdrant point
   │      └── ...
   │
   └── Version 2 (current)
          ├── Page 1 → reused/new Qdrant point
          ├── Page 2 → reused/new Qdrant point
          └── ...
```

And the end-to-end question flow is:

```text
User
 │
 ▼
POST /generate
 │
 ▼
QueryRequest
 │
 ▼
retrieve()
 │
 ├── ColPali query embedding
 ├── scope resolution
 ├── Qdrant MaxSim
 └── ranked PDF pages
 │
 ▼
generation.py
 │
 ├── release ColPali
 ├── build multimodal context
 ├── load Qwen3-VL
 ├── generate answer
 └── release Qwen
 │
 ▼
GenerationResponse
 │
 ├── answer
 ├── sources
 └── generation metadata
```

---

# 55. Source Files Covered

```text
api(6).py
docker_manager(1).py
generate(1).py
ingest(3).py
qdrant_store(1).py
query(2).py
registry(2).py
```

This document intentionally follows the current implementation rather than introducing a separate proposed architecture.
