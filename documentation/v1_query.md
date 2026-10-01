# AMG Multimodal RAG — Query Pipeline Contract

## 1. Overview

The Query Pipeline provides asynchronous retrieval of relevant PDF pages from the Qdrant vector database using a ColPali text-query encoder.

The pipeline is retrieval-only in V1. It does **not** generate a natural-language answer.

The current retrieval flow is:

```text
Client
  |
  | POST /query
  v
FastAPI
  |
  | Validate QueryRequest
  | Generate query_job_id
  | Persist job in SQLite
  | Submit background job
  v
Background Query Worker
  |
  | retrieve(...)
  v
ColPali Query Encoder
  |
  | Text query -> multi-vector embedding
  v
Qdrant
  |
  | MaxSim multivector search
  v
Ranked PDF Pages
  |
  | Persist result in SQLite
  | Save query/<query_job_id>.json
  v
Client
  |
  | GET /query/{query_job_id}
  v
Query Result
```

The API uses a single background worker in V1 so that multiple requests do not load competing ColPali models into the same GPU memory.

---

# 2. Query Endpoints

## 2.1 Submit Query

```http
POST /query
```

### Purpose

Submit a retrieval request.

The endpoint is asynchronous and returns immediately with a `query_job_id`.

### HTTP Status

```http
202 Accepted
```

### Request Body

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What are the safety requirements?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 10
}
```

---

# 3. Query Request Contract

## Schema

| Field | Type | Required | Default | Description |
|---|---|---:|---:|---|
| `user_id` | string | Yes | — | User identifier |
| `organization` | string | Yes | — | Organization identifier/name |
| `role` | string | Yes | — | User role |
| `query` | string | Yes | — | Natural-language retrieval query |
| `document_ids` | array[string] | No | `[]` | Restrict search to specified documents |
| `version_ids` | array[string] | No | `[]` | Restrict search to specified document versions |
| `recursive_search` | boolean | No | `false` | Search all documents and all versions |
| `top_k` | integer | No | `5` | Number of ranked pages to return |

### Validation Rules

#### Required strings

`user_id`, `organization`, `role`, and `query`:

- Must be present.
- Must contain at least one character.
- Leading/trailing whitespace is removed.
- Empty strings after stripping are rejected.

#### IDs

`document_ids` and `version_ids`:

- Empty strings are rejected.
- Leading/trailing whitespace is removed.
- Duplicate IDs are removed while preserving order.

#### `top_k`

```text
top_k >= 1
```

---

# 4. Query Submission Response

A successful `POST /query` does not return retrieval results.

It returns an accepted job:

```json
{
  "job_id": "query_20261001102530_a81c92ef",
  "status": "accepted",
  "message": "Query job accepted."
}
```

### Contract

| Field | Type | Description |
|---|---|---|
| `job_id` | string | Unique query job identifier |
| `status` | string | Submission status |
| `message` | string | Human-readable status message |

The returned `job_id` is the identifier used for polling and query logging.

---

# 5. Query Job ID

Query job IDs follow this format:

```text
query_<UTC_TIMESTAMP>_<UUID8>
```

Example:

```text
query_20261001102530_a81c92ef
```

The API accepts IDs matching:

```text
^query_[a-zA-Z0-9_-]+$
```

---

# 6. Query Processing Flow

After receiving `POST /query`:

### Step 1 — Generate Job ID

The API creates a unique `query_job_id`.

```text
query_20261001102530_a81c92ef
```

### Step 2 — Persist Request

The complete request metadata is stored in SQLite before the background job starts.

Stored fields include:

```text
query_job_id
user_id
organization
role
query
document_ids
version_ids
recursive_search
top_k
status
created_at
```

Initial status:

```text
queued
```

### Step 3 — Submit Background Job

The request is submitted to the background executor.

V1 uses:

```text
max_workers = 1
```

This keeps ColPali retrieval sequential and avoids multiple simultaneous model executions competing for GPU memory.

### Step 4 — Mark Job Running

The job status changes:

```text
queued -> running
```

and `started_at` is recorded.

### Step 5 — Execute Retrieval

The API calls:

```python
retrieve(
    query=request.query,
    document_ids=request.document_ids,
    version_ids=request.version_ids,
    recursive_search=request.recursive_search,
    top_k=request.top_k,
)
```

### Step 6 — Encode Query

ColPali converts the text query into a multivector embedding.

Conceptually:

```text
Text Query
    |
    v
ColPali Processor
    |
    v
ColPali Model
    |
    v
[query_tokens, embedding_dim]
```

The query embedding is converted into a Qdrant-compatible multivector.

### Step 7 — Resolve Retrieval Scope

The retrieval scope is resolved according to the V1 policy described below.

### Step 8 — Search Qdrant

Qdrant is queried using:

```text
collection = personal_collection
vector = page_embedding
limit = top_k
with_payload = true
with_vectors = false
```

The retrieval uses the stored page multivectors and Qdrant MaxSim-based multivector search.

### Step 9 — Rank Results

Each returned page is serialized with a rank:

```text
rank = 1
rank = 2
rank = 3
...
rank = top_k
```

### Step 10 — Persist Result

The result is saved in the query job database.

The job becomes:

```text
running -> completed
```

### Step 11 — Save Debug/Audit JSON

The complete query execution record is saved as:

```text
data/query/<query_job_id>.json
```

Example:

```text
data/query/query_20261001102530_a81c92ef.json
```

---

# 7. Retrieval Scope Contract

The V1 retrieval policy has four possible modes.

## 7.1 Default Search

Request:

```json
{
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false
}
```

Scope:

```text
all_documents_current_versions
```

Meaning:

```text
Search every document
+
only the current/latest version of each document
```

---

## 7.2 Document-Specific Search

Request:

```json
{
  "document_ids": [
    "doc_001",
    "doc_002"
  ],
  "version_ids": [],
  "recursive_search": false
}
```

Scope:

```text
documents_current_versions
```

Meaning:

```text
Search only doc_001 and doc_002
+
only their current/latest versions
```

---

## 7.3 Exact Version Search

Request:

```json
{
  "document_ids": [],
  "version_ids": [
    "version_001",
    "version_002"
  ],
  "recursive_search": false
}
```

Scope:

```text
explicit_versions
```

Meaning:

```text
Search only the explicitly supplied versions.
```

Explicit `version_ids` take precedence over `document_ids`.

---

## 7.4 Recursive Search

Request:

```json
{
  "document_ids": [],
  "version_ids": [],
  "recursive_search": true
}
```

Scope:

```text
all_documents_all_versions
```

Meaning:

```text
Search every document
+
every version
```

When `recursive_search=true`, supplied `document_ids` and `version_ids` are ignored.

---

# 8. Retrieval Scope Precedence

The effective precedence is:

```text
1. recursive_search=True
        |
        v
   all documents / all versions

2. version_ids supplied
        |
        v
   exact versions

3. document_ids supplied
        |
        v
   specified documents / current versions

4. nothing supplied
        |
        v
   all documents / current versions
```

Equivalent decision flow:

```text
                    recursive_search?
                     /           \
                   YES            NO
                   |               |
                   v               v
        all docs/all versions   version_ids?
                                  /      \
                                YES       NO
                                |          |
                                v          v
                         exact versions  document_ids?
                                          /      \
                                        YES       NO
                                        |          |
                                        v          v
                                  selected docs   all docs
                                  current vers.   current vers.
```

---

# 9. Qdrant Retrieval Contract

The current query pipeline uses the following configuration:

```text
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION = personal_collection
QDRANT_VECTOR_NAME = page_embedding
```

The query encoder uses:

```text
COLPALI_MODEL = vidore/colpali-v1.3-merged
COLPALI_MODEL_REVISION = main
```

Quantization configuration:

```text
QUANTIZATION = 4bit
BNB_4BIT_QUANT_TYPE = nf4
BNB_4BIT_COMPUTE_DTYPE = bfloat16
BNB_4BIT_USE_DOUBLE_QUANT = true
```

The exact values can be overridden through environment variables.

---

# 10. Retrieval Result Contract

The `retrieve()` function returns:

```json
{
  "query": "What are the safety requirements?",
  "top_k": 10,
  "scope": {
    "mode": "all_documents_current_versions",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false
  },
  "result_count": 10,
  "results": []
}
```

## Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `query` | string | Original query |
| `top_k` | integer | Requested number of results |
| `scope` | object | Effective retrieval scope |
| `result_count` | integer | Number of results actually returned |
| `results` | array | Ranked page-level results |

---

# 11. Result Item Contract

Each result contains:

```json
{
  "rank": 1,
  "score": 0.8734,
  "point_id": "123456",
  "document_id": "doc_001",
  "version_id": "version_001",
  "version_number": 1,
  "filename": "manual.pdf",
  "page_number": 12,
  "page_hash": "abc123...",
  "page_image_path": "/path/to/page.png",
  "page_text": "...",
  "is_current": true,
  "payload": {}
}
```

## Result Fields

| Field | Type | Description |
|---|---|---|
| `rank` | integer | Retrieval rank, starting from 1 |
| `score` | number | Qdrant similarity score |
| `point_id` | string | Qdrant point identifier |
| `document_id` | string/null | Logical document identifier |
| `version_id` | string/null | Document version identifier |
| `version_number` | integer/null | Version number |
| `filename` | string/null | Original PDF filename |
| `page_number` | integer/null | PDF page number |
| `page_hash` | string/null | Page content hash |
| `page_image_path` | string/null | Stored page image path |
| `page_text` | string/null | Stored page text, when available |
| `is_current` | boolean/null | Whether the page belongs to the current version |
| `payload` | object | Complete Qdrant payload stored with the point |

---

# 12. Poll Query Job

```http
GET /query/{query_job_id}
```

Example:

```bash
curl "http://localhost:8000/query/query_20261001102530_a81c92ef"
```

### Purpose

Retrieve the current state and persisted result of a query job.

---

# 13. Query Job Response

Example completed response:

```json
{
  "query_job_id": "query_20261001102530_a81c92ef",
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What are the safety requirements?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 10,
  "status": "completed",
  "created_at": "2026-10-01T10:25:30.123456+00:00",
  "started_at": "2026-10-01T10:25:30.130000+00:00",
  "finished_at": "2026-10-01T10:25:34.421000+00:00",
  "result": {
    "query": "What are the safety requirements?",
    "top_k": 10,
    "scope": {
      "mode": "all_documents_current_versions",
      "document_ids": [],
      "version_ids": [],
      "recursive_search": false
    },
    "result_count": 10,
    "results": []
  },
  "error": null
}
```

---

# 14. Query Job Statuses

The query job lifecycle is:

```text
queued
  |
  v
running
  |
  +----> completed
  |
  +----> failed
```

## `queued`

The request has been accepted and persisted but retrieval has not started yet.

## `running`

The background worker is executing retrieval.

## `completed`

Retrieval finished successfully and the result is persisted.

## `failed`

An exception occurred during query processing.

The error is persisted in the job record and the query log.

---

# 15. Query JSON Logging Contract

Every completed or failed query should have a corresponding JSON file:

```text
data/query/<query_job_id>.json
```

Example:

```text
data/query/query_20261001102530_a81c92ef.json
```

The JSON is intended for:

- debugging
- auditing
- reproducing retrieval behavior
- fault investigation
- inspecting the exact request
- inspecting the returned pages
- inspecting execution timing
- investigating failed queries

---

# 16. Query Log Structure

The query log follows this structure:

```json
{
  "job": {
    "query_job_id": "...",
    "status": "completed",
    "created_at": "...",
    "started_at": "...",
    "finished_at": "...",
    "duration_ms": 1234.56
  },

  "request": {
    "user_id": "...",
    "organization": "...",
    "role": "...",
    "query": "...",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 10
  },

  "retrieval": {
    "scope": "all_documents_current_versions",
    "result_count": 10
  },

  "execution": {
    "model_id": "vidore/colpali-v1.3-merged",
    "quantization": "bitsandbytes_4bit_nf4",
    "embedding_type": "colpali_multivector"
  },

  "response": {
    "status": "success",
    "result": {}
  },

  "error": null
}
```

For a failed query:

```json
{
  "job": {
    "query_job_id": "...",
    "status": "failed"
  },
  "request": {
    "..."
  },
  "retrieval": {
    "..."
  },
  "execution": {
    "..."
  },
  "response": {
    "status": "failed",
    "result": null
  },
  "error": "Actual exception message"
}
```

---

# 17. Error Contract

## Invalid Request

Pydantic validation errors are returned by FastAPI when the request does not satisfy `QueryRequest`.

Typical causes:

```text
Missing user_id
Missing organization
Missing role
Missing query
Empty query
Empty document ID
Empty version ID
top_k < 1
```

## Job Not Found

```http
GET /query/{query_job_id}
```

returns:

```http
404 Not Found
```

when the query job does not exist.

Example:

```json
{
  "detail": "Query job not found: query_..."
}
```

## Invalid Query Job ID

If the path does not match:

```text
^query_[a-zA-Z0-9_-]+$
```

the API returns:

```http
400 Bad Request
```

## Retrieval Failure

If ColPali, Qdrant, configuration, or another retrieval operation raises an exception:

```text
running -> failed
```

The exception is stored in:

```text
query_jobs.db
```

and:

```text
data/query/<query_job_id>.json
```

---

# 18. cURL Examples

## 18.1 Basic Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What are the safety requirements?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 10
  }'
```

Expected response:

```json
{
  "job_id": "query_20261001102530_a81c92ef",
  "status": "accepted",
  "message": "Query job accepted."
}
```

---

## 18.2 Poll Query

Replace the job ID with the ID returned by `POST /query`.

```bash
curl "http://localhost:8000/query/query_20261001102530_a81c92ef"
```

---

## 18.3 Search Specific Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What are the safety requirements?",
    "document_ids": [
      "document_001",
      "document_002"
    ],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 10
  }'
```

---

## 18.4 Search Exact Versions

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What changed in the latest revision?",
    "document_ids": [],
    "version_ids": [
      "version_001",
      "version_002"
    ],
    "recursive_search": false,
    "top_k": 10
  }'
```

---

## 18.5 Recursive Search

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "Find all references to emergency procedures.",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": true,
    "top_k": 10
  }'
```

---

# 19. End-to-End Contract

The complete client interaction is:

```text
1. Client sends POST /query
                |
                v
2. API validates request
                |
                v
3. API generates query_job_id
                |
                v
4. API persists job as queued
                |
                v
5. API returns HTTP 202
                |
                v
6. Background worker starts
                |
                v
7. Job becomes running
                |
                v
8. Query encoded by ColPali
                |
                v
9. Retrieval scope resolved
                |
                v
10. Qdrant searched using MaxSim
                |
                v
11. Pages ranked
                |
                v
12. Result stored in SQLite
                |
                v
13. query/<query_job_id>.json written
                |
                v
14. Job becomes completed/failed
                |
                v
15. Client polls GET /query/{query_job_id}
```

---

# 20. Important V1 Behavior

### Retrieval only

The current `query.py` explicitly performs retrieval and does not generate an answer from an LLM.

The output is therefore:

```text
Query
    -> relevant PDF pages
    -> ranked results
```

not:

```text
Query
    -> retrieved pages
    -> LLM answer
```

An answer-generation stage can be added later.

### Latest-version default

Normal retrieval searches current/latest document versions.

### Explicit version selection

`version_ids` restrict retrieval to exact versions.

### Recursive search

`recursive_search=true` searches all documents and all versions.

### Top-K

`top_k` controls the maximum number of ranked page results returned.

### Page-level retrieval

The unit of retrieval is a PDF page represented by a Qdrant point.

---

# 21. Current V1 Architecture

```text
                         ┌─────────────────────┐
                         │       Client        │
                         └──────────┬──────────┘
                                    │
                              POST /query
                                    │
                                    v
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │                     │
                         │ QueryRequest        │
                         │ Validation          │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │   QueryJobStore     │
                         │      SQLite         │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │ Background Worker   │
                         │ max_workers = 1     │
                         └──────────┬──────────┘
                                    │
                                    v
                         ┌─────────────────────┐
                         │   Query Encoder     │
                         │      ColPali        │
                         └──────────┬──────────┘
                                    │
                              Multi-vector
                                    │
                                    v
                         ┌─────────────────────┐
                         │       Qdrant        │
                         │  page_embedding     │
                         │      MaxSim         │
                         └──────────┬──────────┘
                                    │
                              Ranked Pages
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         v                     v
                ┌─────────────────┐   ┌─────────────────┐
                │   SQLite Job    │   │  Query JSON Log │
                │     Result      │   │ query/<job>.json│
                └─────────────────┘   └─────────────────┘
                         │
                         v
                  GET /query/{id}
```

---

# 22. Future Extension Point

The current pipeline ends at ranked retrieval:

```text
Query
  ↓
ColPali Query Embedding
  ↓
Qdrant MaxSim Search
  ↓
Top-K Pages
```

A future answer-generation pipeline can be added after retrieval:

```text
Query
  ↓
ColPali Query Embedding
  ↓
Qdrant MaxSim Search
  ↓
Top-K Pages
  ↓
Context Builder
  ↓
LLM / Vision-Language Model
  ↓
Generated Answer
  ↓
Citations / Source Pages
```

This should be treated as a separate stage from the current V1 retrieval contract.
