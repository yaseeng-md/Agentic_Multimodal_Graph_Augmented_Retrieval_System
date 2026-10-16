# AMG Multimodal RAG — Redis Integration & Caching Reference

## 1. Overview

AMG V1 integrates Redis as a high-performance ephemeral caching, fast-polling, coordination, and idempotency layer. Redis acts strictly as an acceleration and coordination layer, while local SQLite databases and JSON audit logs remain the durable, authoritative source of truth.

```text
                               ┌─────────────────────────────────┐
                               │           FastAPI               │
                               │           api.py                │
                               └───────────────┬─────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
       POST /ingest_document              POST /query                    POST /generate
               │                               │                               │
     [Idempotency Check]                       │                               │
               │                               ▼                               ▼
               ▼                     [Retrieval Cache Check]         [Retrieval Cache Check]
       SQLite Job Store                        │                               │
               │                         (Miss)│                         (Miss)│
               ▼                               ▼                               ▼
       Background Ingest             ColPali + Qdrant Search         ColPali + Qdrant Search
               │                               │                               │
               ▼                               ▼                               ▼
       [Update Job Cache]             [Update Retrieval Cache]        [Answer Cache Check]
                                               │                               │
                                               ▼                         (Miss)│
                                      [Update Job Cache]                       ▼
                                                                        Qwen3-VL Generation
                                                                               │
                                                                               ▼
                                                                      [Update Answer Cache]
                                                                               │
                                                                               ▼
                                                                      [Update Job Cache]
```

### Key Capabilities
- **Multi-Tenant Retrieval Caching**: Skips heavy ColPali query encoding and Qdrant late-interaction `MAX_SIM` vector scoring for repeated queries.
- **Vision-Language Answer Caching**: Skips GPU-intensive Qwen3-VL generation when the prompt, retrieval evidence, and model parameters are identical.
- **Fast Job-State Polling**: Serves high-frequency client polling (`/ingest_document/{id}`, `/query/{id}`, `/generate/{id}`) directly from memory to offload SQLite disk I/O.
- **Request Idempotency**: Prevents accidental duplicate ingestion jobs and duplicate LLM generation requests via `Idempotency-Key` HTTP headers.
- **Safe Fallback**: All cache and fast-polling paths fail open/gracefully degrade to the primary pipeline if Redis is offline or unreachable.

---

## 2. Infrastructure & Configuration

### 2.1 Redis Container Specification
Redis runs in a managed Docker container alongside Qdrant:

| Property | Default Value | Environment Variable |
|---|---|---|
| **Container Name** | `amg-redis` | `REDIS_CONTAINER_NAME` |
| **Image** | `redis:7-alpine` | `REDIS_IMAGE` |
| **Host Port** | `6380` | `REDIS_PORT` |
| **Internal Port** | `6379` | N/A |
| **Host Address** | `127.0.0.1` | `REDIS_HOST` |
| **Database Index** | `0` | `REDIS_DB` |
| **Persistent Volume**| `amg_redis_data` | `REDIS_VOLUME` |

### 2.2 Client Implementation
[`redis_manager.py`](file:///home/yaseeng/projects/amg/redis_manager.py) implements `RedisRawClient`, a self-contained client written entirely using Python's standard library `socket` communicating via the RESP2 protocol. It introduces **zero** third-party Python package dependencies and avoids name conflicts with `redis-py`.

### 2.3 Cache TTL Configuration
Configurable via environment variables or `.env`:
- `REDIS_CACHE_TTL` (default: `600` seconds / 10 minutes): TTL for query retrieval and answer generation caches.
- `REDIS_JOB_TTL` (default: `259200` seconds / 3 days): TTL for fast job-state records.
- `REDIS_IDEMPOTENCY_TTL` (default: `86400` seconds / 24 hours): TTL for completed idempotency records.
- `REDIS_IDEMPOTENCY_RESERVATION_TTL` (default: `120` seconds / 2 minutes): Short reservation window preventing orphaned locks on crash.

---

## 3. Key Namespacing & Fingerprinting

All Redis keys adhere to structured namespacing to prevent collisions:

```text
amg:{category}:{schema_version}:{scope}:{fingerprint}
```

### Deterministic Fingerprinting
Payloads are serialized using `canonical_json` (keys sorted alphabetically, compact separators `","` and `":"`, UTF-8 encoded) and hashed using SHA-256 (`stable_hash`).

### Namespace Summary
| Namespace | Key Pattern | Purpose |
|---|---|---|
| **Retrieval Cache** | `amg:cache:v1:retrieval:{sha256}` | Cached ranked page results |
| **Answer Cache** | `amg:cache:v1:answer:{sha256}` | Cached vision-language answers |
| **Job State** | `amg:job:{job_type}:{job_id}` | In-memory job state cache |
| **Idempotency** | `amg:idempotency:v1:{identity}:{endpoint}:{key_hash}` | Idempotency reservation records |
| **Distributed Lock** | `amg:lock:{lock_name}` | Cross-process coordination lock |
| **Rate Limiter** | `amg:ratelimit:v1:{bucket}:{identity_hash}` | Fixed-window request counts |

---

## 4. Multi-Tenant Retrieval Caching

### 4.1 Fingerprint Scope
The retrieval cache key is generated by [`retrieval_cache_key`](file:///home/yaseeng/projects/amg/redis_manager.py#L507-L536) and incorporates:
```json
{
  "organization": "AMG",
  "user_id": "engineer_1",
  "role": "analyst",
  "query": "What are the torque specifications?",
  "document_ids": ["DOC-100"],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5,
  "corpus_revision": null
}
```

> [!IMPORTANT]
> Because identity fields (`organization`, `user_id`, `role`) are part of the cache fingerprint, user A's retrieval results are never served to user B, maintaining strict multi-tenant isolation.

### 4.2 Pipeline Execution
1. In `_run_query_job` and stage 1 of `_run_generate_job`, `cache_get_json(cache_key)` is checked.
2. **Cache Hit**:
   - `retrieve()` is skipped entirely.
   - ColPali model inference and Qdrant network queries are avoided.
   - Result is returned immediately.
3. **Cache Miss**:
   - Executes `retrieve()`.
   - On completion, `cache_set_json(cache_key, result, ttl=600)` stores the ranked results.

---

## 5. Answer Generation Caching

### 5.1 Fingerprint Scope
The answer cache key is generated by [`answer_cache_key`](file:///home/yaseeng/projects/amg/redis_manager.py#L538-L559) and incorporates:
```json
{
  "organization": "AMG",
  "user_id": "engineer_1",
  "role": "analyst",
  "query": "What are the torque specifications?",
  "retrieval_fingerprint": "<sha256_of_retrieval_results>",
  "model_id": "Qwen/Qwen3-VL-2B-Instruct",
  "generation_config": {}
}
```

### 5.2 Pipeline Execution
1. Prior to loading or executing `Qwen3-VL-2B-Instruct`, `cache_get_json(ans_cache_key)` is checked.
2. **Cache Hit**:
   - Returns the previously generated response text, citations, and metadata.
   - Avoids loading the generation model or allocating GPU VRAM.
3. **Cache Miss**:
   - Executes `generate_answer(query, retrieval_result)`.
   - Populates Redis cache with `cache_set_json(ans_cache_key, response, ttl=600)`.

---

## 6. Fast Job-State Polling Cache

High-frequency client polling on asynchronous jobs can saturate SQLite database locks. Redis caches job states to make polling sub-millisecond:

### 6.1 Polling Endpoints
- `GET /ingest_document/{job_id}`: Checked via `get_job_state("ingest", job_id)`.
- `GET /query/{query_job_id}`: Checked via `get_job_state("query", query_job_id)`.
- `GET /generate/{generate_job_id}`: Checked via `get_job_state("generate", generate_job_id)`.

### 6.2 Cache Lifecycle
- **Read Path**: The API endpoint queries Redis first. If found, it returns the cached JSON dictionary directly without hitting SQLite. If missing, it queries SQLite and populates Redis.
- **Write Path**: State updates in background workers (`mark_running`, `mark_completed`, `mark_failed`, `finish_job`) write to SQLite first, then immediately call `set_job_state(...)`.

---

## 7. Idempotency Contract

Endpoints `POST /ingest_document` and `POST /generate` support the optional `Idempotency-Key` HTTP header.

```http
POST /generate HTTP/1.1
Host: localhost:8000
Content-Type: application/json
Idempotency-Key: 7b35f2a1-e489-497d-8153-6a56e7b4c022
```

### 7.1 Reservation Behavior
1. **New Request (`state: new`)**:
   - Redis atomically registers the idempotency key with a 120-second reservation.
   - The API creates the durable job in SQLite, then updates status to `queued` (or `completed`) with a 24-hour TTL.
2. **Identical Retry (`state: existing`)**:
   - If the client retries with the same `Idempotency-Key` and matching request payload:
     - For `POST /ingest_document`: Returns `AcceptedResponse` with the existing `job_id`.
     - For `POST /generate`: Returns the completed answer result directly if already finished.
3. **Payload Conflict (`state: conflict`)**:
   - If a client reuses an `Idempotency-Key` with a different request payload:
     - The server rejects the request immediately with **HTTP 409 Conflict**.

```json
{
  "detail": "Idempotency key reused with different request payload."
}
```

---

## 8. Health Check Contract

The service health check includes real-time Redis connectivity:

### Request
```bash
curl http://localhost:8000/health
```

### Response (Redis Ready)
```json
{
  "status": "ok",
  "redis": {
    "status": "ready",
    "host": "127.0.0.1",
    "port": 6380,
    "db": 0
  }
}
```

### Response (Redis Offline / Docker Down)
```json
{
  "status": "ok",
  "redis": {
    "status": "unavailable",
    "host": "127.0.0.1",
    "port": 6380,
    "db": 0
  }
}
```

---

## 9. Failure Modes & Graceful Degradation

Redis is designed as an optional accelerator:

| Scenario | Behavior |
|---|---|
| **Redis container stopped** | `/health` returns `redis: unavailable`. Endpoints fall back to SQLite and run model inference on every request. No HTTP 500 errors. |
| **Cache corrupted / decoding error** | Key is deleted from Redis and recomputed from primary pipelines. |
| **Docker not started** | API starts normally, logs warning, and continues operating without Redis caching. |
| **Redis memory pressure (OOM)** | Redis handles key eviction according to `maxmemory-policy`. Evicted keys simply trigger cache misses. |

---

## 10. Verification & Diagnostics

Verify Redis connectivity and container health directly from the command line:

```bash
# Check Redis connectivity via Python manager
python -c "import redis_manager; print(redis_manager.health_check())"

# Inspect running Redis container
docker ps -f name=amg-redis

# Ping Redis container port directly
nc -zv 127.0.0.1 6380
```
