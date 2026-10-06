# AMG Multimodal RAG — Generate Pipeline Contract

## 1. Purpose

The **Generate Pipeline** is the answer-generation stage of the AMG Multimodal RAG system.

It combines the existing retrieval pipeline with a local multimodal Qwen3-VL model:

```text
Client
  |
  | POST /generate
  | QueryRequest
  v
FastAPI
  |
  | create internal generate_job_id
  | persist job metadata in SQLite
  v
Single shared GPU executor
  |
  | retrieve(...)
  v
ColPali Query Encoder
  |
  | text -> multivector query embedding
  v
Qdrant
  |
  | MaxSim search over page_embedding
  v
Ranked PDF pages
  |
  | release ColPali from GPU
  v
Generation Context Builder
  |
  | page text + selected page image(s)
  v
Qwen3-VL-2B-Instruct
  |
  | local 4-bit NF4 generation
  v
Generated Answer
  |
  +------------------------------+
  |                              |
  v                              v
HTTP response                JSON audit log
                             data/generate/<job_id>.json
```

The current API is **synchronous at the HTTP layer**: `POST /generate` waits for the GPU job to finish and returns only the generated response. The internal generation job ID is retained for database tracking and debugging.

---

# 2. Pipeline Responsibilities

The generate pipeline has five logical responsibilities:

1. Validate the incoming generation request using the same request contract as `/query`.
2. Retrieve relevant document pages using the existing `query.py::retrieve()` implementation.
3. Release ColPali GPU memory before loading Qwen3-VL on the 6 GB GPU.
4. Build a constrained multimodal context from retrieved page text and selected page images.
5. Generate the final answer with local Qwen3-VL and persist a complete execution/audit record.

The generation layer does **not** implement a second retrieval engine. It calls the existing retrieval implementation.

---

# 3. Main Components

| Component | File / Location | Responsibility |
|---|---|---|
| FastAPI application | `api.py` | HTTP contract, request validation, job creation, execution, response/error handling |
| Retrieval | `query.py` | ColPali query embedding, Qdrant search, retrieval scope |
| Generation | `generate.py` | Context construction, Qwen3-VL loading, inference, source construction, GPU lifecycle |
| Job store | `api.py` | SQLite status/result persistence for generation jobs |
| Audit log | `data/generate/<generate_job_id>.json` | Complete request, retrieval, generation, timing and error record |
| Configuration | `.env` / generation env settings | Model, quantization, GPU lifecycle, context and decoding controls |
| Vector database | Qdrant | Page-level multimodal embeddings and payload metadata |
| Retrieval model | ColPali | Text-to-multivector query encoding |
| Generation model | Qwen3-VL | Local multimodal answer generation |

---

# 4. HTTP API Contract

## 4.1 Endpoint

```http
POST /generate
```

### Current behavior

The endpoint is **synchronous**.

The client sends a request and waits until:

```text
retrieval -> context build -> Qwen generation -> persistence
```

has completed.

The endpoint then returns **only the generated result**.

---

## 4.2 Request Contract

The generate endpoint uses the same `QueryRequest` Pydantic model used by `/query`.

### Request schema

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What are attention mechanisms?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

### Field contract

| Field | Type | Required | Default | Validation / Meaning |
|---|---|---:|---:|---|
| `user_id` | string | Yes | — | Minimum length 1; stripped; cannot be empty |
| `organization` | string | Yes | — | Minimum length 1; stripped; cannot be empty |
| `role` | string | Yes | — | Minimum length 1; stripped; cannot be empty |
| `query` | string | Yes | — | Minimum length 1; stripped; cannot be empty |
| `document_ids` | array of strings | No | `[]` | IDs are stripped, empty IDs rejected, duplicates removed while preserving order |
| `version_ids` | array of strings | No | `[]` | IDs are stripped, empty IDs rejected, duplicates removed while preserving order |
| `recursive_search` | boolean | No | `false` | Enables all-documents/all-versions retrieval policy |
| `top_k` | integer | No | `5` | Must be `>= 1`; controls retrieval count |

### Important contract rule

The request contract for `/generate` is intentionally aligned with `/query` so the same retrieval scope can be used for answer generation.

---

# 5. Retrieval Scope Contract

Generation delegates retrieval to `query.py::retrieve()`.

The retrieval policy is:

| Request condition | Retrieval scope |
|---|---|
| `document_ids=[]`, `version_ids=[]`, `recursive_search=false` | All documents, latest/current versions |
| `document_ids` supplied | Supplied documents, latest/current versions |
| `version_ids` supplied | Exact versions; explicit version selection takes precedence |
| `recursive_search=true` | All documents and all versions; ignores document/version filters |

The generate layer records the retrieval result returned by `retrieve()` in the audit JSON.

---

# 6. End-to-End Execution Flow

## Step 1 — Request arrives

Client sends:

```http
POST /generate
Content-Type: application/json
```

with the `QueryRequest` payload.

FastAPI validates the payload before generation begins.

---

## Step 2 — Internal generation job is created

Even though the endpoint is synchronous, the service creates an internal ID:

```text
generate_<UTC timestamp>_<8-char UUID>
```

Example:

```text
generate_20261001173257_492650e9
```

The ID is stored in the generation SQLite database.

The ID is primarily used for:

- audit logging
- debugging
- failure investigation
- execution tracing
- database status tracking

The current successful HTTP response does **not** return this ID.

---

## Step 3 — Single shared executor

The API uses one shared `ThreadPoolExecutor`:

```python
ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="ingestion",
)
```

The generate job is submitted to this executor and the HTTP request calls:

```python
future.result()
```

Therefore, the HTTP handler waits for completion.

### Why one worker is used

The same process handles ingestion, retrieval and generation on a GPU with approximately 6 GB VRAM.

Serial execution prevents multiple GPU-heavy operations from competing for VRAM.

This means that generation jobs, query jobs and ingestion jobs share the same execution queue.

---

# 7. Internal Generation Job Lifecycle

The generation job status is persisted in SQLite.

### Status sequence

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

### SQLite database

Default:

```text
data/generate_jobs.db
```

Configurable with:

```text
GENERATE_JOB_DB_PATH
```

---

# 8. Generation Job Database Contract

The `generate_jobs` table contains:

| Column | Type / Meaning |
|---|---|
| `generate_job_id` | Primary key |
| `user_id` | Request user identifier |
| `organization` | Organization |
| `role` | Request role |
| `query` | User query |
| `document_ids` | JSON array |
| `version_ids` | JSON array |
| `recursive_search` | Integer representation of boolean |
| `top_k` | Retrieval top-k |
| `status` | `queued`, `running`, `completed`, `failed` |
| `created_at` | Job creation timestamp |
| `started_at` | Execution start timestamp |
| `finished_at` | Execution finish timestamp |
| `result` | JSON-serialized generation response when completed |
| `error` | Error text when failed |

---

# 9. Generation Worker Flow

The internal `_run_generate_job()` performs:

```text
mark job running
      |
      v
retrieve()
      |
      v
retrieval_result
      |
      v
generate_answer()
      |
      v
Qwen3-VL result
      |
      +----> mark completed
      |
      +----> save full generation JSON log
```

On failure:

```text
exception
   |
   +----> mark job failed
   |
   +----> save failed JSON log
   |
   +----> preserve retrieval result when retrieval succeeded
```

This allows generation failures to retain the evidence that was retrieved before generation failed.

---

# 10. Retrieval Stage

The first actual model stage is the existing retrieval pipeline.

```text
User query
    |
    v
ColPali query encoder
    |
    v
query multivector
    |
    v
Qdrant `page_embedding`
    |
    v
MaxSim search
    |
    v
ranked PDF pages
```

Retrieval returns page-level metadata including fields such as:

```text
rank
score
point_id
document_id
version_id
version_number
filename
page_number
page_hash
page_image_path
page_text
is_current
```

The generation module normalizes these fields before building the Qwen context.

---

# 11. Retrieval-to-Generation Handoff

The generation layer receives:

```python
retrieval_result: dict[str, Any]
```

Expected primary structure:

```json
{
  "scope": "latest_versions",
  "result_count": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.91,
      "document_id": "...",
      "version_id": "...",
      "version_number": 1,
      "filename": "document.pdf",
      "page_number": 12,
      "page_hash": "...",
      "page_image_path": "/path/to/page.png",
      "page_text": "...",
      "is_current": true
    }
  ]
}
```

The generation layer does not query Qdrant directly. It consumes the retrieval result.

---

# 12. Context Construction

The generation context is intentionally smaller than the retrieval result.

Qdrant can return `top_k` pages, but only the configured number of pages is passed to Qwen.

Default:

```text
GENERATOR_CONTEXT_TOP_K=2
```

Therefore:

```text
retrieval top_k = 5
        |
        v
retrieve up to 5 pages
        |
        v
context builder selects up to 2 usable pages
        |
        v
Qwen context
```

---

# 13. Text Context Limits

Per page:

```text
GENERATOR_MAX_PAGE_TEXT_CHARS=5000
```

Total generation text context:

```text
GENERATOR_MAX_TOTAL_CONTEXT_CHARS=12000
```

The context builder truncates page text to the per-page limit and then enforces the total character budget.

A page is considered usable when it has text, an allowed image, or both.

---

# 14. Image Context Limits

Images are enabled by default:

```text
GENERATOR_INCLUDE_IMAGES=true
```

Maximum page images:

```text
GENERATOR_MAX_IMAGES=1
```

The image is included only when:

1. image input is enabled,
2. a `page_image_path` exists,
3. the configured image limit has not been reached,
4. the referenced file exists on disk.

The page text and image are therefore **hybrid evidence**.

---

# 15. Qwen Prompt Contract

The generation prompt instructs the model to:

- answer only from retrieved document pages,
- avoid outside knowledge,
- avoid inventing facts, numbers, page numbers, document IDs, versions or quotes,
- use extracted text for textual evidence,
- use the page image when visual/layout information matters,
- explicitly state when evidence is insufficient,
- keep the answer concise,
- use `[S1]`, `[S2]`, etc. as retrieved source identifiers.

Each source is presented with metadata such as:

```text
========== S1 ==========
Document: <filename>
Document ID: <document_id>
Version ID: <version_id>
Version: <version_number>
Page: <page_number>
Retrieval rank: <rank>
Retrieval score: <score>
====================================
```

Then the extracted page text and, when selected, a local page image are appended.

---

# 16. Generation Model

## Model

Default model:

```text
Qwen/Qwen3-VL-2B-Instruct
```

Configuration:

```text
GENERATOR_MODEL=Qwen/Qwen3-VL-2B-Instruct
```

A local model directory can also be used with:

```text
GENERATOR_MODEL_PATH=/path/to/model
```

When `GENERATOR_MODEL_PATH` is set, that local path is used as the model source.

Model revision:

```text
GENERATOR_MODEL_REVISION=main
```

---

# 17. Quantization Contract

Default loading mode:

```text
GENERATOR_USE_4BIT=true
```

The model is loaded through Hugging Face `BitsAndBytesConfig` with:

```text
load_in_4bit=true
bnb_4bit_quant_type=nf4
bnb_4bit_use_double_quant=true
bnb_4bit_compute_dtype=bfloat16
```

Default configuration:

```text
GENERATOR_BNB_4BIT_QUANT_TYPE=nf4
GENERATOR_BNB_4BIT_DOUBLE_QUANT=true
GENERATOR_BNB_4BIT_COMPUTE_DTYPE=bfloat16
```

The resulting generation metadata reports the active quantization as:

```text
bitsandbytes_4bit_nf4
```

when 4-bit loading is enabled.

---

# 18. Pre-Quantized Model Support

The code has a configuration flag:

```text
GENERATOR_PREQUANTIZED=false
```

This flag is reported in generation metadata.

The actual model source is selected by `GENERATOR_MODEL_PATH` when provided; otherwise the configured Hugging Face model ID is used.

The current implementation still uses the BitsAndBytes 4-bit loading path when `GENERATOR_USE_4BIT=true`.

---

# 19. Attention Implementation

Default attention implementation:

```text
GENERATOR_ATTN_IMPLEMENTATION=sdpa
```

The value is passed to the Qwen model loading call as:

```python
attn_implementation=GENERATOR_ATTN_IMPLEMENTATION
```

---

# 20. Qwen Processor / Vision Processing

The generation module uses:

```python
AutoProcessor
```

and:

```python
qwen_vl_utils.process_vision_info
```

The processor is configured with:

```text
min_pixels=GENERATOR_MIN_PIXELS
max_pixels=GENERATOR_MAX_PIXELS
```

Current defaults:

```text
GENERATOR_MIN_PIXELS=262144
GENERATOR_MAX_PIXELS=524288
```

These correspond to:

```text
256 * 32 * 32 = 262,144 pixels
512 * 32 * 32 = 524,288 pixels
```

The pixel budget limits the visual input size and helps control visual-token / VRAM usage.

---

# 21. Generation / Decoding Contract

Default generation parameters:

```text
GENERATOR_MAX_NEW_TOKENS=256
GENERATOR_TEMPERATURE=0.1
GENERATOR_TOP_P=0.9
```

The generation call always enables cache usage:

```text
use_cache=true
```

When `temperature > 0`, sampling is enabled:

```text
do_sample=true
temperature=<configured value>
top_p=<configured value>
```

When `temperature == 0`, deterministic decoding is used:

```text
do_sample=false
```

---

# 22. GPU Memory Lifecycle

The target runtime is a GPU with approximately 6 GB VRAM.

The most important lifecycle rule is:

```text
ColPali on GPU
      |
      | retrieval complete
      v
release ColPali
      |
      v
clear Python/CUDA cache
      |
      v
load Qwen3-VL
      |
      v
generate
      |
      v
release Qwen3-VL
```

### ColPali release

Controlled by:

```text
GENERATOR_RELEASE_RETRIEVER=true
```

When enabled, the generation layer calls the existing query encoder release logic before Qwen is loaded.

### Qwen release

Controlled by:

```text
GENERATOR_RELEASE_AFTER_GENERATION=true
```

When enabled, the Qwen model and processor references are released after generation.

### Garbage collection

```text
GENERATOR_GC_ON_MODEL_SWITCH=true
```

### CUDA cache cleanup

```text
GENERATOR_EMPTY_CUDA_CACHE=true
```

The implementation uses:

```python
gc.collect()
torch.cuda.empty_cache()
```

when configured.

---

# 23. GPU Memory Observability

The generation module records CUDA memory snapshots using:

```text
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

The logged values are converted to GiB and rounded.

Example structure:

```json
{
  "allocated_gib": 4.812,
  "reserved_gib": 5.102
}
```

These snapshots are emitted in application logs around model loading/release and generation.

---

# 24. Generation Serialization

Generation has its own lock:

```text
GENERATOR_SERIALIZE=true
```

When enabled, only one generation operation enters the Qwen generation critical section at a time.

In addition, the API executor is configured with one worker.

Therefore V1 has two layers of protection:

```text
FastAPI shared executor
        |
        | max_workers=1
        v
one GPU-heavy job at a time
        |
        v
Qwen generation lock
```

---

# 25. HTTP Response Contract

## Success

`POST /generate` returns the actual generation response directly.

Example shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.91,
      "document_id": "doc_123",
      "version_id": "ver_1",
      "version_number": 1,
      "filename": "attention.pdf",
      "page_number": 12,
      "page_hash": "...",
      "page_image_path": "/data/pages/page_12.png"
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
    "retrieval_scope": "latest_versions",
    "retrieval_result_count": 5
  }
}
```

The exact values depend on the request and current configuration.

---

# 26. Response Contract: `answer`

```text
answer: string
```

This is the generated natural-language answer produced by Qwen3-VL.

The model is instructed to answer only from the supplied retrieved evidence.

---

# 27. Response Contract: `sources`

Each generation source contains:

| Field | Meaning |
|---|---|
| `source_id` | Application source label such as `S1` |
| `rank` | Original retrieval rank |
| `score` | Retrieval score |
| `document_id` | Source document identifier |
| `version_id` | Source version identifier |
| `version_number` | Document version number |
| `filename` | Original PDF filename |
| `page_number` | Source page number |
| `page_hash` | Page content hash when available |
| `page_image_path` | Local image used for multimodal generation, or null |

The source list represents the pages actually selected for generation, not necessarily every page returned by Qdrant.

---

# 28. Response Contract: `generation`

The response contains model/execution metadata.

| Field | Meaning |
|---|---|
| `model_id` | Hugging Face model ID or resolved local model path |
| `prequantized` | Value of `GENERATOR_PREQUANTIZED` |
| `quantization` | Active quantization mode |
| `compute_dtype` | Active compute dtype for 4-bit inference |
| `context_pages` | Number of pages passed to generation |
| `images_used` | Number of images passed to Qwen |
| `min_pixels` | Image processor lower pixel budget |
| `max_pixels` | Image processor upper pixel budget |
| `max_new_tokens` | Output token limit |
| `duration_ms` | Generation duration in milliseconds |
| `retrieval_scope` | Scope reported by retrieval |
| `retrieval_result_count` | Number of results returned by retrieval |

---

# 29. Error Contract

If generation fails, the endpoint returns an HTTP 500 response.

The response detail contains the generation failure message.

Example:

```json
{
  "detail": "Generation failed."
}
```

The precise detail depends on the exception raised.

The failure is also recorded in:

```text
data/generate/<generate_job_id>.json
```

---

# 30. Empty Retrieval Behavior

The generation layer requires at least one usable retrieved page.

If no usable page is available, generation raises:

```text
No usable retrieved pages were available for generation.
```

A page is usable when it contains text, an allowed existing image, or both.

---

# 31. Complete Generation Audit Log

Every generation execution writes:

```text
data/generate/<generate_job_id>.json
```

The directory can be changed with:

```text
GENERATE_LOG_DIR
```

The log is written using a temporary file followed by atomic replacement:

```text
.<generate_job_id>.tmp
        |
        v
atomic replace
        |
        v
<generate_job_id>.json
```

This reduces the chance of leaving a partially written final log file.

---

# 32. Generation Log Contract

The full log has the following top-level structure:

```json
{
  "job": {},
  "request": {},
  "retrieval": {},
  "execution": {},
  "response": {},
  "error": null
}
```

---

# 33. Log: `job`

Example:

```json
{
  "generate_job_id": "generate_20261001173257_492650e9",
  "status": "completed",
  "created_at": "2026-10-01T17:32:57+00:00",
  "started_at": "2026-10-01T17:32:57+00:00",
  "finished_at": "2026-10-01T17:33:12+00:00",
  "duration_ms": 15000.123
}
```

### Fields

| Field | Meaning |
|---|---|
| `generate_job_id` | Internal generation identifier |
| `status` | `completed` or `failed` |
| `created_at` | Job created timestamp |
| `started_at` | Worker execution start |
| `finished_at` | Execution completion |
| `duration_ms` | Difference between start and finish in milliseconds |

All timestamps are generated in UTC ISO-8601 form.

---

# 34. Log: `request`

The complete request contract is copied into the audit log:

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What are attention mechanisms?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

This allows debugging without reconstructing the original HTTP request.

---

# 35. Log: `retrieval`

The complete retrieval result returned by `retrieve()` is stored.

This means the generation audit log preserves the retrieved evidence used before answer generation.

The field may contain:

```json
{
  "scope": "latest_versions",
  "result_count": 5,
  "results": [
    {
      "rank": 1,
      "score": 0.91,
      "document_id": "...",
      "version_id": "...",
      "page_number": 12,
      "page_text": "...",
      "page_image_path": "..."
    }
  ]
}
```

The exact retrieval object is preserved rather than reconstructed by the log writer.

---

# 36. Log: `execution`

`execution` contains the `generation` metadata returned by `generate.py`.

Example:

```json
{
  "model_id": "Qwen/Qwen3-VL-2B-Instruct",
  "prequantized": false,
  "quantization": "bitsandbytes_4bit_nf4",
  "compute_dtype": "bfloat16",
  "context_pages": 2,
  "images_used": 1,
  "min_pixels": 262144,
  "max_pixels": 524288,
  "max_new_tokens": 256,
  "duration_ms": 1287.52,
  "retrieval_scope": "latest_versions",
  "retrieval_result_count": 5
}
```

This section is intended to answer: **how was this answer generated?**

---

# 37. Log: `response`

The full externally returned generation object is preserved under:

```json
"response": {
  "status": "success",
  "result": {
    "answer": "...",
    "sources": [],
    "generation": {}
  }
}
```

When the job fails:

```json
"response": {
  "status": "failed",
  "result": null
}
```

---

# 38. Log: `error`

Successful execution:

```json
"error": null
```

Failed execution:

```json
"error": "<exception message>"
```

The error is stored both in SQLite and the JSON audit log.

---

# 39. Example Complete Success Log

```json
{
  "job": {
    "generate_job_id": "generate_20261001173257_492650e9",
    "status": "completed",
    "created_at": "2026-10-01T17:32:57+00:00",
    "started_at": "2026-10-01T17:32:57+00:00",
    "finished_at": "2026-10-01T17:33:11+00:00",
    "duration_ms": 14021.782
  },
  "request": {
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What are attention mechanisms?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 5
  },
  "retrieval": {
    "scope": "latest_versions",
    "result_count": 5,
    "results": [
      {
        "rank": 1,
        "score": 0.91,
        "document_id": "doc_123",
        "version_id": "ver_1",
        "page_number": 12,
        "page_text": "...",
        "page_image_path": "/data/pages/doc_123/page_12.png"
      }
    ]
  },
  "execution": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "prequantized": false,
    "quantization": "bitsandbytes_4bit_nf4",
    "compute_dtype": "bfloat16",
    "context_pages": 2,
    "images_used": 1,
    "min_pixels": 262144,
    "max_pixels": 524288,
    "max_new_tokens": 256,
    "duration_ms": 1260.31,
    "retrieval_scope": "latest_versions",
    "retrieval_result_count": 5
  },
  "response": {
    "status": "success",
    "result": {
      "answer": "Attention mechanisms allow a model to focus on relevant parts of the input...",
      "sources": [],
      "generation": {}
    }
  },
  "error": null
}
```

The actual log includes the complete retrieval result and actual source metadata.

---

# 40. Failed Generation Log Behavior

If retrieval succeeds but Qwen generation fails, the audit log still records:

```text
request
retrieval
execution: possibly null
response: failed
error
```

This is important for debugging because the retrieved evidence is not lost when the generation stage fails.

Example structure:

```json
{
  "job": {
    "status": "failed"
  },
  "request": {},
  "retrieval": {
    "scope": "latest_versions",
    "result_count": 5,
    "results": []
  },
  "execution": null,
  "response": {
    "status": "failed",
    "result": null
  },
  "error": "CUDA out of memory"
}
```

---

# 41. Generation Environment Configuration

Recommended V1 configuration for a 6 GB RTX 4050:

```dotenv
# Qwen model
GENERATOR_MODEL=Qwen/Qwen3-VL-2B-Instruct
GENERATOR_MODEL_REVISION=main

# Quantization
GENERATOR_USE_4BIT=true
GENERATOR_PREQUANTIZED=false
GENERATOR_BNB_4BIT_QUANT_TYPE=nf4
GENERATOR_BNB_4BIT_DOUBLE_QUANT=true
GENERATOR_BNB_4BIT_COMPUTE_DTYPE=bfloat16

# Attention
GENERATOR_ATTN_IMPLEMENTATION=sdpa

# GPU lifecycle
GENERATOR_RELEASE_RETRIEVER=true
GENERATOR_RELEASE_AFTER_GENERATION=true
GENERATOR_GC_ON_MODEL_SWITCH=true
GENERATOR_EMPTY_CUDA_CACHE=true
GENERATOR_SERIALIZE=true

# Context budget
GENERATOR_CONTEXT_TOP_K=2
GENERATOR_MAX_IMAGES=1
GENERATOR_INCLUDE_IMAGES=true
GENERATOR_MAX_PAGE_TEXT_CHARS=5000
GENERATOR_MAX_TOTAL_CONTEXT_CHARS=12000

# Visual input budget
GENERATOR_MIN_PIXELS=262144
GENERATOR_MAX_PIXELS=524288

# Decoding
GENERATOR_MAX_NEW_TOKENS=256
GENERATOR_TEMPERATURE=0.1
GENERATOR_TOP_P=0.9

# PyTorch CUDA allocator
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Generation persistence
GENERATE_JOB_DB_PATH=./data/generate_jobs.db
GENERATE_LOG_DIR=./data/generate
```

---

# 42. Configuration Reference

## Model Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Hugging Face model ID |
| `GENERATOR_MODEL_PATH` | unset | Optional local model directory; takes precedence over model ID |
| `GENERATOR_MODEL_REVISION` | `main` | Model revision |

## Quantization

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_USE_4BIT` | `true` | Enable BitsAndBytes 4-bit loading |
| `GENERATOR_PREQUANTIZED` | `false` | Metadata/config flag indicating pre-quantized model usage |
| `GENERATOR_BNB_4BIT_QUANT_TYPE` | `nf4` | 4-bit quantization type |
| `GENERATOR_BNB_4BIT_DOUBLE_QUANT` | `true` | Double quantization |
| `GENERATOR_BNB_4BIT_COMPUTE_DTYPE` | `bfloat16` | Compute dtype |

## Attention

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_ATTN_IMPLEMENTATION` | `sdpa` | Transformers attention implementation |

## GPU Lifecycle

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_RELEASE_RETRIEVER` | `true` | Release ColPali before Qwen |
| `GENERATOR_RELEASE_AFTER_GENERATION` | `true` | Release Qwen after generation |
| `GENERATOR_GC_ON_MODEL_SWITCH` | `true` | Python GC on model transition |
| `GENERATOR_EMPTY_CUDA_CACHE` | `true` | Empty PyTorch CUDA cache on transition |
| `GENERATOR_SERIALIZE` | `true` | Serialize Qwen generation |

## Context

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_CONTEXT_TOP_K` | `2` | Maximum retrieved pages forwarded to Qwen |
| `GENERATOR_MAX_IMAGES` | `1` | Maximum page images |
| `GENERATOR_INCLUDE_IMAGES` | `true` | Enable image context |
| `GENERATOR_MAX_PAGE_TEXT_CHARS` | `5000` | Per-page text limit |
| `GENERATOR_MAX_TOTAL_CONTEXT_CHARS` | `12000` | Total text budget |

## Visual Input

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_MIN_PIXELS` | `262144` | Minimum processor pixel budget |
| `GENERATOR_MAX_PIXELS` | `524288` | Maximum processor pixel budget |

## Decoding

| Variable | Default | Purpose |
|---|---|---|
| `GENERATOR_MAX_NEW_TOKENS` | `256` | Maximum answer length in generated tokens |
| `GENERATOR_TEMPERATURE` | `0.1` | Sampling temperature |
| `GENERATOR_TOP_P` | `0.9` | Nucleus sampling threshold |

## Persistence

| Variable | Default | Purpose |
|---|---|---|
| `GENERATE_JOB_DB_PATH` | `./data/generate_jobs.db` | Generation SQLite database |
| `GENERATE_LOG_DIR` | `./data/generate` | JSON audit directory |

---

# 43. Model Dependencies

The generation implementation imports:

```python
import torch
from dotenv import load_dotenv
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
```

The model stack therefore depends on:

```text
PyTorch
Transformers
BitsAndBytes
qwen-vl-utils
python-dotenv
```

The existing retrieval stack separately provides ColPali and Qdrant dependencies.

---

# 44. CLI / Library Functions in `generate.py`

## `generate_answer()`

Primary API integration function:

```python
generate_answer(
    query: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]
```

It expects retrieval to already be complete.

It:

```text
optional ColPali release
        |
        v
context build
        |
        v
Qwen generation
        |
        v
GenerationResponse -> dict
```

## `query_and_generate()`

Standalone end-to-end function:

```python
query_and_generate(
    query,
    document_ids=None,
    version_ids=None,
    recursive_search=False,
    top_k=5,
)
```

It performs:

```text
retrieve()
   -> release retriever
   -> generate_from_retrieval()
```

and returns:

```json
{
  "query": "...",
  "retrieval": {},
  "response": {}
}
```

The FastAPI `/generate` endpoint uses the split form (`retrieve()` followed by `generate_answer()`) so that retrieval can be stored independently in the generation audit log.

---

# 45. Source Construction

Generation sources are built only from the pages actually passed to Qwen.

Each selected page becomes:

```text
S1
S2
S3
...
```

The source labels are local application labels and are not document-provided IDs.

They allow the model and final response to refer back to retrieved evidence consistently.

---

# 46. HTTP Example

## Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "organization": "AMG",
    "role": "engineer",
    "query": "What are attention mechanisms?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 5
  }'
```

## Successful response

```json
{
  "answer": "...",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.91,
      "document_id": "doc_123",
      "version_id": "ver_1",
      "version_number": 1,
      "filename": "document.pdf",
      "page_number": 12,
      "page_hash": "...",
      "page_image_path": "/data/pages/page_12.png"
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
    "retrieval_scope": "latest_versions",
    "retrieval_result_count": 5
  }
}
```

Only this result is returned to the client.

---

# 47. Current `/generate/{generate_job_id}` Endpoint

Endpoint:

```http
GET /generate/{generate_job_id}
```

It remains available for internal job inspection.

### Running job

```json
{
  "job_id": "generate_...",
  "status": "running"
}
```

### Completed job

Returns only the stored generation result.

### Failed job

Returns HTTP 500 with the stored error.

Because the current `POST /generate` is synchronous and does not return the job ID, this GET endpoint is primarily useful when the internal job ID is obtained from logs, the generation database, or debugging tooling.

---

# 48. API Restart Behavior

The API uses an in-process executor, so generation jobs do not survive a process restart.

At startup:

```python
generate_job_store.recover_running_jobs()
```

Queued/running generation jobs are changed to:

```text
status = failed
```

with error:

```text
API process restarted while generation job was running.
```

Jobs are **not automatically resumed**.

The JSON audit file generated after a completed/failed execution remains on disk.

---

# 49. Concurrency Model

V1 intentionally prioritizes GPU safety over throughput.

```text
Request A ----\
Request B -----+--> shared executor (1 worker) --> GPU pipeline
Request C ----/
```

Therefore:

- only one executor job executes at a time,
- additional requests wait in the executor queue,
- retrieval and generation do not run concurrently in the same process,
- GPU-heavy models do not intentionally overlap.

This is especially important because ColPali and Qwen can both consume significant VRAM.

---

# 50. GPU Memory Failure Scenario

A common failure mode is:

```text
ColPali remains on GPU
        +
Qwen loads
        |
        v
CUDA out-of-memory
```

The current design addresses this using:

```text
GENERATOR_RELEASE_RETRIEVER=true
GENERATOR_GC_ON_MODEL_SWITCH=true
GENERATOR_EMPTY_CUDA_CACHE=true
GENERATOR_RELEASE_AFTER_GENERATION=true
GENERATOR_SERIALIZE=true
```

The design is intentionally serial rather than trying to keep both large models resident simultaneously.

---

# 51. Logging / Observability Checklist

For every generation call, the system can recover the following from logs + SQLite + JSON audit data:

### Request

- user ID
- organization
- role
- query
- document filters
- version filters
- recursive search flag
- retrieval top-k

### Retrieval

- retrieval scope
- result count
- ranked pages
- retrieval scores
- document/version metadata
- page text
- page image paths

### Model execution

- Qwen model source
- quantization mode
- compute dtype
- prequantized flag
- number of context pages
- number of images
- image pixel budget
- max output token count
- generation duration

### Job execution

- job ID
- status
- creation time
- start time
- finish time
- total duration

### Failure

- exception text
- retrieval preserved when available
- failed status in SQLite
- failed JSON audit file

---

# 52. Directory Layout

Recommended runtime layout:

```text
project/
├── api.py
├── query.py
├── generate.py
├── ingest.py
├── registry.py
├── docker_manager.py
├── .env
└── data/
    ├── generate_jobs.db
    ├── generate/
    │   ├── generate_20261001173257_492650e9.json
    │   ├── generate_....json
    │   └── ...
    └── query/
        └── ...
```

---

# 53. Design Principles of V1

The current generation pipeline follows these principles:

1. **Reuse existing retrieval** instead of duplicating retrieval logic.
2. **Keep the API contract aligned with `/query`** so clients can reuse the same request shape.
3. **Use local inference** with Qwen3-VL instead of a hosted answer-generation API.
4. **Use 4-bit NF4 quantization** to fit the selected model on constrained VRAM.
5. **Release ColPali before Qwen** to avoid simultaneous high GPU memory use.
6. **Keep context small** by limiting page count, page text and image count.
7. **Return only the application result** from `POST /generate`.
8. **Retain full forensic information** in JSON logs for debugging and fault tolerance.
9. **Serialize GPU-heavy jobs** in V1.
10. **Persist job state in SQLite** even though the current HTTP request waits synchronously.

---

# 54. Operational Notes

## Recommended startup

The API is designed to run with a single Uvicorn worker:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

The application lifecycle starts Qdrant before accepting work.

## CUDA library configuration

The runtime environment must be able to load the CUDA libraries required by the installed PyTorch/BitsAndBytes stack.

In the current development environment, CUDA runtime library availability was configured through the environment activation setup.

## Allocator configuration

Recommended:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is an allocator configuration intended to reduce fragmentation-related allocation failures.

---

# 55. What the Client Should Depend On

The stable client-facing contract should be considered:

```text
POST /generate
```

Request:

```json
{
  "user_id": "string",
  "organization": "string",
  "role": "string",
  "query": "string",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

Response:

```json
{
  "answer": "string",
  "sources": [],
  "generation": {}
}
```

Clients should not depend on:

- internal SQLite schema,
- internal generation job IDs,
- temporary files,
- internal model-loading locks,
- internal executor implementation,
- internal JSON audit-log paths.

Those are operational/debugging interfaces.

---

# 56. Current V1 Limitations

The current implementation intentionally has these limitations:

### 1. Single-worker execution

Throughput is limited because only one executor worker is used.

### 2. Synchronous `/generate`

The HTTP connection remains open while the model loads and generates the answer.

### 3. In-process execution

Generation jobs are not resumed after an API process restart.

### 4. Single-GPU optimization

The configuration is tuned around a constrained GPU rather than high-throughput multi-GPU serving.

### 5. Small generation context

Only a configured subset of retrieved pages and images is forwarded to Qwen.

### 6. Audit logs may contain large payloads

The generation JSON log stores the full retrieval result, which can become large depending on retrieval payload size.

---

# 57. Recommended Next Evolution

Potential next phases, without changing the current V1 contract prematurely:

```text
V1
single worker
    |
    v
V2
separate retrieval/generation workers
    |
    v
V3
GPU-aware model pool
    |
    v
V4
queue / task broker + horizontally scalable workers
```

Possible future improvements include:

- asynchronous job submission with polling for long-running generations,
- dedicated retrieval and generation worker processes,
- GPU-aware scheduling,
- model residency management,
- batched generation,
- response streaming,
- structured citation verification,
- log retention policies,
- per-user/organization authorization,
- generation metrics and latency dashboards.

These are **future design options**, not part of the current V1 generate contract.

---

# 58. Contract Summary

## Endpoint

```http
POST /generate
```

## Request

```json
{
  "user_id": "user_123",
  "organization": "AMG",
  "role": "engineer",
  "query": "What are attention mechanisms?",
  "document_ids": [],
  "version_ids": [],
  "recursive_search": false,
  "top_k": 5
}
```

## Internal flow

```text
Request
  -> QueryRequest validation
  -> internal generation job
  -> single worker
  -> retrieve()
  -> release ColPali
  -> build text/image context
  -> load Qwen3-VL-2B-Instruct
  -> 4-bit NF4 inference
  -> answer + sources + generation metadata
  -> SQLite result
  -> JSON audit log
  -> HTTP response
```

## HTTP response

Only:

```json
{
  "answer": "...",
  "sources": [],
  "generation": {}
}
```

## Audit location

```text
data/generate/<generate_job_id>.json
```

## Job database

```text
data/generate_jobs.db
```

## Default model

```text
Qwen/Qwen3-VL-2B-Instruct
```

## Default quantization

```text
BitsAndBytes 4-bit NF4
```

## Default context

```text
2 pages
1 image
5000 chars/page
12000 chars total
```

## Default decoding

```text
max_new_tokens=256
temperature=0.1
top_p=0.9
```

## GPU lifecycle

```text
ColPali -> release -> Qwen -> release
```

---

# 59. Source of Truth for This Document

This document describes the current implementation represented by:

```text
api.py
query.py
generate.py
generation.env.example
```

Where behavior is controlled through environment variables, the configured/default values documented above are the current V1 defaults in the generation module and example environment configuration.

