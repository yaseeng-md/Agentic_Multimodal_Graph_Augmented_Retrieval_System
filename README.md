# AMG — Multimodal Document Retrieval-Augmented Generation (RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.11+-dc2626.svg)](https://qdrant.tech)
[![ColPali](https://img.shields.io/badge/ColPali-v1.3_4bit_NF4-8b5cf6.svg)](https://huggingface.co/vidore/colpali-v1.3-merged)
[![Qwen3-VL](https://img.shields.io/badge/Qwen3--VL-2B_Instruct_4bit_NF4-0284c7.svg)](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)

**AMG** is an end-to-end multimodal Retrieval-Augmented Generation (RAG) system engineered for complex, layout-intensive documents (e.g., technical papers, financial filings, manuals, and reports containing tables, charts, infovis, and figures).

Traditional RAG architectures rely on text extraction and Optical Character Recognition (OCR), which often destroys layout spatial awareness, garbles table headers, and ignores graphics. **AMG eliminates OCR heuristics entirely**. It treats document pages as high-resolution visual images, indexes them using multi-vector late-interaction representations via **ColPali v1.3** (quantized to 4-bit NF4) inside **Qdrant**, and generates answers with **Qwen3-VL-2B-Instruct** (4-bit NF4) with grounded page citations.

---

## Table of Contents

1. [What is in this Repository?](#what-is-in-this-repository)
2. [System Architecture](#system-architecture)
3. [Evaluation Results & Benchmarks](#evaluation-results--benchmarks)
4. [How to Use AMG (Setup & Quickstart)](#how-to-use-amg-setup--quickstart)
5. [CLI Usage Guide](#cli-usage-guide)
6. [API Reference & cURL Requests (Ingest & Generate)](#api-reference--curl-requests-ingest--generate)
   - [Authentication (Create User & Login)](#1-authentication)
   - [Document Ingestion (`POST /ingest_document`)](#2-document-ingestion-post-ingest_document)
   - [Document Query / Retrieval (`POST /query`)](#3-document-retrieval-post-query)
   - [Multimodal Answer Generation (`POST /generate`)](#4-multimodal-answer-generation-post-generate)
   - [Document Listing & Health Inspection](#5-document-listing--health-inspection)
7. [Hardware & VRAM Optimization](#hardware--vram-optimization)
8. [License & Acknowledgments](#license--acknowledgments)

---

## What is in this Repository?

The repository contains a production-ready, modular codebase supporting visual document ingestion, multi-vector retrieval, vision-language question answering, container orchestration, caching, and evaluation:

```text
amg/
├── api.py                    # FastAPI service exposing async ingestion, query, generation, and auth
├── authorization.py          # User management, PBKDF2 password hashing, JWT creation & RBAC
├── ingest.py                 # PDF intake, PyMuPDF 150 DPI rendering, ColPali embedding, Qdrant upsert
├── query.py                  # ColPali multi-vector query encoding and late-interaction MaxSim search
├── generate.py               # Multimodal QA pipeline combining Qdrant context with Qwen3-VL-2B
├── docker_manager.py         # Automated Docker container lifecycle manager for Qdrant (port 6333)
├── redis_manager.py          # Redis manager, RESP2 client, caching, idempotency, and distributed locks
├── registry.py               # SQLite persistence layer for documents, versions, and page hashes
├── qdrant_store.py           # Qdrant client abstraction for multi-vector late-interaction collections
├── test_server.py            # Comprehensive end-to-end test suite (11 endpoints, auth, idempotency)
├── documentation/            # In-depth architectural references and pipeline specifications
│   ├── v1_ingest.md          # Ingestion pipeline specifications and edge cases
│   ├── v1_query.md           # Retrieval pipeline specifications and scoring logic
│   ├── v1_generate.md        # Generation pipeline specifications and prompt templates
│   ├── v1_redis_integrate.md # Redis caching, locks, and idempotency guarantees
│   └── v1_eval.md            # Benchmark harness guide, metric formulas, and logging schema
├── eval/                     # Automated evaluation and benchmarking suite
│   ├── run_vidore.py         # ViDoRe v3 benchmark runner (Recall@K, MRR@K, nDCG@K, latency)
│   ├── run_generation.py     # OpenRAG generation benchmark runner (faithfulness, relevancy)
│   ├── datasets/             # ViDoRe and OpenRAG dataset loaders
│   ├── generation/           # Lexical evaluation metrics (Exact Match, Token F1, ROUGE-L, BLEU-4)
│   └── ragas_eval/           # OpenRouter LLM-as-a-judge (Nemotron / GPT-4o-mini)
├── reports/                  # Evaluation scorecards and detailed JSON run telemetry
│   └── vidore/               # ViDoRe benchmark outputs and latest scorecard
├── .env_sample               # Annotated template of all system and model configuration variables
└── requirements.txt          # Complete pinned Python dependencies
```

### Core Capabilities

- **Zero-OCR Visual Pipeline**: Reads documents exactly as humans see them by rendering pages to 150 DPI PNGs via PyMuPDF.
- **ColPali v1.3 Multi-Vector Search**: Encodes visual document pages into patch-level multi-vectors ($1024 \times 128$) and calculates late-interaction `MAX_SIM` directly inside Qdrant.
- **Document Versioning & Scoping**: Built-in SQLite registry handles document versioning (`is_current` flags). Queries can target specific documents, exact versions, or recursively traverse history.
- **Grounded Vision-Language Generation**: `Qwen3-VL-2B-Instruct` ingests high-resolution page context directly, providing grounded, hallucination-resistant answers with page citations.
- **6 GB VRAM Hardware Lifecycle**: Sequential model loading, proactive CUDA cache eviction, and explicit model unloading (`release_query_encoder()` and `unload_generator()`) allow running both heavy models on a single consumer GPU (e.g., RTX 3060 / 4050 6GB).
- **Enterprise-Grade FastAPI Service**: Complete JWT authentication, multi-tenant isolation, SQLite durable job stores, Redis state caching, and atomic idempotency reservations (`Idempotency-Key` header).
- **Automated Benchmarking Harness**: Out-of-the-box support for the ViDoRe v3 retrieval benchmark and OpenRAG generation benchmark with OpenRouter LLM-as-a-judge.

---

## System Architecture

AMG separates ingestion, retrieval, generation, and orchestration into loosely coupled, observable modules.

### High-Level Architecture

```mermaid
flowchart TD
    subgraph ClientLayer["Client Layer"]
        Client["Client / Frontend / Script"]
    end

    subgraph APILayer["FastAPI Orchestration (api.py)"]
        AuthMiddleware["JWT Auth & RBAC (authorization.py)"]
        Endpoints["REST Endpoints (/health, /ingest_document, /query, /generate)"]
        JobStores["SQLite Durable Job Stores (ingestion, query, generate)"]
        WorkerPool["Thread Pool Worker"]
    end

    subgraph ServiceCoordination["Coordination & Caching"]
        RedisManager["Redis Manager (redis_manager.py : 6380)"]
        Idempotency["Atomic Idempotency Engine"]
        ResultCache["Retrieval & Answer Cache"]
    end

    subgraph PipelineLayer["Pipelines"]
        IngestEngine["Ingestion Engine (ingest.py)"]
        QueryEngine["Retrieval Engine (query.py)"]
        GenEngine["Generation Engine (generate.py)"]
    end

    subgraph ModelsLayer["Quantized Inference (BitsAndBytes 4-bit NF4)"]
        ColPali["ColPali v1.3 Multivector Embedder"]
        QwenVL["Qwen3-VL-2B-Instruct Generator"]
    end

    subgraph StorageLayer["Persistence Layer"]
        RegistryDB[("SQLite Registry (ingestion_registry.db)")]
        LocalStore["Local Disk (data/pdfs, data/pages)"]
        QdrantDB[("Qdrant Vector DB (:6333) MAX_SIM")]
    end

    Client -->|HTTP Requests + Bearer Token| AuthMiddleware
    AuthMiddleware --> Endpoints
    Endpoints <--> Idempotency
    Endpoints <--> ResultCache
    Endpoints --> JobStores
    Endpoints --> WorkerPool

    WorkerPool --> IngestEngine
    WorkerPool --> QueryEngine
    WorkerPool --> GenEngine

    IngestEngine --> ColPali
    IngestEngine --> RegistryDB
    IngestEngine --> LocalStore
    IngestEngine --> QdrantDB

    QueryEngine --> ColPali
    QueryEngine --> QdrantDB

    GenEngine --> QueryEngine
    GenEngine --> QwenVL
    GenEngine --> LocalStore

    RedisManager -.-> Idempotency
    RedisManager -.-> ResultCache
```

### End-to-End Processing Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant API as FastAPI (api.py)
    participant Redis as Redis (:6380)
    participant Ingest as ingest.py
    participant Query as query.py
    participant Gen as generate.py
    participant Qdrant as Qdrant (:6333)

    Note over Client,API: 1. Ingestion Phase
    Client->>API: POST /ingest_document (Bearer JWT, PDF source)
    API->>Redis: Check Idempotency-Key
    API->>API: Queue async job in SQLite
    API-->>Client: 202 Accepted (job_id)
    API->>Ingest: Background worker triggers ingest
    Ingest->>Ingest: SHA-256 PDF hash & PyMuPDF 150 DPI page rendering
    Ingest->>Qdrant: Check cached vectors by page hash
    Ingest->>Ingest: ColPali v1.3 NF4 embeds new pages (1024x128 multivector)
    Ingest->>Qdrant: Upsert points with payload (is_current=True)
    Ingest->>API: Update job status to completed

    Note over Client,API: 2. Retrieval & Generation Phase
    Client->>API: POST /generate (Bearer JWT, query, top_k)
    API->>Redis: Check Idempotency & Answer Cache
    API->>Gen: Execute query_and_generate()
    Gen->>Query: retrieve(query, top_k)
    Query->>Query: Embed query with ColPali
    Query->>Qdrant: Late-interaction MaxSim search
    Qdrant-->>Query: Scored top-k page candidates & image paths
    Query-->>Gen: Return retrieved page metadata
    Gen->>Gen: release_query_encoder() [GPU VRAM reclamation]
    Gen->>Gen: Load Qwen3-VL-2B (4-bit NF4) & build multimodal prompt
    Gen->>Gen: Synthesize visual context & produce cited answer
    Gen->>Gen: unload_generator() [GPU VRAM reclamation]
    Gen-->>API: Return answer, sources, and generation telemetry
    API->>Redis: Cache answer & update idempotency
    API-->>Client: 200 OK (answer, sources, latency metrics)
```

---

## Evaluation Results & Benchmarks

AMG includes a dedicated evaluation framework ([`eval/`]) to evaluate both retrieval and generation against standard benchmarks.

### 1. ViDoRe v3 Retrieval Benchmark

Visual Document Retrieval is benchmarked on the official **ViDoRe v3 (Visual Document Retrieval Benchmark)** using the `vidore/arxivqa_test_subsampled` dataset.

- **Benchmark Dataset**: `vidore/arxivqa_test_subsampled` (ArXivQA)
- **Model**: `vidore/colpali-v1.3-merged` (quantized 4-bit NF4)
- **Comparator**: Multi-Vector Late-Interaction `MAX_SIM` in isolated Qdrant collection `eval_vidore_v3`
- **Queries Evaluated**: 100 queries against 100 visual document pages
- **Timestamped Scorecard**: [`reports/vidore/latest_scorecard.md`](reports/vidore/latest_scorecard.md)

#### Official Benchmark Scorecard

| Metric | Score | Target | Description |
| :--- | :--- | :--- | :--- |
| **Recall@1** | **0.8400** (84.0%) | $\ge 0.80$ | The top-ranked candidate is the exact ground-truth page |
| **Recall@3** | **0.8900** (89.0%) | $\ge 0.85$ | Ground-truth page retrieved within the top 3 candidates |
| **Recall@5** | **0.9400** (94.0%) | $\ge 0.90$ | Ground-truth page retrieved within the top 5 candidates |
| **Recall@10** | **0.9600** (96.0%) | $\ge 0.95$ | **96%** of relevant pages retrieved in the top 10 candidates |
| **MRR@10** | **0.8745** | $\ge 0.80$ | Mean Reciprocal Rank (evaluates first-hit placement) |
| **nDCG@10** | **0.8951** | $\ge 0.80$ | Normalized Discounted Cumulative Gain at rank 10 |
| **Average Latency** | **77.41 ms** | $\le 150\text{ ms}$ | High-throughput sub-100ms multi-vector late-interaction retrieval |
| **Total Benchmark Time** | **7.76 s** | — | Total duration for all 100 queries |

### 2. OpenRAG Generation Benchmark

Generation quality is measured using **OpenRAG** (e.g. HotpotQA, 2WikiMultihopQA) using an **LLM-as-a-Judge** (via OpenRouter, default: `nvidia/nemotron-3.5-super-vl:free` or `openai/gpt-4o-mini`) alongside standard lexical metrics:

- **Faithfulness (Groundedness)**: Deconstructs the generated answer into atomic claims and verifies context support ($0.0$ to $1.0$, target $\ge 0.85$).
- **Answer Relevancy**: Evaluates how directly the answer addresses the query without off-topic tangents ($0.0$ to $1.0$, target $\ge 0.80$).
- **Context Precision**: Evaluates whether ground-truth evidence pages are prioritized at top ranks.
- **Lexical Overlap**: Exact Match (EM), Token F1, ROUGE-L, and BLEU-4.

### Running Evaluations

```bash
# Run ViDoRe v3 Retrieval Benchmark (100 samples)
python eval/run_vidore.py --dataset arxivqa --sample 100

# Run OpenRAG Generation Benchmark with LLM-as-a-judge
python eval/run_generation.py --dataset openrag --config hotpotqa --sample 10
```

---

## How to Use AMG (Setup & Quickstart)

### Prerequisites

- **OS**: Linux (tested on Ubuntu 22.04 / 24.04 LTS)
- **Python**: 3.10+
- **Hardware**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (CUDA 12.0+)
- **Docker**: Docker Engine or Docker Desktop running

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone <repo-link>
cd amg

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Environment Configuration

Create your `.env` file from the sample:

```bash
cp .env_sample .env
```

Key environment settings in [`.env`]:

```env
# Storage & Hardware
DEVICE=auto
DATA_DIR=./data
LOG_LEVEL=INFO

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=personal_collection
QDRANT_VECTOR_NAME=page_embedding

# Redis Cache & Coordination
REDIS_HOST=127.0.0.1
REDIS_PORT=6380

# Security (Set a strong 32+ character key in production)
JWT_SECRET_KEY=replace-with-a-very-strong-secret-key-min-32-chars!
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Models (4-bit NF4 quantized)
COLPALI_MODEL=vidore/colpali-v1.3-merged
QWEN_MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
QUANTIZATION=4bit
```

### Step 3: Start Supporting Infrastructure

AMG provides automatic Docker container management. When the FastAPI service or ingestion scripts start, [`docker_manager.py`] and [`redis_manager.py`] verify Docker and launch Qdrant and Redis automatically.

Alternatively, launch them manually:

```bash
# Launch Qdrant
docker run -d --name amg-qdrant \
  -p 6333:6333 \
  -v amg_qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Launch Redis
docker run -d --name amg-redis \
  -p 6380:6379 \
  redis:7-alpine
```

### Step 4: Start the FastAPI Server

```bash
python api.py
# Or directly via uvicorn (single worker required for GPU state management):
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

Access the interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.

---

## CLI Usage Guide

AMG can be operated entirely from the terminal without starting the web server.

### 1. Ingestion CLI ([`ingest.py`])

```bash
# Ingest one or more PDF files
python ingest.py --pdf path/to/document1.pdf path/to/document2.pdf

# Update an existing document with a new revision (versioning)
python ingest.py --update-doc doc_472b4946-f9ba-4bf0 path/to/revised_document.pdf

# Run automated smoke test with dummy generated PDFs
python ingest.py --dummy-test
```

### 2. Retrieval CLI ([`query.py`])

```bash
# Retrieve top-5 relevant pages across all current documents
python query.py --query "What are the performance metrics on ArXivQA?" --top-k 5

# Scoped retrieval targeting a specific document
python query.py --query "Summarize chapter 2" --document-id doc_472b4946-f9ba-4bf0 --top-k 3

# Recursive retrieval searching across all historical versions
python query.py --query "Find previous revisions of the budget table" --recursive-search
```

### 3. Generation CLI ([`generate.py`])

```bash
# End-to-end retrieve and generate grounded answer
python generate.py \
  --query "What is ColPali and how does late interaction work?" \
  --top-k 3 \
  --output result.json
```

---

## API Reference & cURL Requests (Ingest & Generate)

All requests to protected endpoints require an `Authorization: Bearer <token>` header obtained from `/login`.

### 1. Authentication

#### Step 1A: Register a User (`POST /create_user`)

```bash
curl -X POST http://localhost:8000/create_user \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Jane Doe",
    "email": "jane@example.com",
    "password": "SecurePassword123!",
    "organization": "AMG-Research",
    "role": "researcher"
  }'
```

**Response (`200 OK`)**:
```json
{
  "user_id": "usr_94b5f889",
  "name": "Jane Doe",
  "email": "jane@example.com",
  "organization": "AMG-Research",
  "role": "researcher",
  "created_at": "2026-10-22T10:00:00Z"
}
```

#### Step 1B: Authenticate to Get JWT Token (`POST /login`)

```bash
curl -X POST http://localhost:8000/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "jane@example.com",
    "password": "SecurePassword123!"
  }'
```

**Response (`200 OK`)**:
```json
{
  "user_id": "usr_94b5f889",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "email": "jane@example.com",
    "name": "Jane Doe"
  }
}
```

> [!TIP]
> Export the access token in your terminal for subsequent requests:
> ```bash
> export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
> ```

---

### 2. Document Ingestion (`POST /ingest_document`)

Ingestion accepts local server file paths or downloadable S3 presigned URLs. It is processed asynchronously in the background.

#### A. Ingest Local PDF Files

```bash
curl -X POST http://localhost:8000/ingest_document \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: ingest-req-001" \
  -d '{
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "local_pdf_paths": [
      "/home/user/documents/financial_report_2025.pdf"
    ],
    "s3_links": []
  }'
```

#### B. Ingest via S3 Presigned URLs

```bash
curl -X POST http://localhost:8000/ingest_document \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "local_pdf_paths": [],
    "s3_links": [
      "https://my-bucket.s3.amazonaws.com/paper.pdf?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE&Signature=..."
    ]
  }'
```

**Response (`202 Accepted`)**:
```json
{
  "job_id": "ingest_20261022104500_a1b2c3d4",
  "status": "accepted",
  "message": "Ingestion job accepted."
}
```

#### C. Poll Ingestion Job Status (`GET /ingest_document/{job_id}`)

```bash
curl -X GET http://localhost:8000/ingest_document/ingest_20261022104500_a1b2c3d4 \
  -H "Authorization: Bearer $TOKEN"
```

**Response (`200 OK` - Completed)**:
```json
{
  "job_id": "ingest_20261022104500_a1b2c3d4",
  "status": "completed",
  "created_at": "2026-10-22T10:45:00.123456",
  "started_at": "2026-10-22T10:45:00.456789",
  "completed_at": "2026-10-22T10:45:15.789012",
  "result": {
    "total_files": 1,
    "successful_files": 1,
    "failed_files": 0,
    "documents": [
      {
        "filename": "financial_report_2025.pdf",
        "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
        "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
        "version_number": 1,
        "is_current": true,
        "total_pages": 12,
        "indexed_pages": 12
      }
    ]
  }
}
```

---

### 3. Document Retrieval (`POST /query`)

Submits an asynchronous multi-vector retrieval job using ColPali late-interaction `MAX_SIM`.

#### A. Submit Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "query": "What was the operating cash flow in Q3?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 3
  }'
```

**Response (`202 Accepted`)**:
```json
{
  "job_id": "query_20261022104600_b2c3d4e5",
  "status": "accepted",
  "message": "Query job accepted."
}
```

#### B. Poll Query Results (`GET /query/{query_job_id}`)

```bash
curl -X GET http://localhost:8000/query/query_20261022104600_b2c3d4e5 \
  -H "Authorization: Bearer $TOKEN"
```

**Response (`200 OK` - Completed)**:
```json
{
  "job_id": "query_20261022104600_b2c3d4e5",
  "status": "completed",
  "result": {
    "query": "What was the operating cash flow in Q3?",
    "retrieval_scope": "latest_versions",
    "results_count": 3,
    "results": [
      {
        "rank": 1,
        "score": 0.9412,
        "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
        "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
        "version_number": 1,
        "filename": "financial_report_2025.pdf",
        "page_number": 8,
        "page_image_path": "data/pages/doc_e3f28d8b/ver_01f92e3a/page_8.png"
      }
    ]
  }
}
```

---

### 4. Multimodal Answer Generation (`POST /generate`)

Executes visual retrieval followed by `Qwen3-VL-2B-Instruct` multimodal inference. Returns the synthesized answer, ranked sources, and execution telemetry. Supports atomic idempotency through `Idempotency-Key`.

#### A. Submit Generation Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: gen-req-unique-987" \
  -d '{
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "query": "What was the operating cash flow in Q3 according to the financial statement?",
    "document_ids": [],
    "version_ids": [],
    "recursive_search": false,
    "top_k": 3
  }'
```

**Response (`200 OK`)**:
```json
{
  "answer": "According to the Statement of Cash Flows (Source S1, Page 8), the operating cash flow in Q3 was $4.18 billion, representing a 14% year-over-year increase driven primarily by strength in cloud enterprise revenue.",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.9412,
      "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
      "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
      "version_number": 1,
      "filename": "financial_report_2025.pdf",
      "page_number": 8,
      "page_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90...",
      "page_image_path": "/home/user/amg/data/pages/doc_e3f28d8b/ver_01f92e3a/page_8.png"
    },
    {
      "source_id": "S2",
      "rank": 2,
      "score": 0.8875,
      "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
      "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
      "version_number": 1,
      "filename": "financial_report_2025.pdf",
      "page_number": 9,
      "page_hash": "b2c3d4e5f60718293a4b5c6d7e8f90a1...",
      "page_image_path": "/home/user/amg/data/pages/doc_e3f28d8b/ver_01f92e3a/page_9.png"
    }
  ],
  "generation": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "prequantized": false,
    "quantization": "bitsandbytes_4bit_nf4",
    "compute_dtype": "bfloat16",
    "context_pages": 3,
    "images_used": 3,
    "min_pixels": 262144,
    "max_pixels": 524288,
    "max_new_tokens": 256,
    "duration_ms": 3412.85,
    "retrieval_scope": "latest_versions",
    "retrieval_result_count": 3
  }
}
```

#### B. Fetch Historical Generation Job (`GET /generate/{generate_job_id}`)

```bash
curl -X GET http://localhost:8000/generate/gen_20261022104800_c3d4e5f6 \
  -H "Authorization: Bearer $TOKEN"
```

---

### 5. Document Listing & Health Inspection

#### List Ingested Documents & Versions (`GET /list_documents`)

```bash
curl -X GET http://localhost:8000/list_documents \
  -H "Authorization: Bearer $TOKEN"
```

**Response (`200 OK`)**:
```json
{
  "documents": [
    {
      "filename": "financial_report_2025.pdf",
      "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
      "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
      "version_number": 1,
      "is_current": true,
      "page_count": 12,
      "ingested_at": "2026-10-22T10:45:15"
    }
  ]
}
```

#### Check System Health (`GET /health`)

```bash
curl -X GET http://localhost:8000/health
```

**Response (`200 OK`)**:
```json
{
  "status": "ok",
  "redis": "healthy",
  "version": "1.0.0"
}
```

---

## Hardware & VRAM Optimization

AMG was built from the ground up to operate on consumer-tier GPUs with constrained VRAM ($\ge 6\text{ GB}$, e.g. NVIDIA RTX 3060 / 4050 Laptop).

### Dynamic GPU Memory Lifecycle

Concurrent loading of ColPali v1.3 and Qwen3-VL-2B easily exceeds 8 GB of VRAM. AMG eliminates CUDA Out-Of-Memory (OOM) errors through structured sequential execution:

1. **Quantization**: Both models are quantized with `bitsandbytes` 4-bit NF4 (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`), shrinking model weights to ~1.8 GB each.
2. **Sequential Model Swapping**: In [`generate.py`](amg/generate.py):
   - Retrieval is executed with ColPali $\rightarrow$ vectors are generated.
   - Query encoder is freed from memory via `release_query_encoder()`, followed by `torch.cuda.empty_cache()` and `gc.collect()`.
   - Qwen3-VL-2B is loaded to generate the answer.
   - Generator is unhooked via `unload_generator()`, keeping the GPU idle and available.
3. **Resolution Clamping**: Image inputs are normalized to `min_pixels: 262144` and `max_pixels: 524288` to keep vision-transformer patch tokens manageable.

---


## License & Acknowledgments

This project is licensed under the Apache 2.0 License.

- **ColPali**: Efficient Document Retrieval with Vision Language Models ([HuggingFace](https://huggingface.co/vidore/colpali-v1.3-merged))
- **ViDoRe Benchmark**: The Visual Document Retrieval Benchmark ([ViDoRe](https://huggingface.co/vidore))
- **Qwen3-VL**: Alibaba Cloud's state-of-the-art vision-language model family ([Qwen](https://github.com/QwenLM/Qwen-VL))
- **Qdrant**: High-performance Vector Database with native multi-vector late interaction support ([Qdrant](https://qdrant.tech))
