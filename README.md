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
├── utils/
│   ├── reranker.py           # Jina AI multimodal reranker integration (jina-reranker-m0)
│   └── visual_grounding.py   # Patch cross-attention mapping, bounding box extraction & heatmap overlays
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
- **Two-Stage Visual Reranking (`jina-reranker-m0`)**: Optional second-stage multimodal cross-encoder reranking via Jina AI API. Evaluates full-resolution page images as base64 against user queries, filters noise with a $0.75$ relevance threshold, and selects the top 5 most relevant pages for generation.
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
        GraphEngine["Graph Pipeline (graph/pipeline.py)"]
        DoclingProc["Docling Parser (graph/docling_processor.py)"]
    end

    subgraph ModelsLayer["Inference & Extraction"]
        ColPali["ColPali v1.3 Multivector Embedder (NF4)"]
        QwenVL["Qwen3-VL-2B-Instruct Generator (NF4)"]
        GraphExtractor["Schema-Validated Extractor (OpenRouter LLM)"]
    end

    subgraph StorageLayer["Persistence Layer"]
        RegistryDB[("SQLite Registry (ingestion_registry.db)")]
        LocalStore["Local Disk (data/pdfs, data/pages, data/docling_cache)"]
        QdrantDB[("Qdrant Vector DB (:6333) MAX_SIM")]
        Neo4jDB[("Neo4j Knowledge Graph (:7687)")]
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
    IngestEngine --> GraphEngine

    GraphEngine --> DoclingProc
    GraphEngine --> GraphExtractor
    GraphEngine --> Neo4jDB

    QueryEngine --> ColPali
    QueryEngine --> QdrantDB
    QueryEngine --> Neo4jDB

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

## Docling & Neo4j Knowledge Graph Integration

AMG features a dual-ingestion and dual-retrieval architecture combining late-interaction multimodal vector search (ColPali + Qdrant) with structural document parsing (Docling) and a knowledge graph (Neo4j).

### Responsibilities of System Components

| Component | Role in AMG | Key Technologies & APIs |
|---|---|---|
| **ColPali** | Multimodal page embedding | `vidore/colpali-v1.3-merged` (4-bit NF4), MaxSim token patches |
| **Qdrant** | Multi-vector page index & payload store | `personal_collection`, multivector MAX_SIM, Cosine distance |
| **Docling** | Document layout & structural parser | `docling.document_converter.DocumentConverter`, tables & headings |
| **Neo4j** | Knowledge graph & claim provenance | Official `neo4j` Bolt driver (`bolt://localhost:7687`), Cypher MERGE |
| **Jina Reranker** | Cross-modal candidate reranking | `jina-reranker-m0` REST API (`api.jina.ai/v1/rerank`) |
| **Qwen3-VL** | Grounded multimodal answer generator | `Qwen/Qwen3-VL-2B-Instruct` (4-bit NF4) |

### Ingestion Flow: Dual Paths (Path A & Path B)

Every eligible PDF document undergoes two complementary ingestion paths:

1. **Path A: Visual Retrieval Path**:
   - Computes SHA-256 PDF file hash for duplicate prevention.
   - Rasterizes PDF pages at 150 DPI via PyMuPDF (`fitz`) to PNG (`data/pages/<doc_id>/<version_id>/page_0001.png`).
   - Generates ColPali late-interaction multi-vectors with BitsAndBytes 4-bit NF4.
   - Upserts vectors and page metadata into Qdrant (`personal_collection`).

2. **Path B: Structured Document & Graph Construction Path**:
   - Parses document structure with Docling, extracting headings, text blocks, and markdown tables.
   - Preserves 1-based source page numbering matching PyMuPDF rasterization.
   - Caches parsed document structures (`data/docling_cache/<file_hash>.json`) for fast repeat ingestion.
   - Extracts factual entities and provenance-backed claims via schema-enforced OpenRouter LLM.
   - Upserts nodes and relationships into Neo4j using parameterized, idempotent Cypher queries.

### Knowledge Graph Schema & Provenance Model

- **Nodes**:
  - `Document`: Logical document identity (`document_id`, `display_name`).
  - `Version`: Content version (`version_id`, `version_number`, `file_hash`, `stored_pdf_path`, `is_current`).
  - `Page`: Document page (`page_id`, `page_number`, `point_id`, `image_path`, `page_hash`, `is_current`).
  - `Chunk`: Structural chunk (`chunk_id`, `chunk_type`, `text`, `heading_path`).
  - `Entity`: Factual entity (`entity_id`, `name`, `normalized_name`, `entity_type`).
  - `Claim`: Factual relationship assertion (`claim_id`, `predicate`, `target_value`, `confidence`, `evidence_text`).

- **Relationships**:
  - `(Document)-[:HAS_VERSION]->(Version)-[:HAS_PAGE]->(Page)-[:HAS_CHUNK]->(Chunk)`
  - `(Chunk)-[:MENTIONS]->(Entity)`
  - `(Entity)-[:SUBJECT_OF]->(Claim)`
  - `(Claim)-[:OBJECT_OF]->(Entity)` (or property `target_value` for literal/numeric metrics)
  - `(Claim)-[:SOURCED_FROM]->(Chunk)`
  - `(Claim)-[:ON_PAGE]->(Page)`

### Dual Retrieval & Evidence Merging Flow

When a user submits a query:
1. **Visual Retrieval**: ColPali encodes the text query into token multivectors and retrieves top page candidates from Qdrant via MaxSim.
2. **Graph Retrieval**: Identifies candidate entity terms in the query and traverses 1-hop and 2-hop relationships in Neo4j to find verified claims and their source pages.
3. **Candidate Merging & Deduplication**:
   - Candidates are merged by stable page identity `(document_id, version_id, page_number)`.
   - Distinct pages discovered exclusively by the graph are added to the candidate pool (not substituted).
   - Overlapping pages retain dual discovery provenance (`retrieval_sources: ["visual", "graph"]`) and attach verified graph claims.
4. **Multimodal Reranking**: The merged page candidates are reranked using `jina-reranker-m0` on their page images, scoring candidates on a uniform scale.
5. **Grounded Generation**: Qwen3-VL receives selected reranked page images, extracted text, and verified structured knowledge graph claims in its grounded prompt.

### How to Disable Graph Functionality

If Neo4j is offline or graph retrieval is not desired, set in `.env`:
```env
DOCLING_ENABLED=false
NEO4J_ENABLED=false
GRAPH_RETRIEVAL_ENABLED=false
```
Or via CLI:
```bash
python query.py --query "your query" --disable-graph
```
The system will run strictly on the visual ColPali + Qdrant baseline.

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

### 2. Multimodal Reranking Evaluation: ColPali vs. Jina Reranker m0

To evaluate whether second-stage cross-encoder reranking is efficient and advantageous for our visual document RAG pipeline, AMG implements a dual-evaluation harness that runs retrieval **without reranking** and **with reranking** (`jina-reranker-m0` on top-20 retrieved page candidates sending base64 images).

- **Benchmark Dataset**: `vidore/arxivqa_test_subsampled` (ArXivQA)
- **First-Stage Embedder**: `vidore/colpali-v1.3-merged` (4-bit NF4 late-interaction in Qdrant)
- **Second-Stage Reranker**: `jina-reranker-m0` (Multimodal Vision-Language Cross-Encoder via Jina AI API)
- **Retrieval Depth**: Top-20 initial candidate pages per query
- **Relevance Score Threshold**: `0.75` (only candidates with score $\ge 0.75$ retained, capped at top-5 for generation)
- **Report & Scorecard**: [`reports/vidore/latest_rerank_comparison.md`](reports/vidore/latest_rerank_comparison.md)

#### Comparative Benchmark Results (25 Queries, ArXivQA)

| Metric | Without Reranking (ColPali) | With Reranking (ColPali + Jina m0) | Delta (Abs) | Rel Improvement (%) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recall@1** | 0.8000 | **0.8400** | **+0.0400** | **+5.00%** | Ground-truth page placed at rank #1 |
| **Recall@3** | 0.8000 | **0.9600** | **+0.1600** | **+20.00%** | Ground-truth page within top 3 candidates |
| **Recall@5** | 0.9200 | **0.9600** | **+0.0400** | **+4.35%** | Ground-truth page within top 5 candidates |
| **Recall@10** | 0.9200 | **0.9600** | **+0.0400** | **+4.35%** | Ground-truth page within top 10 candidates |
| **Recall@20** | 0.9600 | **0.9600** | 0.0000 | 0.00% | Ground-truth page within initial top 20 pool |
| **MRR@20** | 0.8336 | **0.8933** | **+0.0597** | **+7.16%** | Mean Reciprocal Rank at cutoff 20 |
| **nDCG@20** | 0.8628 | **0.9105** | **+0.0477** | **+5.53%** | Normalized Discounted Cumulative Gain at 20 |

#### Efficiency & Latency Breakdown

| Pipeline Stage | Avg Latency / Query | Total Duration | Throughput | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ColPali Vector Search** | **112.45 ms** | 2.82 s | 8.9 q/s | Multi-vector late-interaction `MAX_SIM` in Qdrant (top-20) |
| **Jina Reranker API** | **3942.60 ms** | 98.57 s | 0.25 q/s | Vision-language cross-encoder on 20 base64 page images |
| **Total End-to-End** | **4055.05 ms** | 101.39 s | 0.25 q/s | Combined two-stage retrieval + rerank pipeline |

#### Thresholding & Context Pruning ($\ge 0.75$, Top-5)

- **100% of queries** had at least 1 document exceeding the $0.75$ relevance score threshold.
- On average, only **1.84 documents** per query met the threshold (out of 20 retrieved).
- **~90% of noise and distracting pages were successfully pruned** prior to generation.

#### Architecture Decision: Is Reranking a Good Decision for Our Use Case?

> [!IMPORTANT]
> **Verdict: YES for High-Fidelity Multimodal Generation, NO for Low-Latency Interactive Search.**
> 
> 1. **Massive Quality Gain for Generation**: Reranking boosts **Recall@3 by +20.0%** and **MRR by +7.16%**, while pruning distracting context pages from 20 down to an average of **1.84 highly relevant pages**. This dramatically cuts hallucinations in `Qwen3-VL-2B`.
> 2. **Latency Trade-Off**: ColPali alone responds in **~112 ms**, whereas transmitting 20 base64 images over the network to Jina AI's cross-encoder takes **~3.9 seconds**.
> 3. **Recommended Production Hybrid**:
>    - For fast search / document discovery: Use ColPali baseline (`enable_rerank=False`, sub-150ms response).
>    - For deep question answering and report generation: Enable Jina reranking (`enable_rerank=True`, retrieve top-20 $\rightarrow$ rerank $\rightarrow$ threshold $\ge 0.75$ $\rightarrow$ top-5 to Qwen).

### 3. OpenRAG Generation Benchmark

Generation quality is measured using **OpenRAG** (e.g. HotpotQA, 2WikiMultihopQA) using an **LLM-as-a-Judge** (via OpenRouter, default: `nvidia/nemotron-3.5-super-vl:free` or `openai/gpt-4o-mini`) alongside standard lexical metrics:

- **Faithfulness (Groundedness)**: Deconstructs the generated answer into atomic claims and verifies context support ($0.0$ to $1.0$, target $\ge 0.85$).
- **Answer Relevancy**: Evaluates how directly the answer addresses the query without off-topic tangents ($0.0$ to $1.0$, target $\ge 0.80$).
- **Context Precision**: Evaluates whether ground-truth evidence pages are prioritized at top ranks.
- **Lexical Overlap**: Exact Match (EM), Token F1, ROUGE-L, and BLEU-4.

### Running Evaluations

```bash
# Run ViDoRe v3 Retrieval Benchmark (Baseline ColPali)
python eval/run_vidore.py --dataset arxivqa --sample 100

# Run Dual-Mode Benchmark (Without Reranking vs. With Jina Reranker m0)
python eval/run_vidore.py --dataset arxivqa --sample 25 --rerank

# Run OpenRAG Generation Benchmark with LLM-as-a-judge
python eval/run_generation.py --dataset openrag --config hotpotqa --sample 10
```

---

## How to Use AMG (Setup & Quickstart)

### Prerequisites

- **OS**: Linux (tested on Ubuntu 22.04 / 24.04 LTS)
- **Python**: 3.10+ (for local bare-metal / virtualenv runs)
- **Hardware**: NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (CUDA 12.0+)
- **Docker**: Docker Engine (v24+) with Docker Compose (v2.20+)

---

### Option A: Run Fully Containerized with Docker Compose (Recommended)

Run the entire AMG stack—FastAPI core engine, Qdrant vector database, Redis cache, and Neo4j graph database—with a single command:

```bash
# 1. Clone repository
git clone <repo-link>
cd amg

# 2. Configure environment
cp .env_sample .env
# Edit .env with your OPENROUTER_API_KEY (for graph extraction) and JINA_RERANKER_KEY

# 3. Start all services in the background
docker compose up --build -d

# Check status and health of all containers
docker compose ps
```

#### With GPU Passthrough (NVIDIA Container Toolkit)
If your host has the `nvidia-container-toolkit` installed, launch with the GPU overlay for accelerated ColPali embedding and Qwen-VL generation:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

#### Monitoring & Teardown
```bash
# View API service logs
docker compose logs -f api

# Stop all containers
docker compose down
```

Access the interactive OpenAPI Swagger UI at `http://localhost:8000/docs`.

---

### Option B: Local Virtual Environment Setup

#### Step 1: Clone and Set Up Virtual Environment

```bash
git clone <repo-link>
cd amg

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 2: Environment Configuration

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

# Neo4j Graph DB
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=amg_neo4j_password_2026

# Security (Set a strong 32+ character key in production)
JWT_SECRET_KEY=replace-with-a-very-strong-secret-key-min-32-chars!
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Models (4-bit NF4 quantized)
COLPALI_MODEL=vidore/colpali-v1.3-merged
QWEN_MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
QUANTIZATION=4bit
```

#### Step 3: Start Supporting Infrastructure

AMG provides automatic Docker container management. When the FastAPI service or ingestion scripts start, [`docker_manager.py`], [`redis_manager.py`], and [`graph/neo4j_manager.py`] verify Docker and launch Qdrant, Redis, and Neo4j automatically.

Alternatively, launch supporting containers via Compose without the API:

```bash
docker compose up -d qdrant redis neo4j
```

#### Step 4: Start the FastAPI Server

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

# Retrieve with Jina multimodal reranking (jina-reranker-m0, threshold 0.75)
python query.py --query "What are the performance metrics on ArXivQA?" --top-k 5 --enable-rerank

# Scoped retrieval targeting a specific document
python query.py --query "Summarize chapter 2" --document-id doc_472b4946-f9ba-4bf0 --top-k 3

# Recursive retrieval searching across all historical versions
python query.py --query "Find previous revisions of the budget table" --recursive-search
```

### 3. Generation CLI ([`generate.py`])

```bash
# End-to-end retrieve and generate grounded answer (baseline ColPali)
python generate.py \
  --query "What is ColPali and how does late interaction work?" \
  --top-k 3 \
  --output result.json

# End-to-end retrieve, rerank with Jina m0 (top-20 -> threshold 0.75 -> top-5), and generate
python generate.py \
  --query "What is ColPali and how does late interaction work?" \
  --enable-rerank \
  --top-k 5 \
  --output result_reranked.json
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
    "top_k": 5,
    "enable_rerank": true,
    "rerank_threshold": 0.75
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
      "score": 0.9888,
      "document_id": "doc_e3f28d8b-821c-4b53-9c02-76326bb178d2",
      "version_id": "ver_01f92e3a-0e92-411a-b328-9842bf27ce52",
      "version_number": 1,
      "filename": "financial_report_2025.pdf",
      "page_number": 8,
      "page_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90...",
      "page_image_path": "/home/user/amg/data/pages/doc_e3f28d8b/ver_01f92e3a/page_8.png",
      "reranked": true,
      "original_rank": 2,
      "original_score": 0.9124
    }
  ],
  "generation": {
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "prequantized": false,
    "quantization": "bitsandbytes_4bit_nf4",
    "compute_dtype": "bfloat16",
    "context_pages": 1,
    "images_used": 1,
    "min_pixels": 262144,
    "max_pixels": 524288,
    "max_new_tokens": 256,
    "duration_ms": 3412.85,
    "retrieval_scope": "latest_versions",
    "retrieval_result_count": 1,
    "rerank": {
      "reranked": true,
      "model": "jina-reranker-m0",
      "initial_candidates_count": 20,
      "final_candidates_count": 1,
      "threshold": 0.75,
      "latency_ms": 3840.12
    }
  }
}
```
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

#### C. Real-Time Streaming Generation (`POST /generate/stream`)

Streams tokens incrementally via Server-Sent Events (SSE) using Hugging Face `TextIteratorStreamer` and FastAPI `StreamingResponse`. Reduces Time-To-First-Token (TTFT) from ~10s to <1s.

```bash
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "user_id": "usr_94b5f889",
    "organization": "AMG-Research",
    "role": "researcher",
    "query": "What was the operating cash flow in Q3?",
    "top_k": 2
  }'
```

##### Server-Sent Events (SSE) Stream Output:
```http
event: token
data: {"token": "Based"}

event: token
data: {"token": " on"}

event: token
data: {"token": " the"}

event: token
data: {"token": " financial"}

...

event: complete
data: {
  "answer": "Based on the financial statement on page 3, the operating cash flow in Q3 was $142.5M...",
  "sources": [
    {
      "source_id": "S1",
      "rank": 1,
      "score": 0.892,
      "document_id": "doc_e3f28d8b",
      "page_number": 3,
      "page_image_path": "data/pages/doc_e3f28d8b/ver_01f92e3a/page_3.png"
    }
  ],
  "generation": {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "duration_ms": 3120.4,
    "ttft_ms": 780.2
  },
  "latency_ms": 3120.4,
  "ttft_ms": 780.2
}
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
