# AMG Multimodal RAG — Evaluation Framework Reference (`v1_eval.md`)

## 1. Overview & Architecture

The AMG Evaluation Suite provides automated, standardized benchmarking for both the visual retrieval engine and the vision-language generation pipeline.

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                 AMG Evaluation Suite                   │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                      ┌────────────────────────────────────┴────────────────────────────────────┐
                      ▼                                                                         ▼
   ┌─────────────────────────────────────┐                                   ┌─────────────────────────────────────┐
   │         Pillar 1: Retrieval         │                                   │        Pillar 2: Generation         │
   │          eval/run_vidore.py         │                                   │        eval/run_generation.py       │
   └──────────────────┬──────────────────┘                                   └──────────────────┬──────────────────┘
                      │                                                                         │
                      ▼                                                                         ▼
   ┌─────────────────────────────────────┐                                   ┌─────────────────────────────────────┐
   │          ViDoRe v3 Benchmark        │                                   │          OpenRAG Benchmark          │
   │    DocVQA, ArxivQA, InfoVQA, etc.   │                                   │   HotpotQA, 2WikiMultihopQA, etc.   │
   └──────────────────┬──────────────────┘                                   └──────────────────┬──────────────────┘
                      │                                                                         │
                      ▼                                                                         ▼
   ┌─────────────────────────────────────┐                                   ┌─────────────────────────────────────┐
   │     ColPali v1.3 (4-bit NF4)        │                                   │     Qwen3-VL-2B (4-bit NF4)         │
   │  Embeds images & queries to Qdrant  │                                   │   Generates grounded answer from    │
   │      Collection: eval_vidore_v3     │                                   │       retrieved context/pages       │
   └──────────────────┬──────────────────┘                                   └──────────────────┬──────────────────┘
                      │                                                                         │
                      │ release_embedder()                                                      │ release_generator()
                      ▼                                                                         ▼
   ┌─────────────────────────────────────┐                                   ┌─────────────────────────────────────┐
   │          Retrieval Metrics          │                                   │      LLM-as-a-Judge Evaluation      │
   │     • Recall@[1, 2, 5, 10]          │                                   │   OpenRouter API (e.g. Nemotron)    │
   │     • Precision@[1, 2, 5, 10]       │                                   │     • Faithfulness / Groundedness   │
   │     • MRR@10                        │                                   │     • Answer Relevancy              │
   │     • nDCG@10                       │                                   │     • Context Precision             │
   └──────────────────┬──────────────────┘                                   │     • Lexical: BLEU-4, ROUGE-L, F1  │
                      │                                                      └──────────────────┬──────────────────┘
                      ▼                                                                         ▼
   ┌─────────────────────────────────────────┐                           ┌─────────────────────────────────────────┐
   │ reports/vidore/<timestamp>/             │                           │ reports/generation/<timestamp>/         │
   │ • <dataset>_scorecard.md                │                           │ • <dataset>_<config>_scorecard.md       │
   │ • <dataset>_report.json                 │                           │ • <dataset>_<config>_report.json        │
   └─────────────────────────────────────────┘                           └─────────────────────────────────────────┘
```

### Key Principles & Design Decisions
1. **Isolated Qdrant Test Collection**: Retrieval tests run in an isolated Qdrant collection (`eval_vidore_v3`) configured with multivector late-interaction (`MultiVectorComparator.MAX_SIM`), keeping production collections (`personal_collection`) untouched.
2. **6 GB VRAM Hardware Lifecycle**: On constrained GPUs (e.g., RTX 3060/4050 6GB), loading ColPali and Qwen3-VL concurrently causes immediate Out-Of-Memory (OOM). The evaluation harness executes sequentially:
   - Evaluates retrieval $\rightarrow$ explicitly calls `release_embedder()` and flushes CUDA cache.
   - Evaluates generation in 4-bit NF4 with `batch_size=1` $\rightarrow$ calls `release_generator()`.
   - Frees GPU memory before running the LLM-as-a-judge.
3. **Strict LLM-as-a-Judge with Retry & Backoff**: Evaluates generation fidelity using OpenRouter's universal API (default: `nvidia/nemotron-3.5-super-vl:free`). Built-in exponential backoff automatically recovers from transient upstream `502 Bad Gateway` blips, and request pacing (0.5s) avoids free-tier rate limits.
4. **Timestamped Historical Logging**: Reports are permanently saved with ISO-8601 timestamps (`reports/vidore/` and `reports/generation/`) so different model configurations and revisions can be systematically compared over time.

---

## 2. Environment Configuration

The evaluation framework reads the following variables from your `.env` file:

```env
# --- Qdrant Settings ---
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
EVAL_COLLECTION_NAME=eval_vidore_v3
EVAL_VECTOR_NAME=page_embedding

# --- ColPali Model Settings ---
COLPALI_MODEL=vidore/colpali-v1.3-merged
COLPALI_MODEL_REVISION=main
QUANTIZATION=4bit
BNB_4BIT_QUANT_TYPE=nf4
BNB_4BIT_COMPUTE_DTYPE=bfloat16
BNB_4BIT_USE_DOUBLE_QUANT=true

# --- OpenRouter LLM-as-a-Judge Settings ---
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=nvidia/nemotron-3.5-super-vl:free
JUDGE_TIMEOUT_SEC=30
```

> [!IMPORTANT]
> Ensure the Qdrant container is active before running retrieval benchmarks (`docker start amg-qdrant`).

---

## 3. Pillar 1: Retrieval Evaluation (`run_vidore.py`)

The retrieval evaluation runner measures late-interaction multi-vector search quality over visual document pages using the ViDoRe v3 benchmark.

### 3.1 CLI Options & Parameters

```bash
python eval/run_vidore.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `str` | `vidore/docvqa_test_subsampled` | Dataset alias or Hugging Face ID (e.g. `docvqa`, `arxivqa`, `infovqa`, `tabfquad`, `ai`, `energy`, `healthcare`). |
| `--split` | `str` | `test` | Split to evaluate (`test`, `train`, `dev`). |
| `--sample` | `int` | `None` (all) | Limit number of samples for rapid benchmarking (e.g. `10`). |
| `--batch-size` | `int` | `1` | Batch size for ColPali image embedding (keep at 1 for 6GB VRAM). |
| `--k` | `str` | `1,3,5,10` | Comma-separated list of cutoff depths for metric evaluation. |
| `--collection` | `str` | `eval_vidore_v3` | Isolated Qdrant collection name to index into. |
| `--reindex` | `flag` | `False` | Force wipe and re-index the evaluation collection from scratch. |
| `--output-dir` | `Path` | `reports/vidore/` | Directory where markdown scorecards and JSON logs are saved. |

### 3.2 Supported Dataset Aliases
- `docvqa` $\rightarrow$ `vidore/docvqa_test_subsampled`
- `arxivqa` $\rightarrow$ `vidore/arxivqa_test_subsampled`
- `infovqa` $\rightarrow$ `vidore/infovqa_test_subsampled`
- `tabfquad` $\rightarrow$ `vidore/tabfquad_test_subsampled`
- `tatdqa` $\rightarrow$ `vidore/tatdqa_test`
- `shiftproject` $\rightarrow$ `vidore/shiftproject_test`
- `ai` $\rightarrow$ `vidore/syntheticDocQA_artificial_intelligence_test`
- `energy` $\rightarrow$ `vidore/syntheticDocQA_energy_test`
- `healthcare` $\rightarrow$ `vidore/syntheticDocQA_healthcare_industry_test`
- `government` $\rightarrow$ `vidore/syntheticDocQA_government_reports_test`

### 3.3 Command Usage Examples

#### Rapid Smoke Test (10 samples)
```bash
python eval/run_vidore.py --dataset docvqa --sample 10
```

#### Full Test Split Evaluation with Custom Cutoffs
```bash
python eval/run_vidore.py --dataset docvqa --split test --k 1,2,5,10
```

#### Reindex Evaluation Corpus
```bash
python eval/run_vidore.py --dataset docvqa --sample 50 --reindex
```

### 3.4 Metrics Computed
- **`Recall@K`**: Proportion of ground-truth relevant document pages successfully retrieved within the top-$K$ candidates:
  $$\text{Recall}@K = \frac{|\text{Retrieved}@K \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **`MRR@K`**: Mean Reciprocal Rank across queries, measuring where the first relevant page appears:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **`nDCG@K`**: Normalized Discounted Cumulative Gain, evaluating ranking order quality and penalizing relevant pages retrieved at lower ranks.
- **Latency & Indexing Stats**: Average search latency (ms/query), indexing throughput (pages/sec), and GPU VRAM snapshot.

---

## 4. Pillar 2: Generation Evaluation (`run_generation.py`)

The generation evaluation runner benchmarks `Qwen3-VL-2B-Instruct` on multi-hop question answering over document context, using an OpenRouter LLM-as-a-judge to score factual accuracy and grounding.

### 4.1 CLI Options & Parameters

```bash
python eval/run_generation.py [OPTIONS]
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `str` | `openrag` | Benchmark suite to evaluate (`openrag`). |
| `--config` | `str` | `hotpotqa` | Sub-benchmark configuration (e.g. `hotpotqa`, `2wikimultihopqa`). |
| `--split` | `str` | `test` | Split to evaluate (auto-falls back to `dev` if `test` is private). |
| `--sample` | `int` | `None` (all) | Limit number of queries for rapid benchmarking (e.g. `5`). |
| `--mode` | `str` | `given_context` | Evaluation mode (`given_context` for isolated generator evaluation). |
| `--judge-key` | `str` | `None` | Optional override for `OPENROUTER_API_KEY`. |
| `--judge-model` | `str` | `None` | Optional override for `OPENROUTER_MODEL` (e.g. `openai/gpt-4o-mini`). |
| `--judge-base-url` | `str` | `None` | Optional override for `OPENROUTER_BASE_URL`. |
| `--output-dir` | `Path` | `reports/generation/` | Directory where markdown scorecards and JSON logs are saved. |

### 4.2 Command Usage Examples

#### Rapid Smoke Test (2 samples, isolated mode)
```bash
python eval/run_generation.py --dataset openrag --config hotpotqa --mode given_context --sample 2
```

#### Larger Evaluation Run (20 samples)
```bash
python eval/run_generation.py --dataset openrag --config hotpotqa --mode given_context --sample 20
```

#### Override Judge Model with GPT-4o-Mini
```bash
python eval/run_generation.py --dataset openrag --config hotpotqa --mode given_context --sample 10 --judge-model openai/gpt-4o-mini
```

### 4.3 Metrics Computed

#### 1. LLM-as-a-Judge Metrics (OpenRouter)
- **`Faithfulness` (Groundedness)**:
  Deconstructs the generated answer into atomic claims and verifies whether each claim is supported by the supplied context without hallucination. Score range: `0.0` to `1.0` (target: $\ge 0.85$).
- **`Answer Relevancy`**:
  Judges whether the answer directly and concisely addresses the user's query, penalizing off-topic tangents or non-responsive answers. Score range: `0.0` to `1.0` (target: $\ge 0.80$).
- **`Context Precision`**:
  Evaluates whether the passages carrying ground-truth evidence are prioritized at top ranks.

#### 2. Lexical & Reference Comparison Metrics
- **`Exact Match (EM)`**: Binary exact match against normalized ground-truth answers.
- **`Token F1`**: Token-level harmonic mean of precision and recall against ground truth.
- **`ROUGE-L`**: Longest Common Subsequence (LCS) overlap measuring sentence structure alignment.
- **`BLEU-4`**: Sentence-level 4-gram precision with Chen & Cherry smoothing.

---

## 5. Report Artifacts & Historical Logging

All benchmark executions produce two synchronized artifacts in dedicated timestamped subdirectories:
`reports/<suite>/<timestamp>/` (e.g. `reports/vidore/20261021_112030/` or `reports/generation/20261021_112030/`).

### 5.1 Timestamped Markdown Scorecards
Saved to `reports/<suite>/<timestamp>/<dataset>_scorecard.md`. Also duplicated to `reports/<suite>/latest_scorecard.md` for quick inspection.

Example output:
```markdown
# Generation & Ragas Evaluation Scorecard: OPENRAG (hotpotqa)

- **Timestamp**: `2026-10-19T19:50:21.763759`
- **Generator Model**: `Qwen/Qwen3-VL-2B-Instruct` (4bit nf4)
- **Evaluation Mode**: `given_context`
- **Evaluated Samples**: `2`
- **Judge**: `OpenRouter (nvidia/nemotron-3.5-super-vl:free)`
- **Average Generation Latency**: `10804.01 ms / query`

### 1. Grounding & Ragas Metrics

| Metric | Score | Target | Description |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **1.0000** | $\ge 0.85$ | Ratio of claims verified by context (no hallucinations) |
| **Answer Relevancy** | **1.0000** | $\ge 0.80$ | Measure of how well answer directly addresses query |
| **Context Precision** | **0.5000** | $\ge 0.75$ | Quality and ranking of context passages |

### 2. Lexical & Semantic Accuracy Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Exact Match (EM)** | **0.0000** | Normalized exact match against reference answers |
| **Token F1** | **0.1059** | Token-level harmonic precision and recall |
| **ROUGE-L** | **0.1059** | Longest common subsequence recall and precision |
| **BLEU-4** | **0.0448** | Sentence-level 4-gram precision with brevity penalty |
```

### 5.2 Structured JSON Run Logs
Saved alongside the scorecard as `reports/<suite>/<timestamp>/<dataset>_report.json`.

Contains full benchmark telemetry, aggregate scores, and detailed per-sample records:
1. **ViDoRe Retrieval JSON**:
   - `timestamp`, `dataset`, `model`, `quantization`, `metrics` (`Recall@K`, `MRR@K`, `nDCG@K`), `stats`.
   - `samples`: List of sample objects containing:
     - `sample_id` & `query`
     - `desired_result`: `relevant_doc_ids` and `ground_truth_answer`
     - `retrieval_output`: Ranked candidates with `rank`, `doc_id`, `score`, and `is_hit` boolean
     - `sample_metadata`: Image filename, source, page, data split
2. **Generation & Judge JSON**:
   - `timestamp`, `dataset`, `config`, `mode`, `model`, `judge`, `metrics`, `avg_latency_ms`.
   - `samples`: List of sample objects containing:
     - `sample_id` & `query`
     - `desired_result`: `ground_truth` and `gold_contexts`
     - `generation_output`: `answer` and `latency_ms`
     - `retrieval_output`: `retrieved_sources` and `contexts` provided to model
     - `judge_evaluation`: `faithfulness`, `relevancy`, `context_precision`, `exact_match`, `f1`, `rouge_l`, `bleu_4`
     - `sample_metadata`: Document titles, multi-hop info, and dataset tags

---

## 6. Troubleshooting & FAQs

### Q: Why did an OpenRouter call fail with `HTTP 502: Bad Gateway`?
Free-tier endpoints on OpenRouter (`:free`) occasionally experience upstream drops during high-traffic periods. `OpenRouterJudge` includes automatic 3-stage exponential backoff (retries at 1.5s, 3.0s) and a 0.5s pacing pause between calls. If issues persist on the free tier, switch to `openai/gpt-4o-mini` via `--judge-model openai/gpt-4o-mini`.

### Q: What prevents CUDA Out-Of-Memory (OOM) during evaluation?
AMG runs inference sequentially and explicitly unloads models:
```python
# Free retrieval encoder before generation:
release_query_encoder()

# Free generator before evaluation:
generator._release_model()
_cleanup_cuda_memory()
```
Always ensure `--batch-size 1` is maintained on GPUs with $\le 6\text{ GB}$ VRAM.
