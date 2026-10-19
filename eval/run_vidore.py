"""Main CLI entry point for ViDoRe v3 Visual Document Retrieval Benchmark."""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

# Ensure repo root is on sys.path and remove script dir to avoid shadowing HuggingFace 'datasets'
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.config import (
    DEFAULT_DATASET,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CUTOFFS,
    EVAL_COLLECTION_NAME,
    EVAL_VECTOR_NAME,
    REPORTS_DIR,
    COLPALI_MODEL,
    QUANTIZATION,
    BNB_4BIT_QUANT_TYPE,
)
from eval.datasets.vidore_loader import load_vidore_dataset
from eval.retrieval.indexer import ViDoReIndexer
from eval.retrieval.runner import ViDoReRunner
from eval.retrieval.metrics import compute_all_metrics
from query import release_query_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("amg-eval-vidore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViDoRe v3 Visual Document Retrieval Benchmark on AMG",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="ViDoRe dataset name or alias (e.g. 'docvqa', 'infovqa', 'arxivqa', 'tabf_factual')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: 'test')",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional limit on number of samples for fast testing (e.g. 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding images (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=",".join(map(str, DEFAULT_CUTOFFS)),
        help="Comma-separated cutoffs (e.g. '1,3,5,10')",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=EVAL_COLLECTION_NAME,
        help=f"Qdrant evaluation collection name (default: {EVAL_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing by deleting existing evaluation collection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save JSON/Markdown benchmark reports",
    )
    return parser.parse_args()


def format_markdown_scorecard(
    dataset_name: str,
    metrics: dict[str, float],
    stats: dict[str, float],
    cutoffs: list[int],
    sample_size: int,
    num_corpus: int,
) -> str:
    lines = [
        "# ViDoRe v3 Retrieval Benchmark Scorecard",
        "",
        f"- **Timestamp**: `{datetime.now().isoformat()}`",
        f"- **Dataset**: `{dataset_name}`",
        f"- **Model**: `{COLPALI_MODEL}` ({QUANTIZATION} {BNB_4BIT_QUANT_TYPE})",
        f"- **Total Queries Evaluated**: `{sample_size}`",
        f"- **Total Corpus Pages**: `{num_corpus}`",
        f"- **Avg Latency per Query**: `{stats.get('avg_latency_ms', 0)} ms`",
        f"- **Total Benchmark Time**: `{stats.get('total_time_sec', 0)} s`",
        "",
        "### Key Retrieval Metrics",
        "",
        "| Metric | Score | Description |",
        "| :--- | :--- | :--- |",
    ]

    for k in cutoffs:
        r_key = f"Recall@{k}"
        if r_key in metrics:
            lines.append(f"| **{r_key}** | **{metrics[r_key]:.4f}** | Ground-truth page within top {k} |")

    max_k = max(cutoffs)
    mrr_key = f"MRR@{max_k}"
    if mrr_key in metrics:
        lines.append(f"| **{mrr_key}** | **{metrics[mrr_key]:.4f}** | Mean Reciprocal Rank at {max_k} |")

    ndcg_key = f"nDCG@{max_k}"
    if ndcg_key in metrics:
        lines.append(f"| **{ndcg_key}** | **{metrics[ndcg_key]:.4f}** | Normalized Discounted Cumulative Gain at {max_k} |")

    lines.append("")
    return "\n".join(lines)


def run_benchmark():
    args = parse_args()
    cutoffs = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    max_k = max(cutoffs) if cutoffs else 10

    # 1. Load Dataset
    print("\n" + "=" * 60)
    print(" 🚀 AMG ViDoRe v3 Retrieval Evaluation Harness")
    print("=" * 60)
    dataset = load_vidore_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.sample,
    )

    if dataset.num_queries == 0 or dataset.num_documents == 0:
        logger.error("Dataset has no queries or documents. Exiting.")
        sys.exit(1)

    # 2. Index Corpus into Isolated Collection
    indexer = ViDoReIndexer(
        collection_name=args.collection,
        vector_name=EVAL_VECTOR_NAME,
    )
    indexer.index_corpus(
        corpus=dataset.corpus,
        dataset_name=dataset.dataset_name,
        batch_size=args.batch_size,
        reindex=args.reindex,
    )
    indexer.release_embedder()

    # 3. Execute Queries
    runner = ViDoReRunner(
        collection_name=args.collection,
        vector_name=EVAL_VECTOR_NAME,
    )
    run_results, stats = runner.run_queries(
        queries=dataset.queries,
        top_k=max_k,
    )

    # 4. Compute Metrics
    metrics = compute_all_metrics(
        qrels=dataset.qrels,
        run_results=run_results,
        cutoffs=cutoffs,
    )

    # 5. Format & Display Results
    md_report = format_markdown_scorecard(
        dataset_name=dataset.dataset_name,
        metrics=metrics,
        stats=stats,
        cutoffs=cutoffs,
        sample_size=dataset.num_queries,
        num_corpus=dataset.num_documents,
    )

    print("\n" + md_report + "\n")

    # 6. Save Reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = dataset.dataset_name.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = args.output_dir / f"{slug}_{ts}.json"
    md_path = args.output_dir / f"{slug}_{ts}.md"
    latest_md_path = args.output_dir / "latest_scorecard.md"

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset.dataset_name,
        "model": COLPALI_MODEL,
        "quantization": f"{QUANTIZATION}_{BNB_4BIT_QUANT_TYPE}",
        "num_queries": dataset.num_queries,
        "num_corpus": dataset.num_documents,
        "metrics": metrics,
        "stats": stats,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    with open(latest_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info("Saved JSON report to %s", json_path)
    logger.info("Saved Markdown scorecard to %s", md_path)

    # Clean up GPU memory
    release_query_encoder()
    print("=" * 60)
    print(" Benchmark completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
