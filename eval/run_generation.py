"""CLI runner for End-to-End Generation & Ragas Evaluation (OpenRAG & RAGBench)."""

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

from eval.datasets.openrag_loader import load_openrag_dataset
from eval.generation.runner import GenerationRunner
from eval.ragas_eval.evaluator import RagasEvaluator
from generate import GENERATOR_MODEL, GENERATOR_BNB_4BIT_QUANT_TYPE, GENERATOR_USE_4BIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("amg-eval-generation-cli")

DEFAULT_OUTPUT_DIR = _REPO_ROOT / "reports" / "generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Generation and Ragas Evaluation on OpenRAG or RAGBench",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openrag",
        choices=["openrag", "ragbench"],
        help="Benchmark suite ('openrag' or 'ragbench')",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hotpotqa",
        help="Sub-benchmark configuration (OpenRAG: 'hotpotqa', '2wikimultihopqa'; RAGBench: 'covidqa', 'finqa', etc.)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate ('test', 'dev', 'train')",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional limit on number of samples for rapid benchmarking",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="given_context",
        choices=["given_context", "end_to_end"],
        help="'given_context' evaluates generation on benchmark context; 'end_to_end' retrieves first via Qdrant",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K documents to retrieve when using --mode end_to_end (default: 3)",
    )
    parser.add_argument(
        "--judge-key",
        type=str,
        default=None,
        help="Optional OpenRouter API key override (defaults to OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model name for OpenRouter (e.g. 'openai/gpt-4o-mini', 'google/gemini-flash-1.5')",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="OpenRouter / OpenAI-compatible base URL (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for scorecards and JSON predictions",
    )
    return parser.parse_args()


def format_generation_scorecard(
    dataset_name: str,
    config: str,
    mode: str,
    num_samples: int,
    metrics: dict[str, float],
    avg_latency_ms: float,
    judge_name: str,
) -> str:
    quant_str = f"4bit {GENERATOR_BNB_4BIT_QUANT_TYPE}" if GENERATOR_USE_4BIT else "fp16"
    lines = [
        f"# Generation & Ragas Evaluation Scorecard: {dataset_name.upper()} ({config})",
        "",
        f"- **Timestamp**: `{datetime.now().isoformat()}`",
        f"- **Generator Model**: `{GENERATOR_MODEL}` ({quant_str})",
        f"- **Evaluation Mode**: `{mode}`",
        f"- **Evaluated Samples**: `{num_samples}`",
        f"- **Judge**: `{judge_name}`",
        f"- **Average Generation Latency**: `{avg_latency_ms:.2f} ms / query`",
        "",
        "### 1. Grounding & Ragas Metrics",
        "",
        "| Metric | Score | Target | Description |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Faithfulness** | **{metrics.get('Faithfulness', 0):.4f}** | $\\ge 0.85$ | Ratio of claims verified by context (no hallucinations) |",
        f"| **Answer Relevancy** | **{metrics.get('AnswerRelevancy', 0):.4f}** | $\\ge 0.80$ | Measure of how well answer directly addresses query |",
        f"| **Context Precision** | **{metrics.get('ContextPrecision', 0):.4f}** | $\\ge 0.75$ | Quality and ranking of context passages |",
        "",
        "### 2. Lexical & Semantic Accuracy Metrics",
        "",
        "| Metric | Score | Description |",
        "| :--- | :--- | :--- |",
        f"| **Exact Match (EM)** | **{metrics.get('ExactMatch', 0):.4f}** | Normalized exact match against reference answers |",
        f"| **Token F1** | **{metrics.get('F1', 0):.4f}** | Token-level harmonic precision and recall |",
        f"| **ROUGE-L** | **{metrics.get('ROUGE-L', 0):.4f}** | Longest common subsequence recall and precision |",
        f"| **BLEU-4** | **{metrics.get('BLEU-4', 0):.4f}** | Sentence-level 4-gram precision with brevity penalty |",
        "",
    ]
    return "\n".join(lines)


def run_eval():
    args = parse_args()

    print("\n" + "=" * 65)
    print(f" 🚀 AMG Generation & Ragas Evaluation Harness: {args.dataset.upper()} ({args.config})")
    print(f"    Mode: {args.mode.upper()} | Top-K: {args.top_k}")
    print("=" * 65)

    # 1. Load Dataset
    dataset = load_openrag_dataset(
        config=args.config,
        split=args.split,
        max_samples=args.sample,
    )

    if len(dataset) == 0:
        logger.error("Dataset has 0 samples. Exiting.")
        sys.exit(1)

    # 2. Run Inference (Mode 1: given_context or Mode 2: end_to_end)
    runner = GenerationRunner()
    predictions = runner.generate_for_dataset(
        dataset=dataset,
        mode=args.mode,
        top_k=args.top_k,
        release_after=True,
    )

    # 3. Evaluate with Ragas / OpenRouter Judge
    from eval.ragas_eval.judge_factory import get_judge
    judge = get_judge(
        api_key=args.judge_key,
        base_url=args.judge_base_url,
        model=args.judge_model,
    )
    judge_name = f"OpenRouter ({judge.model})" if hasattr(judge, "model") else "HeuristicJudge (Offline fallback)"

    evaluator = RagasEvaluator(judge=judge)
    eval_results = evaluator.evaluate_predictions(predictions)
    metrics = eval_results["metrics"]

    latencies = [p["latency_ms"] for p in predictions if p.get("latency_ms")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # 4. Display Scorecard
    scorecard_md = format_generation_scorecard(
        dataset_name=args.dataset,
        config=args.config,
        mode=args.mode,
        num_samples=len(predictions),
        metrics=metrics,
        avg_latency_ms=avg_latency,
        judge_name=judge_name,
    )

    print("\n" + scorecard_md + "\n")

    # 5. Save Reports
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.dataset}_{args.config}"

    json_path = run_dir / f"{slug}_report.json"
    md_path = run_dir / f"{slug}_scorecard.md"
    latest_path = args.output_dir / "latest_scorecard.md"

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "mode": args.mode,
        "model": GENERATOR_MODEL,
        "judge": judge_name,
        "num_samples": len(predictions),
        "avg_latency_ms": avg_latency,
        "metrics": metrics,
        "samples": eval_results["sample_scores"],
        "sample_scores": eval_results["sample_scores"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(scorecard_md)

    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(scorecard_md)

    logger.info("Saved JSON report to %s", json_path)
    logger.info("Saved Markdown scorecard to %s", md_path)

    print("=" * 65)
    print(" Generation evaluation completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    run_eval()
