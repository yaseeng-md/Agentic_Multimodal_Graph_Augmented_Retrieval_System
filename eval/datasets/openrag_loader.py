"""Loader for OpenRAG Benchmark (Text & Multimodal RAG)."""

import logging
from typing import Optional
from PIL import Image

try:
    from datasets import load_dataset
except ImportError as exc:
    raise RuntimeError("The 'datasets' package is required. Run `pip install datasets`.") from exc

from eval.datasets.common import EvalDataset, EvalSample

logger = logging.getLogger("amg-eval-openrag")

DEFAULT_OPENRAG_BENCHMARK = "shayekh/openrag_bench"
VALID_CONFIGS = ("hotpotqa", "2wikimultihopqa", "musique")


def load_openrag_dataset(
    config: str = "hotpotqa",
    split: str = "test",
    max_samples: Optional[int] = None,
    dataset_name: str = DEFAULT_OPENRAG_BENCHMARK,
) -> EvalDataset:
    """
    Load an OpenRAG benchmark split.

    Parameters
    ----------
    config:
        Benchmark config name ('hotpotqa', '2wikimultihopqa', 'musique').
    split:
        Data split, defaults to 'test' (falls back to 'validation' or 'train' if test absent).
    max_samples:
        Optional sample limit for rapid evaluation.
    dataset_name:
        Hugging Face repo name.

    Returns
    -------
    EvalDataset:
        Normalized dataset with query, ground-truth answers, and reference contexts.
    """
    cfg = config.lower().strip()
    logger.info("Loading OpenRAG benchmark: %s (config=%s, split=%s)", dataset_name, cfg, split)

    try:
        ds = load_dataset(dataset_name, cfg, split=split)
    except Exception as exc:
        logger.info("Split '%s' not directly available for %s (%s). Attempting fallbacks...", split, dataset_name, cfg)
        found = False
        for fallback in ("dev", "validation", "test", "train"):
            try:
                ds = load_dataset(dataset_name, cfg, split=fallback)
                logger.info("Loaded fallback split '%s'", fallback)
                found = True
                break
            except Exception:
                continue
        if not found:
            loaded = load_dataset(dataset_name, cfg)
            first_split = list(loaded.keys())[0]
            ds = loaded[first_split]
            logger.info("Loaded first available split '%s'", first_split)

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples: list[EvalSample] = []
    for idx, row in enumerate(ds):
        sample_id = str(row.get("id") or f"openrag_{cfg}_{idx:05d}")
        query = str(row.get("question") or row.get("query") or "").strip()
        answers = row.get("answers") or [row.get("answer")] or []

        # Extract contexts
        raw_ctxs = row.get("ctxs") or row.get("contexts") or []
        contexts: list[str] = []
        for ctx in raw_ctxs:
            if isinstance(ctx, dict):
                text = ctx.get("text") or ctx.get("content") or ""
                if text:
                    contexts.append(str(text))
            elif isinstance(ctx, str):
                contexts.append(ctx)

        # Check for image contexts if multimodal
        context_images: list[Image.Image] = []
        raw_img = row.get("image")
        if raw_img is not None:
            if isinstance(raw_img, Image.Image):
                context_images.append(raw_img.convert("RGB"))
            else:
                context_images.append(Image.open(raw_img).convert("RGB"))

        samples.append(
            EvalSample(
                id=sample_id,
                query=query,
                ground_truth=answers,
                contexts=contexts,
                context_images=context_images,
                metadata={"config": cfg, "source": dataset_name},
            )
        )

    logger.info("Loaded OpenRAG (%s): %d samples", cfg, len(samples))
    return EvalDataset(name=f"openrag_{cfg}", samples=samples)
