"""Inference runner for generation evaluation on multimodal and text datasets."""

import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional
from tqdm import tqdm

import torch

from eval.datasets.common import EvalDataset, EvalSample
from generate import (
    get_generator,
    _cleanup_cuda_memory,
    query_and_generate,
)

logger = logging.getLogger("amg-eval-generation")


class GenerationRunner:
    """Orchestrates VLM/LLM answer generation for evaluation benchmarks with VRAM safety."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.generator = get_generator()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "amg_eval_images"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_pages_from_sample(self, sample: EvalSample) -> list[dict[str, Any]]:
        """Convert EvalSample contexts into generate.py's expected page structures."""
        pages: list[dict[str, Any]] = []

        # Handle multimodal PIL images
        image_paths: list[Optional[str]] = []
        for img_idx, img in enumerate(sample.context_images, start=1):
            img_path = self.temp_dir / f"{sample.id}_page_{img_idx}.png"
            if not img_path.exists():
                img.save(str(img_path))
            image_paths.append(str(img_path))

        # Merge contexts and images
        num_contexts = max(len(sample.contexts), len(image_paths), 1)
        for i in range(num_contexts):
            rank = i + 1
            ctx_text = sample.contexts[i] if i < len(sample.contexts) else ""
            img_path = image_paths[i] if i < len(image_paths) else None

            pages.append(
                {
                    "rank": rank,
                    "score": 1.0,
                    "document_id": sample.id,
                    "version_id": "eval_v1",
                    "version_number": 1,
                    "filename": f"context_{rank}",
                    "page_number": rank,
                    "page_hash": f"eval_hash_{rank}",
                    "page_text": ctx_text,
                    "page_image_path": img_path,
                }
            )

        return pages

    def generate_for_dataset(
        self,
        dataset: EvalDataset,
        mode: str = "given_context",
        top_k: int = 3,
        release_after: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run inference across all samples in the dataset.

        Parameters
        ----------
        dataset:
            The evaluation dataset to process.
        mode:
            'given_context': generate from benchmark's ground-truth contexts.
            'end_to_end': retrieve top-K from Qdrant first, then generate.
        top_k:
            Top-K documents to retrieve if mode is 'end_to_end'.
        release_after:
            If True, unload generator model from GPU VRAM after completion.

        Returns
        -------
        list[dict]:
            Structured predictions containing query, ground_truth, and generated_answer.
        """
        predictions: list[dict[str, Any]] = []
        logger.info(
            "Starting generation for dataset '%s' (%d samples, mode=%s)...",
            dataset.name,
            len(dataset),
            mode,
        )

        t_start = time.perf_counter()

        for sample in tqdm(dataset.samples, desc=f"Generating Answers ({dataset.name})"):
            q_start = time.perf_counter()

            retrieved_contexts = sample.contexts
            retrieved_sources = []

            try:
                if mode == "end_to_end":
                    res = query_and_generate(query=sample.query, top_k=top_k)
                    gen_response = res.get("response", {})
                    answer = gen_response.get("answer", "")
                    duration_ms = gen_response.get("generation", {}).get("duration_ms", 0)
                    retrieval_info = res.get("retrieval", {})
                    retrieved_pages = retrieval_info.get("results", [])
                    retrieved_sources = retrieved_pages
                    # In end_to_end, judge evaluates faithfulness against the actual retrieved pages
                    retrieved_contexts = [
                        p.get("page_text") or f"[Image from {p.get('filename')} page {p.get('page_number')}]"
                        for p in retrieved_pages
                    ]
                else:
                    pages = self._prepare_pages_from_sample(sample)
                    retrieval_stub = {"results": pages}
                    gen_response = self.generator.generate_from_retrieval(
                        query=sample.query,
                        retrieval_result=retrieval_stub,
                    )
                    answer = gen_response.answer
                    duration_ms = gen_response.generation.get("duration_ms", 0)
                    retrieved_sources = gen_response.sources

            except Exception as exc:
                logger.error("Generation failed on sample %s: %s", sample.id, exc)
                answer = f"ERROR: {exc}"
                duration_ms = round((time.perf_counter() - q_start) * 1000, 2)

            predictions.append(
                {
                    "sample_id": sample.id,
                    "query": sample.query,
                    "ground_truth": sample.ground_truth,
                    "ground_truth_text": sample.ground_truth_text,
                    "generated_answer": answer,
                    "contexts": retrieved_contexts,
                    "gold_contexts": sample.contexts,
                    "retrieved_sources": retrieved_sources,
                    "is_multimodal": sample.is_multimodal,
                    "latency_ms": duration_ms,
                    "hallucination_label": sample.hallucination_label,
                    "metadata": sample.metadata,
                }
            )

        total_time = round(time.perf_counter() - t_start, 2)
        logger.info(
            "Generation completed: %d samples in %.2fs",
            len(predictions),
            total_time,
        )

        if release_after:
            self.release_generator()

        return predictions

    def release_generator(self) -> None:
        """Release Qwen-VL model weights from GPU memory to free VRAM for evaluation/other tasks."""
        if self.generator is not None:
            self.generator._release_model()
            _cleanup_cuda_memory()
            logger.info("Released Qwen-VL generator model from GPU VRAM.")
