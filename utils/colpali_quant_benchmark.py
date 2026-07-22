#!/usr/bin/env python3
"""ColPali local cache + FP16 vs NF4 benchmark.

What this script does:
1) Downloads ColPali once and stores it in a local directory.
2) Loads the local model in FP16.
3) Loads the same local model in NF4 (bitsandbytes 4-bit) when supported.
4) Benchmarks memory, latency, and retrieval agreement on a PDF.
5) Writes a Markdown report and JSON metrics file.

Important design note:
- bitsandbytes NF4 is applied at load time from the local snapshot. The script
  keeps the original snapshot on disk and loads NF4 from that local copy.
- This is the most reliable production pattern for Transformers-based models.

Requirements:
    pip install -U torch transformers bitsandbytes accelerate pymupdf pillow numpy
    pip install -U colpali_engine

Optional:
    pip install -U pandas

Usage example:
    python colpali_quant_benchmark.py \
        --pdf "data/pdfs/Yaseen_Resume.pdf" \
        --query "What is the Name of Candidate?" \
        --expected-page-index 1 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
import pymupdf  # PyMuPDF
from transformers import BitsAndBytesConfig

from colpali_engine.models import ColPali, ColPaliProcessor
from importlib.metadata import version, PackageNotFoundError
import colpali_engine

LOGGER = logging.getLogger("colpali_benchmark")


# -----------------------------
# Configuration
# -----------------------------


@dataclass(frozen=True)
class AppConfig:
    model_id: str = "vidore/colpali-v1.3"
    local_model_dir: Path = Path("./models/colpali-v1.3")
    output_dir: Path = Path("./benchmark_results")
    pdf_path: Path = Path("./data/pdfs/Yaseen_Resume.pdf")
    device: str = "cuda:0"
    dtype: str = "float16"
    dpi: int = 96
    max_pages: int | None = None
    batch_size: int = 1
    queries: tuple[str, ...] = ("What is the Name of Candidate?",)
    expected_page_index: int | None = None
    use_local_files_only: bool = True


@dataclass
class VariantMetrics:
    name: str
    load_time_s: float = 0.0
    model_allocated_gb: float = 0.0
    model_reserved_gb: float = 0.0
    peak_load_gb: float = 0.0
    peak_infer_gb: float = 0.0
    embedding_pages_s: float = 0.0
    query_embed_s: float = 0.0
    retrieval_top1_page_index: int | None = None
    retrieval_top1_score: float | None = None
    retrieval_hit: bool | None = None
    disk_size_gb: float = 0.0
    notes: str = ""


@dataclass
class BenchmarkReport:
    timestamp_utc: str
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    transformers_version: str
    colpali_engine_version: str
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    fp16: VariantMetrics = field(default_factory=lambda: VariantMetrics(name="fp16"))
    nf4: VariantMetrics = field(default_factory=lambda: VariantMetrics(name="nf4"))
    recommendation: str = ""


# -----------------------------
# Logging / utilities
# -----------------------------


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "benchmark.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    LOGGER.info("Logging to %s", log_path)


def select_device(requested_device: str) -> tuple[str, torch.dtype]:
    if requested_device.lower().startswith("cuda") and torch.cuda.is_available():
        return requested_device, torch.float16
    if requested_device.lower().startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def safe_json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def gb(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -----------------------------
# PDF utilities
# -----------------------------


def render_pdf_to_images(pdf_path: Path, dpi: int, max_pages: int | None = None) -> list[Image.Image]:
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    try:
        for idx, page in enumerate(doc):
            if max_pages is not None and idx >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        doc.close()
    return images


def image_sha256(image: Image.Image) -> str:
    return sha256_bytes(image.convert("RGB").tobytes())


# -----------------------------
# Model cache and loaders
# -----------------------------


def snapshot_local_model(model_id: str, local_dir: Path, force_refresh: bool = False) -> None:
    """Download the base model once and store it locally.

    If the directory already exists and looks populated, we reuse it unless
    force_refresh is True.
    """
    marker = local_dir / "config.json"
    if marker.exists() and not force_refresh:
        LOGGER.info("Using existing local snapshot at %s", local_dir)
        return

    ensure_dir(local_dir)
    LOGGER.info("Downloading %s to %s", model_id, local_dir)

    # Load the original model once (usually FP16 on GPU) and persist it locally.
    model = ColPali.from_pretrained(model_id).eval()
    processor = ColPaliProcessor.from_pretrained(model_id)

    model.save_pretrained(local_dir)
    processor.save_pretrained(local_dir)

    metadata = {
        "model_id": model_id,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Base snapshot for local FP16/NF4 loading",
    }
    safe_json_dump(local_dir / "snapshot_metadata.json", metadata)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_disk_usage_gb(local_dir: Path) -> float:
    return gb(dir_size_bytes(local_dir))


def gpu_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "allocated_gb": gb(torch.cuda.memory_allocated()),
        "reserved_gb": gb(torch.cuda.memory_reserved()),
        "peak_allocated_gb": gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
    }


def reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def load_fp16_model(local_dir: Path, device: str, local_files_only: bool = True) -> tuple[ColPali, ColPaliProcessor]:
    LOGGER.info("Loading FP16 model from local dir: %s", local_dir)
    model = ColPali.from_pretrained(
        str(local_dir),
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=local_files_only,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(str(local_dir), local_files_only=local_files_only)
    return model, processor


def load_nf4_model(local_dir: Path, local_files_only: bool = True) -> tuple[ColPali, ColPaliProcessor]:
    LOGGER.info("Loading NF4 model from local dir: %s", local_dir)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = ColPali.from_pretrained(
        str(local_dir),
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=local_files_only,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(str(local_dir), local_files_only=local_files_only)
    return model, processor


# -----------------------------
# Embedding / retrieval
# -----------------------------


@torch.inference_mode()
def embed_page_batch(
    model: ColPali,
    processor: ColPaliProcessor,
    images: Sequence[Image.Image],
) -> list[dict[str, torch.Tensor]]:
    if not images:
        return []

    processed = processor.process_images(list(images)).to(model.device)
    outputs = model(**processed)

    results: list[dict[str, torch.Tensor]] = []
    for idx, image in enumerate(images):
        image_embedding = outputs[idx]
        mask = processed.input_ids[idx] == processor.image_token_id
        x_patches, y_patches = processor.get_n_patches(
            image.size,
            patch_size=model.config.vision_config.patch_size,
        )
        image_patch_embeddings = image_embedding[mask].view(x_patches, y_patches, model.dim)

        pooled_rows = image_patch_embeddings.mean(dim=0)
        pooled_cols = image_patch_embeddings.mean(dim=1)

        special_tokens = image_embedding[~mask]
        pooled_rows = torch.cat([pooled_rows, special_tokens], dim=0)
        pooled_cols = torch.cat([pooled_cols, special_tokens], dim=0)

        results.append(
            {
                "original": image_embedding.detach().cpu(),
                "rows": pooled_rows.detach().cpu(),
                "cols": pooled_cols.detach().cpu(),
            }
        )

    return results


@torch.inference_mode()
def embed_query(model: ColPali, processor: ColPaliProcessor, query: str) -> torch.Tensor:
    processed = processor.process_queries([query]).to(model.device)
    embedding = model(**processed)[0]
    return embedding.detach().cpu()


def maxsim_score(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
    """ColBERT-style late interaction score.

    query_emb: (q_tokens, dim)
    doc_emb:   (d_tokens, dim)
    """
    if query_emb.ndim != 2 or doc_emb.ndim != 2:
        raise ValueError("Expected 2D embeddings")

    # Similarity matrix: q_tokens x d_tokens
    sims = query_emb @ doc_emb.T
    score = sims.max(dim=1).values.sum().item()
    return float(score)


@dataclass
class PageEmbeddings:
    original: list[torch.Tensor]
    rows: list[torch.Tensor]
    cols: list[torch.Tensor]


def score_pages(query_emb: torch.Tensor, page_embs: Sequence[torch.Tensor]) -> tuple[int, float, list[float]]:
    scores = [maxsim_score(query_emb, page) for page in page_embs]
    top_idx = int(np.argmax(scores))
    return top_idx, float(scores[top_idx]), scores


# -----------------------------
# Benchmark core
# -----------------------------


def benchmark_variant(
    name: str,
    load_fn,
    local_dir: Path,
    cfg: AppConfig,
    images: Sequence[Image.Image],
    queries: Sequence[str],
    expected_page_index: int | None,
) -> tuple[VariantMetrics, PageEmbeddings]:
    metrics = VariantMetrics(name=name)

    if torch.cuda.is_available():
        reset_cuda_stats()

    start_load = time.perf_counter()
    model, processor = load_fn(local_dir)
    load_time = time.perf_counter() - start_load

    stats_after_load = gpu_stats()
    metrics.load_time_s = load_time
    metrics.model_allocated_gb = stats_after_load["allocated_gb"]
    metrics.model_reserved_gb = stats_after_load["reserved_gb"]
    metrics.peak_load_gb = stats_after_load["peak_allocated_gb"]

    LOGGER.info(
        "[%s] loaded in %.2fs | alloc=%.2f GB | reserved=%.2f GB",
        name,
        metrics.load_time_s,
        metrics.model_allocated_gb,
        metrics.model_reserved_gb,
    )

    # Disk usage is tied to the local snapshot, not the runtime quantized weights.
    metrics.disk_size_gb = model_disk_usage_gb(local_dir)

    # Page embeddings
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_pages = time.perf_counter()
    page_batches = embed_page_batch(model, processor, images)
    metrics.embedding_pages_s = time.perf_counter() - start_pages
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics.peak_infer_gb = gb(torch.cuda.max_memory_allocated())

    # Query embeddings + retrieval agreement on the first query only for a simple production signal.
    query_times: list[float] = []
    top1_page_index: int | None = None
    top1_score: float | None = None
    hit: bool | None = None

    page_originals = [batch["original"] for batch in page_batches]

    for qi, query in enumerate(queries):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        q_start = time.perf_counter()
        q_emb = embed_query(model, processor, query)
        query_times.append(time.perf_counter() - q_start)

        top_idx, score, _ = score_pages(q_emb, page_originals)
        if qi == 0:
            top1_page_index = top_idx
            top1_score = score
            hit = expected_page_index is not None and top_idx == expected_page_index

    metrics.query_embed_s = float(sum(query_times) / max(len(query_times), 1))
    metrics.retrieval_top1_page_index = top1_page_index
    metrics.retrieval_top1_score = top1_score
    metrics.retrieval_hit = hit

    # Housekeeping
    del model, processor, page_batches, page_originals
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, PageEmbeddings(original=[], rows=[], cols=[])


# -----------------------------
# Reporting
# -----------------------------


def choose_recommendation(fp16: VariantMetrics, nf4: VariantMetrics) -> str:
    """Simple, explainable decision rule.

    Prefer NF4 when it materially reduces memory and the retrieval result is
    either identical or not worse on the benchmark query.
    """
    if fp16.retrieval_hit is True and nf4.retrieval_hit is False:
        return "FP16 is safer for retrieval quality on this benchmark."

    # If NF4 cuts memory by at least 30% and keeps the top-1 hit, prefer NF4.
    memory_drop = 0.0
    if fp16.model_allocated_gb > 0:
        memory_drop = (fp16.model_allocated_gb - nf4.model_allocated_gb) / fp16.model_allocated_gb

    if memory_drop >= 0.30 and (nf4.retrieval_hit is True or fp16.retrieval_hit == nf4.retrieval_hit):
        return "NF4 is the better trade-off: lower VRAM with comparable retrieval on this benchmark."

    if nf4.model_allocated_gb < fp16.model_allocated_gb and nf4.retrieval_hit == fp16.retrieval_hit:
        return "NF4 is preferable because it uses less VRAM with no visible quality regression here."

    return "FP16 is the safer default unless VRAM pressure is the main constraint."


def render_markdown_report(report: BenchmarkReport) -> str:
    def fmt(v: float | None, digits: int = 2) -> str:
        if v is None:
            return "-"
        return f"{v:.{digits}f}"

    rows = [
        ("Load time (s)", report.fp16.load_time_s, report.nf4.load_time_s),
        ("Model allocated VRAM (GB)", report.fp16.model_allocated_gb, report.nf4.model_allocated_gb),
        ("Model reserved VRAM (GB)", report.fp16.model_reserved_gb, report.nf4.model_reserved_gb),
        ("Peak infer VRAM (GB)", report.fp16.peak_infer_gb, report.nf4.peak_infer_gb),
        ("Page embedding time (s)", report.fp16.embedding_pages_s, report.nf4.embedding_pages_s),
        ("Query embedding time (s)", report.fp16.query_embed_s, report.nf4.query_embed_s),
        ("Top-1 page index", report.fp16.retrieval_top1_page_index, report.nf4.retrieval_top1_page_index),
        ("Top-1 score", report.fp16.retrieval_top1_score, report.nf4.retrieval_top1_score),
        ("Hit expected page", report.fp16.retrieval_hit, report.nf4.retrieval_hit),
        ("Local snapshot disk (GB)", report.fp16.disk_size_gb, report.nf4.disk_size_gb),
    ]

    lines: list[str] = []
    lines.append(f"# ColPali FP16 vs NF4 Benchmark")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {report.timestamp_utc}")
    lines.append(f"- Hostname: {report.hostname}")
    lines.append(f"- Platform: {report.platform}")
    lines.append(f"- Python: {report.python_version}")
    lines.append(f"- Torch: {report.torch_version}")
    lines.append(f"- Transformers: {report.transformers_version}")
    lines.append(f"- ColPali engine: {report.colpali_engine_version}")
    lines.append(f"- GPU: {report.gpu_name or 'CPU / unavailable'}")
    lines.append(f"- GPU total memory: {fmt(report.gpu_total_memory_gb)} GB")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(report.recommendation)
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | FP16 | NF4 |")
    lines.append("|---|---:|---:|")
    for metric, fp16_val, nf4_val in rows:
        if isinstance(fp16_val, bool) or isinstance(nf4_val, bool):
            lines.append(f"| {metric} | {fp16_val} | {nf4_val} |")
        elif isinstance(fp16_val, int) or isinstance(nf4_val, int):
            lines.append(f"| {metric} | {fp16_val} | {nf4_val} |")
        else:
            lines.append(f"| {metric} | {fmt(fp16_val)} | {fmt(nf4_val)} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The local snapshot is reused on subsequent runs, so the model is not fetched from the Hub every time.")
    lines.append("- NF4 is loaded from the local snapshot using bitsandbytes quantization at runtime.")
    lines.append("- Retrieval score here is a ColBERT-style MaxSim approximation computed directly on embeddings.")
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Benchmark ColPali FP16 vs NF4 with a local snapshot.")
    parser.add_argument("--model-id", default=AppConfig.model_id)
    parser.add_argument("--local-model-dir", default=str(AppConfig.local_model_dir))
    parser.add_argument("--output-dir", default=str(AppConfig.output_dir))
    parser.add_argument("--pdf", default=str(AppConfig.pdf_path))
    parser.add_argument("--device", default=AppConfig.device)
    parser.add_argument("--dpi", type=int, default=AppConfig.dpi)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=AppConfig.batch_size)
    parser.add_argument("--expected-page-index", type=int, default=None)
    parser.add_argument(
        "--query",
        action="append",
        default=list(AppConfig.queries),
        help="Query to benchmark. Repeat --query to add more.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Redownload and resave the local snapshot.")
    args = parser.parse_args()

    return AppConfig(
        model_id=args.model_id,
        local_model_dir=Path(args.local_model_dir),
        output_dir=Path(args.output_dir),
        pdf_path=Path(args.pdf),
        device=args.device,
        dpi=args.dpi,
        max_pages=args.max_pages,
        batch_size=args.batch_size,
        queries=tuple(args.query),
        expected_page_index=args.expected_page_index,
        use_local_files_only=True,
    ), args.force_refresh


def main() -> int:
    cfg, force_refresh = parse_args()
    setup_logging(cfg.output_dir)

    device, dtype = select_device(cfg.device)
    LOGGER.info("Device: %s", device)
    LOGGER.info("DType: %s", dtype)

    if not cfg.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {cfg.pdf_path}")

    # Collect environment details.
    gpu_name = None
    gpu_total_memory_gb = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_memory_gb = gb(props.total_memory)

    # Step 1: save a local snapshot once.
    snapshot_local_model(cfg.model_id, cfg.local_model_dir, force_refresh=force_refresh)

    # Step 2: prepare benchmark inputs.
    images = render_pdf_to_images(cfg.pdf_path, dpi=cfg.dpi, max_pages=cfg.max_pages)
    if not images:
        raise RuntimeError(f"No pages rendered from {cfg.pdf_path}")
    LOGGER.info("Rendered %d pages from %s", len(images), cfg.pdf_path)

    # Step 3: benchmark FP16.
    fp16_metrics, _ = benchmark_variant(
        name="fp16",
        load_fn=lambda local_dir: load_fp16_model(local_dir, device=device, local_files_only=cfg.use_local_files_only),
        local_dir=cfg.local_model_dir,
        cfg=cfg,
        images=images,
        queries=cfg.queries,
        expected_page_index=cfg.expected_page_index,
    )

    # Step 4: benchmark NF4.
    nf4_metrics, _ = benchmark_variant(
        name="nf4",
        load_fn=lambda local_dir: load_nf4_model(local_dir, local_files_only=cfg.use_local_files_only),
        local_dir=cfg.local_model_dir,
        cfg=cfg,
        images=images,
        queries=cfg.queries,
        expected_page_index=cfg.expected_page_index,
    )


    try:
        colpali_version = version("colpali_engine")
    except PackageNotFoundError:
        colpali_version = "Unknown"

    report = BenchmarkReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        colpali_engine_version=colpali_version,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_memory_gb,
        fp16=fp16_metrics,
        nf4=nf4_metrics,
    )

    report.recommendation = choose_recommendation(
        report.fp16,
        report.nf4,
    )

    ensure_dir(cfg.output_dir)
    report_json = cfg.output_dir / "colpali_benchmark_report.json"
    report_md = cfg.output_dir / "colpali_benchmark_report.md"

    safe_json_dump(report_json, asdict(report))
    report_md.write_text(render_markdown_report(report), encoding="utf-8")

    LOGGER.info("Wrote report: %s", report_json)
    LOGGER.info("Wrote report: %s", report_md)
    print("\n" + render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
