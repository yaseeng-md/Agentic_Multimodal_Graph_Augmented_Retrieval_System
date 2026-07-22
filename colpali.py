"""
PDF -> ColPali visual multivectors -> Qdrant (MaxSim) pipeline with explainability.

Features:
- Local model snapshot download once, then reuse from disk
- BitsAndBytes NF4 loading from the local snapshot
- Batch ingestion into Qdrant
- Retrieval by MaxSim
- Explainability mode:
  - rank
  - similarity score
  - page number
  - extracted page snippet
  - top contributing query tokens
  - patch-level overlay images for highest-scoring regions

Run this file directly and change the PDF path/query in __main__.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pymupdf  # PyMuPDF
import torch
from PIL import Image, ImageDraw
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from transformers import BitsAndBytesConfig
from uuid import uuid4

LOGGER = logging.getLogger("colpali_qdrant")

MODEL_NAME = "vidore/colpali-v1.3"
LOCAL_MODEL_DIR = Path("./models/colpali-v1.3")
OUTPUT_DIR = Path("./explainability_output")
LOG_FILE = "pipeline_logs.md"



def setup_logging(log_file: str = LOG_FILE) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="## %(asctime)s | %(levelname)s\n%(message)s\n",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n---\n\n")


def load_visual_config() -> dict:
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("VISUAL_RETRIVAL", {})
    except FileNotFoundError:
        LOGGER.warning("config.json not found; using defaults")
        return {}
    except Exception as exc:
        LOGGER.warning("Failed to read config.json: %s", exc)
        return {}


VISUAL_CONFIG = load_visual_config()


@dataclass(frozen=True)
class AppConfig:
    model_name: str = MODEL_NAME
    collection_name: str = "colpali_documents"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    batch_size: int = 1
    max_upload_docs: int = 5
    dpi: int = 96
    recreate_collection: bool = True
    search_limit: int = 5
    prefetch_limit: int = 50
    latency_budget_ms: float = 150.0


@dataclass
class PageArtifacts:
    page_index: int
    page_number: int
    image: Image.Image
    original: torch.Tensor
    image_patch_embeddings: torch.Tensor
    x_patches: int
    y_patches: int
    payload: dict


def override_config() -> AppConfig:
    return AppConfig(
        model_name=VISUAL_CONFIG.get("VLM_MODEL", MODEL_NAME),
        collection_name=VISUAL_CONFIG.get("COLLECTION_NAME", "colpali_documents"),
        qdrant_url=VISUAL_CONFIG.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=VISUAL_CONFIG.get("QDRANT_API_KEY"),
        batch_size=int(VISUAL_CONFIG.get("BATCH_SIZE", 1)),
        max_upload_docs=int(VISUAL_CONFIG.get("MAX_UPLOAD_DOCS", 5)),
        dpi=int(VISUAL_CONFIG.get("PDF_RENDER_DPI", 96)),
        recreate_collection=bool(VISUAL_CONFIG.get("RECREATE_COLLECTION", True)),
        search_limit=int(VISUAL_CONFIG.get("SEARCH_LIMIT", 5)),
        prefetch_limit=int(VISUAL_CONFIG.get("PREFETCH_LIMIT", 50)),
        latency_budget_ms=float(VISUAL_CONFIG.get("LATENCY_BUDGET_MS", 150.0)),
    )


def select_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def gb(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def image_sha256(image: Image.Image) -> str:
    buf = image.convert("RGB").tobytes()
    return hashlib.sha256(buf).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def render_pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        doc.close()
    return images


def extract_page_snippet(pdf_path: Path, page_index: int, max_chars: int = 1200) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_index]
        text = page.get_text("text").strip()
    finally:
        doc.close()
    if not text:
        return "(no extractable text on this page)"
    return text[:max_chars]


def render_single_page_image(pdf_path: Path, page_index: int, dpi: int) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def document_payload(
    pdf_path: Path,
    document_id: str,
    entity_id: str | None,
    page_index: int,
    page_count: int,
    image: Image.Image,
) -> dict:
    return {
        "entity_id": entity_id,
        "document_id": document_id,
        "document_name": pdf_path.name,
        "source_pdf": str(pdf_path),
        "document_sha256": file_sha256(pdf_path),
        "page_index": page_index,
        "page_number": page_index + 1,
        "page_count": page_count,
        "page_sha256": image_sha256(image),
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def document_already_ingested(client: QdrantClient, cfg: AppConfig, document_sha256: str) -> bool:
    try:
        result = client.scroll(
            collection_name=cfg.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_sha256",
                        match=models.MatchValue(value=document_sha256),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        points = result[0] if isinstance(result, tuple) else result.points
        return len(points) > 0
    except Exception:
        return False


def download_model_if_needed() -> None:
    marker = LOCAL_MODEL_DIR / "config.json"
    if marker.exists():
        LOGGER.info("Using local model snapshot: %s", LOCAL_MODEL_DIR)
        return

    LOGGER.info("Downloading model from Hugging Face and saving locally to %s", LOCAL_MODEL_DIR)
    ensure_dir(LOCAL_MODEL_DIR)

    model = ColPali.from_pretrained(MODEL_NAME).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

    model.save_pretrained(LOCAL_MODEL_DIR)
    processor.save_pretrained(LOCAL_MODEL_DIR)

    metadata = {
        "model_name": MODEL_NAME,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Base snapshot saved locally; NF4 is applied at load time.",
    }
    (LOCAL_MODEL_DIR / "snapshot_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_and_processor(device: str, dtype: torch.dtype):
    download_model_if_needed()

    LOGGER.info("Loading local ColPali model in NF4")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = ColPali.from_pretrained(
        str(LOCAL_MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    ).eval()

    processor = ColPaliProcessor.from_pretrained(
        str(LOCAL_MODEL_DIR),
        local_files_only=True,
    )

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        allocated = 0.0
        reserved = 0.0

    LOGGER.info("Model VRAM Allocated : %.2f GB", allocated)
    LOGGER.info("Model VRAM Reserved  : %.2f GB", reserved)

    return model, processor


@torch.inference_mode()
def embed_page_batch(
    model: ColPali,
    processor: ColPaliProcessor,
    images: Sequence[Image.Image],
) -> list[dict]:
    if not images:
        return []

    processed = processor.process_images(list(images)).to(model.device)
    outputs = model(**processed)

    results: list[dict] = []
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
                "image_patch_embeddings": image_patch_embeddings.detach().cpu(),
                "mean_pooling_rows": pooled_rows.detach().cpu(),
                "mean_pooling_columns": pooled_cols.detach().cpu(),
                "x_patches": x_patches,
                "y_patches": y_patches,
            }
        )

    return results


@torch.inference_mode()
def embed_query(model: ColPali, processor: ColPaliProcessor, query: str) -> tuple[torch.Tensor, torch.Tensor]:
    processed = processor.process_queries([query]).to(model.device)
    query_embedding = model(**processed)[0]
    return query_embedding.detach().cpu(), processed.input_ids[0].detach().cpu()


def ensure_collection(client: QdrantClient, cfg: AppConfig, vector_size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(cfg.collection_name):
        LOGGER.warning("Deleting existing collection %s", cfg.collection_name)
        client.delete_collection(cfg.collection_name)

    if client.collection_exists(cfg.collection_name):
        LOGGER.info("Collection %s already exists; reusing it", cfg.collection_name)
        return

    LOGGER.info("Creating collection %s", cfg.collection_name)
    client.create_collection(
        collection_name=cfg.collection_name,
        vectors_config={
            "original": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
            "mean_pooling_rows": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            "mean_pooling_columns": models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
    )


def chunked(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest_pdf(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    pdf_path: Path,
    batch_size: int,
    dpi: int,
) -> list[PageArtifacts]:
    images = render_pdf_to_images(pdf_path, dpi=dpi)
    LOGGER.info("Rendered %d pages from %s", len(images), pdf_path)

    if not images:
        raise RuntimeError(f"No pages rendered from {pdf_path}")

    sample_batch = embed_page_batch(model, processor, images[:1])
    vector_size = int(sample_batch[0]["original"].shape[-1])
    ensure_collection(client, cfg, vector_size=vector_size, recreate=cfg.recreate_collection)

    artifacts: list[PageArtifacts] = []
    points: list[models.PointStruct] = []
    point_id = str(uuid4())

    for batch_index, image_batch in enumerate(chunked(images, batch_size), start=1):
        LOGGER.info("Embedding batch %d (%d pages)", batch_index, len(image_batch))
        vectors = embed_page_batch(model, processor, image_batch)

        for image, vecs in zip(image_batch, vectors):
            page_index = point_id
            payload = {
                "source_pdf": str(pdf_path),
                "page_index": page_index,
                "page_number": page_index + 1,
                "page_sha256": image_sha256(image),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "original": vecs["original"].tolist(),
                        "mean_pooling_rows": vecs["mean_pooling_rows"].tolist(),
                        "mean_pooling_columns": vecs["mean_pooling_columns"].tolist(),
                    },
                    payload=payload,
                )
            )

            artifacts.append(
                PageArtifacts(
                    page_index=page_index,
                    page_number=page_index + 1,
                    image=image,
                    original=vecs["original"],
                    image_patch_embeddings=vecs["image_patch_embeddings"],
                    x_patches=int(vecs["x_patches"]),
                    y_patches=int(vecs["y_patches"]),
                    payload=payload,
                )
            )

            point_id += 1

    LOGGER.info("Upserting %d page points into Qdrant", len(points))
    client.upsert(collection_name=cfg.collection_name, points=points, wait=True)
    return artifacts


def ingest_documents(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    pdf_paths: Sequence[Path],
    entity_id: str | None = None,
) -> list[PageArtifacts]:
    if len(pdf_paths) == 0:
        return []
    if len(pdf_paths) > cfg.max_upload_docs:
        raise ValueError(f"At most {cfg.max_upload_docs} documents can be uploaded per batch")

    all_artifacts: list[PageArtifacts] = []
    collection_initialized = False

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        images = render_pdf_to_images(pdf_path, dpi=cfg.dpi)
        LOGGER.info("Rendered %d pages from %s", len(images), pdf_path)

        if not images:
            raise RuntimeError(f"No pages rendered from {pdf_path}")

        doc_sha256 = file_sha256(pdf_path)
        if document_already_ingested(client, cfg, doc_sha256):
            LOGGER.info("Skipping already ingested document: %s", pdf_path.name)
            continue

        if not collection_initialized:
            sample_batch = embed_page_batch(model, processor, images[:1])
            vector_size = int(sample_batch[0]["original"].shape[-1])
            ensure_collection(client, cfg, vector_size=vector_size, recreate=cfg.recreate_collection)
            collection_initialized = True

        document_id = uuid4().hex
        points: list[models.PointStruct] = []
        artifacts: list[PageArtifacts] = []

        for batch_index, image_batch in enumerate(chunked(images, cfg.batch_size), start=1):
            LOGGER.info("Embedding batch %d (%d pages) for %s", batch_index, len(image_batch), pdf_path.name)
            vectors = embed_page_batch(model, processor, image_batch)

            for page_offset, (image, vecs) in enumerate(zip(image_batch, vectors), start=len(artifacts)):
                payload = document_payload(
                    pdf_path=pdf_path,
                    document_id=document_id,
                    entity_id=entity_id,
                    page_index=page_offset,
                    page_count=len(images),
                    image=image,
                )

                point_id = f"{document_id}:{page_offset}"
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "original": vecs["original"].tolist(),
                            "mean_pooling_rows": vecs["mean_pooling_rows"].tolist(),
                            "mean_pooling_columns": vecs["mean_pooling_columns"].tolist(),
                        },
                        payload=payload,
                    )
                )

                artifacts.append(
                    PageArtifacts(
                        page_index=page_offset,
                        page_number=page_offset + 1,
                        image=image,
                        original=vecs["original"],
                        image_patch_embeddings=vecs["image_patch_embeddings"],
                        x_patches=int(vecs["x_patches"]),
                        y_patches=int(vecs["y_patches"]),
                        payload=payload,
                    )
                )

        LOGGER.info("Upserting %d page points into Qdrant for %s", len(points), pdf_path.name)
        client.upsert(collection_name=cfg.collection_name, points=points, wait=True)
        all_artifacts.extend(artifacts)

    return all_artifacts


def query_pdf(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    entity_id: str | None = None,
) -> tuple[list[models.ScoredPoint], float, torch.Tensor, torch.Tensor]:
    query_embedding, query_input_ids = embed_query(model, processor, query)

    start = time.perf_counter()
    query_filter = None
    if entity_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="entity_id",
                    match=models.MatchValue(value=entity_id),
                )
            ]
        )

    response = client.query_points(
        collection_name=cfg.collection_name,
        query=query_embedding.tolist(),
        using="original",
        prefetch=[
            models.Prefetch(
                query=query_embedding.tolist(),
                limit=cfg.prefetch_limit,
                using="mean_pooling_rows",
            ),
            models.Prefetch(
                query=query_embedding.tolist(),
                limit=cfg.prefetch_limit,
                using="mean_pooling_columns",
            ),
        ],
        query_filter=query_filter,
        limit=cfg.search_limit,
        with_payload=True,
        with_vectors=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return list(response.points), latency_ms, query_embedding, query_input_ids


def maxsim_query_contributions(query_embedding: torch.Tensor, page_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sims = query_embedding @ page_embedding.T
    token_scores, token_best_patch_idx = sims.max(dim=1)
    return token_scores, token_best_patch_idx


def patch_saliency(query_embedding: torch.Tensor, patch_embeddings: torch.Tensor) -> torch.Tensor:
    sims = query_embedding @ patch_embeddings.T
    patch_scores = sims.max(dim=0).values
    return patch_scores


def save_patch_overlay(
    image: Image.Image,
    patch_scores: torch.Tensor,
    x_patches: int,
    y_patches: int,
    output_path: Path,
    top_fraction: float = 0.10,
) -> None:
    ensure_dir(output_path.parent)

    scores = patch_scores.detach().cpu().float().numpy().reshape(x_patches, y_patches)
    flat = scores.reshape(-1)
    if flat.size == 0:
        image.save(output_path)
        return

    threshold = np.quantile(flat, 1.0 - top_fraction)
    max_score = float(flat.max()) if flat.max() > 0 else 1.0

    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    patch_w = rgba.width / y_patches
    patch_h = rgba.height / x_patches

    for row in range(x_patches):
        for col in range(y_patches):
            score = float(scores[row, col])
            if score < threshold:
                continue

            norm = max(0.0, min(1.0, score / max_score))
            alpha = int(40 + 180 * norm)  # 40..220
            x0 = int(col * patch_w)
            y0 = int(row * patch_h)
            x1 = int((col + 1) * patch_w)
            y1 = int((row + 1) * patch_h)

            draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, alpha), outline=(255, 0, 0, min(255, alpha + 30)))

    composite = Image.alpha_composite(rgba, overlay).convert("RGB")
    composite.save(output_path)


def explain_hits(
    hits: list[models.ScoredPoint],
    query: str,
    query_embedding: torch.Tensor,
    query_input_ids: torch.Tensor,
    model: ColPali,
    processor: ColPaliProcessor,
    cfg: AppConfig,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)

    tokenizer = processor.tokenizer
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        source_pdf_raw = payload.get("source_pdf")
        page_index = int(payload.get("page_index", -1))

        if not source_pdf_raw or page_index < 0:
            LOGGER.warning("Skipping explainability for invalid hit payload: %s", payload)
            continue

        source_pdf = Path(str(source_pdf_raw))
        if not source_pdf.exists():
            LOGGER.warning("Source PDF not found on disk for explainability: %s", source_pdf)
            continue

        image = render_single_page_image(source_pdf, page_index, dpi=cfg.dpi)
        vecs = embed_page_batch(model, processor, [image])[0]
        page_embedding = vecs["original"]
        patch_embeddings = vecs["image_patch_embeddings"].reshape(-1, vecs["image_patch_embeddings"].shape[-1])

        token_scores, _ = maxsim_query_contributions(query_embedding, page_embedding)
        patch_scores = patch_saliency(query_embedding, patch_embeddings)

        top_k = min(8, token_scores.numel())
        top_indices = torch.topk(token_scores, k=top_k).indices.tolist()

        print("\n" + "=" * 88)
        print(f"Rank            : {rank}")
        print(f"Similarity score: {hit.score:.6f}")
        print(f"Document        : {payload.get('document_name')}")
        print(f"Page number     : {payload.get('page_number')}")

        snippet = extract_page_snippet(source_pdf, page_index)
        print("\nSnippet")
        print("-" * 88)
        print(snippet)

        print("\nTop contributing query tokens")
        print("-" * 88)
        for idx in top_indices:
            token_id = int(query_input_ids[idx])
            token_text = tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if not token_text:
                token_text = tokenizer.convert_ids_to_tokens(token_id)
            print(f"{token_text:<20} {float(token_scores[idx]):.6f}")

        x_patches, y_patches = vecs["x_patches"], vecs["y_patches"]
        overlay_path = output_dir / f"rank_{rank:02d}_{payload.get('document_name', 'doc')}_page_{payload.get('page_number', 0):04d}_overlay.png"
        save_patch_overlay(
            image,
            patch_scores,
            x_patches,
            y_patches,
            overlay_path,
            top_fraction=0.10,
        )
        print(f"\nOverlay saved to : {overlay_path}")

        LOGGER.info("Explainability done for rank %d, page %s", rank, payload.get("page_number"))


def ingest_orchestrator(
    pdf_paths: Sequence[Path],
    query: str,
    explain: bool = True,
    entity_id: str | None = None,
) -> None:
    cfg = override_config()

    device, dtype = select_device()
    LOGGER.info("Device: %s", device)
    LOGGER.info("DType : %s", dtype)
    
    model, processor = load_model_and_processor(device, dtype)
    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)

    artifacts = ingest_documents(client, cfg, model, processor, pdf_paths, entity_id=entity_id)
    hits, latency_ms, query_embedding, query_input_ids = query_pdf(client, cfg, model, processor, query, entity_id=entity_id)

    if not hits:
        raise RuntimeError("Qdrant returned no hits")

    top = hits[0]
    top_page_index = top.payload.get("page_index") if top.payload else None

    LOGGER.info("Query: %s", query)
    LOGGER.info("Top hit page_index=%s score=%.6f latency=%.2fms", top_page_index, top.score, latency_ms)

    print("\nTop results")
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        print(
            f"{rank:>2}. document={payload.get('document_name')} "
            f"page_number={payload.get('page_number')} score={hit.score:.6f}"
        )

    print(f"\nRetrieved page index: {top_page_index}")

    if latency_ms <= cfg.latency_budget_ms:
        print(f"Latency passed: {latency_ms:.2f} ms <= {cfg.latency_budget_ms:.2f} ms")
    else:
        print(f"Latency warning: {latency_ms:.2f} ms > {cfg.latency_budget_ms:.2f} ms")

    if explain:
        explain_hits(
            hits=hits,
            query=query,
            query_embedding=query_embedding,
            query_input_ids=query_input_ids,
            model=model,
            processor=processor,
            cfg=cfg,
            output_dir=OUTPUT_DIR,
        )

   

if __name__ == "__main__":
    setup_logging()
    
    DEFAULT_PDF_PATH = [Path(r"E:\Study Material\Research Papers\Explicit Content Detection\Procdia_SalViT.pdf")]
    DEFAULT_QUERY = "What is the accuracy of the model discussed in this paper ?"

    # Change these two lines only.
    PDF_PATH = DEFAULT_PDF_PATH
    QUERY = DEFAULT_QUERY

    ingest_orchestrator(
        pdf_paths=PDF_PATH,
        query=QUERY,
        explain=True,
    )
