from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import shutil
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
import asyncio
from fastapi import UploadFile
from config import AppConfig, override_config
from data import PageArtifacts, IngestJob

LOGGER = logging.getLogger("colpali_qdrant")

MODEL_NAME = "vidore/colpali-v1.3"
LOCAL_MODEL_DIR = Path("./models/colpali-v1.3")
OUTPUT_DIR = Path("./explainability_output")

UPLOAD_DIR = Path("./uploaded_pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



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


def document_already_ingested(
    client: QdrantClient,
    cfg: AppConfig,
    tenant_id: str,
    document_sha256: str,
) -> bool:
    results = client.scroll(
        collection_name=cfg.collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="document_sha256",
                    match=models.MatchValue(value=document_sha256),
                ),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    points, _ = results
    return len(points) > 0


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
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def chunked(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def document_payload(
    pdf_path: Path,
    document_id: str,
    tenant_id: str,
    page_index: int,
    page_count: int,
    image: Image.Image,
    allowed_roles: Sequence[str] | None = None,
    allowed_user_ids: Sequence[str] | None = None,
    department: str | None = None,
    doc_type: str | None = None,
    sensitivity: str | None = None,
) -> dict:
    """
    Payload stored for every page-point in Qdrant.

    tenant_id         -> company/workspace boundary
    allowed_roles     -> roles that may query this document
    allowed_user_ids  -> explicit user exceptions/allow-list
    department        -> optional grouping, e.g. finance, engineering
    doc_type          -> optional type, e.g. invoice, blueprint, T&C
    sensitivity       -> optional sensitivity label
    """
    return {
        "tenant_id": tenant_id,
        "document_id": document_id,
        "document_name": pdf_path.name,
        "source_pdf": str(pdf_path),
        "document_sha256": file_sha256(pdf_path),
        "page_index": page_index,
        "page_number": page_index + 1,
        "page_count": page_count,
        "page_sha256": image_sha256(image),
        "uploaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_roles": list(allowed_roles or []),
        "allowed_user_ids": list(allowed_user_ids or []),
        "department": department,
        "doc_type": doc_type,
        "sensitivity": sensitivity,
    }


# ----------------------------
# Ingestion
# ----------------------------

def ingest_documents(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    pdf_paths: Sequence[Path],
    tenant_id: str,
    allowed_roles: Sequence[str],
    allowed_user_ids: Sequence[str] | None = None,
    department: str | None = None,
    doc_type: str | None = None,
    sensitivity: str | None = None,
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

        # IMPORTANT:
        # Duplicate detection must be tenant-scoped, not global.
        # Same PDF uploaded by a different company should still be allowed.
        if document_already_ingested(client, cfg, tenant_id, doc_sha256):
            LOGGER.info("Skipping already ingested document for tenant %s: %s", tenant_id, pdf_path.name)
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
            LOGGER.info(
                "Embedding batch %d (%d pages) for %s",
                batch_index,
                len(image_batch),
                pdf_path.name,
            )
            vectors = embed_page_batch(model, processor, image_batch)

            for page_offset, (image, vecs) in enumerate(zip(image_batch, vectors), start=len(artifacts)):
                payload = document_payload(
                    pdf_path=pdf_path,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    page_index=page_offset,
                    page_count=len(images),
                    image=image,
                    allowed_roles=allowed_roles,
                    allowed_user_ids=allowed_user_ids,
                    department=department,
                    doc_type=doc_type,
                    sensitivity=sensitivity,
                )
                point_id = str(uuid4())
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




def save_upload_sync(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "document.pdf").suffix or ".pdf"
    dst = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    with dst.open("wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)

    return dst



class IngestManager:
    def __init__(self, cfg, client, model, processor):
        self.cfg = cfg
        self.client = client
        self.model = model
        self.processor = processor
        self.queue: asyncio.Queue[IngestJob | None] = asyncio.Queue(
            maxsize=getattr(cfg, "ingest_queue_size", 100)
        )
        self.worker_count = getattr(cfg, "ingest_workers", 2)
        self._workers: list[asyncio.Task] = []

    async def start(self) -> None:
        for i in range(self.worker_count):
            task = asyncio.create_task(self._worker(i + 1))
            self._workers.append(task)
        LOGGER.info("Started %d ingest workers", self.worker_count)

    async def stop(self) -> None:
        for _ in self._workers:
            await self.queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        LOGGER.info("Stopped ingest workers")

    async def enqueue(self, job: IngestJob) -> None:
        await self.queue.put(job)

    async def _worker(self, worker_id: int) -> None:
        LOGGER.info("Worker %d started", worker_id)
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                break

            try:
                LOGGER.info(
                    "Worker %d processing job %s with %d file(s)",
                    worker_id,
                    job.job_id,
                    len(job.pdf_paths),
                )

                # ingest_documents is blocking, so run it off the event loop
                await asyncio.to_thread(
                    ingest_documents,
                    client=self.client,
                    cfg = self.cfg,
                    model = self.model,
                    processor = self.processor,
                    pdf_paths = job.pdf_paths,
                    tenant_id = job.tenant_id,
                    allowed_roles = job.allowed_roles 
                )

                LOGGER.info("Job %s completed", job.job_id)

            except Exception:
                LOGGER.exception("Job %s failed", job.job_id)

            finally:
                self.queue.task_done()



if __name__ == "__main__":
    pass