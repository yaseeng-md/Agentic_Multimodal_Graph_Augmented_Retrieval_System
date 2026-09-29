
"""
AMG Multimodal RAG - Multi-file PDF ingestion
===============================================

Ingestion architecture:
    PDF(s)
      -> exact PDF SHA-256 (duplicate detection)
      -> document/version registry (SQLite)
      -> render pages to images
      -> rendered-page hash
      -> reuse existing page embedding when possible
      -> ColPali v1.3 merged + BitsAndBytes 4-bit NF4 for new pages
      -> Qdrant personal_collection (MAX_SIM multivectors)

Versioning:
    * A new upload is a new logical document by default.
    * To create a new version, explicitly pass --update-doc DOC_ID PATH.
    * An exact file-hash duplicate is never re-ingested.

Batching:
    * Pass any number of --pdf paths in one command.
    * The ColPali model is loaded once and reused for all files.
    * Files are processed sequentially to avoid multiplying GPU memory use.

Examples:
    python ingest.py --pdf a.pdf b.pdf c.pdf
    python ingest.py --update-doc DOC-123 /path/welding_manual_v2.pdf
    python ingest.py --pdf a.pdf b.pdf --update-doc DOC-123 c.pdf
    python ingest.py --dummy-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from colpali_engine.models import ColPali, ColPaliProcessor

from registry import Registry
from qdrant_store import QdrantStore
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "personal_collection")
MODEL_ID = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3-merged")
MODEL_REVISION = os.getenv("COLPALI_MODEL_REVISION", "main")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"

# BitsAndBytes 4-bit NF4
QUANTIZATION = os.getenv("QUANTIZATION", "4bit").lower()
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4").lower()
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16").lower()
BNB_4BIT_USE_DOUBLE_QUANT = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"
SAVE_QUANTIZED_MODEL = os.getenv("SAVE_QUANTIZED_MODEL", "false").lower() == "true"
QUANTIZED_MODEL_DIR = Path(
    os.getenv(
        "QUANTIZED_MODEL_DIR",
        "./models/colpali-v1.3-merged-4bit-nf4",
    )
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "8"))

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PDF_DIR = Path(os.getenv("PDF_DIR", str(DATA_DIR / "pdfs")))
PAGE_IMAGE_DIR = Path(os.getenv("PAGE_IMAGE_DIR", str(DATA_DIR / "pages")))
DUMMY_DIR = Path(os.getenv("DUMMY_DIR", str(DATA_DIR / "dummy")))
REGISTRY_DB_PATH = Path(
    os.getenv("REGISTRY_DB_PATH", str(DATA_DIR / "ingestion_registry.db"))
)

PDF_DPI = int(os.getenv("PDF_DPI", "150"))
BATCH_SIZE = int(os.getenv("COLPALI_BATCH_SIZE", "1"))
STORE_PAGE_TEXT = os.getenv("STORE_PAGE_TEXT", "true").lower() == "true"
DEVICE_ENV = os.getenv("DEVICE", "auto").lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("multimodal-rag-ingest")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestJob:
    """One uploaded PDF and its optional explicit version relationship."""

    pdf_path: Path
    update_document_id: str | None = None


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "document.pdf"


def new_document_id() -> str:
    return f"DOC-{uuid.uuid4().hex[:10].upper()}"


def new_version_id() -> str:
    return f"VER-{uuid.uuid4().hex[:12].upper()}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_page_image(image: Image.Image) -> str:
    """Hash the rendered RGB pixels, not PNG container metadata."""
    rgb = image.convert("RGB")
    h = hashlib.sha256()
    h.update(str(rgb.size).encode("utf-8"))
    h.update(rgb.tobytes())
    return h.hexdigest()


def embedding_signature() -> str:
    payload = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
        "bnb_quant_type": BNB_4BIT_QUANT_TYPE,
        "bnb_compute_dtype": BNB_4BIT_COMPUTE_DTYPE,
        "bnb_double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
        "pdf_dpi": PDF_DPI,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def page_point_id(document_id: str, version_id: str, page_number: int) -> str:
    """Stable point ID for one document version + page position."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{document_id}:{version_id}:page:{page_number}",
        )
    )


def chunked(items: list, size: int) -> Iterable[list]:
    if size <= 0:
        raise ValueError("Batch size must be > 0.")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def resolve_device() -> str:
    if DEVICE_ENV != "auto":
        return DEVICE_ENV
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quant_compute_dtype() -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if BNB_4BIT_COMPUTE_DTYPE not in mapping:
        raise ValueError(
            f"Unsupported BNB_4BIT_COMPUTE_DTYPE={BNB_4BIT_COMPUTE_DTYPE}. "
            "Use bfloat16, float16, or float32."
        )
    return mapping[BNB_4BIT_COMPUTE_DTYPE]


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []

    logger.info("Rendering PDF: %s", pdf_path)

    with fitz.open(pdf_path) as pdf:
        for page_index in tqdm(
            range(len(pdf)),
            desc=f"Rendering {pdf_path.name}",
            unit="page",
        ):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            image_path = output_dir / f"page_{page_number:04d}.png"

            if not image_path.exists():
                pix = page.get_pixmap(
                    dpi=dpi,
                    colorspace=fitz.csRGB,
                    alpha=False,
                )
                pix.save(str(image_path))

            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                page_hash = sha256_page_image(rgb)

            page_text = page.get_text("text").strip() if STORE_PAGE_TEXT else ""

            pages.append(
                {
                    "page_number": page_number,
                    "image_path": image_path,
                    "page_hash": page_hash,
                    "page_text": page_text,
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
            )

    logger.info("Rendered %d pages: %s", len(pages), pdf_path.name)
    return pages


# ---------------------------------------------------------------------------
# ColPali
# ---------------------------------------------------------------------------

class ColPaliEmbedder:
    """One shared 4-bit NF4 ColPali model for the complete batch."""

    def __init__(self, model_id: str, device: str):
        if QUANTIZATION != "4bit":
            raise ValueError(
                f"This pipeline is configured for 4-bit NF4 only; got {QUANTIZATION!r}."
            )
        if BNB_4BIT_QUANT_TYPE != "nf4":
            raise ValueError(
                f"This pipeline is configured for NF4 only; got {BNB_4BIT_QUANT_TYPE!r}."
            )
        if not device.startswith("cuda"):
            raise RuntimeError(
                "BitsAndBytes 4-bit NF4 requires CUDA. "
                "Set DEVICE=cuda:0 and verify your CUDA PyTorch installation."
            )

        compute_dtype = quant_compute_dtype()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )

        logger.info(
            "Loading ColPali once | model=%s | revision=%s | NF4 | compute=%s",
            model_id,
            MODEL_REVISION,
            compute_dtype,
        )

        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": compute_dtype,
            "device_map": "auto",
            "local_files_only": HF_LOCAL_FILES_ONLY,
            "revision": MODEL_REVISION,
        }

        self.model = ColPali.from_pretrained(model_id, **model_kwargs).eval()
        self.processor = ColPaliProcessor.from_pretrained(
            model_id,
            revision=MODEL_REVISION,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )

        try:
            self.model_device = next(self.model.parameters()).device
        except StopIteration:
            self.model_device = torch.device(device)

        logger.info("ColPali loaded on %s", self.model_device)
        if torch.cuda.is_available():
            logger.info(
                "GPU VRAM | allocated=%.2f GB | reserved=%.2f GB",
                torch.cuda.memory_allocated() / (1024 ** 3),
                torch.cuda.memory_reserved() / (1024 ** 3),
            )

        if SAVE_QUANTIZED_MODEL:
            self.save_quantized_model(QUANTIZED_MODEL_DIR)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = self.processor.process_images(images).to(self.model_device)
        embeddings = self.model(**batch)
        return embeddings.detach().to("cpu", dtype=torch.float32)

    def save_quantized_model(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.processor.save_pretrained(output_dir)
        logger.info("Saved quantized model to %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# Application-level ingestion
# ---------------------------------------------------------------------------

def copy_pdf_to_storage(
    pdf_path: Path,
    document_id: str,
    version_id: str,
    version_number: int,
) -> Path:
    target_dir = PDF_DIR / document_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"v{version_number}_{safe_filename(pdf_path.name)}"
    shutil.copy2(pdf_path, target)
    return target


def prepare_document(
    registry: Registry,
    job: IngestJob,
    file_hash: str,
) -> tuple[sqlite3.Row, bool] | tuple[str, str, int, str, str]:
    """Return either an existing version record or a new processing version."""
    existing = registry.get_by_file_hash(file_hash)
    if existing is not None:
        return existing, True

    if job.update_document_id:
        document = registry.get_document(job.update_document_id)
        if document is None:
            raise ValueError(
                f"Cannot update {job.update_document_id}: document does not exist."
            )
        document_id = job.update_document_id
        versions = registry.get_versions(document_id)
        version_number = (versions[-1]["version_number"] + 1) if versions else 1
    else:
        document_id = new_document_id()
        registry.create_document(document_id, job.pdf_path.name)
        version_number = 1

    version_id = new_version_id()
    stored_path = copy_pdf_to_storage(
        job.pdf_path,
        document_id,
        version_id,
        version_number,
    )
    registry.create_version(
        version_id=version_id,
        document_id=document_id,
        version_number=version_number,
        file_hash=file_hash,
        original_filename=job.pdf_path.name,
        stored_pdf_path=str(stored_path),
    )

    return document_id, version_id, version_number, str(stored_path), job.pdf_path.name


def ingest_one_document(
    *,
    job: IngestJob,
    registry: Registry,
    embedder: ColPaliEmbedder,
    qdrant: QdrantStore,
    signature: str,
) -> dict:
    pdf_path = job.pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    logger.info("\
=== Processing %s ===", pdf_path.name)
    file_hash = sha256_file(pdf_path)

    prepared = prepare_document(registry, job, file_hash)
    if isinstance(prepared[0], sqlite3.Row):
        existing = prepared[0]
        logger.info(
            "Exact file already known: %s -> %s/%s. Skipping ColPali.",
            pdf_path.name,
            existing["document_id"],
            existing["version_id"],
        )
        return {
            "status": "duplicate",
            "document_id": existing["document_id"],
            "version_id": existing["version_id"],
            "version_number": existing["version_number"],
            "filename": existing["original_filename"],
            "file_hash": existing["file_hash"],
        }

    document_id, version_id, version_number, stored_pdf_path, original_filename = prepared
    output_dir = PAGE_IMAGE_DIR / document_id / version_id

    try:
        pages = render_pdf_pages(pdf_path, output_dir, PDF_DPI)
        if not pages:
            raise RuntimeError(f"PDF has no pages: {pdf_path}")

        registry.update_version(version_id, page_count=len(pages))

        current_version = registry.get_current_version(document_id)

        total_new_embeddings = 0
        total_reused_embeddings = 0
        upsert_buffer: list[PointStruct] = []

        def flush_buffer() -> None:
            nonlocal upsert_buffer
            if not upsert_buffer:
                return
            qdrant.upsert(upsert_buffer)
            for point in upsert_buffer:
                # Page records are written after Qdrant acknowledges the upsert.
                payload = point.payload or {}
                registry.upsert_page(
                    version_id=version_id,
                    page_number=int(payload["page_number"]),
                    page_hash=str(payload["page_hash"]),
                    point_id=str(point.id),
                )
            upsert_buffer = []

        # First pass: identify pages that can reuse an existing compatible vector.
        work_pages: list[dict] = []
        for page in pages:
            cached_vector = qdrant.find_cached_vector(
                page_hash=page["page_hash"],
                signature=signature,
            )
            page["cached_vector"] = cached_vector
            if cached_vector is not None:
                total_reused_embeddings += 1
            else:
                work_pages.append(page)

        logger.info(
            "%s | pages=%d | reusable=%d | new_embeddings=%d",
            pdf_path.name,
            len(pages),
            total_reused_embeddings,
            len(work_pages),
        )

        # Reuse / upsert cached page embeddings first.
        for page in pages:
            vector = page.get("cached_vector")
            if vector is None:
                continue

            point_id = page_point_id(document_id, version_id, page["page_number"])
            payload = {
                "document_id": document_id,
                "version_id": version_id,
                "version_number": version_number,
                "file_hash": file_hash,
                "filename": original_filename,
                "page_number": page["page_number"],
                "page_hash": page["page_hash"],
                "page_image_path": str(page["image_path"]),
                "page_width": page["width"],
                "page_height": page["height"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "embedding_signature": signature,
                "quantization": "bitsandbytes_4bit_nf4",
                "embedding_type": "colpali_multivector",
                "is_current": False,
            }
            if STORE_PAGE_TEXT:
                payload["page_text"] = page["page_text"]

            upsert_buffer.append(
                PointStruct(
                    id=point_id,
                    vector={QdrantStore.VECTOR_NAME: vector},
                    payload=payload,
                )
            )
            if len(upsert_buffer) >= QDRANT_UPSERT_BATCH_SIZE:
                flush_buffer()

        flush_buffer()

        # New pages go through ColPali. The model is shared across all files.
        for batch in chunked(work_pages, BATCH_SIZE):
            images: list[Image.Image] = []
            for page in batch:
                with Image.open(page["image_path"]) as image:
                    images.append(image.convert("RGB").copy())

            embeddings = embedder.embed_images(images)
            if embeddings.ndim != 3:
                raise RuntimeError(
                    f"Unexpected ColPali embedding shape: {tuple(embeddings.shape)}. "
                    "Expected [batch, tokens, dimensions]."
                )

            total_new_embeddings += len(batch)

            for index, page in enumerate(batch):
                page_embedding = embeddings[index].tolist()
                point_id = page_point_id(document_id, version_id, page["page_number"])

                payload = {
                    "document_id": document_id,
                    "version_id": version_id,
                    "version_number": version_number,
                    "file_hash": file_hash,
                    "filename": original_filename,
                    "page_number": page["page_number"],
                    "page_hash": page["page_hash"],
                    "page_image_path": str(page["image_path"]),
                    "page_width": page["width"],
                    "page_height": page["height"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "embedding_signature": signature,
                    "quantization": "bitsandbytes_4bit_nf4",
                    "embedding_type": "colpali_multivector",
                    "embedding_shape": list(embeddings[index].shape),
                    "is_current": False,
                }
                if STORE_PAGE_TEXT:
                    payload["page_text"] = page["page_text"]

                upsert_buffer.append(
                    PointStruct(
                        id=point_id,
                        vector={QdrantStore.VECTOR_NAME: page_embedding},
                        payload=payload,
                    )
                )

                if len(upsert_buffer) >= QDRANT_UPSERT_BATCH_SIZE:
                    flush_buffer()

        flush_buffer()

        # New version is complete. Mark its pages current and demote the
        # previous current version. Old vectors remain available for history.
        qdrant.mark_version_current(
            document_id=document_id,
            version_id=version_id,
        )
        if current_version is not None and current_version["version_id"] != version_id:
            qdrant.demote_previous_versions(
                document_id=document_id,
                current_version_id=current_version["version_id"],
            )

        registry.update_version(
            version_id,
            page_count=len(pages),
            status="completed",
            completed_at=utc_now(),
        )
        registry.set_current_version(document_id, version_id)

        return {
            "status": "ingested",
            "document_id": document_id,
            "version_id": version_id,
            "version_number": version_number,
            "filename": original_filename,
            "file_hash": file_hash,
            "pages": len(pages),
            "new_embeddings": total_new_embeddings,
            "reused_embeddings": total_reused_embeddings,
            "stored_pdf": stored_pdf_path,
            "page_image_dir": str(output_dir),
        }

    except Exception:
        registry.update_version(version_id, status="failed")
        logger.exception("Ingestion failed for %s", pdf_path)
        raise


def ingest_files(jobs: list[IngestJob]) -> list[dict]:
    """Ingest N files while loading ColPali only once."""
    if not jobs:
        return []

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    registry = Registry(REGISTRY_DB_PATH)
    qdrant = QdrantStore(
        client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
        collection_name=COLLECTION_NAME,
    )

    try:
        # We only know vector size after the first embedding. If a previous
        # collection exists, it already has the correct configuration.
        if qdrant.collection_exists():
            qdrant.create_payload_indexes()

        embedder: ColPaliEmbedder | None = None
        signature = embedding_signature()
        results: list[dict] = []

        for job in jobs:
            # Exact duplicate check happens before model initialization for
            # duplicate-only batches. This avoids loading GPU weights needlessly.
            if job.pdf_path.exists():
                file_hash = sha256_file(job.pdf_path)
                existing = registry.get_by_file_hash(file_hash)
                if existing is not None:
                    results.append(
                        {
                            "status": "duplicate",
                            "document_id": existing["document_id"],
                            "version_id": existing["version_id"],
                            "version_number": existing["version_number"],
                            "filename": existing["original_filename"],
                            "file_hash": existing["file_hash"],
                        }
                    )
                    logger.info(
                        "Skipping exact duplicate: %s",
                        job.pdf_path,
                    )
                    continue
            else:
                raise FileNotFoundError(f"PDF does not exist: {job.pdf_path}")

            if embedder is None:
                embedder = ColPaliEmbedder(MODEL_ID, device)

            result = ingest_one_document(
                job=job,
                registry=registry,
                embedder=embedder,
                qdrant=qdrant,
                signature=signature,
            )
            results.append(result)

        return results
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Dummy test
# ---------------------------------------------------------------------------

def create_dummy_pdf(output_path: Path, title: str, second_line: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((72, 100), title, fontsize=22)
    page1.insert_text((72, 150), second_line, fontsize=14)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 100), "Page 2", fontsize=18)
    page2.insert_text((72, 140), "This page is for ingestion testing.", fontsize=14)

    doc.save(str(output_path))
    doc.close()
    return output_path


def dummy_ingestion_test() -> list[dict]:
    DUMMY_DIR.mkdir(parents=True, exist_ok=True)
    pdf_a = DUMMY_DIR / "dummy_manual_a.pdf"
    pdf_b = DUMMY_DIR / "dummy_manual_b.pdf"

    if not pdf_a.exists():
        create_dummy_pdf(pdf_a, "Dummy Manual A", "ColPali NF4 multi-file test")
    if not pdf_b.exists():
        create_dummy_pdf(pdf_b, "Dummy Manual B", "Second document in the batch")

    return ingest_files([
        IngestJob(pdf_a),
        IngestJob(pdf_b),
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch PDF ingestion for ColPali NF4 + Qdrant."
    )

    parser.add_argument(
        "--pdf",
        nargs="+",
        help="One or more new PDF files to ingest.",
    )

    parser.add_argument(
        "--update-doc",
        action="append",
        nargs=2,
        metavar=("DOCUMENT_ID", "PDF"),
        help=(
            "Ingest PDF as the next version of an existing document. "
            "Repeat this option for multiple updates."
        ),
    )

    parser.add_argument(
        "--dummy-test",
        action="store_true",
        help="Create two dummy PDFs and ingest them in one batch.",
    )

    parser.add_argument(
        "--reset-registry",
        action="store_true",
        help="Delete the local ingestion registry database. Does not delete Qdrant data.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_registry:
        registry = Registry(REGISTRY_DB_PATH)
        registry.reset()
        registry.close()
        logger.warning("Reset ingestion registry: %s", REGISTRY_DB_PATH)

    if args.dummy_test:
        results = dummy_ingestion_test()
    else:
        jobs: list[IngestJob] = []

        for pdf in args.pdf or []:
            jobs.append(IngestJob(Path(pdf)))

        for document_id, pdf in args.update_doc or []:
            jobs.append(
                IngestJob(
                    pdf_path=Path(pdf),
                    update_document_id=document_id,
                )
            )

        if not jobs:
            parser = argparse.ArgumentParser()
            print(
                "No files supplied. Examples:\
"
                "  python ingest.py --pdf a.pdf b.pdf c.pdf\
"
                "  python ingest.py --update-doc DOC-123 updated.pdf\
"
                "  python ingest.py --pdf a.pdf b.pdf --update-doc DOC-123 c.pdf\
"
                "  python ingest.py --dummy-test"
            )
            return

        results = ingest_files(jobs)

    print("\
Ingestion summary:")
    for result in results:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()