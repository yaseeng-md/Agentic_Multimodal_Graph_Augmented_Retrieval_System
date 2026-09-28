"""
Multimodal PDF ingestion pipeline
---------------------------------

Pipeline:
    PDF
      -> render pages to images
      -> ColPali page embeddings (multi-vector)
      -> Qdrant multivector collection (MAX_SIM)
      -> store page metadata + optional extracted page text

The collection defaults to: personal_collection

Environment variables are loaded from .env.

"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import uuid
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "personal_collection")
MODEL_ID = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3-merged")
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"

# BitsAndBytes 4-bit NF4 configuration
QUANTIZATION = os.getenv("QUANTIZATION", "4bit").lower()
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4").lower()
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16").lower()
BNB_4BIT_USE_DOUBLE_QUANT = os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"
SAVE_QUANTIZED_MODEL = os.getenv("SAVE_QUANTIZED_MODEL", "false").lower() == "true"
QUANTIZED_MODEL_DIR = Path(os.getenv(
    "QUANTIZED_MODEL_DIR",
    "./models/colpali-v1.3-merged-4bit-nf4",
))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PDF_DIR = Path(os.getenv("PDF_DIR", str(DATA_DIR / "pdfs")))
PAGE_IMAGE_DIR = Path(os.getenv("PAGE_IMAGE_DIR", str(DATA_DIR / "pages")))
DUMMY_DIR = Path(os.getenv("DUMMY_DIR", str(DATA_DIR / "dummy")))

PDF_DPI = int(os.getenv("PDF_DPI", "150"))
BATCH_SIZE = int(os.getenv("COLPALI_BATCH_SIZE", "1"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))
STORE_PAGE_TEXT = os.getenv("STORE_PAGE_TEXT", "true").lower() == "true"

DEVICE_ENV = os.getenv("DEVICE", "auto").lower()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("multimodal-rag-ingest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device() -> str:
    """Resolve the torch device from .env."""
    if DEVICE_ENV != "auto":
        return DEVICE_ENV

    if torch.cuda.is_available():
        return "cuda:0"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def model_dtype(device: str) -> torch.dtype:
    """Choose a safe inference dtype for the current device."""
    if device.startswith("cuda"):
        return torch.bfloat16
    if device.startswith("mps"):
        return torch.float16
    return torch.float32


def quant_compute_dtype() -> torch.dtype:
    """Resolve the BitsAndBytes compute dtype from .env."""
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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 hash for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def chunked(items: list, size: int) -> Iterable[list]:
    """Yield fixed-size chunks from a list."""
    if size <= 0:
        raise ValueError("Batch size must be > 0.")

    for start in range(0, len(items), size):
        yield items[start:start + size]


def page_point_id(document_sha256: str, page_number: int) -> str:
    """
    Generate a deterministic Qdrant point ID.

    This makes ingestion idempotent for the same PDF + page number.
    """
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{document_sha256}:page:{page_number}",
        )
    )


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_path: Path,
    document_id: str,
    output_dir: Path,
    dpi: int,
) -> list[dict]:
    """
    Render every PDF page to PNG and collect page metadata.

    Also extracts text from the PDF page when STORE_PAGE_TEXT=true.
    For scanned PDFs this may return an empty string because OCR is not
    performed here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []

    logger.info("Rendering PDF: %s", pdf_path)

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)

        for page_index in tqdm(
            range(total_pages),
            desc="Rendering pages",
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

            page_text = page.get_text("text").strip() if STORE_PAGE_TEXT else ""

            pages.append(
                {
                    "page_number": page_number,
                    "image_path": image_path,
                    "page_text": page_text,
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
            )

    logger.info("Rendered %d pages.", len(pages))
    return pages


# ---------------------------------------------------------------------------
# ColPali
# ---------------------------------------------------------------------------

class ColPaliEmbedder:
    """Loads merged ColPali and creates page-level multi-vector embeddings."""

    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device

        if QUANTIZATION != "4bit":
            raise ValueError(
                f"This ingestion pipeline is configured for 4-bit NF4 only. "
                f"Got QUANTIZATION={QUANTIZATION!r}."
            )

        if BNB_4BIT_QUANT_TYPE != "nf4":
            raise ValueError(
                f"This ingestion pipeline is configured for NF4 only. "
                f"Got BNB_4BIT_QUANT_TYPE={BNB_4BIT_QUANT_TYPE!r}."
            )

        if not device.startswith("cuda"):
            raise RuntimeError(
                "BitsAndBytes 4-bit NF4 is configured, but a CUDA device "
                "was not selected. Check DEVICE and your PyTorch/CUDA setup."
            )

        compute_dtype = quant_compute_dtype()

        logger.info(
            "Loading ColPali model=%s | device=%s | quantization=4bit NF4 | "
            "compute_dtype=%s | double_quant=%s | local_only=%s",
            model_id,
            device,
            compute_dtype,
            BNB_4BIT_USE_DOUBLE_QUANT,
            HF_LOCAL_FILES_ONLY,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )

        self.model = ColPali.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            local_files_only=HF_LOCAL_FILES_ONLY,
        ).eval()

        self.processor = ColPaliProcessor.from_pretrained(
            model_id,
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
        """
        Return page embeddings as:
            [batch, num_visual_tokens, embedding_dim]
        """
        batch = self.processor.process_images(images).to(self.model_device)
        embeddings = self.model(**batch)
        return embeddings.detach().to("cpu", dtype=torch.float32)

    def save_quantized_model(self, output_dir: Path) -> None:
        """Save the currently loaded BitsAndBytes quantized model and processor."""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving 4-bit NF4 model to %s", output_dir.resolve())
        self.model.save_pretrained(output_dir, safe_serialization=True)
        self.processor.save_pretrained(output_dir)
        logger.info("Quantized model saved.")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantStore:
    """Qdrant storage for ColPali page multi-vectors."""

    VECTOR_NAME = "page_embedding"

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def create_collection(self, vector_size: int) -> None:
        """
        Create a Qdrant multivector collection using MAX_SIM.

        HNSW is disabled for the original multivector because the exact
        late-interaction vector set is what we want to store and score.
        """
        if self.collection_exists():
            logger.info(
                "Qdrant collection already exists: %s",
                self.collection_name,
            )
            return

        logger.info(
            "Creating Qdrant collection=%s | vector_size=%d",
            self.collection_name,
            vector_size,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            },
        )

    def recreate_collection(self, vector_size: int) -> None:
        """Delete and recreate the collection."""
        if self.collection_exists():
            logger.warning(
                "Deleting existing collection: %s",
                self.collection_name,
            )
            self.client.delete_collection(self.collection_name)

        self.create_collection(vector_size)

    def upsert_pages(
        self,
        points: list[PointStruct],
    ) -> None:
        """Upsert a batch of page embeddings."""
        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: str | Path,
    *,
    recreate_collection: bool = False,
) -> dict:
    """
    Ingest one PDF into Qdrant.

    Returns a summary dictionary.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    document_sha256 = sha256_file(pdf_path)
    document_id = document_sha256[:16]

    # Keep page images grouped by document.
    document_image_dir = PAGE_IMAGE_DIR / document_id

    logger.info("Document ID: %s", document_id)

    pages = render_pdf_pages(
        pdf_path=pdf_path,
        document_id=document_id,
        output_dir=document_image_dir,
        dpi=PDF_DPI,
    )

    if not pages:
        raise RuntimeError("PDF contains no pages.")

    device = resolve_device()
    embedder = ColPaliEmbedder(
        model_id=MODEL_ID,
        device=device,
    )

    qdrant = QdrantStore(
        client=QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        ),
        collection_name=COLLECTION_NAME,
    )

    total_upserted = 0

    for batch_index, batch_pages in enumerate(
        chunked(pages, BATCH_SIZE),
        start=1,
    ):
        images: list[Image.Image] = []

        for page in batch_pages:
            with Image.open(page["image_path"]) as image:
                images.append(image.convert("RGB").copy())

        embeddings = embedder.embed_images(images)

        if embeddings.ndim != 3:
            raise RuntimeError(
                f"Unexpected ColPali embedding shape: {tuple(embeddings.shape)}. "
                "Expected [batch, tokens, dimensions]."
            )

        vector_size = int(embeddings.shape[-1])

        if batch_index == 1:
            if recreate_collection:
                qdrant.recreate_collection(vector_size)
            else:
                qdrant.create_collection(vector_size)

        points: list[PointStruct] = []

        for index, page in enumerate(batch_pages):
            page_embedding = embeddings[index]

            # page_embedding shape:
            #   [num_visual_tokens, embedding_dim]
            multi_vector = page_embedding.tolist()

            payload = {
                "document_id": document_id,
                "document_sha256": document_sha256,
                "source_pdf": str(pdf_path),
                "filename": pdf_path.name,
                "page_number": page["page_number"],
                "page_image_path": str(page["image_path"]),
                "page_width": page["width"],
                "page_height": page["height"],
                "model_id": MODEL_ID,
                "quantization": "bitsandbytes_4bit_nf4",
                "embedding_type": "colpali_multivector",
                "embedding_shape": list(page_embedding.shape),
            }

            if STORE_PAGE_TEXT:
                payload["page_text"] = page["page_text"]

            points.append(
                PointStruct(
                    id=page_point_id(
                        document_sha256=document_sha256,
                        page_number=page["page_number"],
                    ),
                    vector={
                        qdrant.VECTOR_NAME: multi_vector,
                    },
                    payload=payload,
                )
            )

        qdrant.upsert_pages(points)
        total_upserted += len(points)

        logger.info(
            "Upserted batch %d | pages=%d | embedding_shape=%s",
            batch_index,
            len(points),
            tuple(embeddings.shape),
        )

    summary = {
        "status": "success",
        "document_id": document_id,
        "filename": pdf_path.name,
        "pages": len(pages),
        "upserted_points": total_upserted,
        "collection": COLLECTION_NAME,
        "qdrant_url": QDRANT_URL,
        "model": MODEL_ID,
        "quantization": "bitsandbytes_4bit_nf4",
        "compute_dtype": BNB_4BIT_COMPUTE_DTYPE,
        "double_quant": BNB_4BIT_USE_DOUBLE_QUANT,
        "device": str(device),
        "pdf": str(pdf_path),
        "page_image_dir": str(document_image_dir),
    }

    logger.info("Ingestion completed: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Dummy test
# ---------------------------------------------------------------------------

def create_dummy_pdf(output_path: Path) -> Path:
    """
    Create a tiny local PDF for ingestion testing.

    This tests the complete rendering -> embedding -> Qdrant path without
    requiring the user to provide a real document.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (72, 100),
        "Dummy Multimodal RAG Document",
        fontsize=22,
    )
    page1.insert_text(
        (72, 150),
        "Page 1: This page contains a simple ingestion test.",
        fontsize=14,
    )

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (72, 100),
        "Page 2: Welding safety requires appropriate protection.",
        fontsize=14,
    )
    page2.insert_text(
        (72, 140),
        "This page exists to verify page-level retrieval metadata.",
        fontsize=14,
    )

    doc.save(str(output_path))
    doc.close()

    return output_path


def dummy_ingestion_test() -> dict:
    """
    Run a full end-to-end local ingestion test.

    Creates a 2-page PDF in DUMMY_DIR and ingests it into
    personal_collection.
    """
    DUMMY_DIR.mkdir(parents=True, exist_ok=True)
    dummy_pdf = DUMMY_DIR / "dummy_rag_test.pdf"

    if not dummy_pdf.exists():
        logger.info("Creating dummy PDF: %s", dummy_pdf)
        create_dummy_pdf(dummy_pdf)
    else:
        logger.info("Using existing dummy PDF: %s", dummy_pdf)

    return ingest_pdf(dummy_pdf)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into a Qdrant ColPali multivector collection."
    )

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument(
        "--pdf",
        type=str,
        help="Path to a PDF to ingest.",
    )

    group.add_argument(
        "--dummy-test",
        action="store_true",
        help="Create and ingest a small dummy PDF.",
    )

    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before ingestion.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dummy_test:
        result = dummy_ingestion_test()
        print("\nDummy ingestion result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return

    if args.pdf:
        result = ingest_pdf(
            args.pdf,
            recreate_collection=args.recreate_collection,
        )
        print("\nIngestion result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return

    print(
        "Nothing to ingest.\n\n"
        "Examples:\n"
        "  python ingest.py --pdf ./data/pdfs/manual.pdf\n"
        "  python ingest.py --dummy-test\n"
        "  python ingest.py --pdf ./data/pdfs/manual.pdf --recreate-collection"
    )


if __name__ == "__main__":
    main()
