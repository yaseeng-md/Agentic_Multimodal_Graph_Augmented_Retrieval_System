"""
AMG Multimodal RAG - V1 Answer Generation

This module is built on top of the EXISTING query.py retrieval pipeline.

Flow:
    query.py::retrieve()
        -> ColPali query embedding
        -> Qdrant MaxSim
        -> ranked PDF pages

    generate.py
        -> optional ColPali release
        -> context builder
        -> Qwen3-VL
        -> optional Qwen release

All generation/GPU-lifecycle controls are configuration driven through .env.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from dotenv import load_dotenv
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

# Existing retrieval implementation.
from query import retrieve, release_query_encoder


load_dotenv()

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


GENERATOR_MODEL = os.getenv(
    "GENERATOR_MODEL",
    "Qwen/Qwen3-VL-2B-Instruct",
)

GENERATOR_MODEL_PATH = (
    os.getenv("GENERATOR_MODEL_PATH", "").strip() or None
)

GENERATOR_REVISION = os.getenv(
    "GENERATOR_MODEL_REVISION",
    "main",
)

# Model loading.
GENERATOR_USE_4BIT = _env_bool(
    "GENERATOR_USE_4BIT",
    True,
)

GENERATOR_PREQUANTIZED = _env_bool(
    "GENERATOR_PREQUANTIZED",
    False,
)

GENERATOR_BNB_4BIT_QUANT_TYPE = os.getenv(
    "GENERATOR_BNB_4BIT_QUANT_TYPE",
    "nf4",
)

GENERATOR_BNB_4BIT_DOUBLE_QUANT = _env_bool(
    "GENERATOR_BNB_4BIT_DOUBLE_QUANT",
    True,
)

GENERATOR_BNB_4BIT_COMPUTE_DTYPE = os.getenv(
    "GENERATOR_BNB_4BIT_COMPUTE_DTYPE",
    "bfloat16",
).lower()

GENERATOR_ATTN_IMPLEMENTATION = os.getenv(
    "GENERATOR_ATTN_IMPLEMENTATION",
    "sdpa",
)

# ---------------------------------------------------------------------------
# GPU lifecycle.
#
# This is the important part for a 6 GB RTX 4050.
# ---------------------------------------------------------------------------

# Release ColPali after retrieval and before loading Qwen.
GENERATOR_RELEASE_RETRIEVER = _env_bool(
    "GENERATOR_RELEASE_RETRIEVER",
    True,
)

# Release Qwen after generation.
# Recommended true when the same process will perform retrieval again.
GENERATOR_RELEASE_AFTER_GENERATION = _env_bool(
    "GENERATOR_RELEASE_AFTER_GENERATION",
    True,
)

# Force Python GC before/after model transitions.
GENERATOR_GC_ON_MODEL_SWITCH = _env_bool(
    "GENERATOR_GC_ON_MODEL_SWITCH",
    True,
)

# Empty CUDA cache during model transitions.
GENERATOR_EMPTY_CUDA_CACHE = _env_bool(
    "GENERATOR_EMPTY_CUDA_CACHE",
    True,
)

# Serialize generation on a single 6 GB GPU.
GENERATOR_SERIALIZE = _env_bool(
    "GENERATOR_SERIALIZE",
    True,
)

# ---------------------------------------------------------------------------
# Context / multimodal budget.
# ---------------------------------------------------------------------------

GENERATOR_CONTEXT_TOP_K = int(
    os.getenv("GENERATOR_CONTEXT_TOP_K", "2")
)

GENERATOR_MAX_IMAGES = int(
    os.getenv("GENERATOR_MAX_IMAGES", "1")
)

GENERATOR_INCLUDE_IMAGES = _env_bool(
    "GENERATOR_INCLUDE_IMAGES",
    True,
)

GENERATOR_MAX_PAGE_TEXT_CHARS = int(
    os.getenv("GENERATOR_MAX_PAGE_TEXT_CHARS", "5000")
)

GENERATOR_MAX_TOTAL_CONTEXT_CHARS = int(
    os.getenv("GENERATOR_MAX_TOTAL_CONTEXT_CHARS", "12000")
)

# Qwen image pixel budget. These control visual token usage.
# 256 * 32 * 32 ~= 262k pixels
GENERATOR_MIN_PIXELS = int(
    os.getenv("GENERATOR_MIN_PIXELS", str(256 * 32 * 32))
)

# 512 * 32 * 32 ~= 524k pixels
GENERATOR_MAX_PIXELS = int(
    os.getenv("GENERATOR_MAX_PIXELS", str(512 * 32 * 32))
)

# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------

GENERATOR_MAX_NEW_TOKENS = int(
    os.getenv("GENERATOR_MAX_NEW_TOKENS", "256")
)

GENERATOR_TEMPERATURE = float(
    os.getenv("GENERATOR_TEMPERATURE", "0.1")
)

GENERATOR_TOP_P = float(
    os.getenv("GENERATOR_TOP_P", "0.9")
)


# ===========================================================================
# Result structures
# ===========================================================================

@dataclass(frozen=True)
class GenerationSource:
    source_id: str
    rank: int
    score: float
    document_id: Optional[str]
    version_id: Optional[str]
    version_number: Optional[int]
    filename: Optional[str]
    page_number: Optional[int]
    page_hash: Optional[str]
    page_image_path: Optional[str]


@dataclass
class GenerationResponse:
    answer: str
    sources: list[dict[str, Any]]
    generation: dict[str, Any]


# ===========================================================================
# Utilities
# ===========================================================================

def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported GENERATOR_BNB_4BIT_COMPUTE_DTYPE={name!r}"
        ) from exc


def _model_source() -> str:
    if GENERATOR_MODEL_PATH:
        return str(
            Path(GENERATOR_MODEL_PATH)
            .expanduser()
            .resolve()
        )
    return GENERATOR_MODEL


def _local_file_uri(path: str) -> str:
    return Path(path).expanduser().resolve().as_uri()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cleanup_cuda_memory() -> None:
    """
    Release Python garbage and cached CUDA blocks between model phases.
    """
    if GENERATOR_GC_ON_MODEL_SWITCH:
        gc.collect()

    if (
        GENERATOR_EMPTY_CUDA_CACHE
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()


def _cuda_memory_snapshot() -> dict[str, float]:
    """
    Return a small VRAM snapshot in GiB for debugging/logging.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
        }

    return {
        "allocated_gib": round(
            torch.cuda.memory_allocated() / (1024 ** 3),
            3,
        ),
        "reserved_gib": round(
            torch.cuda.memory_reserved() / (1024 ** 3),
            3,
        ),
    }


def _normalise_retrieval_page(
    result: dict[str, Any],
) -> dict[str, Any]:
    payload = result.get("payload")
    payload = payload if isinstance(payload, dict) else {}

    def get(name: str) -> Any:
        value = result.get(name)
        return value if value is not None else payload.get(name)

    return {
        "rank": _safe_int(result.get("rank")) or 0,
        "score": _safe_float(result.get("score")) or 0.0,
        "point_id": result.get("point_id"),
        "document_id": get("document_id"),
        "version_id": get("version_id"),
        "version_number": _safe_int(get("version_number")),
        "filename": get("filename"),
        "page_number": _safe_int(get("page_number")),
        "page_hash": get("page_hash"),
        "page_image_path": get("page_image_path"),
        "page_text": (
            str(get("page_text")).strip()
            if get("page_text") is not None
            else ""
        ),
        "is_current": get("is_current"),
    }


def _build_generation_pages(
    retrieval_result: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_results = retrieval_result.get("results", [])

    if not isinstance(raw_results, list):
        raise TypeError(
            "retrieval_result['results'] must be a list"
        )

    pages: list[dict[str, Any]] = []
    total_chars = 0
    images_used = 0

    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue

        page = _normalise_retrieval_page(raw_result)

        page_text = page["page_text"][
            :GENERATOR_MAX_PAGE_TEXT_CHARS
        ]

        image_path = page["page_image_path"]

        if image_path:
            image_path = str(
                Path(image_path).expanduser()
            )

        usable_text = bool(page_text)

        usable_image = (
            GENERATOR_INCLUDE_IMAGES
            and bool(image_path)
            and images_used < GENERATOR_MAX_IMAGES
            and Path(image_path).is_file()
        )

        if not usable_text and not usable_image:
            continue

        remaining = (
            GENERATOR_MAX_TOTAL_CONTEXT_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        page_text = page_text[:remaining]

        if usable_image:
            images_used += 1
        else:
            image_path = None

        page["page_text"] = page_text
        page["page_image_path"] = image_path

        pages.append(page)
        total_chars += len(page_text)

        if len(pages) >= GENERATOR_CONTEXT_TOP_K:
            break

    return pages


def _build_messages(
    query: str,
    pages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are the answer-generation stage of a multimodal "
                "RAG system.\n\n"
                "Answer the user's question ONLY from the retrieved "
                "document pages below.\n\n"
                "Rules:\n"
                "1. Do not use outside knowledge.\n"
                "2. Do not invent facts, numbers, page numbers, document "
                "IDs, versions, or quotes.\n"
                "3. Use extracted text for textual evidence.\n"
                "4. Use the page image when visual/layout information is "
                "relevant.\n"
                "5. If evidence is insufficient, explicitly say so.\n"
                "6. Keep the answer concise.\n"
                "7. [S1], [S2], etc. refer to the supplied retrieved pages.\n\n"
                f"USER QUESTION:\n{query.strip()}"
            ),
        }
    ]

    for index, page in enumerate(pages, start=1):
        source_id = f"S{index}"

        content.append(
            {
                "type": "text",
                "text": (
                    f"\n\n========== {source_id} ==========\n"
                    f"Document: {page['filename'] or 'unknown'}\n"
                    f"Document ID: {page['document_id'] or 'unknown'}\n"
                    f"Version ID: {page['version_id'] or 'unknown'}\n"
                    f"Version: "
                    f"{page['version_number'] if page['version_number'] is not None else 'unknown'}\n"
                    f"Page: "
                    f"{page['page_number'] if page['page_number'] is not None else 'unknown'}\n"
                    f"Retrieval rank: {page['rank']}\n"
                    f"Retrieval score: {page['score']}\n"
                    "====================================\n"
                ),
            }
        )

        if page["page_text"]:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Extracted page text:\n"
                        f"{page['page_text']}"
                    ),
                }
            )

        if page["page_image_path"]:
            content.append(
                {
                    "type": "image",
                    "image": _local_file_uri(
                        page["page_image_path"]
                    ),
                }
            )

    content.append(
        {
            "type": "text",
            "text": (
                "\n\nAnswer the user's question now using only "
                "the supplied sources."
            ),
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def _build_sources(
    pages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        asdict(
            GenerationSource(
                source_id=f"S{index}",
                rank=page["rank"],
                score=page["score"],
                document_id=page["document_id"],
                version_id=page["version_id"],
                version_number=page["version_number"],
                filename=page["filename"],
                page_number=page["page_number"],
                page_hash=page["page_hash"],
                page_image_path=page["page_image_path"],
            )
        )
        for index, page in enumerate(pages, start=1)
    ]

class _NoOpLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

# ===========================================================================
# Qwen generator
# ===========================================================================

class QwenGenerator:
    """
    Process-local generation model.

    By configuration, the model can be released after every generation.
    """

    def __init__(self) -> None:
        self.model: Optional[
            Qwen3VLForConditionalGeneration
        ] = None
        self.processor: Optional[Any] = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    def _load_model(self) -> None:
        if self.model is not None:
            return

        with self._load_lock:
            if self.model is not None:
                return

            source = _model_source()

            logger.info(
                "Loading Qwen3-VL: %s",
                source,
            )

            if GENERATOR_USE_4BIT:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "GENERATOR_USE_4BIT=true requires CUDA."
                    )

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=(
                        GENERATOR_BNB_4BIT_QUANT_TYPE
                    ),
                    bnb_4bit_compute_dtype=_resolve_dtype(
                        GENERATOR_BNB_4BIT_COMPUTE_DTYPE
                    ),
                    bnb_4bit_use_double_quant=(
                        GENERATOR_BNB_4BIT_DOUBLE_QUANT
                    ),
                )
            else:
                quantization_config = None

            model_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "revision": GENERATOR_REVISION,
                "attn_implementation": (
                    GENERATOR_ATTN_IMPLEMENTATION
                ),
            }

            if quantization_config is not None:
                model_kwargs["quantization_config"] = (
                    quantization_config
                )
                model_kwargs["dtype"] = _resolve_dtype(
                    GENERATOR_BNB_4BIT_COMPUTE_DTYPE
                )
            else:
                model_kwargs["dtype"] = (
                    torch.bfloat16
                    if (
                        torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported()
                    )
                    else torch.float16
                    if torch.cuda.is_available()
                    else torch.float32
                )

            self.model = (
                Qwen3VLForConditionalGeneration.from_pretrained(
                    source,
                    **model_kwargs,
                ).eval()
            )

            self.processor = AutoProcessor.from_pretrained(
                source,
                revision=GENERATOR_REVISION,
                min_pixels=GENERATOR_MIN_PIXELS,
                max_pixels=GENERATOR_MAX_PIXELS,
            )

            logger.info(
                "Qwen loaded. GPU memory: %s",
                _cuda_memory_snapshot(),
            )

    def _release_model(self) -> None:
        if self.model is None and self.processor is None:
            return

        self.model = None
        self.processor = None

        _cleanup_cuda_memory()

        logger.info(
            "Qwen released. GPU memory: %s",
            _cuda_memory_snapshot(),
        )

    def generate_from_retrieval(
        self,
        query: str,
        retrieval_result: dict[str, Any],
    ) -> GenerationResponse:
        if not query or not query.strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        pages = _build_generation_pages(
            retrieval_result
        )

        if not pages:
            raise RuntimeError(
                "No usable retrieved pages were available "
                "for generation."
            )

        messages = _build_messages(
            query=query,
            pages=pages,
        )

        started = time.perf_counter()

        lock = (
            self._generation_lock
            if GENERATOR_SERIALIZE
            else _NoOpLock()
        )

        with lock:
            self._load_model()

            assert self.model is not None
            assert self.processor is not None

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs, video_kwargs = (
                process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            )

            video_metadata = None

            if video_inputs is not None:
                video_inputs, video_metadata = zip(
                    *video_inputs
                )
                video_inputs = list(video_inputs)
                video_metadata = list(video_metadata)

            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )

            inputs = inputs.to(self.model.device)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": GENERATOR_MAX_NEW_TOKENS,
                "use_cache": True,
            }

            if GENERATOR_TEMPERATURE > 0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": GENERATOR_TEMPERATURE,
                        "top_p": GENERATOR_TOP_P,
                    }
                )
            else:
                generation_kwargs["do_sample"] = False

            logger.info(
                "Starting generation. GPU memory before generate: %s",
                _cuda_memory_snapshot(),
            )

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                )

            trimmed_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            output_text = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Release input/generation tensors before optional model release.
            del inputs
            del generated_ids
            del trimmed_ids

        answer = (
            output_text[0].strip()
            if output_text
            else ""
        )

        duration_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        response = GenerationResponse(
            answer=answer,
            sources=_build_sources(pages),
            generation={
                "model_id": _model_source(),
                "prequantized": GENERATOR_PREQUANTIZED,
                "quantization": (
                    (
                        "bitsandbytes_4bit_"
                        f"{GENERATOR_BNB_4BIT_QUANT_TYPE}"
                    )
                    if GENERATOR_USE_4BIT
                    else "none"
                ),
                "compute_dtype": (
                    GENERATOR_BNB_4BIT_COMPUTE_DTYPE
                    if GENERATOR_USE_4BIT
                    else "auto"
                ),
                "context_pages": len(pages),
                "images_used": sum(
                    1
                    for page in pages
                    if page["page_image_path"]
                ),
                "min_pixels": GENERATOR_MIN_PIXELS,
                "max_pixels": GENERATOR_MAX_PIXELS,
                "max_new_tokens": GENERATOR_MAX_NEW_TOKENS,
                "duration_ms": duration_ms,
                "retrieval_scope": retrieval_result.get(
                    "scope"
                ),
                "retrieval_result_count": retrieval_result.get(
                    "result_count"
                ),
            },
        )

        if GENERATOR_RELEASE_AFTER_GENERATION:
            self._release_model()

        return response


# ===========================================================================
# Process-local generator
# ===========================================================================

_GENERATOR: Optional[QwenGenerator] = None
_GENERATOR_LOCK = threading.Lock()


def get_generator() -> QwenGenerator:
    global _GENERATOR

    if _GENERATOR is None:
        with _GENERATOR_LOCK:
            if _GENERATOR is None:
                _GENERATOR = QwenGenerator()

    return _GENERATOR


# ===========================================================================
# End-to-end function
# ===========================================================================

def query_and_generate(
    query: str,
    document_ids: Optional[Sequence[str]] = None,
    version_ids: Optional[Sequence[str]] = None,
    recursive_search: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Existing retrieve() -> optional ColPali release -> Qwen generation.
    """
    retrieval_result = retrieve(
        query=query,
        document_ids=document_ids,
        version_ids=version_ids,
        recursive_search=recursive_search,
        top_k=top_k,
    )

    logger.info(
        "Retrieval completed. GPU memory: %s",
        _cuda_memory_snapshot(),
    )

    if GENERATOR_RELEASE_RETRIEVER:
        release_query_encoder()
        _cleanup_cuda_memory()

    logger.info(
        "After retriever release. GPU memory: %s",
        _cuda_memory_snapshot(),
    )

    generator = get_generator()

    generation_result = generator.generate_from_retrieval(
        query=query,
        retrieval_result=retrieval_result,
    )

    return {
        "query": query,
        "retrieval": retrieval_result,
        "response": asdict(generation_result),
    }


def generate_answer(
    query: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate from an already completed retrieve() call.

    This function also honors GENERATOR_RELEASE_RETRIEVER for callers that
    retrieved data before entering this module.
    """
    if GENERATOR_RELEASE_RETRIEVER:
        release_query_encoder()
        _cleanup_cuda_memory()

    generator = get_generator()

    result = generator.generate_from_retrieval(
        query=query,
        retrieval_result=retrieval_result,
    )

    return asdict(result)


# ===========================================================================
# JSON helper
# ===========================================================================

def save_generation_result(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run existing AMG retrieval followed by Qwen3-VL generation."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query.",
    )

    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        default=None,
        help="Restrict retrieval to a document ID. Repeat if needed.",
    )

    parser.add_argument(
        "--version-id",
        dest="version_ids",
        action="append",
        default=None,
        help="Restrict retrieval to a version ID. Repeat if needed.",
    )

    parser.add_argument(
        "--recursive-search",
        action="store_true",
        help="Search all documents and all versions.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retrieval top-k. Default: 5.",
    )

    parser.add_argument(
        "--output",
        default="generation_result.json",
        help="Optional JSON output file.",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = _build_parser().parse_args()

    result = query_and_generate(
        query=args.query,
        document_ids=args.document_ids,
        version_ids=args.version_ids,
        recursive_search=args.recursive_search,
        top_k=args.top_k,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    if args.output:
        save_generation_result(
            result=result,
            output_path=args.output,
        )
        print(
            f"Saved end-to-end result to: {args.output}"
        )


if __name__ == "__main__":
    main()
