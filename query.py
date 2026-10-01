"""
AMG Multimodal RAG - V1 Retrieval

Independent retrieval module for testing before wiring retrieval into FastAPI.

Responsibilities in this file:
    1. Load the same ColPali model configuration used by ingestion.
    2. Encode a text query into a ColPali multi-vector query embedding.
    3. Build the retrieval scope according to the V1 policy.
    4. Search Qdrant's `page_embedding` multivector using MaxSim.
    5. Return ranked page-level results with their stored payload metadata.

V1 retrieval policy:
    - No document_ids / version_ids:
        Search all documents, latest/current versions only.
    - document_ids supplied:
        Search only those documents, latest/current versions only.
    - version_ids supplied:
        Search only those exact versions. Explicit version selection takes
        precedence over document_ids.
    - recursive_search=True:
        Search all documents and all versions, ignoring document_ids and
        version_ids.

This file is intentionally retrieval-only. It does not generate an answer.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


load_dotenv()


# ---------------------------------------------------------------------------
# Runtime dependencies
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("PyTorch is required to run query.py") from exc

try:
    from qdrant_client import QdrantClient, models as qdrant_models
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("qdrant-client is required to run query.py") from exc

try:
    from colpali_engine.models import ColPali, ColPaliProcessor
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "colpali-engine is required to run query.py. "
        "Install the same ColPali dependency used by the ingestion pipeline."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "personal_collection")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "page_embedding")

COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3-merged")
COLPALI_MODEL_REVISION = os.getenv("COLPALI_MODEL_REVISION", "main")
QUANTIZATION = os.getenv("QUANTIZATION", "4bit").lower()
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16").lower()
BNB_4BIT_USE_DOUBLE_QUANT = (
    os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"
)
DEVICE = os.getenv("DEVICE", "auto").lower()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalScope:
    """Human-readable description of the scope used for a query."""

    mode: str
    document_ids: tuple[str, ...]
    version_ids: tuple[str, ...]
    recursive_search: bool


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _resolve_compute_dtype(name: str) -> torch.dtype:
    """Resolve a torch dtype from an environment string."""
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
        valid = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported BNB_4BIT_COMPUTE_DTYPE={name!r}. Valid values: {valid}"
        ) from exc


def _resolve_device() -> str:
    """Resolve the requested model device."""
    if DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
        return "cuda"

    if DEVICE == "cpu":
        return "cpu"

    return DEVICE


def _build_quantization_config() -> Optional[Any]:
    """Build the BitsAndBytes 4-bit configuration when enabled."""
    if QUANTIZATION not in {"4bit", "nf4"}:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit BitsAndBytes quantization is configured, but CUDA is not available. "
            "Use a CUDA environment or set QUANTIZATION=none for a CPU test."
        )

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "transformers is required for BitsAndBytes quantization."
        ) from exc

    compute_dtype = _resolve_compute_dtype(BNB_4BIT_COMPUTE_DTYPE)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


class QueryEncoder:
    """Loads ColPali once and encodes text queries into multivectors."""

    def __init__(self) -> None:
        self.device = _resolve_device()
        self.model, self.processor = self._load()

    def _load(self):
        quantization_config = _build_quantization_config()

        load_kwargs: dict[str, Any] = {
            "revision": COLPALI_MODEL_REVISION,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = _resolve_compute_dtype(
                BNB_4BIT_COMPUTE_DTYPE
            )
        else:
            dtype = (
                torch.bfloat16
                if self.device == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if self.device == "cuda"
                else torch.float32
            )
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["device_map"] = self.device

        print(
            "[QueryEncoder] Loading ColPali: "
            f"model={COLPALI_MODEL}, revision={COLPALI_MODEL_REVISION}, "
            f"quantization={QUANTIZATION}, device={self.device}"
        )

        model = ColPali.from_pretrained(COLPALI_MODEL, **load_kwargs).eval()
        processor = ColPaliProcessor.from_pretrained(
            COLPALI_MODEL,
            revision=COLPALI_MODEL_REVISION,
        )

        return model, processor

    def encode(self, query: str) -> list[list[float]]:
        """Encode one text query as a Qdrant-compatible multivector."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        batch = self.processor.process_queries([query]).to(self.model.device)

        with torch.no_grad():
            embeddings = self.model(**batch)

        # ColPali returns [batch, query_tokens, embedding_dim].
        # Qdrant multivector search expects [query_tokens, embedding_dim]
        # for a single query.
        if hasattr(embeddings, "detach"):
            embedding = embeddings[0].detach().float().cpu().tolist()
        elif hasattr(embeddings, "embeddings"):
            value = embeddings.embeddings
            embedding = value[0].detach().float().cpu().tolist()
        else:
            raise TypeError(
                "Unexpected ColPali query embedding type: "
                f"{type(embeddings).__name__}"
            )

        if not embedding or not isinstance(embedding[0], list):
            raise ValueError("ColPali returned an invalid multivector shape")

        return embedding


_QUERY_ENCODER: Optional[QueryEncoder] = None


def get_query_encoder() -> QueryEncoder:
    """Return a process-local singleton query encoder."""
    global _QUERY_ENCODER

    if _QUERY_ENCODER is None:
        _QUERY_ENCODER = QueryEncoder()

    return _QUERY_ENCODER


# ---------------------------------------------------------------------------
# Retrieval policy
# ---------------------------------------------------------------------------
def _normalise_ids(values: Optional[Iterable[str]]) -> list[str]:
    """Normalize optional ID collections and reject blank IDs."""
    if values is None:
        return []

    result = []
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if value:
            result.append(value)

    # Preserve caller order while removing duplicates.
    return list(dict.fromkeys(result))


def resolve_retrieval_scope(
    document_ids: Optional[Sequence[str]] = None,
    version_ids: Optional[Sequence[str]] = None,
    recursive_search: bool = False,
) -> RetrievalScope:
    """
    Resolve the V1 retrieval policy.

    Precedence:
        1. recursive_search=True -> all documents/all versions
        2. version_ids -> exact versions
        3. document_ids -> current version(s) of those documents
        4. otherwise -> current version of every document
    """
    docs = _normalise_ids(document_ids)
    versions = _normalise_ids(version_ids)

    if recursive_search:
        return RetrievalScope(
            mode="all_documents_all_versions",
            document_ids=(),
            version_ids=(),
            recursive_search=True,
        )

    if versions:
        return RetrievalScope(
            mode="explicit_versions",
            document_ids=(),
            version_ids=tuple(versions),
            recursive_search=False,
        )

    if docs:
        return RetrievalScope(
            mode="documents_current_versions",
            document_ids=tuple(docs),
            version_ids=(),
            recursive_search=False,
        )

    return RetrievalScope(
        mode="all_documents_current_versions",
        document_ids=(),
        version_ids=(),
        recursive_search=False,
    )


def _build_qdrant_filter(scope: RetrievalScope) -> Optional[qdrant_models.Filter]:
    """Convert a resolved retrieval scope into a Qdrant payload filter."""
    must_conditions: list[Any] = []

    if scope.mode == "all_documents_all_versions":
        return None

    if scope.mode == "explicit_versions":
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="version_id",
                match=qdrant_models.MatchAny(any=list(scope.version_ids)),
            )
        )

    else:
        # Default retrieval always searches the latest/current version only.
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="is_current",
                match=qdrant_models.MatchValue(value=True),
            )
        )

        if scope.mode == "documents_current_versions":
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(any=list(scope.document_ids)),
                )
            )

    return qdrant_models.Filter(must=must_conditions)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------
def create_qdrant_client() -> QdrantClient:
    """Create a Qdrant client using the same connection settings as V1."""
    kwargs: dict[str, Any] = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY

    return QdrantClient(**kwargs)


def _check_collection(client: QdrantClient) -> None:
    """Fail early with a useful message if the configured collection is absent."""
    collections = client.get_collections().collections
    names = {collection.name for collection in collections}

    if QDRANT_COLLECTION not in names:
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' does not exist. "
            f"Available collections: {sorted(names)}"
        )


def _serialise_point(point: Any, rank: int) -> dict[str, Any]:
    """Convert a Qdrant ScoredPoint to JSON-friendly retrieval output."""
    point_id = point.id
    payload = dict(point.payload or {})

    return {
        "rank": rank,
        "score": float(point.score),
        "point_id": str(point_id),
        "document_id": payload.get("document_id"),
        "version_id": payload.get("version_id"),
        "version_number": payload.get("version_number"),
        "filename": payload.get("filename") or payload.get("original_filename"),
        "page_number": payload.get("page_number"),
        "page_hash": payload.get("page_hash"),
        "page_image_path": payload.get("page_image_path"),
        "page_text": payload.get("page_text"),
        "is_current": payload.get("is_current"),
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Public retrieval function
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    document_ids: Optional[Sequence[str]] = None,
    version_ids: Optional[Sequence[str]] = None,
    recursive_search: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Retrieve the top-k relevant PDF pages for a text query.

    Parameters
    ----------
    query:
        Natural-language user query.
    document_ids:
        Optional logical document IDs to restrict retrieval to.
    version_ids:
        Optional exact version IDs to restrict retrieval to. These take
        precedence over document_ids unless recursive_search=True.
    recursive_search:
        If True, ignore document_ids/version_ids and search every version of
        every document.
    top_k:
        Number of ranked pages to return.

    Returns
    -------
    dict
        JSON-serialisable retrieval response.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be an integer >= 1")

    scope = resolve_retrieval_scope(
        document_ids=document_ids,
        version_ids=version_ids,
        recursive_search=recursive_search,
    )

    print(f"[Retrieval] Query: {query}")
    print(f"[Retrieval] Scope: {scope.mode}")
    print(f"[Retrieval] Top-K: {top_k}")

    encoder = get_query_encoder()
    query_embedding = encoder.encode(query)

    client = create_qdrant_client()
    _check_collection(client)

    query_filter = _build_qdrant_filter(scope)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        using=QDRANT_VECTOR_NAME,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = [
        _serialise_point(point, rank=index)
        for index, point in enumerate(response.points, start=1)
    ]

    return {
        "query": query,
        "top_k": top_k,
        "scope": {
            "mode": scope.mode,
            "document_ids": list(scope.document_ids),
            "version_ids": list(scope.version_ids),
            "recursive_search": scope.recursive_search,
        },
        "result_count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Independent CLI test
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AMG V1 multimodal PDF retrieval directly from query.py."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query to search for.",
    )
    parser.add_argument(
        "--document-id",
        dest="document_ids",
        action="append",
        default=None,
        help="Restrict search to a document ID. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--version-id",
        dest="version_ids",
        action="append",
        default=None,
        help="Restrict search to an exact version ID. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--recursive-search",
        action="store_true",
        help="Search all documents and all versions, ignoring supplied IDs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = retrieve(
        query=args.query,
        document_ids=args.document_ids,
        version_ids=args.version_ids,
        recursive_search=args.recursive_search,
        top_k=args.top_k,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "000"

    filename = f"{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"Saved result to: {filename}")


if __name__ == "__main__":
    main()
