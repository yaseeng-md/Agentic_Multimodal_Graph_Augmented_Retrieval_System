"""
Jina AI Multimodal Reranker Integration (jina-reranker-m0).

Provides a client for re-ranking multimodal document retrieval candidates using
Jina AI's Vision-Language Reranker API (jina-reranker-m0).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
import time
from typing import Any, Optional, Sequence, Union

from PIL import Image
import requests
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

logger = logging.getLogger("amg-reranker")

# Default configuration from environment
JINA_RERANKER_BASE_URL = os.getenv(
    "JINA_RERANKER_BASE_URL",
    "https://api.jina.ai/v1/rerank",
).strip()

JINA_RERANKER_MODEL = os.getenv(
    "JINA_RERANKER_MODEL",
    "jina-reranker-m0",
).strip()

JINA_RERANKER_TIMEOUT_SEC = int(
    os.getenv("JINA_RERANKER_TIMEOUT_SEC", "30")
)

# Standard default thresholds
DEFAULT_RERANK_RETRIEVAL_TOP_K = 20
DEFAULT_RERANK_RELEVANCE_THRESHOLD = 0.75
DEFAULT_RERANK_GENERATION_TOP_K = 5


def get_jina_api_key() -> str:
    """Retrieve the Jina API key from environment variables."""
    key = os.getenv("JINA_RERANKER_KEY") or os.getenv("JINA_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "JINA_API_KEY or JINA_RERANKER_KEY environment variable is not set. "
            "Please provide a valid Jina API key."
        )
    return key.strip()


def encode_image_to_base64(
    image_input: Union[Image.Image, str, Path, bytes],
    max_size: tuple[int, int] = (1024, 1024),
    format: str = "JPEG",
    quality: int = 85,
) -> str:
    """
    Convert an image (PIL Image, file path, bytes, or existing base64/URL) to base64.

    Parameters
    ----------
    image_input:
        A PIL Image, local file path (str or Path), raw image bytes, or existing base64 string/URL.
    max_size:
        Max width/height to resize the image to, keeping aspect ratio and reducing network payload.
    format:
        Output format ('JPEG' or 'PNG'). JPEG with 85 quality provides excellent readability at small size.
    quality:
        Compression quality for JPEG (1-100).

    Returns
    -------
    str:
        Base64-encoded image string or existing HTTP URL.
    """
    if isinstance(image_input, str):
        # If already a URL, return directly
        if image_input.startswith(("http://", "https://")):
            return image_input
        # If already base64 data URI or raw base64 string
        if image_input.startswith("data:image/") or (len(image_input) > 200 and " " not in image_input and not Path(image_input).exists()):
            return image_input
        # Otherwise treat as file path
        path = Path(image_input).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            return _pil_to_base64(img, max_size=max_size, format=format, quality=quality)

    if isinstance(image_input, Path):
        path = image_input.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        with Image.open(path) as img:
            return _pil_to_base64(img, max_size=max_size, format=format, quality=quality)

    if isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as img:
            return _pil_to_base64(img, max_size=max_size, format=format, quality=quality)

    if isinstance(image_input, Image.Image):
        return _pil_to_base64(image_input, max_size=max_size, format=format, quality=quality)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _pil_to_base64(
    img: Image.Image,
    max_size: tuple[int, int] = (1024, 1024),
    format: str = "JPEG",
    quality: int = 85,
) -> str:
    """Helper to resize and serialize a PIL Image to base64."""
    img_rgb = img.convert("RGB")
    if max_size and (img_rgb.width > max_size[0] or img_rgb.height > max_size[1]):
        img_rgb.thumbnail(max_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img_rgb.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class JinaReranker:
    """
    Client for Jina AI Multimodal Reranker API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = JINA_RERANKER_MODEL,
        base_url: str = JINA_RERANKER_BASE_URL,
        timeout: int = JINA_RERANKER_TIMEOUT_SEC,
    ) -> None:
        self.api_key = (api_key or os.getenv("JINA_RERANKER_KEY") or os.getenv("JINA_API_KEY") or "").strip()
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: Optional[int] = None,
        max_retries: int = 3,
        backoff_sec: float = 1.5,
    ) -> dict[str, Any]:
        """
        Send reranking request to Jina AI API.

        Parameters
        ----------
        query:
            The natural language search query.
        documents:
            List of document dictionaries, each containing 'image' and/or 'text'.
            Example: [{'image': base64_str}, {'text': '...'}]
        top_n:
            Optional number of top scoring results to return.
        max_retries:
            Maximum retry attempts for transient server errors / rate limits.
        backoff_sec:
            Initial backoff delay in seconds between retries.

        Returns
        -------
        dict[str, Any]:
            Parsed API response containing 'results' list with 'index' and 'relevance_score'.
        """
        if not self.api_key:
            self.api_key = get_jina_api_key()

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty for reranking.")

        if not documents:
            return {"model": self.model, "results": [], "usage": {}}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "query": query.strip(),
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None and top_n > 0:
            payload["top_n"] = top_n

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.perf_counter()
                response = self._session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if response.status_code == 200:
                    data = response.json()
                    logger.debug(
                        "Jina rerank completed in %.2f ms | %d candidates",
                        elapsed_ms,
                        len(documents),
                    )
                    return data

                # Retry on rate limit (429) or transient gateway errors (500, 502, 503, 504)
                if response.status_code in {429, 500, 502, 503, 504}:
                    logger.warning(
                        "Jina API returned %d (attempt %d/%d). Retrying in %.2fs...",
                        response.status_code,
                        attempt,
                        max_retries,
                        backoff_sec * attempt,
                    )
                    time.sleep(backoff_sec * attempt)
                    continue

                response.raise_for_status()

            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < max_retries:
                    logger.warning(
                        "Jina rerank request failed (%s). Retrying in %.2fs...",
                        exc,
                        backoff_sec * attempt,
                    )
                    time.sleep(backoff_sec * attempt)
                else:
                    logger.error("Jina rerank failed after %d attempts: %s", max_retries, exc)

        raise RuntimeError(f"Failed to rerank via Jina AI API after {max_retries} attempts: {last_error}")

    def rerank_id_image_pairs(
        self,
        query: str,
        id_image_pairs: Sequence[tuple[str, Any]],
        top_n: Optional[int] = None,
    ) -> tuple[list[tuple[str, float]], float]:
        """
        Rerank a sequence of (doc_id, image) pairs for a query.

        Parameters
        ----------
        query:
            Natural language query text.
        id_image_pairs:
            Sequence of (doc_id, PIL.Image or path).
        top_n:
            Optional limit on number of returned results.

        Returns
        -------
        tuple[list[tuple[str, float]], float]:
            Ranked list of (doc_id, relevance_score) sorted descending, plus elapsed time in ms.
        """
        if not id_image_pairs:
            return [], 0.0

        t_start = time.perf_counter()
        doc_payloads = []
        doc_ids = []

        for doc_id, image_item in id_image_pairs:
            b64_image = encode_image_to_base64(image_item)
            doc_payloads.append({"image": b64_image})
            doc_ids.append(doc_id)

        rerank_response = self.rerank(
            query=query,
            documents=doc_payloads,
            top_n=top_n,
        )

        results = rerank_response.get("results", [])
        ranked_pairs: list[tuple[str, float]] = []

        for item in results:
            idx = item.get("index")
            score = float(item.get("relevance_score", 0.0))
            if idx is not None and 0 <= idx < len(doc_ids):
                ranked_pairs.append((doc_ids[idx], score))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return ranked_pairs, elapsed_ms

    def rerank_retrieval_results(
        self,
        query: str,
        retrieval_candidates: list[dict[str, Any]],
        threshold: float = DEFAULT_RERANK_RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_RERANK_GENERATION_TOP_K,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Rerank standard AMG retrieval candidate dictionaries, apply confidence
        threshold, and select top-K candidates for multimodal generation.

        Parameters
        ----------
        query:
            Natural language query string.
        retrieval_candidates:
            List of candidate dictionaries from query.retrieve() (having 'page_image_path').
        threshold:
            Relevance score cutoff threshold (default: 0.75).
        top_k:
            Maximum number of top-scoring candidates to return (default: 5).

        Returns
        -------
        tuple[list[dict[str, Any]], float]:
            Reranked, filtered candidates with attached rerank metadata, and latency in ms.
        """
        if not retrieval_candidates:
            return [], 0.0

        t_start = time.perf_counter()
        doc_payloads: list[dict[str, Any]] = []
        valid_candidates: list[dict[str, Any]] = []

        for cand in retrieval_candidates:
            img_path = cand.get("page_image_path")
            if not img_path or not Path(img_path).is_file():
                continue
            try:
                b64_img = encode_image_to_base64(img_path)
                doc_payloads.append({"image": b64_img})
                valid_candidates.append(cand)
            except Exception as exc:
                logger.warning("Could not encode page image %s: %s", img_path, exc)

        if not valid_candidates:
            return retrieval_candidates[:top_k], 0.0

        rerank_response = self.rerank(
            query=query,
            documents=doc_payloads,
        )

        results = rerank_response.get("results", [])

        # Map reranked results
        reranked_all: list[dict[str, Any]] = []
        for new_rank, item in enumerate(results, start=1):
            idx = item.get("index")
            score = float(item.get("relevance_score", 0.0))
            if idx is not None and 0 <= idx < len(valid_candidates):
                cand_copy = dict(valid_candidates[idx])
                cand_copy["original_rank"] = cand_copy.get("rank")
                cand_copy["original_score"] = cand_copy.get("score")
                cand_copy["rank"] = new_rank
                cand_copy["score"] = score
                cand_copy["relevance_score"] = score
                cand_copy["reranked"] = True
                cand_copy["rerank_model"] = self.model
                reranked_all.append(cand_copy)

        # Apply threshold filtering (e.g. 0.75)
        filtered = [c for c in reranked_all if c["score"] >= threshold]

        # If none pass threshold, gracefully fall back to top 1 candidate so generation doesn't completely fail
        if not filtered and reranked_all:
            logger.warning(
                "No candidates exceeded rerank threshold %.2f (top score: %.4f). Falling back to top-1.",
                threshold,
                reranked_all[0]["score"],
            )
            fallback_item = dict(reranked_all[0])
            fallback_item["threshold_fallback"] = True
            filtered = [fallback_item]

        final_candidates = filtered[:top_k]

        # Re-index rank numbers from 1 to len(final_candidates)
        for i, item in enumerate(final_candidates, start=1):
            item["rank"] = i

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return final_candidates, elapsed_ms
