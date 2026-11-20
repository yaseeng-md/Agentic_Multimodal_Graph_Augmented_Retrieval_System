"""
AMG Multimodal RAG — Brave Web Search Integration

Provides external web search capabilities via Brave Search REST API.
Gracefully handles missing API keys, timeouts, and rate limits without failing the generation pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

BRAVE_SEARCH_ENDPOINT = os.getenv(
    "BRAVE_SEARCH_ENDPOINT", "https://api.search.brave.com/res/v1/web/search"
)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()
DEFAULT_WEB_SEARCH_COUNT = int(os.getenv("WEB_SEARCH_COUNT", "5"))
DEFAULT_WEB_SEARCH_TIMEOUT = float(os.getenv("WEB_SEARCH_TIMEOUT", "10.0"))


@dataclass
class WebSearchResultItem:
    title: str
    url: str
    description: str
    snippet: str
    published_date: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WebSearchResponse:
    query: str
    status: str  # "success" | "skipped_no_api_key" | "empty" | "error" | "timeout"
    result_count: int
    results: list[WebSearchResultItem]
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "status": self.status,
            "result_count": self.result_count,
            "results": [item.to_dict() for item in self.results],
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    def format_evidence_text(self, max_items: int = 5) -> str:
        """Format web results into a clean text block for LLM prompt context."""
        if not self.results:
            return ""
        lines: list[str] = ["[Web Search Evidence]"]
        for idx, item in enumerate(self.results[:max_items], start=1):
            lines.append(f"[Web {idx}] Title: {item.title}")
            lines.append(f"  URL: {item.url}")
            if item.snippet:
                lines.append(f"  Snippet: {item.snippet}")
            elif item.description:
                lines.append(f"  Summary: {item.description}")
        return "\n".join(lines)


class BraveSearchClient:
    """Client for querying the Brave Web Search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = BRAVE_SEARCH_ENDPOINT,
        timeout: float = DEFAULT_WEB_SEARCH_TIMEOUT,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("BRAVE_API_KEY", "").strip()
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = requests.Session()

    def search(
        self,
        query: str,
        count: int = DEFAULT_WEB_SEARCH_COUNT,
    ) -> WebSearchResponse:
        """
        Execute a web search. If no API key is configured or the search fails,
        returns a WebSearchResponse with an appropriate status rather than raising.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            return WebSearchResponse(
                query=clean_query,
                status="empty",
                result_count=0,
                results=[],
                latency_ms=0.0,
            )

        if not self.api_key:
            logger.info("Brave search skipped: BRAVE_API_KEY not configured.")
            return WebSearchResponse(
                query=clean_query,
                status="skipped_no_api_key",
                result_count=0,
                results=[],
                latency_ms=0.0,
            )

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": clean_query,
            "count": min(max(1, count), 20),
            "text_decorations": "false",
            "search_lang": "en",
        }

        start_time = time.perf_counter()
        try:
            resp = self.session.get(
                self.endpoint,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            if resp.status_code == 200:
                data = resp.json()
                web_data = data.get("web", {})
                raw_items = web_data.get("results", [])

                items: list[WebSearchResultItem] = []
                for item in raw_items:
                    items.append(
                        WebSearchResultItem(
                            title=item.get("title", "").strip(),
                            url=item.get("url", "").strip(),
                            description=item.get("description", "").strip(),
                            snippet=(
                                item.get("extra_snippets", [""])[0]
                                if item.get("extra_snippets")
                                else item.get("description", "")
                            ).strip(),
                            published_date=item.get("page_age"),
                        )
                    )

                return WebSearchResponse(
                    query=clean_query,
                    status="success",
                    result_count=len(items),
                    results=items,
                    latency_ms=latency_ms,
                )

            err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Brave search error for '%s': %s", clean_query, err_msg)
            return WebSearchResponse(
                query=clean_query,
                status="error",
                result_count=0,
                results=[],
                latency_ms=latency_ms,
                error=err_msg,
            )

        except requests.exceptions.Timeout:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("Brave search timed out for '%s'", clean_query)
            return WebSearchResponse(
                query=clean_query,
                status="timeout",
                result_count=0,
                results=[],
                latency_ms=latency_ms,
                error="Request timed out",
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("Brave search failed for '%s': %s", clean_query, exc)
            return WebSearchResponse(
                query=clean_query,
                status="error",
                result_count=0,
                results=[],
                latency_ms=latency_ms,
                error=str(exc),
            )

    def close(self):
        self.session.close()


_BRAVE_CLIENT: Optional[BraveSearchClient] = None


def get_brave_search_client() -> BraveSearchClient:
    global _BRAVE_CLIENT
    if _BRAVE_CLIENT is None:
        _BRAVE_CLIENT = BraveSearchClient()
    return _BRAVE_CLIENT
