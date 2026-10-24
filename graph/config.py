"""Knowledge graph and structured document processing configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


# Feature Flags
DOCLING_ENABLED = _env_bool("DOCLING_ENABLED", True)
NEO4J_ENABLED = _env_bool("NEO4J_ENABLED", True)
GRAPH_RETRIEVAL_ENABLED = _env_bool("GRAPH_RETRIEVAL_ENABLED", True)

# Neo4j Database Connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687").strip()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "amg_neo4j_password_2026").strip()
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j").strip()
NEO4J_CONNECT_TIMEOUT_SEC = _env_float("NEO4J_CONNECT_TIMEOUT_SEC", 10.0)
NEO4J_MAX_CONNECTION_LIFETIME_SEC = _env_float("NEO4J_MAX_CONNECTION_LIFETIME_SEC", 3600.0)

# Neo4j Docker Management
NEO4J_CONTAINER_NAME = os.getenv("NEO4J_CONTAINER_NAME", "amg-neo4j").strip()
NEO4J_IMAGE = os.getenv("NEO4J_IMAGE", "neo4j:5-community").strip()
NEO4J_VOLUME = os.getenv("NEO4J_VOLUME", "amg_neo4j_data").strip()
NEO4J_HTTP_PORT = _env_int("NEO4J_HTTP_PORT", 7474)
NEO4J_BOLT_PORT = _env_int("NEO4J_BOLT_PORT", 7687)

# Docling Parser Configuration
DOCLING_CACHE_DIR = Path(os.getenv("DOCLING_CACHE_DIR", "./data/docling_cache")).resolve()
DOCLING_MAX_PAGE_COUNT = _env_int("DOCLING_MAX_PAGE_COUNT", 0)  # 0 = unlimited
DOCLING_MAX_CHUNK_CHARS = _env_int("DOCLING_MAX_CHUNK_CHARS", 1200)
DOCLING_CHUNK_OVERLAP_CHARS = _env_int("DOCLING_CHUNK_OVERLAP_CHARS", 150)

# Graph Extraction Configuration
GRAPH_EXTRACTOR_PROVIDER = os.getenv("GRAPH_EXTRACTOR_PROVIDER", "openrouter").strip().lower()
GRAPH_EXTRACTOR_MODEL = os.getenv("GRAPH_EXTRACTOR_MODEL", os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")).strip()
GRAPH_EXTRACTOR_API_KEY = os.getenv("GRAPH_EXTRACTOR_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
GRAPH_EXTRACTOR_BASE_URL = os.getenv("GRAPH_EXTRACTOR_BASE_URL", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")).strip()
GRAPH_EXTRACTOR_TIMEOUT_SEC = _env_float("GRAPH_EXTRACTOR_TIMEOUT_SEC", 45.0)
GRAPH_EXTRACTOR_MAX_RETRIES = _env_int("GRAPH_EXTRACTOR_MAX_RETRIES", 3)
GRAPH_EXTRACTOR_MAX_CHUNKS_PER_CALL = _env_int("GRAPH_EXTRACTOR_MAX_CHUNKS_PER_CALL", 3)
GRAPH_EXTRACTOR_MAX_CLAIMS_PER_PAGE = _env_int("GRAPH_EXTRACTOR_MAX_CLAIMS_PER_PAGE", 15)

# Graph Retrieval Configuration
GRAPH_SEARCH_MAX_HOPS = _env_int("GRAPH_SEARCH_MAX_HOPS", 2)
GRAPH_SEARCH_PAGE_LIMIT = _env_int("GRAPH_SEARCH_PAGE_LIMIT", 10)
GRAPH_SEARCH_CLAIM_LIMIT = _env_int("GRAPH_SEARCH_CLAIM_LIMIT", 20)
GRAPH_SEARCH_MIN_CONFIDENCE = _env_float("GRAPH_SEARCH_MIN_CONFIDENCE", 0.6)
