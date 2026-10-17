"""Configuration for AMG Evaluation Harness."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent

# Qdrant Isolation
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
EVAL_COLLECTION_NAME = os.getenv("EVAL_COLLECTION_NAME", "eval_vidore_v3")
EVAL_VECTOR_NAME = os.getenv("EVAL_VECTOR_NAME", "page_embedding")

# Model & Quantization Configuration
COLPALI_MODEL = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3-merged")
COLPALI_MODEL_REVISION = os.getenv("COLPALI_MODEL_REVISION", "main")
QUANTIZATION = os.getenv("QUANTIZATION", "4bit").lower()
BNB_4BIT_QUANT_TYPE = os.getenv("BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = os.getenv("BNB_4BIT_COMPUTE_DTYPE", "bfloat16").lower()
BNB_4BIT_USE_DOUBLE_QUANT = (
    os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true"
)
DEVICE = os.getenv("DEVICE", "auto").lower()

# Benchmark defaults
DEFAULT_DATASET = "vidore/docvqa_test_subsampled"
DEFAULT_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "1"))
DEFAULT_CUTOFFS = (1, 3, 5, 10)

# Paths
REPORTS_DIR = ROOT_DIR / "reports" / "vidore"
CACHE_DIR = ROOT_DIR / "data" / "eval_cache"


@dataclass(frozen=True)
class EvalConfig:
    qdrant_url: str = QDRANT_URL
    qdrant_api_key: Optional[str] = QDRANT_API_KEY
    collection_name: str = EVAL_COLLECTION_NAME
    vector_name: str = EVAL_VECTOR_NAME
    model_id: str = COLPALI_MODEL
    revision: str = COLPALI_MODEL_REVISION
    batch_size: int = DEFAULT_BATCH_SIZE
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS
    reports_dir: Path = REPORTS_DIR
    cache_dir: Path = CACHE_DIR
