"""ViDoRe (Visual Document Retrieval) dataset loader."""

from dataclasses import dataclass, field
import logging
from typing import Any, Optional
from PIL import Image

try:
    from datasets import load_dataset
except ImportError as exc:
    raise RuntimeError(
        "The 'datasets' package is required for ViDoRe evaluation. "
        "Install with `pip install datasets`."
    ) from exc

logger = logging.getLogger("amg-eval-vidore")

DATASET_ALIASES = {
    "docvqa": "vidore/docvqa_test_subsampled",
    "infovqa": "vidore/infovqa_test_subsampled",
    "arxivqa": "vidore/arxivqa_test_subsampled",
    "tabf_factual": "vidore/tabf_factual",
    "tabfact": "vidore/tabf_factual",
    "shiftproject": "vidore/shiftproject_test",
    "ai": "vidore/artificial_intelligence_test",
    "energy": "vidore/energy_test",
    "economics": "vidore/economics_test",
}


@dataclass
class ViDoReDataset:
    """In-memory representation of a ViDoRe benchmark split."""

    dataset_name: str
    corpus: dict[str, Image.Image] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def num_documents(self) -> int:
        return len(self.corpus)

    @property
    def num_queries(self) -> int:
        return len(self.queries)


def resolve_dataset_name(name: str) -> str:
    """Resolve shorthand aliases into full Hugging Face dataset IDs."""
    key = name.strip().lower()
    return DATASET_ALIASES.get(key, name)


def load_vidore_dataset(
    dataset_name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> ViDoReDataset:
    """
    Load a ViDoRe dataset split from Hugging Face and prepare Corpus, Queries, and Qrels.

    Parameters
    ----------
    dataset_name:
        Dataset name or shorthand alias (e.g. 'docvqa', 'vidore/docvqa_test_subsampled').
    split:
        Dataset split, defaults to 'test'.
    max_samples:
        Optional sample limit for quick smoke testing or small benchmarks.

    Returns
    -------
    ViDoReDataset:
        Structured container with deduplicated corpus page images, queries, and qrels.
    """
    hf_name = resolve_dataset_name(dataset_name)
    logger.info("Loading ViDoRe dataset: %s (split: %s)", hf_name, split)

    ds = load_dataset(hf_name, split=split)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    corpus: dict[str, Image.Image] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for idx, row in enumerate(ds):
        # 1. Resolve query identifier and query text
        query_id = (
            str(row.get("questionId"))
            if row.get("questionId") is not None
            else str(row.get("query_id") or row.get("id") or f"query_{idx:05d}")
        )
        query_text = (row.get("query") or row.get("question") or "").strip()
        if not query_text:
            continue

        # 2. Resolve document/image identifier
        doc_id = (
            str(row.get("image_filename"))
            if row.get("image_filename")
            else None
        )
        if not doc_id:
            raw_doc_id = row.get("docId") or row.get("doc_id")
            raw_page = row.get("page")
            if raw_doc_id is not None and raw_page is not None:
                doc_id = f"doc_{raw_doc_id}_page_{raw_page}"
            else:
                doc_id = f"doc_{idx:05d}"

        # 3. Store corpus page image (deduplicating across queries sharing pages)
        if doc_id not in corpus:
            raw_image = row.get("image")
            if raw_image is not None:
                if isinstance(raw_image, Image.Image):
                    corpus[doc_id] = raw_image.convert("RGB")
                else:
                    # In case of file path or raw bytes
                    corpus[doc_id] = Image.open(raw_image).convert("RGB")

        # 4. Record query and ground-truth qrel
        queries[query_id] = query_text
        if query_id not in qrels:
            qrels[query_id] = {}
        qrels[query_id][doc_id] = 1

        # 5. Optional metadata for reporting / debugging
        metadata[query_id] = {
            "doc_id": doc_id,
            "answer": row.get("answer"),
            "source": row.get("source"),
            "data_split": row.get("data_split"),
        }

    logger.info(
        "Loaded ViDoRe dataset: %s | %d queries | %d unique corpus pages",
        hf_name,
        len(queries),
        len(corpus),
    )

    return ViDoReDataset(
        dataset_name=hf_name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata=metadata,
    )
