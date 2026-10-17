"""Indexes ViDoRe benchmark corpus pages into an isolated Qdrant collection."""

import logging
import uuid
from typing import Any, Optional
from PIL import Image
from tqdm import tqdm

import torch
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

from eval.config import (
    EVAL_COLLECTION_NAME,
    EVAL_VECTOR_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
    DEFAULT_BATCH_SIZE,
)
from ingest import (
    ColPaliEmbedder,
    resolve_device,
    MODEL_ID,
)

logger = logging.getLogger("amg-eval-indexer")


def get_eval_qdrant_client(url: str = QDRANT_URL, api_key: Optional[str] = QDRANT_API_KEY) -> QdrantClient:
    """Connect to Qdrant instance for evaluation."""
    kwargs: dict[str, Any] = {"url": url}
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


class ViDoReIndexer:
    """Manages isolated Qdrant collection setup and batch multivector indexing for ViDoRe."""

    def __init__(
        self,
        collection_name: str = EVAL_COLLECTION_NAME,
        vector_name: str = EVAL_VECTOR_NAME,
        client: Optional[QdrantClient] = None,
        embedder: Optional[ColPaliEmbedder] = None,
    ) -> None:
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.client = client or get_eval_qdrant_client()
        self.embedder = embedder

    def ensure_collection(self, vector_size: int = 128, reindex: bool = False) -> None:
        """Create or reset the isolated evaluation collection."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and reindex:
            logger.info("Reindexing requested: deleting existing collection %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creating evaluation collection=%s | vector_size=%d | MAX_SIM",
                self.collection_name,
                vector_size,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                    )
                },
            )

        # Ensure payload index on doc_id and dataset_name
        for field in ("doc_id", "dataset_name"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def get_indexed_doc_ids(self) -> set[str]:
        """Fetch set of doc_ids already present in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return set()

        doc_ids: set[str] = set()
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for record in records:
                if record.payload and "doc_id" in record.payload:
                    doc_ids.add(record.payload["doc_id"])
            if next_offset is None:
                break
            offset = next_offset

        return doc_ids

    def index_corpus(
        self,
        corpus: dict[str, Image.Image],
        dataset_name: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        reindex: bool = False,
    ) -> int:
        """
        Embed and upsert corpus page images into the evaluation collection.

        Returns
        -------
        int: Number of new documents embedded and indexed.
        """
        self.ensure_collection(vector_size=128, reindex=reindex)

        existing_ids = set() if reindex else self.get_indexed_doc_ids()
        pending_items = [
            (doc_id, img) for doc_id, img in corpus.items() if doc_id not in existing_ids
        ]

        if not pending_items:
            logger.info(
                "All %d corpus pages are already indexed in %s.",
                len(corpus),
                self.collection_name,
            )
            return 0

        logger.info(
            "Indexing %d pages (%d already cached) in %s with batch_size=%d",
            len(pending_items),
            len(existing_ids),
            self.collection_name,
            batch_size,
        )

        if self.embedder is None:
            device = resolve_device()
            self.embedder = ColPaliEmbedder(model_id=MODEL_ID, device=device)

        # Batch embed and upsert
        for i in tqdm(range(0, len(pending_items), batch_size), desc="Embedding ViDoRe Pages"):
            batch = pending_items[i : i + batch_size]
            batch_doc_ids = [item[0] for item in batch]
            batch_images = [item[1] for item in batch]

            # Embed using shared ColPali 4-bit NF4
            embeddings = self.embedder.embed_images(batch_images)  # [B, N_patches, 128]

            points: list[PointStruct] = []
            for doc_id, emb in zip(batch_doc_ids, embeddings):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_name}:{doc_id}"))
                multivector = emb.tolist()
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={self.vector_name: multivector},
                        payload={
                            "doc_id": doc_id,
                            "dataset_name": dataset_name,
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "Successfully indexed %d new pages into %s",
            len(pending_items),
            self.collection_name,
        )
        return len(pending_items)

    def release_embedder(self) -> None:
        """Release the ColPali model from GPU memory to make room for queries/LLMs."""
        if self.embedder is not None:
            del self.embedder
            self.embedder = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Released ViDoRe indexer ColPali model from GPU memory.")

