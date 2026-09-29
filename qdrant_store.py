from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import os
import logging
from dotenv import load_dotenv
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("multimodal-rag-ingest")


class QdrantStore:
    VECTOR_NAME = "page_embedding"

    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def create_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            return

        logger.info(
            "Creating Qdrant collection=%s | vector_size=%d | MAX_SIM",
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
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            },
        )

    def create_payload_indexes(self) -> None:
        # Exact identifiers used for filtering/reuse. Qdrant recommends keyword
        # payload indexes for frequently filtered exact strings.
        for field in (
            "document_id",
            "version_id",
            "file_hash",
            "page_hash",
            "embedding_signature",
            "is_current",   
        ):
            try:
                schema = (
                    models.PayloadSchemaType.BOOL
                    if field == "is_current"
                    else models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                # Index already exists or server/client version differs slightly.
                logger.debug("Payload index %s not changed: %s", field, exc)

    def ensure_collection(self, vector_size: int) -> None:
        if not self.collection_exists():
            self.create_collection(vector_size)
        self.create_payload_indexes()

    def upsert(self, points: list[PointStruct]) -> None:
        if not points:
            return

        first_vector = points[0].vector
        if isinstance(first_vector, dict):
            first_vector = first_vector[self.VECTOR_NAME]

        vector_size = len(first_vector[0])
        self.ensure_collection(vector_size)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def find_cached_vector(
        self,
        *,
        page_hash: str,
        signature: str,
    ) -> list[list[float]] | None:
        """Find an existing compatible page embedding by content hash."""
        if not self.collection_exists():
            return None

        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_hash",
                        match=models.MatchValue(value=page_hash),
                    ),
                    models.FieldCondition(
                        key="embedding_signature",
                        match=models.MatchValue(value=signature),
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )

        if not scroll_result:
            return None

        vector = scroll_result[0].vector
        if isinstance(vector, dict):
            vector = vector.get(self.VECTOR_NAME)
        if vector is None:
            return None

        return vector
    def mark_version_current(
        self,
        *,
        document_id: str,
        version_id: str,
    ) -> None:
        # Mark the new version as current.
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_current": True},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchValue(value=version_id),
                    ),
                ]
            ),
            wait=True,
        )

    def demote_previous_versions(
        self,
        document_id: str,
        current_version_id: str,
    ) -> None:
        # Mark all older versions of this document as historical.
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_current": False},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="is_current",
                        match=models.MatchValue(value=True),
                    ),
                ],
                must_not=[
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchValue(value=current_version_id),
                    )
                ],
            ),
            wait=True,
        )

