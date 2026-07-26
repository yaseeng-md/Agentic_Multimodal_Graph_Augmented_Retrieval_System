import time
import torch
from PIL import Image, ImageDraw
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from config import AppConfig
from dataclasses import dataclass
from typing import Sequence

@dataclass(slots=True)
class UserContext:
    tenant_id: str
    user_id: str
    roles: Sequence[str]
    
@torch.inference_mode()
def embed_query(
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    processed = processor.process_queries([query]).to(model.device)
    query_embedding = model(**processed)[0]
    return (
        query_embedding.detach().cpu(),
        processed.input_ids[0].detach().cpu(),
    )


def query_pdf(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    user: UserContext,
) -> tuple[list[models.ScoredPoint], float, torch.Tensor, torch.Tensor]:

    query_embedding, query_input_ids = embed_query(
        model,
        processor,
        query,
    )

    query_filter = models.Filter(
        must=[
            # Company isolation
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=user.tenant_id),
            ),
        ],
        should=[
            # User has one of the allowed roles
            models.FieldCondition(
                key="allowed_roles",
                match=models.MatchAny(any=list(user.roles)),
            ),

            # OR document explicitly shared with this user
            models.FieldCondition(
                key="allowed_user_ids",
                match=models.MatchValue(value=user.user_id),
            ),
        ],
        min_should=1,
    )

    start = time.perf_counter()

    response = client.query_points(
        collection_name=cfg.collection_name,
        query=query_embedding.tolist(),
        using="original",
        prefetch=[
            models.Prefetch(
                query=query_embedding.tolist(),
                using="mean_pooling_rows",
                limit=cfg.prefetch_limit,
            ),
            models.Prefetch(
                query=query_embedding.tolist(),
                using="mean_pooling_columns",
                limit=cfg.prefetch_limit,
            ),
        ],
        query_filter=query_filter,
        limit=cfg.search_limit,
        with_payload=True,
        with_vectors=False,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    return (
        list(response.points),
        latency_ms,
        query_embedding,
        query_input_ids,
    )