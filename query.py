import time
import torch
from PIL import Image, ImageDraw
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient, models
from config import AppConfig



@torch.inference_mode()
def embed_query(model: ColPali, processor: ColPaliProcessor, query: str) -> tuple[torch.Tensor, torch.Tensor]:
    processed = processor.process_queries([query]).to(model.device)
    query_embedding = model(**processed)[0]
    return query_embedding.detach().cpu(), processed.input_ids[0].detach().cpu()


def query_pdf(
    client: QdrantClient,
    cfg: AppConfig,
    model: ColPali,
    processor: ColPaliProcessor,
    query: str,
    entity_id: str | None = None,
) -> tuple[list[models.ScoredPoint], float, torch.Tensor, torch.Tensor]:
    query_embedding, query_input_ids = embed_query(model, processor, query)

    start = time.perf_counter()
    query_filter = None
    if entity_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="entity_id",
                    match=models.MatchValue(value=entity_id),
                )
            ]
        )

    response = client.query_points(
        collection_name=cfg.collection_name,
        query=query_embedding.tolist(),
        using="original",
        prefetch=[
            models.Prefetch(
                query=query_embedding.tolist(),
                limit=cfg.prefetch_limit,
                using="mean_pooling_rows",
            ),
            models.Prefetch(
                query=query_embedding.tolist(),
                limit=cfg.prefetch_limit,
                using="mean_pooling_columns",
            ),
        ],
        query_filter=query_filter,
        limit=cfg.search_limit,
        with_payload=True,
        with_vectors=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return list(response.points), latency_ms, query_embedding, query_input_ids

