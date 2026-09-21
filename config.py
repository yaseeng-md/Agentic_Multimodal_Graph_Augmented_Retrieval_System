from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    model_name: str = None
    collection_name: str = "colpali_documents"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    batch_size: int = 1
    max_upload_docs: int = 5
    dpi: int = 96
    recreate_collection: bool = True
    search_limit: int = 5
    prefetch_limit: int = 50
    latency_budget_ms: float = 150.0
    ingest_workers: int = 1,
    ingest_queue_size: int = 100



def override_config(VISUAL_CONFIG) -> AppConfig:
    return AppConfig(
        model_name=VISUAL_CONFIG.get("VLM_MODEL"),
        collection_name=VISUAL_CONFIG.get("COLLECTION_NAME", "colpali_documents"),
        qdrant_url=VISUAL_CONFIG.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=VISUAL_CONFIG.get("QDRANT_API_KEY"),
        batch_size=int(VISUAL_CONFIG.get("BATCH_SIZE", 1)),
        max_upload_docs=int(VISUAL_CONFIG.get("MAX_UPLOAD_DOCS", 5)),
        dpi=int(VISUAL_CONFIG.get("PDF_RENDER_DPI", 96)),
        recreate_collection=bool(VISUAL_CONFIG.get("RECREATE_COLLECTION", True)),
        search_limit=int(VISUAL_CONFIG.get("SEARCH_LIMIT", 5)),
        prefetch_limit=int(VISUAL_CONFIG.get("PREFETCH_LIMIT", 50)),
        latency_budget_ms=float(VISUAL_CONFIG.get("LATENCY_BUDGET_MS", 150.0)),
        ingest_workers = int(VISUAL_CONFIG.get("INGEST_WORKERS", 2)),
        ingest_queue_size = int(VISUAL_CONFIG.get("INGEST_QUEUE_SIZE", 100))
    )
