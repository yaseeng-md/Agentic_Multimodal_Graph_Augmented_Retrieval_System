"""Knowledge graph ingestion pipeline combining Docling, extraction, and Neo4j."""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from graph.config import DOCLING_ENABLED, NEO4J_ENABLED
from graph.docling_processor import DoclingProcessor
from graph.extractor import GraphExtractor
from graph.neo4j_store import Neo4jStore
from graph.schema import (
    PageReference,
    GraphIngestionReport,
    make_page_id,
)

logger = logging.getLogger("amg-graph-pipeline")


class GraphIngestionPipeline:
    """Orchestrates Path B: Docling parsing, schema-validated extraction, and Neo4j upsert."""

    def __init__(
        self,
        *,
        docling_processor: Optional[DoclingProcessor] = None,
        extractor: Optional[GraphExtractor] = None,
        neo4j_store: Optional[Neo4jStore] = None,
    ):
        self.docling_processor = docling_processor or DoclingProcessor()
        self.extractor = extractor or GraphExtractor()
        self.neo4j_store = neo4j_store or Neo4jStore()

    def ingest_document_graph(
        self,
        pdf_path: Path,
        *,
        document_id: str,
        version_id: str,
        version_number: int,
        file_hash: str,
        original_filename: str,
        stored_pdf_path: str,
        rendered_pages: Sequence[dict[str, Any]],
        is_current: bool = True,
    ) -> GraphIngestionReport:
        """Execute the full Path B graph ingestion pipeline with provenance linking."""
        t_start = time.perf_counter()

        if not DOCLING_ENABLED or not NEO4J_ENABLED:
            logger.info("Graph ingestion disabled (DOCLING=%s, NEO4J=%s)", DOCLING_ENABLED, NEO4J_ENABLED)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="skipped" if not DOCLING_ENABLED else "completed",
                extraction_status="skipped",
                neo4j_status="skipped" if not NEO4J_ENABLED else "completed",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 1. Connect to Neo4j and ensure schema
        try:
            self.neo4j_store.ensure_schema()
            self.neo4j_store.upsert_document_and_version(
                document_id=document_id,
                display_name=original_filename,
                version_id=version_id,
                version_number=version_number,
                file_hash=file_hash,
                stored_pdf_path=stored_pdf_path,
                is_current=is_current,
            )
        except Exception as exc:
            logger.exception("Failed to connect or initialize Neo4j schema: %s", exc)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="skipped",
                extraction_status="skipped",
                neo4j_status="failed",
                error=f"Neo4j connection error: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 2. Build and upsert Page nodes with registry-compatible IDs and image paths
        page_refs: list[PageReference] = []
        for p in rendered_pages:
            page_refs.append(
                PageReference(
                    document_id=document_id,
                    version_id=version_id,
                    page_number=p["page_number"],
                    point_id=p.get("point_id"),
                    page_hash=p.get("page_hash"),
                    image_path=str(p.get("image_path", "")),
                )
            )

        try:
            pages_upserted = self.neo4j_store.upsert_pages(page_refs, is_current=is_current)
        except Exception as exc:
            logger.exception("Failed to upsert pages into Neo4j: %s", exc)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="skipped",
                extraction_status="skipped",
                neo4j_status="failed",
                error=f"Neo4j page upsert error: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 3. Docling parsing
        docling_res = self.docling_processor.process_pdf(
            pdf_path,
            document_id=document_id,
            version_id=version_id,
            file_hash=file_hash,
        )

        if docling_res.status != "completed" or not docling_res.chunks:
            logger.warning("Docling processing status: %s (error: %s)", docling_res.status, docling_res.error)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status=docling_res.status,
                extraction_status="skipped",
                neo4j_status="completed",
                pages_created=pages_upserted,
                error=docling_res.error,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 4. Upsert structured chunks into Neo4j
        try:
            chunks_upserted = self.neo4j_store.upsert_chunks(docling_res.chunks)
        except Exception as exc:
            logger.exception("Failed to upsert chunks into Neo4j: %s", exc)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="completed",
                extraction_status="skipped",
                neo4j_status="failed",
                pages_created=pages_upserted,
                error=f"Neo4j chunk upsert error: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 5. Entity & Claim Extraction
        extraction_res = self.extractor.extract_from_chunks(
            docling_res.chunks,
            document_id=document_id,
            version_id=version_id,
        )

        if extraction_res.status not in ("completed", "partial") or not extraction_res.claims:
            logger.info("Extraction status: %s (claims: %d)", extraction_res.status, len(extraction_res.claims))
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="completed",
                extraction_status=extraction_res.status,
                neo4j_status="completed",
                pages_created=pages_upserted,
                chunks_created=chunks_upserted,
                error=extraction_res.error,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # 6. Upsert Entities & Claims into Neo4j
        try:
            ent_count, claim_count = self.neo4j_store.upsert_entities_and_claims(
                extraction_res.entities,
                extraction_res.claims,
            )
        except Exception as exc:
            logger.exception("Failed to upsert entities/claims into Neo4j: %s", exc)
            return GraphIngestionReport(
                document_id=document_id,
                version_id=version_id,
                docling_status="completed",
                extraction_status=extraction_res.status,
                neo4j_status="failed",
                pages_created=pages_upserted,
                chunks_created=chunks_upserted,
                error=f"Neo4j entity/claim upsert error: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "Graph ingestion complete for %s/%s in %.2fms: %d pages, %d chunks, %d entities, %d claims",
            document_id,
            version_id,
            total_ms,
            pages_upserted,
            chunks_upserted,
            ent_count,
            claim_count,
        )

        return GraphIngestionReport(
            document_id=document_id,
            version_id=version_id,
            docling_status="completed",
            extraction_status=extraction_res.status,
            neo4j_status="completed",
            pages_created=pages_upserted,
            chunks_created=chunks_upserted,
            entities_upserted=ent_count,
            claims_upserted=claim_count,
            duration_ms=total_ms,
        )
