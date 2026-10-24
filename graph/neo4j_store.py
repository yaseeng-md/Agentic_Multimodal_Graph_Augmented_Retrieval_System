"""Neo4j knowledge graph storage and idempotent Cypher persistence module."""

import logging
from typing import Any, Optional, Sequence
from neo4j import GraphDatabase, Driver, Session

from graph.config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    NEO4J_CONNECT_TIMEOUT_SEC,
    NEO4J_MAX_CONNECTION_LIFETIME_SEC,
)
from graph.schema import (
    PageReference,
    DocChunk,
    ExtractedEntity,
    ExtractedClaim,
)

logger = logging.getLogger("amg-neo4j-store")


class Neo4jStore:
    """Manages knowledge-graph schema constraints and idempotent Cypher operations."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.database = database or NEO4J_DATABASE
        self._driver: Optional[Driver] = None

    def get_driver(self) -> Driver:
        """Get or initialize the Neo4j driver with configured connection pools."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                connection_timeout=NEO4J_CONNECT_TIMEOUT_SEC,
                max_connection_lifetime=NEO4J_MAX_CONNECTION_LIFETIME_SEC,
            )
        return self._driver

    def close(self) -> None:
        """Close driver connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def health_check(self) -> dict[str, Any]:
        """Verify Neo4j connectivity and database availability."""
        try:
            driver = self.get_driver()
            with driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS alive").single()
                alive = result and result["alive"] == 1
                return {
                    "status": "ready" if alive else "unhealthy",
                    "uri": self.uri,
                    "database": self.database,
                }
        except Exception as exc:
            return {
                "status": "unavailable",
                "uri": self.uri,
                "error": str(exc),
            }

    def ensure_schema(self) -> None:
        """Initialize uniqueness constraints and indexes idempotently."""
        constraints = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (v:Version) REQUIRE v.version_id IS UNIQUE",
            "CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.page_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.claim_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX claim_predicate_idx IF NOT EXISTS FOR (cl:Claim) ON (cl.predicate)",
            "CREATE INDEX page_doc_ver_idx IF NOT EXISTS FOR (p:Page) ON (p.document_id, p.version_id)",
        ]

        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            for c_query in constraints:
                try:
                    session.run(c_query)
                except Exception as exc:
                    logger.debug("Constraint creation notice (%s): %s", c_query, exc)
            for i_query in indexes:
                try:
                    session.run(i_query)
                except Exception as exc:
                    logger.debug("Index creation notice (%s): %s", i_query, exc)

        logger.info("Neo4j constraints and indexes verified.")

    def upsert_document_and_version(
        self,
        *,
        document_id: str,
        display_name: str,
        version_id: str,
        version_number: int,
        file_hash: str,
        stored_pdf_path: str,
        is_current: bool = True,
    ) -> None:
        """Idempotently upsert Document and Version nodes and their relationship."""
        query = """
        MERGE (d:Document {document_id: $doc_id})
        ON CREATE SET d.display_name = $display_name, d.created_at = datetime()
        ON MATCH SET d.display_name = $display_name, d.updated_at = datetime()

        MERGE (v:Version {version_id: $ver_id})
        ON CREATE SET 
            v.document_id = $doc_id,
            v.version_number = $ver_num,
            v.file_hash = $file_hash,
            v.stored_pdf_path = $pdf_path,
            v.is_current = $is_current,
            v.created_at = datetime()
        ON MATCH SET
            v.stored_pdf_path = $pdf_path,
            v.is_current = $is_current,
            v.updated_at = datetime()

        MERGE (d)-[:HAS_VERSION]->(v)
        """
        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            session.run(
                query,
                doc_id=document_id,
                display_name=display_name,
                ver_id=version_id,
                ver_num=version_number,
                file_hash=file_hash,
                pdf_path=stored_pdf_path,
                is_current=is_current,
            )

    def upsert_pages(self, pages: Sequence[PageReference], *, is_current: bool = True) -> int:
        """Idempotently upsert Page nodes linked to their respective Version."""
        if not pages:
            return 0

        query = """
        UNWIND $pages AS p
        MATCH (v:Version {version_id: p.version_id})
        MERGE (page:Page {page_id: p.page_id})
        ON CREATE SET
            page.document_id = p.document_id,
            page.version_id = p.version_id,
            page.page_number = p.page_number,
            page.point_id = p.point_id,
            page.page_hash = p.page_hash,
            page.image_path = p.image_path,
            page.is_current = $is_current,
            page.created_at = datetime()
        ON MATCH SET
            page.point_id = p.point_id,
            page.page_hash = p.page_hash,
            page.image_path = p.image_path,
            page.is_current = $is_current,
            page.updated_at = datetime()

        MERGE (v)-[:HAS_PAGE]->(page)
        """
        records = [
            {
                "page_id": p.page_id,
                "document_id": p.document_id,
                "version_id": p.version_id,
                "page_number": p.page_number,
                "point_id": p.point_id or "",
                "page_hash": p.page_hash or "",
                "image_path": p.image_path or "",
            }
            for p in pages
        ]

        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            session.run(query, pages=records, is_current=is_current)

        return len(records)

    def upsert_chunks(self, chunks: Sequence[DocChunk]) -> int:
        """Idempotently upsert structured Chunk nodes linked to their parent Page."""
        if not chunks:
            return 0

        query = """
        UNWIND $chunks AS c
        MATCH (p:Page {page_id: c.page_id})
        MERGE (chunk:Chunk {chunk_id: c.chunk_id})
        ON CREATE SET
            chunk.document_id = c.document_id,
            chunk.version_id = c.version_id,
            chunk.page_number = c.page_number,
            chunk.chunk_index = c.chunk_index,
            chunk.chunk_type = c.chunk_type,
            chunk.text = c.text,
            chunk.heading_path = c.heading_path,
            chunk.created_at = datetime()
        ON MATCH SET
            chunk.text = c.text,
            chunk.heading_path = c.heading_path,
            chunk.chunk_type = c.chunk_type,
            chunk.updated_at = datetime()

        MERGE (p)-[:HAS_CHUNK]->(chunk)
        """
        records = [
            {
                "chunk_id": c.chunk_id,
                "page_id": c.page_id,
                "document_id": c.document_id,
                "version_id": c.version_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chunk_type": c.chunk_type,
                "text": c.text,
                "heading_path": c.heading_path,
            }
            for c in chunks
        ]

        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            session.run(query, chunks=records)

        return len(records)

    def upsert_entities_and_claims(
        self,
        entities: Sequence[ExtractedEntity],
        claims: Sequence[ExtractedClaim],
    ) -> tuple[int, int]:
        """Idempotently upsert Entities, Claims, and their provenance edges."""
        driver = self.get_driver()

        # 1. Upsert entities
        ent_count = 0
        if entities:
            ent_query = """
            UNWIND $entities AS e
            MERGE (ent:Entity {entity_id: e.entity_id})
            ON CREATE SET
                ent.name = e.name,
                ent.normalized_name = e.normalized_name,
                ent.entity_type = e.entity_type,
                ent.description = e.description,
                ent.created_at = datetime()
            ON MATCH SET
                ent.name = e.name,
                ent.description = coalesce(e.description, ent.description),
                ent.updated_at = datetime()
            """
            ent_records = [
                {
                    "entity_id": e.entity_id,
                    "name": e.name,
                    "normalized_name": e.normalized_name,
                    "entity_type": e.entity_type,
                    "description": e.description or "",
                }
                for e in entities
            ]
            with driver.session(database=self.database) as session:
                session.run(ent_query, entities=ent_records)
            ent_count = len(ent_records)

        # 2. Upsert claims and link provenance
        claim_count = 0
        if claims:
            claim_query = """
            UNWIND $claims AS cl
            MATCH (sub:Entity {entity_id: cl.subject_entity_id})
            MATCH (chunk:Chunk {chunk_id: cl.source_chunk_id})
            MATCH (page:Page {page_id: cl.source_page_id})

            MERGE (claim:Claim {claim_id: cl.claim_id})
            ON CREATE SET
                claim.predicate = cl.predicate,
                claim.target_value = cl.target_value,
                claim.confidence = cl.confidence,
                claim.evidence_text = cl.evidence_text,
                claim.source_page_id = cl.source_page_id,
                claim.source_chunk_id = cl.source_chunk_id,
                claim.page_number = cl.page_number,
                claim.document_id = cl.document_id,
                claim.version_id = cl.version_id,
                claim.extraction_method = cl.extraction_method,
                claim.created_at = datetime()
            ON MATCH SET
                claim.confidence = cl.confidence,
                claim.evidence_text = cl.evidence_text,
                claim.updated_at = datetime()

            MERGE (sub)-[:SUBJECT_OF]->(claim)
            MERGE (claim)-[:SOURCED_FROM]->(chunk)
            MERGE (claim)-[:ON_PAGE]->(page)

            FOREACH (_ IN CASE WHEN cl.object_entity_id IS NOT NULL AND cl.object_entity_id <> '' THEN [1] ELSE [] END |
                MERGE (obj:Entity {entity_id: cl.object_entity_id})
                MERGE (claim)-[:OBJECT_OF]->(obj)
            )
            """
            claim_records = [
                {
                    "claim_id": cl.claim_id,
                    "subject_entity_id": cl.subject_entity_id,
                    "object_entity_id": cl.object_entity_id or "",
                    "predicate": cl.predicate,
                    "target_value": cl.target_value or "",
                    "confidence": cl.confidence,
                    "evidence_text": cl.evidence_text,
                    "source_page_id": cl.source_page_id,
                    "source_chunk_id": cl.source_chunk_id,
                    "page_number": cl.page_number,
                    "document_id": cl.document_id,
                    "version_id": cl.version_id,
                    "extraction_method": cl.extraction_method,
                }
                for cl in claims
            ]
            with driver.session(database=self.database) as session:
                session.run(claim_query, claims=claim_records)
            claim_count = len(claim_records)

        return ent_count, claim_count

    def demote_previous_versions(self, document_id: str, current_version_id: str) -> None:
        """Mark older versions and pages of a document as is_current = false."""
        query = """
        MATCH (d:Document {document_id: $doc_id})-[:HAS_VERSION]->(v:Version)
        WHERE v.version_id <> $current_ver_id
        SET v.is_current = false
        WITH v
        MATCH (v)-[:HAS_PAGE]->(p:Page)
        SET p.is_current = false
        """
        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            session.run(query, doc_id=document_id, current_ver_id=current_version_id)

    def prune_version_graph(self, version_id: str) -> dict[str, int]:
        """Safely remove a document version and its owned sub-graph (claims, chunks, pages)."""
        query = """
        MATCH (v:Version {version_id: $ver_id})
        OPTIONAL MATCH (v)-[:HAS_PAGE]->(p:Page)
        OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (p)<-[:ON_PAGE]-(cl:Claim)
        DETACH DELETE cl, c, p, v
        """
        driver = self.get_driver()
        with driver.session(database=self.database) as session:
            summary = session.run(query, ver_id=version_id).consume()
            return {
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_deleted": summary.counters.relationships_deleted,
            }
