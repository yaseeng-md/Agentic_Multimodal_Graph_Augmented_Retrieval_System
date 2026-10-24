"""Knowledge graph retriever module for entity-based traversal and source page resolution."""

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from graph.config import (
    GRAPH_RETRIEVAL_ENABLED,
    GRAPH_SEARCH_MAX_HOPS,
    GRAPH_SEARCH_PAGE_LIMIT,
    GRAPH_SEARCH_CLAIM_LIMIT,
    GRAPH_SEARCH_MIN_CONFIDENCE,
)
from graph.neo4j_store import Neo4jStore
from graph.schema import (
    GraphClaimRecord,
    GraphRetrievalResult,
)

logger = logging.getLogger("amg-graph-retriever")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being",
    "what", "where", "when", "which", "who", "whom", "how", "why",
    "can", "could", "should", "would", "do", "does", "did", "from",
    "that", "this", "these", "those", "have", "has", "had", "show",
    "tell", "me", "about", "give", "list", "explain", "detail"
}


def extract_query_terms(query: str) -> list[str]:
    """Extract cleaned candidate entity terms and phrases from a natural language query."""
    clean = re.sub(r"[^\w\s-]", " ", query.lower()).strip()
    tokens = [t.strip() for t in clean.split() if t.strip()]

    meaningful_tokens = [t for t in tokens if len(t) >= 3 and t not in STOP_WORDS]
    phrases = []

    # Include multi-word n-grams (2-grams, 3-grams) for compound entities
    for i in range(len(tokens) - 1):
        if tokens[i] not in STOP_WORDS and tokens[i + 1] not in STOP_WORDS:
            phrases.append(f"{tokens[i]} {tokens[i + 1]}")
            if i + 2 < len(tokens) and tokens[i + 2] not in STOP_WORDS:
                phrases.append(f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}")

    terms = list(dict.fromkeys(phrases + meaningful_tokens))
    return terms[:12]


class GraphRetriever:
    """Retrieves factual claims and resolves them to source pages via Neo4j."""

    def __init__(self, neo4j_store: Optional[Neo4jStore] = None):
        self.store = neo4j_store or Neo4jStore()

    def retrieve(
        self,
        query: str,
        *,
        document_ids: Optional[Sequence[str]] = None,
        version_ids: Optional[Sequence[str]] = None,
        recursive_search: bool = False,
        max_hops: int = GRAPH_SEARCH_MAX_HOPS,
        claim_limit: int = GRAPH_SEARCH_CLAIM_LIMIT,
        page_limit: int = GRAPH_SEARCH_PAGE_LIMIT,
    ) -> GraphRetrievalResult:
        """Search graph for query-relevant entities, claims, and verified source pages."""
        t_start = time.perf_counter()

        if not GRAPH_RETRIEVAL_ENABLED:
            return GraphRetrievalResult(
                query=query,
                status="fallback_empty",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        terms = extract_query_terms(query)
        if not terms:
            return GraphRetrievalResult(
                query=query,
                status="fallback_empty",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        try:
            driver = self.store.get_driver()
        except Exception as exc:
            logger.warning("Graph retrieval unavailable: %s", exc)
            return GraphRetrievalResult(
                query=query,
                status="service_unavailable",
                error=str(exc),
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )

        # Build parameterized Cypher query with optional scope filtering
        scope_conditions = []
        cypher_params: dict[str, Any] = {
            "terms": terms,
            "min_conf": GRAPH_SEARCH_MIN_CONFIDENCE,
            "claim_limit": claim_limit,
        }

        if not recursive_search and not version_ids:
            scope_conditions.append("p.is_current = true")

        if document_ids:
            scope_conditions.append("p.document_id IN $doc_ids")
            cypher_params["doc_ids"] = list(document_ids)

        if version_ids:
            scope_conditions.append("p.version_id IN $ver_ids")
            cypher_params["ver_ids"] = list(version_ids)

        where_scope = f"WHERE {' AND '.join(scope_conditions)}" if scope_conditions else ""

        # Query matching 1-hop and 2-hop connected entities and claims
        cypher = f"""
        MATCH (e:Entity)
        WHERE any(term IN $terms WHERE toLower(e.name) CONTAINS toLower(term))
        WITH collect(DISTINCT e) AS seed_entities, [t IN $terms | toLower(t)] AS query_terms

        UNWIND seed_entities AS seed
        OPTIONAL MATCH (sub:Entity)-[:SUBJECT_OF]->(cl:Claim)
        WHERE (sub = seed OR (cl)-[:OBJECT_OF]->(seed)) AND cl.confidence >= $min_conf
        MATCH (cl)-[:ON_PAGE]->(p:Page)
        {where_scope}
        MATCH (d:Document) WHERE d.document_id = p.document_id
        OPTIONAL MATCH (v:Version) WHERE v.version_id = p.version_id
        OPTIONAL MATCH (cl)-[:OBJECT_OF]->(obj:Entity)
        OPTIONAL MATCH (cl)-[:SOURCED_FROM]->(c:Chunk)

        WITH seed.name AS matched_entity,
             cl.claim_id AS claim_id,
             sub.name AS subject,
             cl.predicate AS predicate,
             coalesce(obj.name, cl.target_value, '') AS object_or_val,
             cl.confidence AS confidence,
             cl.evidence_text AS evidence_text,
             p.page_id AS page_id,
             p.page_number AS page_number,
             p.document_id AS document_id,
             p.version_id AS version_id,
             coalesce(v.version_number, 1) AS version_number,
             p.image_path AS image_path,
             p.point_id AS point_id,
             p.page_hash AS page_hash,
             p.is_current AS is_current,
             d.display_name AS filename,
             coalesce(c.chunk_id, '') AS chunk_id,
             coalesce(c.text, '') AS chunk_text

        WHERE claim_id IS NOT NULL
        RETURN DISTINCT
            matched_entity,
            claim_id,
            subject,
            predicate,
            object_or_val,
            confidence,
            evidence_text,
            page_id,
            page_number,
            document_id,
            version_id,
            version_number,
            image_path,
            point_id,
            page_hash,
            is_current,
            filename,
            chunk_id,
            chunk_text
        ORDER BY confidence DESC
        LIMIT $claim_limit
        """

        matched_entities_set: set[str] = set()
        claims_list: list[GraphClaimRecord] = []
        pages_by_id: dict[str, dict[str, Any]] = {}

        try:
            with driver.session(database=self.store.database) as session:
                rows = session.run(cypher, **cypher_params).data()

            for r in rows:
                if r.get("matched_entity"):
                    matched_entities_set.add(r["matched_entity"])

                claim_rec = GraphClaimRecord(
                    claim_id=r["claim_id"],
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object_or_value=r["object_or_val"],
                    confidence=float(r["confidence"]),
                    evidence_text=r["evidence_text"],
                    page_number=int(r["page_number"]),
                    document_id=r["document_id"],
                    version_id=r["version_id"],
                    page_id=r["page_id"],
                    source_chunk_id=r["chunk_id"],
                )
                claims_list.append(claim_rec)

                # Group by page_id for evidence candidate resolution
                pid = r["page_id"]
                if pid not in pages_by_id:
                    # Validate local image existence
                    img_path = r.get("image_path")
                    if img_path and not Path(img_path).exists():
                        # Try standard fallback location
                        fallback = Path(f"data/pages/{r['document_id']}/{r['version_id']}/page_{r['page_number']:04d}.png")
                        if fallback.exists():
                            img_path = str(fallback)

                    pages_by_id[pid] = {
                        "rank": 0,  # Assigned after ranking/merging
                        "score": round(float(r["confidence"]), 4),
                        "point_id": r.get("point_id") or "",
                        "document_id": r["document_id"],
                        "version_id": r["version_id"],
                        "version_number": r["version_number"],
                        "filename": r.get("filename") or "document.pdf",
                        "page_number": int(r["page_number"]),
                        "page_hash": r.get("page_hash") or "",
                        "page_image_path": img_path,
                        "page_text": r.get("chunk_text") or r.get("evidence_text"),
                        "is_current": bool(r.get("is_current", True)),
                        "retrieval_sources": ["graph"],
                        "graph_claims": [claim_rec.model_dump()],
                    }
                else:
                    pages_by_id[pid]["graph_claims"].append(claim_rec.model_dump())

            resolved_pages = list(pages_by_id.values())[:page_limit]
            for idx, p in enumerate(resolved_pages, start=1):
                p["rank"] = idx

            duration = round((time.perf_counter() - t_start) * 1000, 2)
            logger.info(
                "Graph retrieval for '%s': %d claims, %d source pages in %.2fms",
                query,
                len(claims_list),
                len(resolved_pages),
                duration,
            )

            return GraphRetrievalResult(
                query=query,
                matched_entities=sorted(matched_entities_set),
                claims=claims_list,
                source_pages=resolved_pages,
                traversal_depth=max_hops,
                duration_ms=duration,
                status="success" if claims_list else "fallback_empty",
            )

        except Exception as exc:
            logger.exception("Error executing graph retrieval for query '%s': %s", query, exc)
            return GraphRetrievalResult(
                query=query,
                status="service_unavailable",
                error=str(exc),
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
