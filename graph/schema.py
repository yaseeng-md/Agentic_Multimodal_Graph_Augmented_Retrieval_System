"""Pydantic schemas and deterministic ID contracts for knowledge graph integration."""

import hashlib
import re
import unicodedata
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator


def slugify(text: str) -> str:
    """Produce a deterministic, canonical slug for entity normalization."""
    if not text:
        return "unknown"
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    slug = re.sub(r"[^\w\s-]", "", normalized.lower()).strip()
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:80] or "unknown"


def make_page_id(document_id: str, version_id: str, page_number: int) -> str:
    """Generate a deterministic, globally unique page identifier."""
    return f"PAGE:{document_id}:{version_id}:{page_number:04d}"


def make_chunk_id(document_id: str, version_id: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic, globally unique chunk identifier."""
    return f"CHUNK:{document_id}:{version_id}:{page_number:04d}:{chunk_index:04d}"


def make_entity_id(entity_type: str, name: str) -> str:
    """Generate a canonical, deterministic entity identifier."""
    clean_type = entity_type.upper().strip() or "CONCEPT"
    clean_slug = slugify(name)
    return f"ENT:{clean_type}:{clean_slug}"


def make_claim_id(page_id: str, subject_id: str, predicate: str, target: str) -> str:
    """Generate a deterministic claim ID with page-level provenance."""
    canonical_pred = predicate.upper().strip().replace(" ", "_")
    canonical_target = slugify(target)
    content = f"{subject_id}|{canonical_pred}|{canonical_target}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"CLAIM:{page_id}:{digest}"


class PageReference(BaseModel):
    """Reference to an existing page in the SQLite registry and Qdrant."""
    document_id: str
    version_id: str
    page_number: int = Field(ge=1, description="1-based page number")
    point_id: Optional[str] = None
    page_hash: Optional[str] = None
    image_path: Optional[str] = None

    @property
    def page_id(self) -> str:
        return make_page_id(self.document_id, self.version_id, self.page_number)


class DocChunk(BaseModel):
    """A structured text or tabular chunk extracted by Docling."""
    chunk_id: str
    document_id: str
    version_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    chunk_type: Literal["text", "table", "heading", "caption", "list_item", "other"] = "text"
    text: str = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_id(self) -> str:
        return make_page_id(self.document_id, self.version_id, self.page_number)


VALID_ENTITY_TYPES = {
    "PAPER", "MODEL", "AUTHOR", "DATASET", "METRIC",
    "ORGANIZATION", "CONCEPT", "METHOD", "TASK", "HARDWARE", "TOOL"
}


class ExtractedEntity(BaseModel):
    """A validated entity extracted from structured document content."""
    entity_id: str
    name: str = Field(min_length=1)
    normalized_name: str
    entity_type: str
    description: Optional[str] = None

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_entity_type(cls, v: str) -> str:
        v_clean = str(v).upper().strip()
        if v_clean not in VALID_ENTITY_TYPES:
            return "CONCEPT"
        return v_clean


class ExtractedClaim(BaseModel):
    """A validated claim or relationship with strict source provenance."""
    claim_id: str
    subject_entity_id: str
    subject_name: str
    predicate: str = Field(min_length=1)
    object_entity_id: Optional[str] = None
    object_name: Optional[str] = None
    target_value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence_text: str = Field(min_length=1)
    source_page_id: str
    source_chunk_id: str
    page_number: int = Field(ge=1)
    document_id: str
    version_id: str
    extraction_method: str = "openrouter"


class DoclingProcessingResult(BaseModel):
    """Result of Docling document parsing stage."""
    document_id: str
    version_id: str
    page_count: int = 0
    chunks: list[DocChunk] = Field(default_factory=list)
    tables_count: int = 0
    headings_count: int = 0
    status: Literal["completed", "failed", "skipped"] = "completed"
    error: Optional[str] = None


class ExtractionResult(BaseModel):
    """Result of entity and claim extraction stage."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    chunks_processed: int = 0
    status: Literal["completed", "failed", "skipped", "partial"] = "completed"
    error: Optional[str] = None


class GraphIngestionReport(BaseModel):
    """Comprehensive status report for Path B ingestion."""
    document_id: str
    version_id: str
    docling_status: Literal["completed", "failed", "skipped"]
    extraction_status: Literal["completed", "failed", "skipped", "partial"]
    neo4j_status: Literal["completed", "failed", "skipped"]
    pages_created: int = 0
    chunks_created: int = 0
    entities_upserted: int = 0
    claims_upserted: int = 0
    error: Optional[str] = None
    duration_ms: float = 0.0


class GraphClaimRecord(BaseModel):
    """A claim record returned during graph retrieval."""
    claim_id: str
    subject: str
    predicate: str
    object_or_value: str
    confidence: float
    evidence_text: str
    page_number: int
    document_id: str
    version_id: str
    page_id: str
    source_chunk_id: str


class GraphRetrievalResult(BaseModel):
    """Structured graph retrieval result compatible with orchestration."""
    query: str
    matched_entities: list[str] = Field(default_factory=list)
    claims: list[GraphClaimRecord] = Field(default_factory=list)
    source_pages: list[dict[str, Any]] = Field(default_factory=list)
    traversal_depth: int = 1
    duration_ms: float = 0.0
    status: Literal["success", "fallback_empty", "service_unavailable"] = "success"
    error: Optional[str] = None
