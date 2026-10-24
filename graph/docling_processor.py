"""Docling structured document processing module."""

import json
import logging
from pathlib import Path
from typing import Optional, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from graph.config import (
    DOCLING_CACHE_DIR,
    DOCLING_MAX_PAGE_COUNT,
    DOCLING_MAX_CHUNK_CHARS,
    DOCLING_CHUNK_OVERLAP_CHARS,
)
from graph.schema import (
    DocChunk,
    DoclingProcessingResult,
    make_chunk_id,
)

logger = logging.getLogger("amg-docling-processor")


class DoclingProcessor:
    """Processes PDF documents into page-normalized structured chunks and tables."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DOCLING_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._converter: Optional[DocumentConverter] = None

    def _get_converter(self) -> DocumentConverter:
        """Lazy-initialize Docling DocumentConverter with CPU-optimized layout pipeline."""
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Keep document parsing lightweight and fast
            pipeline_options.do_table_structure = True

            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
            logger.info("Initializing Docling DocumentConverter...")
            self._converter = DocumentConverter(format_options=format_options)
        return self._converter

    def process_pdf(
        self,
        pdf_path: Path,
        *,
        document_id: str,
        version_id: str,
        file_hash: str,
    ) -> DoclingProcessingResult:
        """Parse a PDF document into normalized, page-mapped chunks with caching."""
        if not pdf_path.exists():
            return DoclingProcessingResult(
                document_id=document_id,
                version_id=version_id,
                status="failed",
                error=f"PDF file does not exist: {pdf_path}",
            )

        cache_file = self.cache_dir / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                chunks = [
                    DocChunk(
                        chunk_id=c["chunk_id"],
                        document_id=document_id,
                        version_id=version_id,
                        page_number=c["page_number"],
                        chunk_index=c["chunk_index"],
                        chunk_type=c.get("chunk_type", "text"),
                        text=c["text"],
                        heading_path=c.get("heading_path", []),
                        metadata=c.get("metadata", {}),
                    )
                    for c in cached_data.get("chunks", [])
                ]
                logger.info(
                    "Loaded Docling parse from cache: %s (%d chunks, %d pages)",
                    cache_file.name,
                    len(chunks),
                    cached_data.get("page_count", 0),
                )
                return DoclingProcessingResult(
                    document_id=document_id,
                    version_id=version_id,
                    page_count=cached_data.get("page_count", 0),
                    chunks=chunks,
                    tables_count=cached_data.get("tables_count", 0),
                    headings_count=cached_data.get("headings_count", 0),
                    status="completed",
                )
            except Exception as e:
                logger.warning("Failed to load Docling cache %s: %s. Re-parsing.", cache_file, e)

        try:
            converter = self._get_converter()
            conversion_res = converter.convert(str(pdf_path))
            doc = conversion_res.document
        except Exception as exc:
            logger.exception("Docling conversion failed for %s: %s", pdf_path, exc)
            return DoclingProcessingResult(
                document_id=document_id,
                version_id=version_id,
                status="failed",
                error=f"Docling conversion error: {exc}",
            )

        # Extract items and map to 1-based page numbers
        chunks: list[DocChunk] = []
        page_numbers_seen: set[int] = set()
        tables_count = 0
        headings_count = 0
        current_heading_path: list[str] = []
        chunk_idx = 0

        try:
            # Check page count
            if hasattr(doc, "pages") and doc.pages:
                total_pages = len(doc.pages)
            else:
                total_pages = 1

            for item, level in doc.iterate_items():
                item_type = type(item).__name__

                # Determine 1-based page number
                page_no = 1
                prov = getattr(item, "prov", None)
                if prov and len(prov) > 0 and getattr(prov[0], "page_no", None) is not None:
                    page_no = int(prov[0].page_no)
                page_numbers_seen.add(page_no)

                if DOCLING_MAX_PAGE_COUNT > 0 and page_no > DOCLING_MAX_PAGE_COUNT:
                    continue

                # Handle headings
                if "Header" in item_type or "Heading" in item_type or "Title" in item_type:
                    headings_count += 1
                    h_text = getattr(item, "text", "").strip()
                    if h_text:
                        # Manage heading hierarchy path
                        if level <= len(current_heading_path):
                            current_heading_path = current_heading_path[: max(0, level - 1)]
                        current_heading_path.append(h_text)

                        c_id = make_chunk_id(document_id, version_id, page_no, chunk_idx)
                        chunks.append(
                            DocChunk(
                                chunk_id=c_id,
                                document_id=document_id,
                                version_id=version_id,
                                page_number=page_no,
                                chunk_index=chunk_idx,
                                chunk_type="heading",
                                text=h_text,
                                heading_path=list(current_heading_path),
                                metadata={"level": level, "item_type": item_type},
                            )
                        )
                        chunk_idx += 1
                    continue

                # Handle tables
                if "Table" in item_type:
                    tables_count += 1
                    table_md = ""
                    if hasattr(item, "export_to_markdown"):
                        table_md = item.export_to_markdown()
                    elif hasattr(item, "text"):
                        table_md = item.text

                    if table_md.strip():
                        c_id = make_chunk_id(document_id, version_id, page_no, chunk_idx)
                        chunks.append(
                            DocChunk(
                                chunk_id=c_id,
                                document_id=document_id,
                                version_id=version_id,
                                page_number=page_no,
                                chunk_index=chunk_idx,
                                chunk_type="table",
                                text=table_md.strip(),
                                heading_path=list(current_heading_path),
                                metadata={"item_type": item_type},
                            )
                        )
                        chunk_idx += 1
                    continue

                # Handle standard text paragraphs and list items
                item_text = getattr(item, "text", "").strip()
                if not item_text:
                    continue

                c_type = "list_item" if "List" in item_type else "text"

                # Sub-chunk if text exceeds DOCLING_MAX_CHUNK_CHARS
                if len(item_text) <= DOCLING_MAX_CHUNK_CHARS:
                    c_id = make_chunk_id(document_id, version_id, page_no, chunk_idx)
                    chunks.append(
                        DocChunk(
                            chunk_id=c_id,
                            document_id=document_id,
                            version_id=version_id,
                            page_number=page_no,
                            chunk_index=chunk_idx,
                            chunk_type=c_type,
                            text=item_text,
                            heading_path=list(current_heading_path),
                            metadata={"item_type": item_type},
                        )
                    )
                    chunk_idx += 1
                else:
                    start = 0
                    while start < len(item_text):
                        end = min(start + DOCLING_MAX_CHUNK_CHARS, len(item_text))
                        sub_text = item_text[start:end].strip()
                        if sub_text:
                            c_id = make_chunk_id(document_id, version_id, page_no, chunk_idx)
                            chunks.append(
                                DocChunk(
                                    chunk_id=c_id,
                                    document_id=document_id,
                                    version_id=version_id,
                                    page_number=page_no,
                                    chunk_index=chunk_idx,
                                    chunk_type=c_type,
                                    text=sub_text,
                                    heading_path=list(current_heading_path),
                                    metadata={"item_type": item_type, "is_subchunk": True},
                                )
                            )
                            chunk_idx += 1
                        start += DOCLING_MAX_CHUNK_CHARS - DOCLING_CHUNK_OVERLAP_CHARS

            effective_pages = max(total_pages, max(page_numbers_seen) if page_numbers_seen else 1)

            # Save to cache
            cache_payload = {
                "file_hash": file_hash,
                "page_count": effective_pages,
                "tables_count": tables_count,
                "headings_count": headings_count,
                "chunks": [c.model_dump() for c in chunks],
            }
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f, indent=2)
            except Exception as e:
                logger.warning("Failed to write Docling cache %s: %s", cache_file, e)

            return DoclingProcessingResult(
                document_id=document_id,
                version_id=version_id,
                page_count=effective_pages,
                chunks=chunks,
                tables_count=tables_count,
                headings_count=headings_count,
                status="completed",
            )

        except Exception as exc:
            logger.exception("Error extracting structured content from Docling: %s", exc)
            return DoclingProcessingResult(
                document_id=document_id,
                version_id=version_id,
                status="failed",
                error=f"Error extracting items from Docling: {exc}",
            )
