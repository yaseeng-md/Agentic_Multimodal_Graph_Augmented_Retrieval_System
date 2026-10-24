"""Comprehensive unit and integration test suite for Docling and Neo4j graph ingestion."""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.schema import (
    PageReference,
    DocChunk,
    ExtractedEntity,
    ExtractedClaim,
    make_page_id,
    make_chunk_id,
    make_entity_id,
    make_claim_id,
    slugify,
)
from graph.docling_processor import DoclingProcessor
from graph.extractor import GraphExtractor
from graph.neo4j_store import Neo4jStore
from graph.pipeline import GraphIngestionPipeline
from ingest import create_dummy_pdf, sha256_file


class TestDeterministicIDs(unittest.TestCase):
    """Test deterministic and collision-resistant identifier generation."""

    def test_slugify(self):
        self.assertEqual(slugify("Attention Is All You Need!"), "attention_is_all_you_need")
        self.assertEqual(slugify("Qwen3-VL (2B-Instruct)"), "qwen3_vl_2b_instruct")
        self.assertEqual(slugify("   "), "unknown")

    def test_page_id(self):
        pid1 = make_page_id("DOC-123", "VER-ABC", 1)
        pid2 = make_page_id("DOC-123", "VER-ABC", 1)
        pid3 = make_page_id("DOC-123", "VER-ABC", 2)
        self.assertEqual(pid1, "PAGE:DOC-123:VER-ABC:0001")
        self.assertEqual(pid1, pid2)
        self.assertNotEqual(pid1, pid3)

    def test_chunk_id(self):
        cid1 = make_chunk_id("DOC-123", "VER-ABC", 1, 0)
        cid2 = make_chunk_id("DOC-123", "VER-ABC", 1, 0)
        self.assertEqual(cid1, "CHUNK:DOC-123:VER-ABC:0001:0000")
        self.assertEqual(cid1, cid2)

    def test_entity_id(self):
        eid1 = make_entity_id("MODEL", "Transformer")
        eid2 = make_entity_id("model", "transformer")
        eid3 = make_entity_id("DATASET", "Transformer")
        self.assertEqual(eid1, "ENT:MODEL:transformer")
        self.assertEqual(eid1, eid2)
        self.assertNotEqual(eid1, eid3)

    def test_claim_id(self):
        page_id = "PAGE:DOC-1:VER-1:0001"
        cid1 = make_claim_id(page_id, "ENT:MODEL:transformer", "INTRODUCES", "Self-Attention")
        cid2 = make_claim_id(page_id, "ENT:MODEL:transformer", "introduces", "Self-Attention")
        cid3 = make_claim_id(page_id, "ENT:MODEL:transformer", "EVALUATED_ON", "WMT 2014")
        self.assertEqual(cid1, cid2)
        self.assertNotEqual(cid1, cid3)
        self.assertTrue(cid1.startswith(f"CLAIM:{page_id}:"))


class TestDoclingProcessing(unittest.TestCase):
    """Test Docling PDF parsing and page-number mapping."""

    @classmethod
    def setUpClass(cls):
        cls.dummy_pdf = Path("./data/dummy/test_docling_sample.pdf")
        create_dummy_pdf(
            cls.dummy_pdf,
            "Docling Integration Test",
            "Docling parses PDF documents into structured blocks.",
        )
        cls.file_hash = sha256_file(cls.dummy_pdf)

    def test_docling_parsing_and_page_mapping(self):
        processor = DoclingProcessor()
        res = processor.process_pdf(
            self.dummy_pdf,
            document_id="DOC-TEST-001",
            version_id="VER-TEST-001",
            file_hash=self.file_hash,
        )
        self.assertEqual(res.status, "completed")
        self.assertGreater(len(res.chunks), 0)
        self.assertEqual(res.page_count, 2)

        # Verify all chunk page numbers are 1-based integers
        for c in res.chunks:
            self.assertIn(c.page_number, (1, 2))
            self.assertTrue(c.chunk_id.startswith(f"CHUNK:DOC-TEST-001:VER-TEST-001:{c.page_number:04d}:"))


class TestNeo4jIdempotencyAndProvenance(unittest.TestCase):
    """Integration test with live local Neo4j instance verifying idempotency and provenance."""

    @classmethod
    def setUpClass(cls):
        cls.store = Neo4jStore()
        health = cls.store.health_check()
        if health.get("status") != "ready":
            raise unittest.SkipTest(f"Neo4j is not available at {cls.store.uri}: {health.get('error')}")
        cls.store.ensure_schema()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "store"):
            cls.store.close()

    def test_idempotent_upsert_and_provenance(self):
        doc_id = "DOC-IDEMP-001"
        ver_id = "VER-IDEMP-001"
        file_hash = "hash_abc_123"

        # 1. First upsert of document and version
        self.store.upsert_document_and_version(
            document_id=doc_id,
            display_name="idempotency_test.pdf",
            version_id=ver_id,
            version_number=1,
            file_hash=file_hash,
            stored_pdf_path="data/pdfs/test.pdf",
            is_current=True,
        )

        # 2. Pages
        pages = [
            PageReference(
                document_id=doc_id,
                version_id=ver_id,
                page_number=1,
                point_id="point-uuid-1",
                page_hash="pagehash1",
                image_path="data/pages/doc/ver/page_0001.png",
            ),
            PageReference(
                document_id=doc_id,
                version_id=ver_id,
                page_number=2,
                point_id="point-uuid-2",
                page_hash="pagehash2",
                image_path="data/pages/doc/ver/page_0002.png",
            ),
        ]
        self.store.upsert_pages(pages, is_current=True)

        # 3. Chunks
        chunks = [
            DocChunk(
                chunk_id=make_chunk_id(doc_id, ver_id, 1, 0),
                document_id=doc_id,
                version_id=ver_id,
                page_number=1,
                chunk_index=0,
                chunk_type="text",
                text="The Transformer achieves 28.4 BLEU on English-to-German.",
            )
        ]
        self.store.upsert_chunks(chunks)

        # 4. Entities & Claims
        ent_m = ExtractedEntity(
            entity_id=make_entity_id("MODEL", "Transformer"),
            name="Transformer",
            normalized_name="Transformer",
            entity_type="MODEL",
        )
        ent_d = ExtractedEntity(
            entity_id=make_entity_id("DATASET", "WMT 2014 EN-DE"),
            name="WMT 2014 EN-DE",
            normalized_name="Wmt 2014 En-De",
            entity_type="DATASET",
        )
        claim = ExtractedClaim(
            claim_id=make_claim_id(
                make_page_id(doc_id, ver_id, 1),
                ent_m.entity_id,
                "ACHIEVES_ON",
                ent_d.entity_id,
            ),
            subject_entity_id=ent_m.entity_id,
            subject_name=ent_m.name,
            predicate="ACHIEVES_ON",
            object_entity_id=ent_d.entity_id,
            object_name=ent_d.name,
            target_value="28.4 BLEU",
            confidence=0.98,
            evidence_text="The Transformer achieves 28.4 BLEU on English-to-German.",
            source_page_id=make_page_id(doc_id, ver_id, 1),
            source_chunk_id=chunks[0].chunk_id,
            page_number=1,
            document_id=doc_id,
            version_id=ver_id,
        )

        # First write
        self.store.upsert_entities_and_claims([ent_m, ent_d], [claim])

        # Query counts
        driver = self.store.get_driver()
        with driver.session(database=self.store.database) as session:
            c1 = session.run("MATCH (cl:Claim {claim_id: $cid}) RETURN count(cl) AS cnt", cid=claim.claim_id).single()["cnt"]
            e1 = session.run("MATCH (e:Entity {entity_id: $eid}) RETURN count(e) AS cnt", eid=ent_m.entity_id).single()["cnt"]
            self.assertEqual(c1, 1)
            self.assertEqual(e1, 1)

        # Repeat identical write (IDEMPOTENCY CHECK)
        self.store.upsert_document_and_version(
            document_id=doc_id,
            display_name="idempotency_test.pdf",
            version_id=ver_id,
            version_number=1,
            file_hash=file_hash,
            stored_pdf_path="data/pdfs/test.pdf",
            is_current=True,
        )
        self.store.upsert_pages(pages, is_current=True)
        self.store.upsert_chunks(chunks)
        self.store.upsert_entities_and_claims([ent_m, ent_d], [claim])

        with driver.session(database=self.store.database) as session:
            c2 = session.run("MATCH (cl:Claim {claim_id: $cid}) RETURN count(cl) AS cnt", cid=claim.claim_id).single()["cnt"]
            e2 = session.run("MATCH (e:Entity {entity_id: $eid}) RETURN count(e) AS cnt", eid=ent_m.entity_id).single()["cnt"]
            self.assertEqual(c2, 1, "Duplicate Claim node created on re-ingestion!")
            self.assertEqual(e2, 1, "Duplicate Entity node created on re-ingestion!")

            # Provenance navigation test: Claim -> Page -> image_path
            res = session.run(
                """
                MATCH (cl:Claim {claim_id: $cid})-[:ON_PAGE]->(p:Page)
                MATCH (cl)-[:SOURCED_FROM]->(c:Chunk)
                MATCH (sub:Entity)-[:SUBJECT_OF]->(cl)-[:OBJECT_OF]->(obj:Entity)
                RETURN p.page_number AS page_no, p.image_path AS img, c.text AS chunk_text,
                       sub.name AS sub_name, obj.name AS obj_name
                """,
                cid=claim.claim_id,
            ).single()

            self.assertIsNotNone(res)
            self.assertEqual(res["page_no"], 1)
            self.assertEqual(res["img"], "data/pages/doc/ver/page_0001.png")
            self.assertEqual(res["sub_name"], "Transformer")
            self.assertEqual(res["obj_name"], "WMT 2014 EN-DE")

        # Cleanup test version
        self.store.prune_version_graph(ver_id)

    def test_full_pipeline_with_extraction_and_reingest(self):
        """End-to-end test of Path B pipeline with real Docling parsing, extraction, and Neo4j upsert."""
        pdf_path = Path("./data/dummy/test_pipeline_sample.pdf")
        create_dummy_pdf(
            pdf_path,
            "Attention Is All You Need",
            "Vaswani et al. introduced the Transformer model. The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German.",
        )
        file_hash = sha256_file(pdf_path)
        doc_id = "DOC-E2E-TRANSFORMER"
        ver_id = "VER-E2E-TRANSFORMER"

        pipeline = GraphIngestionPipeline(neo4j_store=self.store)
        rendered_pages = [
            {
                "page_number": 1,
                "point_id": "test-point-1",
                "page_hash": "testhash1",
                "image_path": "data/pages/doc/ver/page_0001.png",
            },
            {
                "page_number": 2,
                "point_id": "test-point-2",
                "page_hash": "testhash2",
                "image_path": "data/pages/doc/ver/page_0002.png",
            },
        ]

        # 1. First Run
        report1 = pipeline.ingest_document_graph(
            pdf_path,
            document_id=doc_id,
            version_id=ver_id,
            version_number=1,
            file_hash=file_hash,
            original_filename="transformer_sample.pdf",
            stored_pdf_path=str(pdf_path),
            rendered_pages=rendered_pages,
            is_current=True,
        )

        self.assertEqual(report1.docling_status, "completed")
        self.assertEqual(report1.neo4j_status, "completed")
        self.assertEqual(report1.pages_created, 2)
        self.assertGreater(report1.chunks_created, 0)

        # Extraction may succeed if OpenRouter is online, or report unconfigured/failed gracefully
        if report1.extraction_status == "completed":
            self.assertGreater(report1.entities_upserted, 0)
            self.assertGreater(report1.claims_upserted, 0)

        # 2. Re-ingestion Run (Idempotency verification)
        report2 = pipeline.ingest_document_graph(
            pdf_path,
            document_id=doc_id,
            version_id=ver_id,
            version_number=1,
            file_hash=file_hash,
            original_filename="transformer_sample.pdf",
            stored_pdf_path=str(pdf_path),
            rendered_pages=rendered_pages,
            is_current=True,
        )

        driver = self.store.get_driver()
        with driver.session(database=self.store.database) as session:
            pages_count = session.run(
                "MATCH (v:Version {version_id: $ver_id})-[:HAS_PAGE]->(p:Page) RETURN count(p) AS cnt",
                ver_id=ver_id,
            ).single()["cnt"]
            self.assertEqual(pages_count, 2, "Page count duplicated on re-ingestion!")

            chunks_count = session.run(
                "MATCH (v:Version {version_id: $ver_id})-[:HAS_PAGE]->()-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS cnt",
                ver_id=ver_id,
            ).single()["cnt"]
            self.assertEqual(chunks_count, report1.chunks_created, "Chunk count duplicated on re-ingestion!")

            claims_count = session.run(
                "MATCH (v:Version {version_id: $ver_id})-[:HAS_PAGE]->()<-[:ON_PAGE]-(cl:Claim) RETURN count(cl) AS cnt",
                ver_id=ver_id,
            ).single()["cnt"]
            self.assertEqual(claims_count, report1.claims_upserted, "Claim count duplicated on re-ingestion!")

        # Clean up
        self.store.prune_version_graph(ver_id)


if __name__ == "__main__":
    unittest.main()
