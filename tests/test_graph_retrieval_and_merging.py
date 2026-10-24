"""Unit and integration tests for graph retrieval, candidate merging, and query integration."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query import _merge_retrieval_candidates, retrieve
from graph.retriever import extract_query_terms, GraphRetriever
from graph.neo4j_store import Neo4jStore
from generate import _normalise_retrieval_page, _build_messages


class TestQueryTermExtraction(unittest.TestCase):
    """Test natural language query term and phrase extraction."""

    def test_term_extraction(self):
        query = "What is the Transformer BLEU score on WMT 2014 English-to-German?"
        terms = extract_query_terms(query)
        self.assertIn("transformer", [t.lower() for t in terms])
        self.assertIn("bleu score", [t.lower() for t in terms])
        # Stop words like 'what', 'is', 'the', 'on' should not be primary single terms
        self.assertNotIn("what", terms)
        self.assertNotIn("is", terms)


class TestCandidateMergingAndDeduplication(unittest.TestCase):
    """Test candidate merging logic ensuring union, deduplication, and provenance retention."""

    def setUp(self):
        self.visual_candidates = [
            {
                "rank": 1,
                "score": 0.85,
                "document_id": "DOC-1",
                "version_id": "VER-1",
                "page_number": 1,
                "filename": "paper.pdf",
                "page_image_path": "data/pages/doc1/ver1/page_0001.png",
                "page_text": "Visual page 1 text",
            },
            {
                "rank": 2,
                "score": 0.72,
                "document_id": "DOC-1",
                "version_id": "VER-1",
                "page_number": 2,
                "filename": "paper.pdf",
                "page_image_path": "data/pages/doc1/ver1/page_0002.png",
                "page_text": "Visual page 2 text",
            },
        ]

    def test_overlapping_candidate_deduplication(self):
        """When the same page is retrieved by both paths, it must be deduplicated with dual provenance."""
        graph_candidates = [
            {
                "rank": 1,
                "score": 0.95,
                "document_id": "DOC-1",
                "version_id": "VER-1",
                "page_number": 1,
                "filename": "paper.pdf",
                "page_image_path": "data/pages/doc1/ver1/page_0001.png",
                "page_text": "Graph evidence text",
                "graph_claims": [{"subject": "Transformer", "predicate": "INTRODUCES", "object_or_value": "Attention"}],
            }
        ]

        merged = _merge_retrieval_candidates(self.visual_candidates, graph_candidates, limit=5)
        self.assertEqual(len(merged), 2, "Candidate count should remain 2 after deduplicating Page 1.")

        page1 = next(p for p in merged if p["page_number"] == 1)
        self.assertEqual(sorted(page1["retrieval_sources"]), ["graph", "visual"])
        self.assertEqual(len(page1["graph_claims"]), 1)
        self.assertEqual(page1["graph_claims"][0]["predicate"], "INTRODUCES")

    def test_disjoint_candidates_merged_not_substituted(self):
        """Pages discovered exclusively by the graph must be added, not substituted for visual candidates."""
        graph_candidates = [
            {
                "rank": 1,
                "score": 0.90,
                "document_id": "DOC-1",
                "version_id": "VER-1",
                "page_number": 3,  # Distinct page not in visual candidates
                "filename": "paper.pdf",
                "page_image_path": "data/pages/doc1/ver1/page_0003.png",
                "page_text": "Graph page 3 table data",
                "graph_claims": [{"subject": "Transformer", "predicate": "ACHIEVES", "object_or_value": "28.4 BLEU"}],
            }
        ]

        merged = _merge_retrieval_candidates(self.visual_candidates, graph_candidates, limit=5)
        self.assertEqual(len(merged), 3, "Candidate count should be 3 (visual pages 1, 2 + graph page 3).")

        page_numbers = [p["page_number"] for p in merged]
        self.assertEqual(page_numbers, [1, 2, 3])

        page3 = next(p for p in merged if p["page_number"] == 3)
        self.assertEqual(page3["retrieval_sources"], ["graph"])
        self.assertEqual(len(page3["graph_claims"]), 1)

    def test_retrieval_sources_provenance_preserved(self):
        """Verify each candidate correctly reflects its discovery source."""
        merged = _merge_retrieval_candidates(self.visual_candidates, [], limit=5)
        for p in merged:
            self.assertEqual(p["retrieval_sources"], ["visual"])


class TestGraphRetrieverLive(unittest.TestCase):
    """Integration test with live Neo4j database."""

    @classmethod
    def setUpClass(cls):
        cls.store = Neo4jStore()
        health = cls.store.health_check()
        if health.get("status") != "ready":
            raise unittest.SkipTest(f"Neo4j is not available at {cls.store.uri}")

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_graph_retrieval_resolves_pages(self):
        retriever = GraphRetriever(self.store)
        res = retriever.retrieve("Vision-Language Model Retrieval and Deduplication")
        self.assertIn(res.status, ("success", "fallback_empty"))
        if res.claims:
            self.assertGreater(len(res.source_pages), 0)
            for page in res.source_pages:
                self.assertIn("page_number", page)
                self.assertIn("document_id", page)
                self.assertIn("retrieval_sources", page)
                self.assertIn("graph", page["retrieval_sources"])


class TestPromptInjectionOfGraphClaims(unittest.TestCase):
    """Test that generator messages include structured graph claims when present."""

    def test_messages_include_graph_claims(self):
        pages = [
            {
                "rank": 1,
                "score": 0.88,
                "document_id": "DOC-1",
                "version_id": "VER-1",
                "version_number": 1,
                "filename": "test.pdf",
                "page_number": 1,
                "page_image_path": None,
                "page_text": "Sample text",
                "graph_claims": [
                    {
                        "subject": "ColPali",
                        "predicate": "OUTPERFORMS",
                        "object_or_value": "BGE-M3",
                        "evidence_text": "ColPali outperforms text retrieval baselines",
                    }
                ],
            }
        ]

        messages = _build_messages("Does ColPali outperform baselines?", pages)
        self.assertEqual(len(messages), 1)
        content_items = messages[0]["content"]

        # Check that graph claims text was added to content
        found_graph_text = any(
            item.get("type") == "text"
            and "Verified Structured Knowledge Graph Claims" in item.get("text", "")
            and "ColPali OUTPERFORMS BGE-M3" in item.get("text", "")
            for item in content_items
        )
        self.assertTrue(found_graph_text, "Knowledge graph claims should be included in generator context.")


if __name__ == "__main__":
    unittest.main()
