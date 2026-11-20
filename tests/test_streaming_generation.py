import json
import sys
import unittest
from unittest.mock import MagicMock, patch

if "colpali_engine" not in sys.modules:
    sys.modules["colpali_engine"] = MagicMock()
    sys.modules["colpali_engine.models"] = MagicMock()

from fastapi.testclient import TestClient

import api
from authorization import AuthContext, get_current_user


class TestStreamingGeneration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.mock_user = AuthContext(
            user_id="user_stream_test",
            name="Stream Tester",
            email="stream@example.com",
            organization_id="org_stream",
            organization_name="Stream Org",
            role="chief",
            status="active",
        )
        self.valid_payload = {
            "user_id": "user_stream_test",
            "organization": "Stream Org",
            "role": "chief",
            "query": "Test query for streaming",
            "top_k": 2,
        }

    def tearDown(self):
        api.app.dependency_overrides.clear()

    def test_stream_unauthorized(self):
        # Without dependency override or token, returns 401
        response = self.client.post("/generate/stream", json=self.valid_payload)
        self.assertEqual(response.status_code, 401)

    @patch("api.cache_get_json", return_value=None)
    @patch("api.cache_set_json")
    @patch("api.retrieve")
    @patch("api.generate_answer_stream")
    def test_stream_successful_sse(
        self,
        mock_generate_stream,
        mock_retrieve,
        mock_cache_set,
        mock_cache_get,
    ):
        api.app.dependency_overrides[get_current_user] = lambda: self.mock_user

        mock_retrieve.return_value = {
            "query": "Test query for streaming",
            "results": [
                {
                    "rank": 1,
                    "score": 0.95,
                    "document_id": "doc1",
                    "version_id": "v1",
                    "version_number": 1,
                    "filename": "doc1.pdf",
                    "page_number": 1,
                    "page_hash": "hash1",
                    "page_image_path": "data/images/doc1_p1.png",
                    "page_text": "Sample text",
                }
            ],
            "scope": {"mode": "current_only"},
            "result_count": 1,
        }

        mock_generate_stream.return_value = iter([
            {"type": "token", "token": "Hello", "ttft_ms": 120.5},
            {"type": "token", "token": " world", "ttft_ms": 120.5},
            {
                "type": "complete",
                "data": {
                    "answer": "Hello world",
                    "sources": [{"source_id": "S1", "rank": 1, "score": 0.95}],
                    "generation": {
                        "model_id": "test-model",
                        "duration_ms": 500.0,
                        "ttft_ms": 120.5,
                    },
                    "latency_ms": 500.0,
                    "ttft_ms": 120.5,
                },
            },
        ])

        response = self.client.post(
            "/generate/stream",
            json=self.valid_payload,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers.get("content-type", ""))

        text = response.text
        self.assertIn("event: token", text)
        self.assertIn('data: {"token": "Hello"}', text)
        self.assertIn('data: {"token": " world"}', text)
        self.assertIn("event: complete", text)
        self.assertIn('"answer": "Hello world"', text)

    @patch("api.cache_get_json")
    @patch("api.retrieve")
    def test_stream_cached_response(
        self,
        mock_retrieve,
        mock_cache_get,
    ):
        api.app.dependency_overrides[get_current_user] = lambda: self.mock_user

        mock_retrieve.return_value = {"results": [], "result_count": 0}
        # Simulate retrieval cache hit (first call) and answer cache hit (second call)
        mock_cache_get.side_effect = [
            {"results": [], "result_count": 0},
            {
                "answer": "Cached answer response",
                "sources": [],
                "generation": {"model_id": "cached-model"},
            },
        ]

        response = self.client.post(
            "/generate/stream",
            json=self.valid_payload,
        )

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("event: token", text)
        self.assertIn("Cached answer response", text)
        self.assertIn("event: complete", text)
        self.assertIn('"cached": true', text)

    @patch("api.cache_get_json", return_value=None)
    @patch("api.retrieve")
    @patch("api.generate_answer_stream")
    def test_stream_error_handling(
        self,
        mock_generate_stream,
        mock_retrieve,
        mock_cache_get,
    ):
        api.app.dependency_overrides[get_current_user] = lambda: self.mock_user

        mock_retrieve.return_value = {"results": [], "result_count": 0}
        mock_generate_stream.side_effect = RuntimeError("GPU out of memory simulation")

        response = self.client.post(
            "/generate/stream",
            json=self.valid_payload,
        )

        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("event: error", text)
        self.assertIn("GPU out of memory simulation", text)


if __name__ == "__main__":
    unittest.main()
