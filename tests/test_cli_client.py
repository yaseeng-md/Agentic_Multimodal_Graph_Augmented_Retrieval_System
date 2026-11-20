"""
Unit tests for cli_client.py verifying SessionManager and ApiDebugger renderers.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cli_client import SessionManager, ApiDebugger, console


def test_session_manager(tmp_path: Path):
    sess_file = tmp_path / "test_session.json"
    sm = SessionManager(session_path=sess_file)

    assert not sm.is_authenticated()
    assert sm.get_auth_headers() == {}

    # Simulate saving token
    fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c3JfMTIzIiwicm9sZSI6ImVuZ2luZWVyIn0.sig"
    sm.set_login_data(fake_token, {"name": "Alice", "role": "engineer", "organization_name": "OrgX"})

    assert sm.is_authenticated()
    headers = sm.get_auth_headers()
    assert "Authorization" in headers
    assert headers["Authorization"] == f"Bearer {fake_token}"

    claims = sm.decode_token_unverified()
    assert claims is not None
    assert claims.get("sub") == "usr_123"

    # Test reload from disk
    sm2 = SessionManager(session_path=sess_file)
    assert sm2.is_authenticated()
    assert sm2.user.get("name") == "Alice"

    # Test job recording
    sm2.record_job("ingest", "ingest_20261027_abc")
    assert sm2.last_jobs["ingest"] == "ingest_20261027_abc"

    # Test clear
    sm2.clear()
    assert not sm2.is_authenticated()
    assert sm2.get_auth_headers() == {}


def test_specialized_renderers(tmp_path: Path):
    sm = SessionManager(session_path=tmp_path / "test.json")
    dbg = ApiDebugger(sm)

    # 1. Documents rendering
    doc_data = {
        "documents": [
            {
                "filename": "sample.pdf",
                "document_id": "doc_1",
                "version_id": "ver_1",
                "version_number": 1,
                "is_current": True,
            }
        ]
    }
    dbg._render_specialized_view(doc_data)

    # 2. Query retrieval results
    query_data = {
        "result": [
            {
                "score": 0.952,
                "document_id": "doc_1",
                "page_number": 2,
                "bbox": [100, 100, 300, 300],
                "claim": "ColPali achieves late-interaction visual embeddings.",
            }
        ]
    }
    dbg._render_specialized_view(query_data)

    # 3. Generated answer
    gen_data = {
        "answer": "This is a **multimodal** grounded answer.",
        "model": "Qwen3-VL-2B-Instruct",
        "generation_time_ms": 340.5,
    }
    dbg._render_specialized_view(gen_data)

    # 4. Error message
    err_data = {"detail": "Invalid credentials or unauthorized."}
    dbg._render_specialized_view(err_data)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_session_manager(p)
        test_specialized_renderers(p)
    print("All cli_client tests passed successfully!")
