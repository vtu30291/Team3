"""
Hardened Production-Grade Integration and Unit Testing Suite.
Organized into three distinct validation planes:
- Plane A: Happy Path Flows (RAG, CRUD, embeddings, semantic chunking)
- Plane B: Boundary Edge Failures (25MB exact, 25.1MB invalid, 2,000-char annotations, empty queries)
- Plane C: Chaos & Infrastructure Mocking (Ollama offline, MySQL drops, corrupted PDF uploads)

Run: c:/Users/dell/Downloads/Team 3/Team 3/.venv/Scripts/python.exe -m pytest test_backend_final_hz.py -v --tb=short
"""

import io
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Fixtures & Test App Factory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _session_setup():
    """Import app and dependencies once with database initialization mocked."""
    # Ensure environment variables are set before any imports
    os.environ["FLASK_PORT"] = "5000"
    os.environ["DB_HOST"] = "localhost"
    os.environ["DB_USER"] = "test_user"
    os.environ["DB_PASSWORD"] = "test_pass"
    os.environ["DB_NAME"] = "test_rag_db"
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
    os.environ["STORAGE_DIR"] = "./test_storage"

    with patch("db.init_db"), \
         patch("db.ensure_database_exists"), \
         patch("db.get_db_connection"):
        import app as app_module
        yield app_module


@pytest.fixture()
def client(_session_setup):
    """Provides a Flask test client for the application."""
    flask_app = _session_setup.app
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture()
def mock_db_conn():
    """Provides a mock database connection and patches get_db_connection in all modules."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {"total": 0}
    mock_cursor.fetchall.return_value = []
    
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    with patch("db.get_db_connection", return_value=mock_conn), \
         patch("app.get_db_connection", return_value=mock_conn), \
         patch("ingestion.get_db_connection", return_value=mock_conn):
        yield mock_conn


@pytest.fixture()
def mock_vector_store(_session_setup):
    """Exposes a mocked VectorStoreManager and patches it on the app."""
    mock_vs = MagicMock()
    mock_vs.search_vectors.return_value = (
        np.array([[0.85, 0.72, 0.55]]),
        np.array([[1, 2, 3]]),
    )
    mock_vs.remove_vectors.return_value = 3
    mock_vs.add_vectors.return_value = None
    mock_vs.get_total_vectors.return_value = 0
    
    old_vs = _session_setup.vector_store
    _session_setup.vector_store = mock_vs
    yield mock_vs
    _session_setup.vector_store = old_vs


def _make_pdf_bytes(content: str = "Test PDF content") -> bytes:
    """Generate minimal valid PDF binary bytes for upload testing."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )
    return pdf


# ===========================================================================
# PLANE A: HAPPY PATH FLOWS
# ===========================================================================

@pytest.mark.plane_a
class TestPlaneA:
    """Plane A validates all successful flows and expected behaviors."""

    def test_index_route(self, client):
        """GET / returns online status with API catalog."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "online"
        assert "endpoints" in data
        assert "chat_query" in data["endpoints"]

    def test_upload_valid_pdf(self, client, mock_db_conn):
        """POST upload with valid PDF returns 201 and tracking token."""
        pdf_data = _make_pdf_bytes()
        data = {"file": (io.BytesIO(pdf_data), "test_doc.pdf")}

        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        body = resp.get_json()
        assert "tracking_token" in body
        assert body["file_name"] == "test_doc.pdf"
        assert len(body["tracking_token"]) == 36

    def test_finalize_material(self, client, mock_db_conn):
        """POST finalize with valid token returns 202 and starts ingestion."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchone.return_value = {
            "id": "test-uuid-1234",
            "file_path": "/fake/path/test.pdf",
        }

        with patch("app.start_async_ingestion") as mock_ingest:
            resp = client.post(
                "/api/v1/admin/materials/finalize",
                json={
                    "tracking_token": "test-uuid-1234",
                    "annotation": "Test annotation for document.",
                },
            )
            assert resp.status_code == 202
            body = resp.get_json()
            assert "tracking_token" in body
            assert body["tracking_token"] == "test-uuid-1234"

    def test_list_materials_paginated(self, client, mock_db_conn):
        """GET materials returns paginated list of metadata."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            {
                "id": "uuid-1",
                "file_name": "doc1.pdf",
                "file_path": "/path/doc1.pdf",
                "annotation": "Test",
                "index_id": "1",
                "chunk_count": 5,
                "status": "active",
                "created_at": None,
                "updated_at": None,
            }
        ]
        mock_cursor.fetchone.return_value = {"total": 1}

        resp = client.get("/api/v1/admin/materials?page=1&size=10")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "materials" in body
        assert body["total"] == 1
        assert len(body["materials"]) == 1

    def test_chat_query_with_context(self, client, mock_db_conn, mock_vector_store):
        """POST chat query returns RAG response with active database context."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Project Nebula-X is a quantum initiative.",
                "faiss_id": 1,
                "file_name": "test_doc.pdf",
            }
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model), \
             patch("app.query_ollama_non_stream", return_value="Nebula-X is a quantum computing project."):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "What is Nebula-X?", "stream": False},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "response" in body
            assert "Nebula-X" in body["response"]

    def test_chat_query_streaming(self, client, mock_db_conn, mock_vector_store):
        """POST chat query with stream=True yields plain text stream."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Some chunk text.",
                "faiss_id": 1,
                "file_name": "doc.pdf",
            }
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        def fake_stream(prompt):
            yield "Hello "
            yield "World"

        with patch("app.get_embedding_model", return_value=mock_model), \
             patch("app.query_ollama_stream", side_effect=fake_stream):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test query", "stream": True},
            )
            assert resp.status_code == 200
            assert resp.content_type == "text/plain; charset=utf-8"
            streamed_data = resp.get_data(as_text=True)
            assert "Hello " in streamed_data

    def test_update_status_to_inactive(self, client, mock_db_conn):
        """PATCH status to inactive successfully updates state."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchone.return_value = ("uuid-1",)

        resp = client.patch(
            "/api/v1/admin/materials/uuid-1/status",
            json={"status": "inactive"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "inactive" in body["message"]

    def test_delete_material(self, client, mock_db_conn, mock_vector_store):
        """DELETE material deletes disk file and purges FAISS index."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchone.return_value = {"file_path": "/fake/path.pdf"}
        mock_cursor.fetchall.return_value = [{"faiss_id": 1}, {"faiss_id": 2}]

        with patch("os.path.exists", return_value=False):
            resp = client.delete("/api/v1/admin/materials/uuid-1")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "deleted" in body["message"].lower()
        mock_vector_store.remove_vectors.assert_called_once()

    def test_semantic_chunking_short_text(self):
        """Unit test: Text shorter than chunk size forms 1 chunk."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer.decode.return_value = "Short text decoded"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        with patch("ingestion.get_embedding_model", return_value=mock_model):
            from ingestion import semantic_chunking
            result = semantic_chunking("Short text", chunk_size=500, overlap=50)
            assert len(result) == 1
            assert result[0] == "Short text decoded"

    def test_semantic_chunking_long_text(self):
        """Unit test: Text longer than chunk size forms overlapping chunks."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(1200))
        mock_tokenizer.decode.return_value = "Chunk text"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        with patch("ingestion.get_embedding_model", return_value=mock_model):
            from ingestion import semantic_chunking
            result = semantic_chunking("Long text " * 200, chunk_size=500, overlap=50)
            assert len(result) == 3

    def test_vector_store_add_vectors_normalizes(self):
        """Unit test: L2 normalization is executed before index insertion."""
        with patch("faiss.read_index"), \
             patch("faiss.write_index"), \
             patch("faiss.normalize_L2") as mock_normalize, \
             patch("faiss.IndexFlatIP") as mock_flat, \
             patch("faiss.IndexIDMap") as mock_idmap:

            mock_index = MagicMock()
            mock_index.ntotal = 0
            mock_idmap.return_value = mock_index

            with patch("os.path.exists", return_value=False):
                from vector_store import VectorStoreManager
                vs = VectorStoreManager.__new__(VectorStoreManager)
                vs.storage_dir = "/tmp"
                vs.vector_store_dir = "/tmp/vs"
                vs.index_path = "/tmp/vs/index.faiss"
                vs.lock = MagicMock()
                vs.lock.__enter__ = MagicMock(return_value=None)
                vs.lock.__exit__ = MagicMock(return_value=False)
                vs.index = mock_index

                embeddings = np.random.rand(3, 384).astype("float32")
                ids = np.array([1, 2, 3], dtype=np.int64)

                vs.add_vectors(embeddings, ids)
                mock_normalize.assert_called_once()
                mock_index.add_with_ids.assert_called_once()


# ===========================================================================
# PLANE B: BOUNDARY EDGE FAILURES
# ===========================================================================

@pytest.mark.plane_b
class TestPlaneB:
    """Plane B stresses system boundaries, validation constraints, and limits."""

    def test_upload_exact_25mb(self, client, mock_db_conn):
        """Accepts file that is exactly 25 MB (boundary pass)."""
        exact_size = 25 * 1024 * 1024
        exact_payload = b"%PDF" + b"\x00" * (exact_size - 4)
        data = {"file": (io.BytesIO(exact_payload), "boundary_25mb.pdf")}

        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        body = resp.get_json()
        assert "tracking_token" in body

    def test_upload_exceeds_25mb(self, client):
        """Rejects files exceeding 25 MB (boundary fail at 25.1 MB)."""
        oversized_size = int(25.1 * 1024 * 1024)
        oversized = b"%PDF" + b"\x00" * (oversized_size - 4)
        data = {"file": (io.BytesIO(oversized), "fail_25_1mb.pdf")}

        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413

    def test_finalize_exact_2000_char_annotation(self, client, mock_db_conn):
        """Accepts annotation that is exactly 2000 characters."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchone.return_value = {
            "id": "test-uuid-2000",
            "file_path": "/fake/path/test.pdf",
        }

        with patch("app.start_async_ingestion"):
            resp = client.post(
                "/api/v1/admin/materials/finalize",
                json={
                    "tracking_token": "test-uuid-2000",
                    "annotation": "A" * 2000,
                },
            )
            assert resp.status_code == 202

    def test_finalize_oversized_annotation(self, client):
        """Rejects annotations exceeding 2000 characters (boundary fail at 2001)."""
        resp = client.post(
            "/api/v1/admin/materials/finalize",
            json={
                "tracking_token": "test-uuid",
                "annotation": "A" * 2001,
            },
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "exceeds" in body["error"].lower() or "limit" in body["error"].lower()

    def test_chat_empty_query(self, client):
        """Rejects empty chat queries with 400."""
        resp = client.post("/api/v1/chat/query", json={"query": ""})
        assert resp.status_code == 400

    def test_chat_whitespace_only_query(self, client):
        """Rejects chat queries containing only whitespace."""
        resp = client.post("/api/v1/chat/query", json={"query": "   \n\t "})
        assert resp.status_code == 400

    def test_upload_non_pdf_extension(self, client):
        """Rejects non-pdf extensions with 400."""
        data = {"file": (io.BytesIO(b"%PDF-1.4 header"), "readme.txt")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_upload_fake_pdf_header(self, client):
        """Rejects files ending in .pdf but missing binary header signature."""
        data = {"file": (io.BytesIO(b"PNG\x89 file format info"), "fake.pdf")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_chat_no_matching_chunks(self, client, mock_vector_store):
        """Returns fallback RAG response when FAISS has zero threshold matches."""
        mock_vector_store.search_vectors.return_value = (
            np.array([[0.1, 0.05]]),
            np.array([[1, 2]]),
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "Who is the Prime Minister?"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "could not find" in body["response"].lower()

    def test_vector_store_add_vectors_wrong_dimension(self):
        """Unit test: Throws ValueError when inserting dimension other than 384."""
        with patch("faiss.IndexFlatIP"), \
             patch("faiss.IndexIDMap"), \
             patch("os.path.exists", return_value=False), \
             patch("os.makedirs"):
            from vector_store import VectorStoreManager
            vs = VectorStoreManager.__new__(VectorStoreManager)
            vs.storage_dir = "/tmp"
            vs.vector_store_dir = "/tmp/vs"
            vs.index_path = "/tmp/vs/index.faiss"
            vs.lock = MagicMock()
            vs.index = MagicMock()

            bad_embeddings = np.random.rand(3, 512).astype("float32")
            ids = np.array([1, 2, 3], dtype=np.int64)

            with pytest.raises(ValueError, match="384"):
                vs.add_vectors(bad_embeddings, ids)


# ===========================================================================
# PLANE C: SYSTEM CHAOS & INFRASTRUCTURE MOCKING
# ===========================================================================

@pytest.mark.plane_c
class TestPlaneC:
    """Plane C simulates failures in surrounding infrastructure (DB, LLM, Files)."""

    def test_chat_ollama_offline_503(self, client, mock_db_conn, mock_vector_store):
        """Simulates Ollama server connection failure (Ollama 503 equivalent)."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Some context information.",
                "faiss_id": 1,
                "file_name": "context.pdf",
            }
        ]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model), \
             patch("app.query_ollama_non_stream", return_value="Error: Unable to connect to local Ollama server at http://localhost:11434."):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test query"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "Unable to connect" in body["response"]

    def test_chat_ollama_timeout(self, client, mock_db_conn, mock_vector_store):
        """Simulates Ollama request timeout."""
        mock_cursor = mock_db_conn.cursor.return_value
        mock_cursor.fetchall.return_value = [{"text_content": "Context", "faiss_id": 1, "file_name": "doc.pdf"}]

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model), \
             patch("app.query_ollama_non_stream", return_value="Error: Ollama request timed out."):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test query"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "timed out" in body["response"].lower()

    def test_mysql_connection_drop_on_chat(self, client, mock_db_conn, mock_vector_store):
        """Simulates MySQL connection pool drop mid-transaction during chat."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        # DB connection drops
        mock_db_conn.cursor.side_effect = Exception("Lost connection to MySQL server during query")

        with patch("app.get_embedding_model", return_value=mock_model):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test query"},
            )
            assert resp.status_code == 500
            body = resp.get_json()
            assert "error" in body

    def test_ingestion_corrupted_pdf_upload(self, mock_db_conn):
        """
        Simulates background ingestion of a corrupted PDF.
        The parser throws an exception, and status is correctly rolled back to 'inactive'.
        """
        mock_cursor = mock_db_conn.cursor.return_value

        # Mock os.path.exists to bypass the initial check, and fitz.open to raise exception (simulate corrupt PDF)
        with patch("os.path.exists", return_value=True), \
             patch("fitz.open", side_effect=Exception("PDF file header corrupted")):
            from ingestion import process_document_ingestion
            mock_vs = MagicMock()

            # Execute synchronous ingestion simulation
            process_document_ingestion("corrupt-token", "/fake/corrupt.pdf", mock_vs)

            # Assert status was reset to 'inactive' in database on failure
            mock_cursor.execute.assert_any_call(
                "UPDATE materials SET status = 'inactive' WHERE id = %s",
                ("corrupt-token",),
            )

    def test_ollama_client_non_stream_url_error(self):
        """Unit test: URLError inside ollama_client returns clean error string."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Ollama service down (503)")):
            from ollama_client import query_ollama_non_stream
            result = query_ollama_non_stream("test prompt")
            assert "Error:" in result
            assert "Unable to connect" in result
