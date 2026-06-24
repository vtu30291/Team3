"""
Comprehensive End-to-End Testing Suite for Local RAG API Backend.
Uses pytest + unittest.mock for full path coverage without requiring
live MySQL, Ollama, or FAISS infrastructure.

Run: python -m pytest test_backend.py -v --tb=short
"""

import io
import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

# ---------------------------------------------------------------------------
# Fixtures & Test App Factory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, tmp_path):
    """Set environment variables BEFORE any app module imports."""
    monkeypatch.setenv("FLASK_PORT", "5000")
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_pass")
    monkeypatch.setenv("DB_NAME", "test_rag_db")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))


@pytest.fixture()
def app(tmp_path):
    """
    Create a fresh Flask test application with all external dependencies mocked.
    Each test gets an isolated app instance with its own tmp storage.
    """
    storage_dir = str(tmp_path / "storage")
    pdfs_dir = os.path.join(storage_dir, "pdfs")
    vector_store_dir = os.path.join(storage_dir, "vector_store")
    os.makedirs(pdfs_dir, exist_ok=True)
    os.makedirs(vector_store_dir, exist_ok=True)

    # Mock the database init and connection pool before importing app
    with patch("db.init_db") as mock_init_db, \
         patch("db.ensure_database_exists"), \
         patch("db.get_db_connection") as mock_get_conn:

        # Setup mock connection and cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"total": 0}
        mock_cursor.fetchall.return_value = []

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_get_conn.return_value = mock_conn

        # Patch config paths before importing app
        with patch("config.STORAGE_DIR", storage_dir), \
             patch("config.PDFS_DIR", pdfs_dir), \
             patch("config.VECTOR_STORE_DIR", vector_store_dir):

            # Force reimport of app module with mocked dependencies
            import importlib
            import app as app_module
            importlib.reload(app_module)

            app_module.app.config["TESTING"] = True

            # Patch the vector store on the app module
            mock_vs = MagicMock()
            mock_vs.search_vectors.return_value = (
                np.array([[0.85, 0.72, 0.55]]),
                np.array([[1, 2, 3]]),
            )
            mock_vs.remove_vectors.return_value = 3
            mock_vs.add_vectors.return_value = None
            mock_vs.get_total_vectors.return_value = 0
            app_module.vector_store = mock_vs

            yield app_module.app, mock_get_conn, mock_vs


@pytest.fixture()
def client(app):
    """Flask test client."""
    flask_app, _, _ = app
    return flask_app.test_client()


@pytest.fixture()
def mock_db_conn(app):
    """Expose the mock get_db_connection for test assertions."""
    _, mock_get_conn, _ = app
    return mock_get_conn


@pytest.fixture()
def mock_vector_store(app):
    """Expose the mock VectorStoreManager for test assertions."""
    _, _, mock_vs = app
    return mock_vs


def _make_pdf_bytes(content: str = "Test PDF content") -> bytes:
    """Generate minimal valid PDF binary bytes for upload testing."""
    # Minimal valid PDF structure
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
# HAPPY PATH TESTS
# ===========================================================================


class TestHappyPaths:
    """Tests for all successful/expected operation flows."""

    @pytest.mark.happy
    def test_index_route(self, client):
        """GET / returns status JSON with endpoint catalog."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "online"
        assert "endpoints" in data
        assert "chat_query" in data["endpoints"]
        assert data["version"] == "v1"

    @pytest.mark.happy
    def test_upload_valid_pdf(self, client, mock_db_conn):
        """POST upload with valid PDF returns 201 and tracking token."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

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
        # Verify UUID format (36 chars with dashes)
        assert len(body["tracking_token"]) == 36

    @pytest.mark.happy
    def test_finalize_material(self, client, mock_db_conn):
        """POST finalize with valid token returns 202 and starts ingestion."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "id": "test-uuid-1234",
            "file_path": "/fake/path/test.pdf",
        }
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

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

    @pytest.mark.happy
    def test_list_materials_paginated(self, client, mock_db_conn):
        """GET materials returns paginated JSON with correct structure."""
        mock_cursor = MagicMock()
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
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        resp = client.get("/api/v1/admin/materials?page=1&size=10")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "materials" in body
        assert "total" in body
        assert "page" in body
        assert "size" in body
        assert "pages" in body
        assert body["total"] == 1
        assert len(body["materials"]) == 1

    @pytest.mark.happy
    def test_chat_query_with_context(self, client, mock_db_conn, mock_vector_store):
        """POST chat query returns RAG response with matched context."""
        # Setup DB to return active chunks
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Project Nebula-X is a quantum initiative.",
                "faiss_id": 1,
                "file_name": "test_doc.pdf",
            }
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        # Mock embedding model
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

    @pytest.mark.happy
    def test_chat_query_streaming(self, client, mock_db_conn, mock_vector_store):
        """POST chat query with stream=True returns text/plain streaming response."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Some chunk text.",
                "faiss_id": 1,
                "file_name": "doc.pdf",
            }
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

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

    @pytest.mark.happy
    def test_update_status_to_inactive(self, client, mock_db_conn):
        """PATCH status to inactive returns 200."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("uuid-1",)  # exists
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        resp = client.patch(
            "/api/v1/admin/materials/uuid-1/status",
            json={"status": "inactive"},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "inactive" in body["message"]

    @pytest.mark.happy
    def test_delete_material(self, client, mock_db_conn, mock_vector_store):
        """DELETE material returns 200 and triggers FAISS purge."""
        mock_cursor = MagicMock()
        # First call: SELECT file_path
        # Second call: SELECT faiss_id
        mock_cursor.fetchone.return_value = {"file_path": "/fake/path.pdf"}
        mock_cursor.fetchall.return_value = [
            {"faiss_id": 1},
            {"faiss_id": 2},
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        with patch("os.path.exists", return_value=False):
            resp = client.delete("/api/v1/admin/materials/uuid-1")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "deleted" in body["message"].lower()
        mock_vector_store.remove_vectors.assert_called_once()


# ===========================================================================
# EDGE CASE TESTS
# ===========================================================================


class TestEdgeCases:
    """Tests for boundary conditions, validation, and edge-case handling."""

    @pytest.mark.edge
    def test_upload_exceeds_25mb(self, client):
        """Rejects files that exceed the 25 MB limit with 413."""
        # Create bytes just over 25 MB with valid PDF header
        oversized = b"%PDF" + b"\x00" * (25 * 1024 * 1024 + 1)
        data = {"file": (io.BytesIO(oversized), "huge.pdf")}

        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        # Flask's MAX_CONTENT_LENGTH triggers 413
        assert resp.status_code == 413

    @pytest.mark.edge
    def test_upload_non_pdf_extension(self, client):
        """Rejects non-PDF file extensions with 400."""
        data = {"file": (io.BytesIO(b"plain text"), "readme.txt")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "PDF" in body["error"]

    @pytest.mark.edge
    def test_upload_no_file(self, client):
        """Rejects request with missing file field with 400."""
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data={},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "No file" in body["error"]

    @pytest.mark.edge
    def test_upload_fake_pdf_header(self, client):
        """Rejects binary file with .pdf extension but no %PDF magic header."""
        fake_pdf = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # PNG header
        data = {"file": (io.BytesIO(fake_pdf), "fake.pdf")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "not a valid PDF" in body["error"]

    @pytest.mark.edge
    def test_upload_empty_filename(self, client):
        """Rejects upload with empty filename."""
        data = {"file": (io.BytesIO(b"%PDF-test"), "")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    @pytest.mark.edge
    def test_finalize_missing_token(self, client):
        """Returns 400 when tracking_token is missing."""
        resp = client.post(
            "/api/v1/admin/materials/finalize",
            json={"annotation": "Some text"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "tracking_token" in body["error"]

    @pytest.mark.edge
    def test_finalize_nonexistent_token(self, client, mock_db_conn):
        """Returns 404 when tracking_token doesn't match any draft record."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # Not found
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        resp = client.post(
            "/api/v1/admin/materials/finalize",
            json={
                "tracking_token": "nonexistent-uuid",
                "annotation": "Test",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.edge
    def test_finalize_oversized_annotation(self, client):
        """Returns 400 when annotation exceeds 2000 characters."""
        resp = client.post(
            "/api/v1/admin/materials/finalize",
            json={
                "tracking_token": "test-uuid",
                "annotation": "A" * 2001,
            },
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "limit" in body["error"].lower() or "exceeds" in body["error"].lower()

    @pytest.mark.edge
    def test_finalize_missing_annotation(self, client):
        """Returns 400 when annotation field is absent."""
        resp = client.post(
            "/api/v1/admin/materials/finalize",
            json={"tracking_token": "test-uuid"},
        )
        assert resp.status_code == 400

    @pytest.mark.edge
    def test_chat_empty_query(self, client):
        """Returns 400 when query string is empty."""
        resp = client.post("/api/v1/chat/query", json={"query": ""})
        assert resp.status_code == 400
        body = resp.get_json()
        assert "empty" in body["error"].lower()

    @pytest.mark.edge
    def test_chat_whitespace_only_query(self, client):
        """Returns 400 when query is only whitespace."""
        resp = client.post("/api/v1/chat/query", json={"query": "   "})
        assert resp.status_code == 400

    @pytest.mark.edge
    def test_update_invalid_status(self, client):
        """Returns 400 for invalid status value."""
        resp = client.patch(
            "/api/v1/admin/materials/uuid-1/status",
            json={"status": "deleted"},
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "Invalid status" in body["error"]

    @pytest.mark.edge
    def test_update_status_missing_body(self, client):
        """Returns 400 when status field is missing."""
        resp = client.patch(
            "/api/v1/admin/materials/uuid-1/status",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.edge
    def test_delete_nonexistent_material(self, client, mock_db_conn):
        """Returns 404 when material doesn't exist."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        resp = client.delete("/api/v1/admin/materials/nonexistent-id")
        assert resp.status_code == 404

    @pytest.mark.edge
    def test_list_materials_default_pagination(self, client, mock_db_conn):
        """GET materials without params uses defaults (page=1, size=12)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = {"total": 0}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        resp = client.get("/api/v1/admin/materials")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["page"] == 1
        assert body["size"] == 12

    @pytest.mark.edge
    def test_chat_no_matching_chunks(self, client, mock_vector_store):
        """Returns fallback message when FAISS returns no matches above threshold."""
        # All distances below threshold (0.30)
        mock_vector_store.search_vectors.return_value = (
            np.array([[0.1, 0.05, -0.2]]),
            np.array([[1, 2, 3]]),
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "Unrelated question about astrophysics"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "could not find" in body["response"].lower()


# ===========================================================================
# FAILURE / MOCK TESTS
# ===========================================================================


class TestFailureMocking:
    """Tests simulating infrastructure failures (Ollama offline, DB drops, etc.)."""

    @pytest.mark.failure
    def test_chat_ollama_offline_503(self, client, mock_db_conn, mock_vector_store):
        """Simulates Ollama server being offline — should return 200 with error message."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Some relevant chunk.",
                "faiss_id": 1,
                "file_name": "doc.pdf",
            }
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        # Simulate Ollama returning an error string (as the real client does)
        with patch("app.get_embedding_model", return_value=mock_model), \
             patch(
                 "app.query_ollama_non_stream",
                 return_value="Error: Unable to connect to local Ollama server at http://localhost:11434.",
             ):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "What is Nebula-X?", "stream": False},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "Error" in body["response"]
            assert "Ollama" in body["response"]

    @pytest.mark.failure
    def test_chat_ollama_timeout(self, client, mock_db_conn, mock_vector_store):
        """Simulates Ollama request timeout — should return 200 with timeout message."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                "text_content": "Some chunk.",
                "faiss_id": 1,
                "file_name": "doc.pdf",
            }
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db_conn.return_value = mock_conn

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model), \
             patch(
                 "app.query_ollama_non_stream",
                 return_value="Error: Ollama request timed out. The model may be overloaded.",
             ):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test", "stream": False},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "timed out" in body["response"].lower()

    @pytest.mark.failure
    def test_mysql_connection_drop_on_list(self, client, mock_db_conn):
        """Simulates MySQL pool exhaustion on materials listing — returns 500."""
        mock_db_conn.side_effect = Exception("Pool exhausted: no available connections")

        resp = client.get("/api/v1/admin/materials")
        assert resp.status_code == 500
        body = resp.get_json()
        assert "error" in body

    @pytest.mark.failure
    def test_mysql_connection_drop_on_upload(self, client, mock_db_conn):
        """Simulates MySQL failure during upload DB insert — returns 500."""
        mock_db_conn.side_effect = Exception("Connection refused")

        pdf_data = _make_pdf_bytes()
        data = {"file": (io.BytesIO(pdf_data), "test.pdf")}
        resp = client.post(
            "/api/v1/admin/materials/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 500

    @pytest.mark.failure
    def test_mysql_connection_drop_on_chat(self, client, mock_db_conn, mock_vector_store):
        """Simulates MySQL connection drop during chat query — returns 500."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        # FAISS returns valid results, but DB connection fails
        mock_db_conn.side_effect = Exception("MySQL has gone away")

        with patch("app.get_embedding_model", return_value=mock_model):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "test query"},
            )
            assert resp.status_code == 500
            body = resp.get_json()
            assert "error" in body

    @pytest.mark.failure
    def test_faiss_search_empty_index(self, client, mock_vector_store):
        """Queries when FAISS index is empty — should return fallback message."""
        mock_vector_store.search_vectors.return_value = (
            np.array([[]]),
            np.array([[]]),
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype("float32")

        with patch("app.get_embedding_model", return_value=mock_model):
            resp = client.post(
                "/api/v1/chat/query",
                json={"query": "anything"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "could not find" in body["response"].lower()


# ===========================================================================
# UNIT TESTS FOR CORE MODULES
# ===========================================================================


class TestIngestionUnit:
    """Unit tests for the ingestion module functions."""

    @pytest.mark.happy
    def test_semantic_chunking_short_text(self):
        """Text shorter than chunk_size returns a single chunk."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))
        mock_tokenizer.decode.return_value = "Short text decoded"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        with patch("ingestion._model", mock_model), \
             patch("ingestion._model_lock", MagicMock()):
            with patch("ingestion.get_embedding_model", return_value=mock_model):
                from ingestion import semantic_chunking
                result = semantic_chunking("Short text", chunk_size=500, overlap=50)
                assert len(result) == 1
                assert result[0] == "Short text decoded"

    @pytest.mark.happy
    def test_semantic_chunking_long_text(self):
        """Text longer than chunk_size produces multiple overlapping chunks."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(1200))
        mock_tokenizer.decode.return_value = "Chunk text"

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        with patch("ingestion._model", mock_model), \
             patch("ingestion._model_lock", MagicMock()):
            with patch("ingestion.get_embedding_model", return_value=mock_model):
                from ingestion import semantic_chunking
                result = semantic_chunking("Long text " * 200, chunk_size=500, overlap=50)
                # With 1200 tokens, step=450: chunks at [0:500], [450:950], [900:1200]
                assert len(result) == 3

    @pytest.mark.edge
    def test_semantic_chunking_empty_text(self):
        """Empty text returns no chunks."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = []

        mock_model = MagicMock()
        mock_model.tokenizer = mock_tokenizer

        with patch("ingestion._model", mock_model), \
             patch("ingestion._model_lock", MagicMock()):
            with patch("ingestion.get_embedding_model", return_value=mock_model):
                from ingestion import semantic_chunking
                result = semantic_chunking("", chunk_size=500, overlap=50)
                assert result == []

    @pytest.mark.edge
    def test_extract_text_missing_file(self):
        """Raises FileNotFoundError for non-existent PDF path."""
        from ingestion import extract_text_from_pdf

        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/non/existent/path.pdf")


class TestOllamaClientUnit:
    """Unit tests for the Ollama client functions."""

    @pytest.mark.failure
    def test_non_stream_url_error(self):
        """URLError during non-stream query returns error string."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            from ollama_client import query_ollama_non_stream
            result = query_ollama_non_stream("test prompt")
            assert "Error" in result
            assert "Ollama" in result

    @pytest.mark.failure
    def test_non_stream_timeout_error(self):
        """TimeoutError during non-stream query returns timeout message."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            from ollama_client import query_ollama_non_stream
            result = query_ollama_non_stream("test prompt")
            assert "timed out" in result.lower()

    @pytest.mark.happy
    def test_non_stream_success(self):
        """Successful non-stream query returns model content."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"message": {"content": "Hello from LLM"}}
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            from ollama_client import query_ollama_non_stream
            result = query_ollama_non_stream("test prompt")
            assert result == "Hello from LLM"


class TestVectorStoreUnit:
    """Unit tests for the VectorStoreManager."""

    @pytest.mark.happy
    def test_add_vectors_normalizes(self):
        """add_vectors normalizes embeddings before insertion."""
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

    @pytest.mark.edge
    def test_add_vectors_wrong_dimension(self):
        """Raises ValueError for incorrect embedding dimensions."""
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

            bad_embeddings = np.random.rand(3, 128).astype("float32")
            ids = np.array([1, 2, 3], dtype=np.int64)

            with pytest.raises(ValueError, match="384"):
                vs.add_vectors(bad_embeddings, ids)


class TestConfigUnit:
    """Unit tests for the config module."""

    @pytest.mark.happy
    def test_config_defaults_loaded(self):
        """Config module loads with sane defaults."""
        import config
        assert config.CHUNK_SIZE == 500
        assert config.CHUNK_OVERLAP == 50
        assert config.MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024
        assert config.EMBEDDING_DIMENSION == 384
        assert config.SIMILARITY_THRESHOLD == 0.30
        assert config.MAX_ANNOTATION_LENGTH == 2000
        assert config.CONTEXT_CHUNKS_LIMIT == 3
