"""
Flask API Backend for Local RAG Pipeline.
Provides admin CRUD endpoints for PDF materials and a RAG chat query endpoint.
"""

import os
import uuid
import math
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import config

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Startup Directory Setup Guardrail
os.makedirs(config.PDFS_DIR, exist_ok=True)
os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)
logger.info(
    f"Storage directory structure verified: {config.PDFS_DIR}, {config.VECTOR_STORE_DIR}"
)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # type: ignore  # Cross-Origin Resource Sharing for Team 1 & Team 2 UI components

app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE_BYTES + 100 * 1024

# Initialize Database Pool & Vector Store Manager
from db import init_db, get_db_connection
from vector_store import VectorStoreManager
from ingestion import start_async_ingestion, get_embedding_model
from ollama_client import query_ollama_stream, query_ollama_non_stream

try:
    init_db()
except Exception as e:
    logger.critical(
        f"Failed to complete database initialization migrations on startup: {e}"
    )

vector_store = VectorStoreManager(storage_dir=config.STORAGE_DIR)


# ==========================================
# ROOT ROUTE: GET /
# ==========================================
@app.route("/", methods=["GET"])
def index():
    """Serves a friendly API documentation and status JSON."""
    return (
        jsonify(
            {
                "status": "online",
                "message": "Local RAG API Backend is running successfully.",
                "version": "v1",
                "endpoints": {
                    "list_materials": "GET /api/v1/admin/materials",
                    "upload_material": "POST /api/v1/admin/materials/upload",
                    "finalize_material": "POST /api/v1/admin/materials/finalize",
                    "update_material_status": "PATCH /api/v1/admin/materials/<id>/status",
                    "delete_material": "DELETE /api/v1/admin/materials/<id>",
                    "chat_query": "POST /api/v1/chat/query",
                },
            }
        ),
        200,
    )


# ==========================================
# ADMIN ROUTE: GET /api/v1/admin/materials
# ==========================================
@app.route("/api/v1/admin/materials", methods=["GET"])
def list_materials():
    """
    Serves the main paginated administrative data table dashboard.
    Accepts page, size, search, and sort_by/order.
    Returns complete annotations.
    """
    page = request.args.get("page", 1, type=int)
    size = request.args.get("size", 12, type=int)
    search = request.args.get("search", "").strip()
    sort_by = request.args.get("sort_by", "created_at").strip()
    order = request.args.get("order", "DESC").upper().strip()

    # Strict sort & order whitelist guards (prevents SQL injection via f-string)
    allowed_sort = ["created_at", "updated_at", "file_name", "status"]
    if sort_by not in allowed_sort:
        sort_by = "created_at"
    if order not in ["ASC", "DESC"]:
        order = "DESC"

    # Defensive bounds clamping
    page = max(1, page)
    size = max(1, min(size, 100))

    offset = (page - 1) * size
    conn = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        if search:
            search_pattern = f"%{search}%"
            query_sql = f"""
                SELECT id, file_name, file_path, annotation, index_id, chunk_count,
                       status, created_at, updated_at
                FROM materials
                WHERE file_name LIKE %s OR annotation LIKE %s
                ORDER BY {sort_by} {order}
                LIMIT %s OFFSET %s
            """
            count_sql = (
                "SELECT COUNT(*) as total FROM materials "
                "WHERE file_name LIKE %s OR annotation LIKE %s"
            )

            cursor.execute(query_sql, (search_pattern, search_pattern, size, offset))
            materials = cursor.fetchall()

            cursor.execute(count_sql, (search_pattern, search_pattern))
            total = cursor.fetchone()["total"]
        else:
            query_sql = f"""
                SELECT id, file_name, file_path, annotation, index_id, chunk_count,
                       status, created_at, updated_at
                FROM materials
                ORDER BY {sort_by} {order}
                LIMIT %s OFFSET %s
            """
            count_sql = "SELECT COUNT(*) as total FROM materials"

            cursor.execute(query_sql, (size, offset))
            materials = cursor.fetchall()

            cursor.execute(count_sql)
            total = cursor.fetchone()["total"]

        cursor.close()

        pages = math.ceil(total / size) if size > 0 else 0

        # Convert datetime objects to string format for JSON compatibility
        for item in materials:
            if item.get("created_at"):
                item["created_at"] = item["created_at"].isoformat()
            if item.get("updated_at"):
                item["updated_at"] = item["updated_at"].isoformat()

        return (
            jsonify(
                {
                    "materials": materials,
                    "total": total,
                    "page": page,
                    "size": size,
                    "pages": pages,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Failed to fetch materials dashboard: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve administrative records"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# ADMIN ROUTE: PATCH /api/v1/admin/materials/<id>/status
# ==========================================
@app.route("/api/v1/admin/materials/<id>/status", methods=["PATCH"])
def update_material_status(id):
    """Toggles row activation status controls ('active' or 'inactive')."""
    data = request.get_json() or {}
    status = data.get("status")

    if status not in ["active", "inactive"]:
        return (
            jsonify(
                {"error": "Invalid status parameter. Must be 'active' or 'inactive'."}
            ),
            400,
        )

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check document existence
        cursor.execute("SELECT id FROM materials WHERE id = %s", (id,))
        if not cursor.fetchone():
            cursor.close()
            return jsonify({"error": "Material record not found."}), 404

        # Update status
        cursor.execute(
            "UPDATE materials SET status = %s WHERE id = %s", (status, id)
        )
        conn.commit()
        cursor.close()

        logger.info(f"Updated status of material {id} to {status}")
        return (
            jsonify({"message": f"Material status successfully changed to {status}"}),
            200,
        )

    except Exception as e:
        logger.error(
            f"Failed to update status for material {id}: {e}", exc_info=True
        )
        return jsonify({"error": "Failed to update status due to system error"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# ADMIN ROUTE: DELETE /api/v1/admin/materials/<id>
# ==========================================
@app.route("/api/v1/admin/materials/<id>", methods=["DELETE"])
def delete_material(id):
    """
    Permanent, non-reversible destruction of document, storage PDF file,
    and associated vector chunk layers in local FAISS.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # 1. Fetch material info to locate PDF file path
        cursor.execute("SELECT file_path FROM materials WHERE id = %s", (id,))
        material = cursor.fetchone()
        if not material:
            cursor.close()
            return jsonify({"error": "Material record not found."}), 404

        file_path = material["file_path"]

        # 2. Retrieve chunk FAISS IDs mapping to purge from index
        cursor.execute(
            "SELECT faiss_id FROM chunks WHERE material_id = %s", (id,)
        )
        chunks_records = cursor.fetchall()
        faiss_ids = [chunk["faiss_id"] for chunk in chunks_records]

        # 3. Purge from vector store files
        if faiss_ids:
            try:
                vector_store.remove_vectors(faiss_ids)
            except Exception as vs_err:
                logger.warning(
                    f"Failed to remove vectors from FAISS for material {id}: {vs_err}"
                )

        # 4. Remove MySQL record (cascades automatically to chunks table)
        cursor.execute("DELETE FROM materials WHERE id = %s", (id,))
        conn.commit()
        cursor.close()

        # 5. Delete physical binary file from server disk
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted storage PDF: {file_path}")

        logger.info(f"Completely deleted material {id} and purged from FAISS.")
        return (
            jsonify({"message": "Document successfully deleted and vectors purged."}),
            200,
        )

    except Exception as e:
        logger.error(
            f"Failure during deletion of material {id}: {e}", exc_info=True
        )
        return jsonify({"error": "Failed to execute document deletion"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# ADMIN ROUTE: POST /api/v1/admin/materials/upload
# ==========================================
@app.route("/api/v1/admin/materials/upload", methods=["POST"])
def upload_material():
    """
    Step 1 of the document creation wizard. Saves PDF file binary to storage,
    checks file size limit (25 MB), and inserts temporary inactive tracking record.
    """
    # Guardrail check using Request Header
    if request.content_length and request.content_length > config.MAX_FILE_SIZE_BYTES + 100 * 1024:
        return jsonify({"error": "File size exceeds strict 25 MB limit."}), 413

    if "file" not in request.files:
        return jsonify({"error": "No file payload in request."}), 400

    file = request.files["file"]
    filename = file.filename or ""
    if filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not filename.lower().endswith(".pdf"):
        return (
            jsonify({"error": "Invalid file format. Only PDF files are allowed."}),
            400,
        )

    # Read binary header signature to verify standard PDF (%PDF-)
    try:
        magic_header = file.read(4)
        file.seek(0)
        if magic_header != b"%PDF":
            return (
                jsonify({"error": "File binary is not a valid PDF document."}),
                400,
            )
    except Exception as e:
        return (
            jsonify({"error": f"Failed to parse file magic signature: {str(e)}"}),
            400,
        )

    # Ensure precise file size limits programmatically
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > config.MAX_FILE_SIZE_BYTES:
            return jsonify({"error": "File size exceeds strict 25 MB limit."}), 413
    except Exception as e:
        return (
            jsonify({"error": f"Failed to run file size check: {str(e)}"}),
            400,
        )

    tracking_token = str(uuid.uuid4())
    dest_path = os.path.join(config.PDFS_DIR, f"{tracking_token}.pdf")

    conn = None
    try:
        file.save(dest_path)

        # Store inactive tracking reference record (includes file_size_bytes)
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO materials (id, file_name, file_path, file_size_bytes, annotation, index_id, status)
            VALUES (%s, %s, %s, %s, NULL, NULL, 'inactive')
        """
        cursor.execute(insert_sql, (tracking_token, file.filename, dest_path, file_size))
        conn.commit()
        cursor.close()

        logger.info(
            f"Uploaded file saved: {file.filename} -> Token: {tracking_token} "
            f"({file_size} bytes)"
        )
        return (
            jsonify(
                {"tracking_token": tracking_token, "file_name": file.filename}
            ),
            201,
        )

    except Exception as e:
        logger.error(f"Error saving uploaded material: {e}", exc_info=True)
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return jsonify({"error": "Failed to store uploaded document"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# ADMIN ROUTE: POST /api/v1/admin/materials/finalize
# ==========================================
@app.route("/api/v1/admin/materials/finalize", methods=["POST"])
def finalize_material():
    """
    Step 2 & 3 of the wizard. Sets annotations (max 2,000 characters),
    and starts the asynchronous vector parsing process immediately in a background thread.
    """
    data = request.get_json() or {}
    tracking_token = data.get("tracking_token")
    annotation = data.get("annotation")

    if not tracking_token:
        return jsonify({"error": "tracking_token is required."}), 400
    if annotation is None:
        return jsonify({"error": "annotation is required."}), 400

    # Hard text length constraints
    if len(annotation) > config.MAX_ANNOTATION_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Annotation length exceeds the maximum limit of "
                    f"{config.MAX_ANNOTATION_LENGTH} characters."
                }
            ),
            400,
        )

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Fetch inactive record to ensure path is valid
        cursor.execute(
            "SELECT id, file_path FROM materials WHERE id = %s",
            (tracking_token,),
        )
        material = cursor.fetchone()
        if not material:
            cursor.close()
            return (
                jsonify(
                    {"error": "No draft upload found with this tracking token."}
                ),
                404,
            )

        file_path = material["file_path"]

        # Save complete annotation
        cursor.execute(
            "UPDATE materials SET annotation = %s WHERE id = %s",
            (annotation, tracking_token),
        )
        conn.commit()
        cursor.close()
        conn.close()
        conn = None

        # Kickoff async parser and embedding builder
        start_async_ingestion(tracking_token, file_path, vector_store)

        logger.info(
            f"Finalized document parameters for {tracking_token}. "
            "Async ingestion started."
        )
        return (
            jsonify(
                {
                    "message": "Document properties finalized. Ingestion pipeline successfully scheduled.",
                    "tracking_token": tracking_token,
                }
            ),
            202,
        )

    except Exception as e:
        logger.error(f"Error finalizing document metadata: {e}", exc_info=True)
        return jsonify({"error": "Failed to finalize document parameters"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# CHAT ROUTE: POST /api/v1/chat/query
# ==========================================
@app.route("/api/v1/chat/query", methods=["POST"])
def chat_query():
    """
    Unified RAG query generation endpoint. Retrieves top chunks
    by similarity (threshold >= 0.30) and forwards to Ollama LLM.
    """
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    stream = data.get("stream", False)

    if not query_text:
        return jsonify({"error": "query parameter cannot be empty."}), 400

    conn = None
    try:
        # 1. Vectorize query using same embedding model
        import numpy as np

        model = get_embedding_model()
        query_vector = np.array(model.encode(query_text))

        # 2. Query local FAISS index
        # Retrieve large K to filter active items downstream
        distances, indices = vector_store.search_vectors(
            query_vector, k=config.SEARCH_TOP_K
        )

        # 3. Apply Cosine Similarity threshold
        matching_ids = []

        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and score >= config.SIMILARITY_THRESHOLD:
                matching_ids.append(int(idx))

        # Check if zero matching chunks found
        if not matching_ids:
            return (
                jsonify(
                    {
                        "response": "I could not find relevant information in the knowledge base."
                    }
                ),
                200,
            )

        # 4. Perform database inner join to verify document is 'active'
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        placeholders = ", ".join(["%s"] * len(matching_ids))
        # FAISS IDs are verified integers
        safe_ids = matching_ids
        order_by_ids = ", ".join(map(str, safe_ids))

        # Select top chunks belonging exclusively to active documents,
        # preserving FAISS similarity rank
        query_sql = f"""
            SELECT c.text_content, c.faiss_id, m.file_name
            FROM chunks c
            INNER JOIN materials m ON c.material_id = m.id
            WHERE m.status = 'active' AND c.faiss_id IN ({placeholders})
            ORDER BY FIELD(c.faiss_id, {order_by_ids})
            LIMIT {config.CONTEXT_CHUNKS_LIMIT}
        """
        cursor.execute(query_sql, tuple(safe_ids))
        active_chunks = cursor.fetchall()

        cursor.close()
        conn.close()
        conn = None

        if not active_chunks:
            return (
                jsonify(
                    {
                        "response": "I could not find relevant information in the knowledge base."
                    }
                ),
                200,
            )

        # 5. Synthesize clean context prompt block
        context_blocks = []
        for idx, chunk in enumerate(active_chunks):
            context_blocks.append(
                f"Source: {chunk['file_name']}\nContent: {chunk['text_content']}"
            )

        system_prompt = (
            "You are a helpful and precise local AI assistant. "
            "Answer the user's question using only the retrieved context below.\n"
            "If the retrieved context does not contain the answer, say exactly: "
            "'I could not find relevant information in the knowledge base.'\n\n"
            "Context Information:\n"
            "---------------------\n"
            f"{chr(10).join(context_blocks)}\n"
            "---------------------\n\n"
            f"Question: {query_text}\n"
            "Answer:"
        )

        # 6. Call Ollama Client API

        if stream:
            # Output token streaming response
            def token_stream_generator():
                for token in query_ollama_stream(system_prompt):
                    yield token

            return Response(token_stream_generator(), mimetype="text/plain")
        else:
            response_text = query_ollama_non_stream(system_prompt)
            return jsonify({"response": response_text}), 200

    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        return jsonify({"error": "Failed to complete AI retrieval loop"}), 500
    finally:
        if conn:
            conn.close()


# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    logger.info(f"Starting local Flask RAG Server on port {config.FLASK_PORT}...")
    app.run(host="0.0.0.0", port=config.FLASK_PORT, debug=False)
