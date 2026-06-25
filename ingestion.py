"""
Document ingestion pipeline: PDF extraction, token-based chunking,
embedding generation, MySQL storage, and FAISS index insertion.
Runs in background threads to avoid blocking Flask request handlers.

This version adds psutil RSS memory logging before and after every
major operation so that Render logs identify the exact operation that
crosses the 512 MB limit.
"""

import gc
import os
import traceback
import numpy as np
import threading
import logging

import config
from db import get_db_connection
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


# ------------------------------------------------------------------ #
# Memory instrumentation helper
# ------------------------------------------------------------------ #
try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())

    def _mem() -> float:
        """Return current RSS memory usage in MB."""
        return _PROC.memory_info().rss / 1024 / 1024

except ImportError:
    def _mem() -> float:
        return -1.0  # psutil not available; sentinel value


def _log_mem(label: str):
    """Emit a standardised RSS log line that is easy to grep in Render."""
    logger.info(f"[MEM] {label}: {_mem():.1f} MB RSS")


# ------------------------------------------------------------------ #
# Embedding model (singleton)
# ------------------------------------------------------------------ #
def get_embedding_model():
    """
    Lazy-load and cache the SentenceTransformer model (thread-safe singleton).
    Only one model instance ever lives in memory.
    """
    global _model
    with _model_lock:
        if _model is None:
            _log_mem("Before SentenceTransformer load")
            logger.info(
                f"Loading SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}'..."
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            _log_mem("After SentenceTransformer load")
            logger.info("SentenceTransformer model loaded successfully.")
        return _model


# ------------------------------------------------------------------ #
# PDF extraction
# ------------------------------------------------------------------ #
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    The fitz.Document is explicitly closed and deleted to free native memory.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    import fitz  # PyMuPDF

    _log_mem("Before PDF extraction")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    del doc
    gc.collect()
    _log_mem("After PDF extraction")
    return text


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #
def semantic_chunking(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list:
    """
    Cuts raw text into overlapping token-spans (~500 tokens each).
    The encoded token list is explicitly freed before returning.
    """
    model = get_embedding_model()
    tokenizer = model.tokenizer

    _log_mem("Before tokenizer.encode (chunking)")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    _log_mem("After tokenizer.encode (chunking)")

    if not tokens:
        return []

    chunks = []
    if len(tokens) <= chunk_size:
        single = tokenizer.decode(tokens, skip_special_tokens=True)
        del tokens
        gc.collect()
        return [single]

    start_idx = 0
    step = chunk_size - overlap

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        del chunk_tokens
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end_idx == len(tokens):
            break
        start_idx += step

    del tokens
    gc.collect()
    _log_mem("After chunking complete (tokens freed)")
    return chunks


# ------------------------------------------------------------------ #
# Single-chunk encoder
# ------------------------------------------------------------------ #
def _encode_one(model, text: str) -> np.ndarray:
    """
    Encode a single chunk of text.

    - batch_size=1         : one sentence at a time inside SentenceTransformer
    - convert_to_tensor=False: returns a plain NumPy array, no live Tensor kept
    - normalize_embeddings : L2-normalise inside encode() — no extra buffer needed
    - show_progress_bar    : off — prevents tqdm buffering overhead
    - torch.no_grad()      : suppresses gradient buffer allocation
    """
    try:
        import torch
        with torch.no_grad():
            vec = model.encode(
                [text],
                batch_size=1,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
    except ImportError:
        vec = model.encode(
            [text],
            batch_size=1,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return np.array(vec, dtype="float32")


# ------------------------------------------------------------------ #
# Main ingestion function
# ------------------------------------------------------------------ #
def process_document_ingestion(
    material_id: str, file_path: str, vector_store: VectorStoreManager
):
    """
    Ingests a PDF document with full psutil RSS memory instrumentation.

    Memory is logged before and after:
      - SentenceTransformer load
      - PDF extraction
      - Chunk generation
      - Every encode() call
      - Every FAISS add_vectors_deferred() call
      - Every MySQL INSERT
      - Before and after conn.commit()

    Key optimisation vs previous versions:
      FAISS save_index() is called ONCE after all vectors are inserted,
      not after every single chunk. This avoids N full index serialisations
      (each of which allocates a write buffer proportional to index size).
    """
    _log_mem("INGESTION START")
    logger.info(
        f"[INGESTION START] material_id={material_id} | file_path={file_path}"
    )

    # ------------------------------------------------------------------ #
    # STEP 1: Extract text from PDF
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 1] Extracting text from PDF for material {material_id}")
        text = extract_text_from_pdf(file_path)   # logs mem inside
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF file.")
        logger.info(
            f"[STEP 1 OK] Extracted {len(text)} characters for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 1 FAILED] PDF extraction failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 2: Chunk text → free raw text immediately
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 2] Running semantic chunking for material {material_id}")
        _log_mem("Before semantic_chunking()")
        chunks = semantic_chunking(text)       # logs mem inside
        del text
        gc.collect()
        _log_mem("After del text + gc.collect()")
        if not chunks:
            raise ValueError("Document was empty or generated zero chunks.")
        total_chunks = len(chunks)
        logger.info(
            f"[STEP 2 OK] Generated {total_chunks} chunks for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 2 FAILED] Chunking failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 3: Open DB, lock, encode + insert + FAISS (deferred save) per chunk
    # ------------------------------------------------------------------ #
    conn = None
    cursor = None
    start_id = None

    try:
        logger.info(
            f"[STEP 3 START] Beginning embedding loop for material {material_id} "
            f"({total_chunks} chunks, batch_size=1, deferred FAISS save)"
        )
        _log_mem("STEP 3 START")

        # Open connection and begin explicit transaction
        conn = get_db_connection()
        conn.autocommit = False
        cursor = conn.cursor()

        # Acquire row-level lock and compute FAISS start ID
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM chunks FOR UPDATE")
        row = cursor.fetchone()
        start_id = row[0] + 1
        logger.info(
            f"[STEP 3] DB lock acquired. FAISS start_id={start_id} "
            f"for material {material_id}"
        )

        model = get_embedding_model()
        insert_query = """
            INSERT INTO chunks (material_id, chunk_index, text_content, faiss_id)
            VALUES (%s, %s, %s, %s)
        """

        for chunk_idx in range(total_chunks):
            chunk_text = chunks[chunk_idx]
            faiss_id = start_id + chunk_idx

            logger.info(
                f"[STEP 3 BATCH {chunk_idx + 1}/{total_chunks}] "
                f"faiss_id={faiss_id} for material {material_id}"
            )

            # 3a: Encode — measure before + after
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} before encode()")
            try:
                batch_embeddings = _encode_one(model, chunk_text)
            except Exception:
                logger.exception(
                    f"[STEP 3 FAILED] encode() failed at chunk "
                    f"{chunk_idx + 1}/{total_chunks} for material {material_id}.\n"
                    + traceback.format_exc()
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} after encode()")

            # 3b: MySQL INSERT — measure before + after
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} before MySQL INSERT")
            try:
                cursor.execute(
                    insert_query,
                    (material_id, chunk_idx, chunk_text, faiss_id),
                )
            except Exception:
                logger.exception(
                    f"[STEP 3 FAILED] MySQL INSERT failed at chunk "
                    f"{chunk_idx + 1}/{total_chunks} for material {material_id}.\n"
                    + traceback.format_exc()
                )
                del batch_embeddings
                gc.collect()
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} after MySQL INSERT")

            # 3c: FAISS add_vectors_deferred (NO disk write yet) — measure before + after
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} before FAISS add_vectors_deferred()")
            try:
                faiss_ids_arr = np.array([faiss_id], dtype=np.int64)
                vector_store.add_vectors_deferred(batch_embeddings, faiss_ids_arr)
            except Exception:
                logger.exception(
                    f"[STEP 3 FAILED] FAISS add_vectors_deferred() failed at chunk "
                    f"{chunk_idx + 1}/{total_chunks} for material {material_id}.\n"
                    + traceback.format_exc()
                )
                del batch_embeddings, faiss_ids_arr
                gc.collect()
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return
            _log_mem(f"STEP 3 BATCH {chunk_idx + 1} after FAISS add_vectors_deferred()")

            # 3d: Free this batch's memory before next iteration
            del batch_embeddings, faiss_ids_arr, chunk_text
            gc.collect()

        # All chunks done — flush FAISS index to disk ONCE
        _log_mem("Before FAISS save_index() (single flush after all vectors)")
        try:
            vector_store.flush_index()
        except Exception:
            logger.exception(
                f"[STEP 3 FAILED] FAISS flush_index() failed for material {material_id}.\n"
                + traceback.format_exc()
            )
            try:
                conn.rollback()
            except Exception:
                pass
            _mark_inactive(material_id)
            return
        _log_mem("After FAISS save_index()")

        # Free the chunks list
        del chunks
        gc.collect()

        logger.info(
            f"[STEP 3 COMPLETE] All {total_chunks} chunks encoded and written to FAISS "
            f"for material {material_id}. Index total: {vector_store.get_total_vectors()}"
        )
        _log_mem("STEP 3 COMPLETE")

    except Exception:
        logger.exception(
            f"[STEP 3 FAILED] Unexpected error in embedding loop for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 4: UPDATE materials — set index_id, chunk_count, status='active'
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 4] Updating materials table to 'active' for material {material_id} "
            f"(chunk_count={total_chunks}, index_id={start_id})"
        )
        update_query = """
            UPDATE materials
            SET index_id = %s, chunk_count = %s, status = 'active'
            WHERE id = %s
        """
        cursor.execute(update_query, (str(start_id), total_chunks, material_id))
        rows_affected = cursor.rowcount
        logger.info(
            f"[STEP 4 OK] UPDATE materials affected {rows_affected} row(s) "
            f"for material {material_id}"
        )
        if rows_affected == 0:
            logger.warning(
                f"[STEP 4 WARN] UPDATE matched 0 rows — "
                f"material {material_id} may not exist in DB!"
            )
    except Exception:
        logger.exception(
            f"[STEP 4 FAILED] UPDATE materials failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 5: Commit the transaction
    # ------------------------------------------------------------------ #
    try:
        _log_mem("Before conn.commit()")
        logger.info(f"[STEP 5] Committing transaction for material {material_id}")
        conn.commit()
        _log_mem("After conn.commit()")
        logger.info(
            f"[STEP 5 OK] Transaction committed successfully for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 5 FAILED] conn.commit() failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # STEP 6: Post-commit verification SELECT
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 6] Verifying DB record for material {material_id}")
        verify_conn = get_db_connection()
        verify_cursor = verify_conn.cursor(dictionary=True)
        verify_cursor.execute(
            "SELECT id, status, chunk_count, index_id FROM materials WHERE id = %s",
            (material_id,),
        )
        record = verify_cursor.fetchone()
        verify_cursor.close()
        verify_conn.close()
        if record:
            logger.info(
                f"[STEP 6 VERIFY] material_id={material_id} -> "
                f"status={record['status']}, chunk_count={record['chunk_count']}, "
                f"index_id={record['index_id']}"
            )
        else:
            logger.warning(
                f"[STEP 6 WARN] material {material_id} not found in DB after commit!"
            )
    except Exception:
        logger.exception(
            f"[STEP 6 WARN] Post-commit verification failed for material {material_id}.\n"
            + traceback.format_exc()
        )

    _log_mem("INGESTION COMPLETE")
    logger.info(
        f"[INGESTION COMPLETE] material_id={material_id} successfully activated."
    )


# ------------------------------------------------------------------ #
# Error recovery helper
# ------------------------------------------------------------------ #
def _mark_inactive(material_id: str):
    """
    Marks the material as 'inactive' after a pipeline failure.
    Always uses a fresh DB connection to avoid inheriting a broken transaction.
    """
    logger.info(
        f"[ROLLBACK] Attempting to mark material {material_id} as inactive..."
    )
    try:
        rollback_conn = get_db_connection()
        rollback_conn.autocommit = False
        rollback_cursor = rollback_conn.cursor()
        rollback_cursor.execute(
            "UPDATE materials SET status = 'inactive' WHERE id = %s",
            (material_id,),
        )
        rollback_conn.commit()
        rollback_cursor.close()
        rollback_conn.close()
        logger.info(
            f"[ROLLBACK OK] material {material_id} marked inactive after failure."
        )
    except Exception:
        logger.exception(
            f"[ROLLBACK FAILED] Could not mark material {material_id} as inactive.\n"
            + traceback.format_exc()
        )


# ------------------------------------------------------------------ #
# Thread launcher
# ------------------------------------------------------------------ #
def start_async_ingestion(
    material_id: str, file_path: str, vector_store: VectorStoreManager
):
    """
    Launches document ingestion in a background thread.
    """
    thread = threading.Thread(
        target=process_document_ingestion,
        args=(material_id, file_path, vector_store),
        daemon=True,
    )
    thread.start()
    return thread
