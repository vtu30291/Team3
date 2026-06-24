"""
FAISS vector index manager for the local RAG pipeline.
Uses IndexIDMap wrapping IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).
Thread-safe via a reentrant lock.
"""

import os
import faiss
import numpy as np
import threading
import logging

import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir or config.STORAGE_DIR
        self.vector_store_dir = os.path.join(self.storage_dir, "vector_store")
        self.index_path = os.path.join(self.vector_store_dir, "index.faiss")
        self.lock = threading.Lock()
        self.index = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        os.makedirs(self.vector_store_dir, exist_ok=True)
        with self.lock:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    if self.index is None:
                        logger.warning(
                            f"faiss.read_index returned None for {self.index_path}. "
                            "Creating new index."
                        )
                        self._create_new_index()
                    else:
                        logger.info(
                            f"Loaded existing FAISS index from {self.index_path}. "
                            f"Total vectors: {self.index.ntotal}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error reading FAISS index, creating new index: {e}"
                    )
                    self._create_new_index()
            else:
                self._create_new_index()

    def _create_new_index(self):
        # IndexFlatIP (Inner Product) with L2 normalized vectors gives cosine similarity
        sub_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self.index = faiss.IndexIDMap(sub_index)
        logger.info("Initialized new FAISS IndexIDMap with IndexFlatIP backend.")

    def save_index(self):
        """Persist the FAISS index to disk. Assumes lock is already held by caller."""
        try:
            assert self.index is not None
            faiss.write_index(self.index, self.index_path)
            logger.info(
                f"Successfully wrote FAISS index to {self.index_path}. "
                f"Total vectors: {self.index.ntotal}"
            )
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise

    def add_vectors(self, embeddings: np.ndarray, ids: np.ndarray):
        """Add L2-normalized embeddings to the index with specific integer IDs."""
        if embeddings.shape[1] != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embeddings dimension must be {config.EMBEDDING_DIMENSION}, "
                f"got {embeddings.shape[1]}"
            )

        # Copy to avoid modifying caller's array
        embeddings = embeddings.astype("float32").copy()
        faiss.normalize_L2(embeddings)

        ids = np.array(ids, dtype=np.int64)

        with self.lock:
            assert self.index is not None
            self.index.add_with_ids(embeddings, ids)
            self.save_index()

    def remove_vectors(self, ids: np.ndarray | list):
        """
        Remove specific IDs from the index.
        BUG FIX: FAISS remove_ids returns the new index size, not the count of removed
        items. We now calculate actual removed count from ntotal before and after.
        """
        ids = np.array(ids, dtype=np.int64)
        with self.lock:
            assert self.index is not None, "FAISS index is not initialized."
            count_before = self.index.ntotal
            self.index.remove_ids(ids)
            count_after = self.index.ntotal
            removed_count = count_before - count_after
            self.save_index()
            logger.info(
                f"Removed {removed_count} vectors from index. "
                f"Index size: {count_before} -> {count_after}"
            )
            return removed_count

    def search_vectors(self, query_embedding: np.ndarray, k: int = 3):
        """Search for top k nearest neighbors. Returns (distances, indices)."""
        query_embedding = query_embedding.astype("float32").copy()
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        faiss.normalize_L2(query_embedding)

        with self.lock:
            assert self.index is not None
            if self.index.ntotal == 0:
                logger.warning("Search query on empty FAISS index.")
                return np.array([[]]), np.array([[]])

            distances, indices = self.index.search(query_embedding, k)
            return distances, indices

    def get_total_vectors(self):
        with self.lock:
            return self.index.ntotal if self.index else 0
