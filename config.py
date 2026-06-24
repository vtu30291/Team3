"""
Centralized configuration for the Local RAG API Backend.
All environment variables and application constants are defined here.
Modules should import from this file instead of calling os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# Flask Server
# ==========================================
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))

# ==========================================
# MySQL Database
# ==========================================
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_USER: str = os.getenv("DB_USER", "root")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_NAME: str = os.getenv("DB_NAME", "rag_db")
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
DB_POOL_NAME: str = "rag_pool"

# ==========================================
# Ollama LLM
# ==========================================
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ==========================================
# Storage
# ==========================================
STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")
PDFS_DIR: str = os.path.join(STORAGE_DIR, "pdfs")
VECTOR_STORE_DIR: str = os.path.join(STORAGE_DIR, "vector_store")

# ==========================================
# Document & Chunking Constraints
# ==========================================
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
CHUNK_SIZE: int = 500  # tokens per chunk
CHUNK_OVERLAP: int = 50  # overlapping tokens between chunks
MAX_ANNOTATION_LENGTH: int = 2000  # characters

# ==========================================
# Embedding Model
# ==========================================
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ==========================================
# RAG Search
# ==========================================
SIMILARITY_THRESHOLD: float = 0.30
SEARCH_TOP_K: int = 50  # candidates retrieved from FAISS
CONTEXT_CHUNKS_LIMIT: int = 3  # chunks sent to LLM
