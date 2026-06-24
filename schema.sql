-- Database schema initialization for RAG API Backend
-- Idempotent: safe to re-run on every startup via CREATE TABLE IF NOT EXISTS

CREATE TABLE IF NOT EXISTS materials (
    id VARCHAR(36) PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT DEFAULT 0,
    annotation TEXT,
    index_id VARCHAR(255) NULL,
    chunk_count INT DEFAULT 0,
    status ENUM('active', 'inactive') DEFAULT 'inactive',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_material_status (status)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    material_id VARCHAR(36) NOT NULL,
    chunk_index INT NOT NULL,
    text_content TEXT NOT NULL,
    faiss_id INT NOT NULL,
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES materials(id) ON DELETE CASCADE,
    INDEX idx_chunk_faiss_id (faiss_id),
    INDEX idx_chunk_material_id (material_id)
);
