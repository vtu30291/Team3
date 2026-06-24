"""
Database connection pool manager and migration runner.
Uses mysql.connector.pooling for connection reuse and automatic cleanup.
"""

import os
import logging

# pyrefly: ignore [missing-import]
import mysql.connector
# pyrefly: ignore [missing-import]
from mysql.connector import pooling

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_pool = None


def ensure_database_exists():
    """Create the target database if it does not already exist."""
    try:
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
        )
        cursor = conn.cursor()
        # DB_NAME is validated via config; this is not user-supplied input.
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME}")
        cursor.close()
        conn.close()
        logger.info(f"Database '{config.DB_NAME}' verified or created.")
    except Exception as e:
        logger.error(f"Error ensuring database exists: {e}")
        raise


def init_db():
    """Initialize the connection pool and run schema migrations."""
    global _pool
    ensure_database_exists()

    try:
        _pool = pooling.MySQLConnectionPool(
            pool_name=config.DB_POOL_NAME,
            pool_size=config.DB_POOL_SIZE,
            pool_reset_session=True,
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
        )
        logger.info("Database connection pool initialized successfully.")
        run_migrations()
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


def get_db_connection():
    """Retrieve a connection from the pool (auto-initializes if needed)."""
    global _pool
    if _pool is None:
        init_db()
    assert _pool is not None
    return _pool.get_connection()


def run_migrations():
    """
    Execute schema.sql statements one at a time.
    BUG FIX: Wrapped in try/finally to guarantee connection release back to pool,
    preventing connection pool exhaustion if any statement fails mid-execution.
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        logger.warning(f"schema.sql not found at {schema_path}, skipping migrations.")
        return

    conn = None
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = get_db_connection()
        cursor = conn.cursor()

        # Split schema into individual statements and execute one at a time.
        # mysql-connector-python 8.4.0 does not support the multi=True parameter.
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for statement in statements:
            cursor.execute(statement)

        conn.commit()
        cursor.close()
        logger.info("Database migrations executed successfully.")
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
