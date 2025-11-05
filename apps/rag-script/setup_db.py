#!/usr/bin/env python3
"""
Database setup for AgriMind RAG System
Creates the necessary tables and extensions for vector storage
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def setup_database():
    """Initialize the database with required extensions and tables"""
    
    # Database connection parameters
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'agrimind'),
        'user': os.getenv('DB_USER', 'agrimind'),
        'password': os.getenv('DB_PASSWORD', 'agrimind')
    }
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create pgvector extension
        logger.info("Creating pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table
        logger.info("Creating documents table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(255) PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type VARCHAR(100),
                metadata JSONB,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for vector similarity search
        logger.info("Creating vector index...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        # Create index for metadata search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_metadata_idx 
            ON documents USING GIN (metadata);
        """)
        
        # Create index for document type
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS documents_type_idx 
            ON documents (doc_type);
        """)
        
        # Create function to update timestamp
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Create trigger for updated_at
        cursor.execute("""
            CREATE TRIGGER update_documents_updated_at 
            BEFORE UPDATE ON documents 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
        
        logger.info("Database setup completed successfully!")
        
        # Verify setup
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()[0]
        logger.info(f"Documents table ready. Current document count: {count}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_database()
