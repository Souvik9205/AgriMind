#!/usr/bin/env python3
"""
Vector store module for AgriMind RAG System
Handles document storage and retrieval from PostgreSQL with pgvector
"""

import os
import json
import psycopg2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class Document:
    """Document data class"""
    id: str
    title: str
    content: str
    doc_type: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
    """Search result data class"""
    document: Document
    similarity_score: float
    rank: int

class VectorStore:
    """Vector database operations using PostgreSQL with pgvector"""
    
    def __init__(self):
        self.db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'agrimind'),
            'user': os.getenv('DB_USER', 'agrimind'),
            'password': os.getenv('DB_PASSWORD', 'agrimind')
        }
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)
    
    def insert_document(self, document: Document) -> bool:
        """Insert a document into the vector store"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert embedding to list for JSON serialization
                embedding_list = document.embedding.tolist() if document.embedding is not None else None
                
                cursor.execute("""
                    INSERT INTO documents (id, title, content, doc_type, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        doc_type = EXCLUDED.doc_type,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    document.id,
                    document.title,
                    document.content,
                    document.doc_type,
                    json.dumps(document.metadata),
                    embedding_list
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert document {document.id}: {e}")
            return False
    
    def insert_documents(self, documents: List[Document]) -> int:
        """Insert multiple documents into the vector store"""
        successful_inserts = 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for document in documents:
                    try:
                        # Convert embedding to list for JSON serialization
                        embedding_list = document.embedding.tolist() if document.embedding is not None else None
                        
                        cursor.execute("""
                            INSERT INTO documents (id, title, content, doc_type, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                doc_type = EXCLUDED.doc_type,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            document.id,
                            document.title,
                            document.content,
                            document.doc_type,
                            json.dumps(document.metadata),
                            embedding_list
                        ))
                        
                        successful_inserts += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to insert document {document.id}: {e}")
                        continue
                
                conn.commit()
                logger.info(f"Successfully inserted {successful_inserts} out of {len(documents)} documents")
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
        
        return successful_inserts
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, 
                         doc_type_filter: Optional[str] = None,
                         metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform similarity search using vector embeddings"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert embedding to proper format for PostgreSQL vector type
                embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
                
                # Build the query
                query = """
                    SELECT id, title, content, doc_type, metadata, embedding,
                           1 - (embedding <=> %s::vector) as similarity_score
                    FROM documents
                    WHERE embedding IS NOT NULL
                """
                params = [embedding_str]
                
                # Add filters
                if doc_type_filter:
                    query += " AND doc_type = %s"
                    params.append(doc_type_filter)
                
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        query += f" AND metadata->>%s = %s"
                        params.extend([key, str(value)])
                
                # Add similarity threshold and ordering
                query += f" AND (1 - (embedding <=> %s::vector)) >= %s"
                params.extend([embedding_str, self.similarity_threshold])
                
                query += " ORDER BY similarity_score DESC LIMIT %s"
                params.append(top_k)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                search_results = []
                for rank, row in enumerate(results, 1):
                    doc_id, title, content, doc_type, metadata_json, embedding_list, similarity_score = row
                    
                    # Parse metadata - it might already be a dict if using JSONB
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json) if metadata_json else {}
                    else:
                        metadata = metadata_json if metadata_json else {}
                    
                    # Create document
                    document = Document(
                        id=doc_id,
                        title=title,
                        content=content,
                        doc_type=doc_type,
                        metadata=metadata,
                        embedding=np.array(embedding_list) if embedding_list else None
                    )
                    
                    search_results.append(SearchResult(
                        document=document,
                        similarity_score=float(similarity_score),
                        rank=rank
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, title, content, doc_type, metadata, embedding
                    FROM documents
                    WHERE id = %s
                """, (doc_id,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                doc_id, title, content, doc_type, metadata_json, embedding_list = result
                
                # Parse metadata
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                return Document(
                    id=doc_id,
                    title=title,
                    content=content,
                    doc_type=doc_type,
                    metadata=metadata,
                    embedding=np.array(embedding_list) if embedding_list else None
                )
                
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the store"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total documents
                cursor.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                # Documents with embeddings
                cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
                docs_with_embeddings = cursor.fetchone()[0]
                
                # Document types
                cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
                doc_types = dict(cursor.fetchall())
                
                # Documents by region (from metadata)
                cursor.execute("""
                    SELECT metadata->>'region' as region, COUNT(*)
                    FROM documents
                    WHERE metadata->>'region' IS NOT NULL
                    GROUP BY metadata->>'region'
                """)
                regions = dict(cursor.fetchall())
                
                return {
                    'total_documents': total_docs,
                    'documents_with_embeddings': docs_with_embeddings,
                    'document_types': doc_types,
                    'regions': regions,
                    'embedding_coverage': docs_with_embeddings / total_docs if total_docs > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
