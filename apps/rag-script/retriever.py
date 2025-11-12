#!/usr/bin/env python3
"""
Document retriever for AgriMind RAG System
Handles the retrieval and ranking of relevant documents
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
from embeddings import EmbeddingGenerator
from vector_store import VectorStore, SearchResult, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DocumentRetriever:
    """Handles document retrieval and ranking for RAG"""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = int(os.getenv('TOP_K_RESULTS', '5'))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))  # Lowered from 0.7 to 0.5
    
    def retrieve_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Extract filters
            doc_type_filter = filters.get('doc_type') if filters else None
            metadata_filter = filters.get('metadata') if filters else None
            
            # Perform similarity search
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                doc_type_filter=doc_type_filter,
                metadata_filter=metadata_filter
            )
            
            # Convert to dictionary format
            documents = []
            for result in search_results:
                doc_dict = {
                    'id': result.document.id,
                    'title': result.document.title,
                    'content': result.document.content,
                    'doc_type': result.document.doc_type,
                    'metadata': result.document.metadata,
                    'similarity_score': result.similarity_score,
                    'rank': result.rank
                }
                documents.append(doc_dict)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def retrieve_with_reranking(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents with additional reranking based on query relevance"""
        # First, get initial results
        documents = self.retrieve_documents(query, filters)
        
        if not documents:
            return documents
        
        # Apply additional ranking factors
        reranked_documents = self._rerank_documents(query, documents)
        
        return reranked_documents
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply additional ranking factors to documents"""
        query_lower = query.lower()
        
        # Define ranking factors
        def calculate_score(doc):
            base_score = doc['similarity_score']
            
            # Boost for exact keyword matches in title
            title_boost = 0.1 if any(word in doc['title'].lower() for word in query_lower.split()) else 0
            
            # Boost for specific document types based on query
            type_boost = 0
            doc_type = doc['doc_type']
            
            if 'market' in query_lower or 'price' in query_lower:
                if 'market' in doc_type:
                    type_boost = 0.15
            elif 'crop' in query_lower or 'farming' in query_lower or 'agriculture' in query_lower:
                if 'pdf_content' in doc_type:
                    type_boost = 0.1
            
            # Boost for West Bengal specific content
            wb_boost = 0
            metadata = doc.get('metadata', {})
            if metadata.get('region') == 'West Bengal' or metadata.get('is_west_bengal_specific'):
                wb_boost = 0.05
            
            # Boost for recent or ICAR content
            source_boost = 0
            source_file = metadata.get('source_file', '').lower()
            if 'icar' in source_file:
                source_boost = 0.1
            elif '2024' in source_file or '2025' in source_file:
                source_boost = 0.05
            
            # Calculate final score
            final_score = base_score + title_boost + type_boost + wb_boost + source_boost
            return min(final_score, 1.0)  # Cap at 1.0
        
        # Calculate new scores and sort
        for doc in documents:
            doc['reranked_score'] = calculate_score(doc)
        
        # Sort by reranked score
        reranked_documents = sorted(documents, key=lambda x: x['reranked_score'], reverse=True)
        
        # Update ranks
        for i, doc in enumerate(reranked_documents):
            doc['rank'] = i + 1
        
        return reranked_documents
    
    def retrieve_by_type(self, query: str, doc_type: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents of a specific type"""
        filters = {'doc_type': doc_type}
        
        # Temporarily override top_k if specified
        original_top_k = self.top_k
        if top_k:
            self.top_k = top_k
        
        try:
            documents = self.retrieve_documents(query, filters)
            return documents
        finally:
            self.top_k = original_top_k
    
    def retrieve_by_region(self, query: str, region: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve documents for a specific region"""
        filters = {'metadata': {'region': region}}
        
        # Temporarily override top_k if specified
        original_top_k = self.top_k
        if top_k:
            self.top_k = top_k
        
        try:
            documents = self.retrieve_documents(query, filters)
            return documents
        finally:
            self.top_k = original_top_k
    
    def get_diverse_results(self, query: str, max_per_type: int = 2) -> List[Dict[str, Any]]:
        """Get diverse results by limiting documents per type"""
        all_documents = self.retrieve_with_reranking(query)
        
        if not all_documents:
            return all_documents
        
        # Group by document type
        type_groups = {}
        for doc in all_documents:
            doc_type = doc['doc_type']
            if doc_type not in type_groups:
                type_groups[doc_type] = []
            type_groups[doc_type].append(doc)
        
        # Select top documents from each type
        diverse_results = []
        for doc_type, docs in type_groups.items():
            # Sort by reranked score and take top max_per_type
            sorted_docs = sorted(docs, key=lambda x: x.get('reranked_score', x['similarity_score']), reverse=True)
            diverse_results.extend(sorted_docs[:max_per_type])
        
        # Sort final results by score
        diverse_results.sort(key=lambda x: x.get('reranked_score', x['similarity_score']), reverse=True)
        
        # Update ranks
        for i, doc in enumerate(diverse_results):
            doc['rank'] = i + 1
        
        return diverse_results[:self.top_k]
