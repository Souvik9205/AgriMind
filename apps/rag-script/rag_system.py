#!/usr/bin/env python3
"""
Main RAG System for AgriMind
Orchestrates the entire RAG pipeline
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from retriever import DocumentRetriever
from llm_client import LLMClient, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self):
        """Initialize the RAG system components"""
        logger.info("Initializing AgriMind RAG System...")
        
        try:
            # Initialize components
            self.embedding_generator = EmbeddingGenerator()
            self.vector_store = VectorStore()
            self.retriever = DocumentRetriever(self.vector_store, self.embedding_generator)
            self.llm_client = LLMClient()
            
            logger.info("RAG System initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, 
              use_reranking: bool = True, diverse_results: bool = False) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User's question
            filters: Optional filters for document retrieval
            use_reranking: Whether to use document reranking
            diverse_results: Whether to ensure diverse document types in results
        
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Retrieve relevant documents
            if diverse_results:
                documents = self.retriever.get_diverse_results(question)
            elif use_reranking:
                documents = self.retriever.retrieve_with_reranking(question, filters)
            else:
                documents = self.retriever.retrieve_documents(question, filters)
            
            if not documents:
                logger.warning("No relevant documents found for query")
                return RAGResponse(
                    answer="I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing your question or ask about West Bengal agriculture, market prices, or farming practices.",
                    sources=[],
                    confidence=0.0,
                    context_used="No relevant context found"
                )
            
            # Generate response using LLM
            response = self.llm_client.generate_response(question, documents)
            
            logger.info(f"Query processed successfully. Retrieved {len(documents)} documents, confidence: {response.confidence:.3f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again later.",
                sources=[],
                confidence=0.0,
                context_used=""
            )
    
    def query_market_data(self, question: str, region: Optional[str] = None) -> RAGResponse:
        """Query specifically for market-related information"""
        filters = {'doc_type': 'market_summary'}
        if region:
            filters['metadata'] = {'region': region}
        
        return self.query(question, filters=filters, diverse_results=False)
    
    def query_agricultural_guides(self, question: str) -> RAGResponse:
        """Query specifically for agricultural guides and ICAR reports"""
        filters = {'doc_type': 'pdf_content'}
        return self.query(question, filters=filters, diverse_results=False)
    
    def query_west_bengal_specific(self, question: str) -> RAGResponse:
        """Query for West Bengal specific information"""
        filters = {'metadata': {'region': 'West Bengal'}}
        return self.query(question, filters=filters, diverse_results=True)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.vector_store.get_statistics()
            
            stats = {
                'vector_store': vector_stats,
                'embedding_model': self.embedding_generator.model_name,
                'embedding_dimension': self.embedding_generator.get_dimension(),
                'llm_model': self.llm_client.model_name,
                'configuration': {
                    'top_k_results': self.retriever.top_k,
                    'similarity_threshold': self.retriever.similarity_threshold,
                    'max_context_length': self.llm_client.max_context_length
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all components"""
        health = {
            'status': 'healthy',
            'components': {},
            'issues': []
        }
        
        try:
            # Check vector store
            doc_count = self.vector_store.get_document_count()
            health['components']['vector_store'] = {
                'status': 'healthy' if doc_count > 0 else 'warning',
                'document_count': doc_count
            }
            if doc_count == 0:
                health['issues'].append("No documents found in vector store")
            
            # Check embedding generator
            test_embedding = self.embedding_generator.generate_embedding("test")
            health['components']['embedding_generator'] = {
                'status': 'healthy' if test_embedding is not None else 'error',
                'model': self.embedding_generator.model_name
            }
            
            # Check LLM client
            try:
                test_response = self.llm_client.generate_simple_response("Hello")
                health['components']['llm_client'] = {
                    'status': 'healthy' if test_response else 'error',
                    'model': self.llm_client.model_name
                }
            except Exception as e:
                health['components']['llm_client'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['issues'].append(f"LLM client error: {e}")
            
            # Overall status
            if any(comp.get('status') == 'error' for comp in health['components'].values()):
                health['status'] = 'error'
            elif any(comp.get('status') == 'warning' for comp in health['components'].values()):
                health['status'] = 'warning'
            
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Health check failed: {e}")
        
        return health


def main():
    """Demo function for testing the RAG system"""
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Perform health check
        health = rag.health_check()
        print(f"System Health: {health['status']}")
        if health['issues']:
            print("Issues:", health['issues'])
        
        # Get system stats
        stats = rag.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"Total Documents: {stats.get('vector_store', {}).get('total_documents', 0)}")
        print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
        print(f"LLM Model: {stats.get('llm_model', 'Unknown')}")
        
        # Example queries
        test_queries = [
            "What are the best crops to grow in West Bengal during Kharif season?",
            "What are the current market prices for rice in Kolkata?",
            "How can farmers improve their crop yield in West Bengal?",
            "What are the weather considerations for farming in West Bengal?"
        ]
        
        print("\n" + "="*50)
        print("DEMO QUERIES")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 40)
            
            response = rag.query(query)
            print(f"Answer: {response.answer[:200]}...")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Sources: {len(response.sources)} documents")
            
            if response.sources:
                print("Top sources:")
                for j, source in enumerate(response.sources[:2], 1):
                    print(f"  {j}. {source['title']} (score: {source['similarity_score']:.3f})")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
