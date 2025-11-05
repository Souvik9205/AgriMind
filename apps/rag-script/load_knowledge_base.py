#!/usr/bin/env python3
"""
Knowledge base loader for AgriMind RAG System
Loads processed documents into the vector database
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from embeddings import EmbeddingGenerator
from vector_store import VectorStore, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class KnowledgeBaseLoader:
    """Loads knowledge base documents into the vector store"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.kb_path = Path(os.getenv('KNOWLEDGE_BASE_PATH', '../../packages/kb/dist'))
        
    def load_all_documents(self) -> int:
        """Load all documents from the knowledge base"""
        logger.info("Loading all documents from knowledge base...")
        
        total_loaded = 0
        
        # Load RAG-ready documents
        rag_ready_path = self.kb_path / "rag_ready" / "rag_documents.json"
        if rag_ready_path.exists():
            total_loaded += self._load_rag_documents(rag_ready_path)
        else:
            logger.warning(f"RAG documents file not found: {rag_ready_path}")
            # Try to load from processed data
            total_loaded += self._load_from_processed_data()
        
        logger.info(f"Knowledge base loading completed. Total documents loaded: {total_loaded}")
        return total_loaded
    
    def _load_rag_documents(self, file_path: Path) -> int:
        """Load documents from RAG-ready JSON file"""
        logger.info(f"Loading RAG documents from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            documents = []
            logger.info(f"Processing {len(documents_data)} documents...")
            
            for doc_data in tqdm(documents_data, desc="Processing documents"):
                try:
                    # Create document object
                    document = Document(
                        id=doc_data['id'],
                        title=doc_data['title'],
                        content=doc_data['content'],
                        doc_type=doc_data['type'],
                        metadata=doc_data.get('metadata', {})
                    )
                    
                    # Generate embedding
                    embedding = self.embedding_generator.generate_embedding(document.content)
                    document.embedding = embedding
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Insert documents into vector store
            loaded_count = self.vector_store.insert_documents(documents)
            logger.info(f"Successfully loaded {loaded_count} RAG documents")
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load RAG documents: {e}")
            return 0
    
    def _load_from_processed_data(self) -> int:
        """Load documents from processed data files (fallback)"""
        logger.info("Loading from processed data files...")
        
        total_loaded = 0
        
        # Load market data
        market_file = self.kb_path / "west_bengal_market_data.json"
        if market_file.exists():
            total_loaded += self._load_market_data(market_file)
        
        # Load PDF processing results
        pdf_file = self.kb_path / "pdf_processing_results.json"
        if pdf_file.exists():
            total_loaded += self._load_pdf_data(pdf_file)
        
        return total_loaded
    
    def _load_market_data(self, file_path: Path) -> int:
        """Load market data documents"""
        logger.info(f"Loading market data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                market_data = json.load(f)
            
            documents = []
            
            # Create document for West Bengal summary
            wb_summary = market_data.get('west_bengal_summary', {})
            if wb_summary:
                content = self._format_market_summary(wb_summary, 'West Bengal')
                document = Document(
                    id='wb_market_summary',
                    title='West Bengal Agricultural Market Summary',
                    content=content,
                    doc_type='market_summary',
                    metadata={
                        'region': 'West Bengal',
                        'data_type': 'market_summary',
                        'source': 'market_data_csv'
                    }
                )
                document.embedding = self.embedding_generator.generate_embedding(content)
                documents.append(document)
            
            # Create documents for districts
            district_data = market_data.get('district_wise_data', {})
            for district_name, district_info in district_data.items():
                content = self._format_district_data(district_name, district_info)
                document = Document(
                    id=f'district_{district_name.lower().replace(" ", "_")}',
                    title=f'{district_name} District Market Information',
                    content=content,
                    doc_type='district_market_data',
                    metadata={
                        'region': district_name,
                        'state': 'West Bengal',
                        'data_type': 'district_market_data',
                        'source': 'market_data_csv'
                    }
                )
                document.embedding = self.embedding_generator.generate_embedding(content)
                documents.append(document)
            
            loaded_count = self.vector_store.insert_documents(documents)
            logger.info(f"Loaded {loaded_count} market data documents")
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return 0
    
    def _load_pdf_data(self, file_path: Path) -> int:
        """Load PDF content documents"""
        logger.info(f"Loading PDF data from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                pdf_data = json.load(f)
            
            documents = []
            
            for pdf_id, pdf_info in pdf_data.items():
                if 'error' in pdf_info:
                    continue
                
                content = pdf_info.get('content', '')
                if not content:
                    continue
                
                # Split content into chunks
                chunks = self._split_text_into_chunks(content)
                
                for i, chunk in enumerate(chunks):
                    document = Document(
                        id=f'{pdf_id}_chunk_{i}',
                        title=f'{pdf_info.get("filename", pdf_id)} - Part {i+1}',
                        content=chunk,
                        doc_type='pdf_content',
                        metadata={
                            'source_file': pdf_info.get('filename', pdf_id),
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'data_type': 'agricultural_document'
                        }
                    )
                    document.embedding = self.embedding_generator.generate_embedding(chunk)
                    documents.append(document)
            
            loaded_count = self.vector_store.insert_documents(documents)
            logger.info(f"Loaded {loaded_count} PDF content documents")
            
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load PDF data: {e}")
            return 0
    
    def _format_market_summary(self, summary: Dict, region: str) -> str:
        """Format market summary data"""
        content = f"Agricultural Market Summary for {region}\n\n"
        content += f"Total market records: {summary.get('total_records', 0)}\n"
        content += f"Number of districts: {summary.get('unique_districts', 0)}\n"
        content += f"Number of commodities: {summary.get('unique_commodities', 0)}\n"
        content += f"Number of markets: {summary.get('unique_markets', 0)}\n\n"
        
        price_summary = summary.get('price_summary', {})
        if price_summary:
            content += "Price Information:\n"
            content += f"Average minimum price: ₹{price_summary.get('avg_min_price', 0):.2f}\n"
            content += f"Average maximum price: ₹{price_summary.get('avg_max_price', 0):.2f}\n"
            content += f"Average modal price: ₹{price_summary.get('avg_modal_price', 0):.2f}\n"
        
        return content
    
    def _format_district_data(self, district_name: str, district_info: Dict) -> str:
        """Format district-specific data"""
        content = f"Agricultural Market Information for {district_name} District, West Bengal\n\n"
        content += f"Total market records: {district_info.get('total_records', 0)}\n"
        content += f"Average modal price: ₹{district_info.get('avg_modal_price', 0):.2f}\n\n"
        
        markets = district_info.get('markets', [])
        if markets:
            content += f"Markets in {district_name}:\n"
            for market in markets[:10]:
                content += f"- {market}\n"
            if len(markets) > 10:
                content += f"... and {len(markets) - 10} more markets\n"
        
        return content
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + max_chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks
    
    def reload_knowledge_base(self) -> int:
        """Reload the entire knowledge base (clear and reload)"""
        logger.info("Reloading knowledge base...")
        
        # Note: This is a simple reload. In production, you might want to
        # implement more sophisticated updating strategies
        
        return self.load_all_documents()


def main():
    """Main function to load knowledge base"""
    try:
        loader = KnowledgeBaseLoader()
        
        # Check if knowledge base path exists
        if not loader.kb_path.exists():
            logger.error(f"Knowledge base path not found: {loader.kb_path}")
            logger.info("Please ensure the knowledge base has been processed first.")
            return
        
        # Load all documents
        total_loaded = loader.load_all_documents()
        
        if total_loaded > 0:
            # Show statistics
            stats = loader.vector_store.get_statistics()
            logger.info("Knowledge base loading completed!")
            logger.info(f"Total documents in vector store: {stats.get('total_documents', 0)}")
            logger.info(f"Documents with embeddings: {stats.get('documents_with_embeddings', 0)}")
            logger.info(f"Document types: {stats.get('document_types', {})}")
        else:
            logger.error("No documents were loaded. Please check the knowledge base files.")
        
    except Exception as e:
        logger.error(f"Knowledge base loading failed: {e}")


if __name__ == "__main__":
    main()
