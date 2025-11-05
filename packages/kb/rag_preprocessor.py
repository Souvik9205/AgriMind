#!/usr/bin/env python3
"""
RAG Data Preprocessor for AgriMind
Prepares processed data for RAG system ingestion
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPreprocessor:
    def __init__(self, processed_data_path: str = "./processed"):
        self.processed_path = Path(processed_data_path)
        self.rag_output_path = self.processed_path / "rag_ready"
        self.rag_output_path.mkdir(exist_ok=True)
        
    def create_rag_documents(self) -> List[Dict[str, Any]]:
        """Create document chunks suitable for RAG system"""
        documents = []
        
        # Process market data
        documents.extend(self._create_market_documents())
        
        # Process PDF content
        documents.extend(self._create_pdf_documents())
        
        # Save RAG documents
        rag_file = self.rag_output_path / "rag_documents.json"
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Created {len(documents)} RAG documents. Saved to {rag_file}")
        return documents
    
    def _create_market_documents(self) -> List[Dict[str, Any]]:
        """Create RAG documents from market data"""
        documents = []
        market_file = self.processed_path / "west_bengal_market_data.json"
        
        if not market_file.exists():
            logger.warning("Market data file not found")
            return documents
        
        with open(market_file, 'r') as f:
            market_data = json.load(f)
        
        # Create document for overall West Bengal market summary
        wb_summary = market_data.get('west_bengal_summary', {})
        if wb_summary:
            documents.append({
                'id': 'wb_market_summary',
                'type': 'market_summary',
                'title': 'West Bengal Agricultural Market Summary',
                'content': self._format_market_summary(wb_summary, 'West Bengal'),
                'metadata': {
                    'region': 'West Bengal',
                    'data_type': 'market_summary',
                    'source': 'market_data_csv',
                    'date': datetime.now().isoformat()
                }
            })
        
        # Create document for Kolkata specific data
        kolkata_summary = market_data.get('kolkata_summary', {})
        if kolkata_summary:
            documents.append({
                'id': 'kolkata_market_summary',
                'type': 'market_summary',
                'title': 'Kolkata Agricultural Market Summary',
                'content': self._format_market_summary(kolkata_summary, 'Kolkata'),
                'metadata': {
                    'region': 'Kolkata',
                    'data_type': 'market_summary',
                    'source': 'market_data_csv',
                    'date': datetime.now().isoformat()
                }
            })
        
        # Create documents for each district
        district_data = market_data.get('district_wise_data', {})
        for district_name, district_info in district_data.items():
            documents.append({
                'id': f'district_{district_name.lower().replace(" ", "_")}',
                'type': 'district_market_data',
                'title': f'{district_name} District Market Information',
                'content': self._format_district_data(district_name, district_info),
                'metadata': {
                    'region': district_name,
                    'state': 'West Bengal',
                    'data_type': 'district_market_data',
                    'source': 'market_data_csv',
                    'date': datetime.now().isoformat()
                }
            })
        
        # Create documents for commodity price analysis
        price_analysis = market_data.get('price_analysis', {})
        if price_analysis:
            documents.append({
                'id': 'wb_price_analysis',
                'type': 'price_analysis',
                'title': 'West Bengal Commodity Price Analysis',
                'content': self._format_price_analysis(price_analysis),
                'metadata': {
                    'region': 'West Bengal',
                    'data_type': 'price_analysis',
                    'source': 'market_data_csv',
                    'date': datetime.now().isoformat()
                }
            })
        
        # Create documents for Kolkata markets
        kolkata_markets = market_data.get('kolkata_markets', {})
        for market_name, market_info in kolkata_markets.items():
            documents.append({
                'id': f'kolkata_market_{market_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}',
                'type': 'market_specific_data',
                'title': f'{market_name} - Kolkata Market Details',
                'content': self._format_market_specific_data(market_name, market_info),
                'metadata': {
                    'region': 'Kolkata',
                    'market': market_name,
                    'data_type': 'market_specific_data',
                    'source': 'market_data_csv',
                    'date': datetime.now().isoformat()
                }
            })
        
        return documents
    
    def _create_pdf_documents(self) -> List[Dict[str, Any]]:
        """Create RAG documents from PDF content"""
        documents = []
        pdf_file = self.processed_path / "pdf_processing_results.json"
        
        if not pdf_file.exists():
            logger.warning("PDF processing results file not found")
            return documents
        
        with open(pdf_file, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
        
        for pdf_id, pdf_info in pdf_data.items():
            if 'error' in pdf_info:
                continue
                
            content = pdf_info.get('content', '')
            if not content:
                continue
            
            # Split content into chunks for better RAG performance
            chunks = self._split_text_into_chunks(content, max_chunk_size=1000, overlap=100)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f'{pdf_id}_chunk_{i}',
                    'type': 'pdf_content',
                    'title': f'{pdf_info.get("filename", pdf_id)} - Part {i+1}',
                    'content': chunk,
                    'metadata': {
                        'source_file': pdf_info.get('filename', pdf_id),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'page_count': pdf_info.get('page_count', 0),
                        'data_type': 'agricultural_document',
                        'is_agriculture_related': pdf_info.get('summary', {}).get('is_agriculture_related', False),
                        'is_west_bengal_specific': pdf_info.get('summary', {}).get('is_west_bengal_specific', False),
                        'west_bengal_mentions': pdf_info.get('summary', {}).get('west_bengal_mentions', 0),
                        'kolkata_mentions': pdf_info.get('summary', {}).get('kolkata_mentions', 0),
                        'date': datetime.now().isoformat()
                    }
                })
        
        return documents
    
    def _format_market_summary(self, summary: Dict, region: str) -> str:
        """Format market summary data for RAG"""
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
        """Format district-specific data for RAG"""
        content = f"Agricultural Market Information for {district_name} District, West Bengal\n\n"
        content += f"Total market records: {district_info.get('total_records', 0)}\n"
        content += f"Average modal price: ₹{district_info.get('avg_modal_price', 0):.2f}\n\n"
        
        markets = district_info.get('markets', [])
        if markets:
            content += f"Markets in {district_name}:\n"
            for market in markets[:10]:  # Limit to first 10 markets
                content += f"- {market}\n"
            if len(markets) > 10:
                content += f"... and {len(markets) - 10} more markets\n"
            content += "\n"
        
        commodities = district_info.get('commodities', [])
        if commodities:
            content += f"Commodities traded in {district_name}:\n"
            for commodity in commodities[:15]:  # Limit to first 15 commodities
                content += f"- {commodity}\n"
            if len(commodities) > 15:
                content += f"... and {len(commodities) - 15} more commodities\n"
        
        return content
    
    def _format_price_analysis(self, price_analysis: Dict) -> str:
        """Format price analysis data for RAG"""
        content = "West Bengal Agricultural Commodity Price Analysis\n\n"
        
        # Most expensive commodities
        expensive = price_analysis.get('top_expensive_commodities', [])
        if expensive:
            content += "Most Expensive Commodities:\n"
            for item in expensive[:5]:
                content += f"- {item.get('Commodity', 'Unknown')} ({item.get('Variety', 'Unknown')}): "
                content += f"₹{item.get('Modal_x0020_Price', 0)} at {item.get('Market', 'Unknown')}, {item.get('District', 'Unknown')}\n"
            content += "\n"
        
        # Cheapest commodities
        cheap = price_analysis.get('top_cheap_commodities', [])
        if cheap:
            content += "Most Affordable Commodities:\n"
            for item in cheap[:5]:
                content += f"- {item.get('Commodity', 'Unknown')} ({item.get('Variety', 'Unknown')}): "
                content += f"₹{item.get('Modal_x0020_Price', 0)} at {item.get('Market', 'Unknown')}, {item.get('District', 'Unknown')}\n"
            content += "\n"
        
        # Commodity statistics
        commodity_stats = price_analysis.get('commodity_price_stats', {})
        if commodity_stats:
            content += "Popular Commodities Price Statistics:\n"
            sorted_commodities = sorted(commodity_stats.items(), key=lambda x: x[1].get('count', 0), reverse=True)
            for commodity, stats in sorted_commodities[:10]:
                content += f"- {commodity}: Avg ₹{stats.get('mean', 0):.2f} "
                content += f"(Range: ₹{stats.get('min', 0)} - ₹{stats.get('max', 0)}, "
                content += f"Markets: {stats.get('count', 0)})\n"
        
        return content
    
    def _format_market_specific_data(self, market_name: str, market_info: Dict) -> str:
        """Format market-specific data for RAG"""
        content = f"Market Information: {market_name}, Kolkata\n\n"
        content += f"Number of commodities: {market_info.get('commodity_count', 0)}\n"
        content += f"Average price: ₹{market_info.get('avg_price', 0):.2f}\n"
        
        price_range = market_info.get('price_range', {})
        if price_range:
            content += f"Price range: ₹{price_range.get('min', 0)} - ₹{price_range.get('max', 0)}\n\n"
        
        commodities = market_info.get('commodities', [])
        if commodities:
            content += "Commodities available:\n"
            for commodity in commodities:
                content += f"- {commodity}\n"
        
        return content
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for better RAG performance"""
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
    
    def create_embeddings_metadata(self) -> Dict[str, Any]:
        """Create metadata for embeddings generation"""
        rag_file = self.rag_output_path / "rag_documents.json"
        
        if not rag_file.exists():
            logger.error("RAG documents file not found. Run create_rag_documents first.")
            return {}
        
        with open(rag_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        metadata = {
            'total_documents': len(documents),
            'document_types': {},
            'regions': {},
            'sources': {},
            'avg_content_length': 0,
            'embeddings_config': {
                'model_recommendation': 'sentence-transformers/all-MiniLM-L6-v2',
                'chunk_size': 'variable (optimized for content)',
                'overlap': '100 words',
                'vector_dimensions': 384  # for all-MiniLM-L6-v2
            }
        }
        
        total_length = 0
        for doc in documents:
            # Count document types
            doc_type = doc.get('type', 'unknown')
            metadata['document_types'][doc_type] = metadata['document_types'].get(doc_type, 0) + 1
            
            # Count regions
            region = doc.get('metadata', {}).get('region', 'unknown')
            metadata['regions'][region] = metadata['regions'].get(region, 0) + 1
            
            # Count sources
            source = doc.get('metadata', {}).get('source', 'unknown')
            metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
            
            # Calculate content length
            content_length = len(doc.get('content', ''))
            total_length += content_length
        
        metadata['avg_content_length'] = total_length / len(documents) if documents else 0
        
        # Save metadata
        metadata_file = self.rag_output_path / "embeddings_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Embeddings metadata created: {metadata_file}")
        return metadata


def main():
    """Main function to run RAG preprocessing"""
    preprocessor = RAGPreprocessor()
    
    # Create RAG documents
    documents = preprocessor.create_rag_documents()
    
    # Create embeddings metadata
    metadata = preprocessor.create_embeddings_metadata()
    
    logger.info(f"RAG preprocessing completed successfully!")
    logger.info(f"Created {len(documents)} documents ready for RAG system")
    logger.info(f"Document types: {metadata.get('document_types', {})}")
    logger.info(f"Regions covered: {list(metadata.get('regions', {}).keys())}")


if __name__ == "__main__":
    main()
