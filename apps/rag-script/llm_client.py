#!/usr/bin/env python3
"""
LLM client for AgriMind RAG System
Handles communication with Google Gemini 2.5 Flash
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class RAGResponse:
    """Response from the RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    context_used: str

class LLMClient:
    """Client for Google Gemini LLM"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
        self.model_name = "gemini-2.0-flash-exp"
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # System prompt for agricultural domain
        self.system_prompt = """You are AgriMind, an expert agricultural assistant specializing in West Bengal agriculture and farming practices. You have access to a comprehensive knowledge base including:

- ICAR (Indian Council of Agricultural Research) reports and advisories
- West Bengal market data and commodity prices
- Kharif season crop recommendations
- Weather and climate information for the region
- Farming best practices and techniques

Your role is to provide accurate, helpful, and contextually relevant information about agriculture in West Bengal. When answering questions:

1. Base your responses on the provided context from the knowledge base
2. Be specific about crops, varieties, and practices suitable for West Bengal
3. Include relevant market information when discussing crops
4. Consider seasonal factors (Kharif, Rabi) in your recommendations
5. Provide practical, actionable advice for farmers
6. Always cite your sources and mention if information is specific to certain districts or regions
7. If the context doesn't contain sufficient information to answer the question, say so clearly

Remember to be helpful, accurate, and farmer-focused in your responses."""

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> RAGResponse:
        """Generate a response using the LLM with retrieved context"""
        try:
            # Prepare context
            context = self._prepare_context(context_documents)
            
            # Create the prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract sources
            sources = self._extract_sources(context_documents)
            
            # Calculate confidence (simple heuristic based on context relevance)
            confidence = self._calculate_confidence(context_documents)
            
            return RAGResponse(
                answer=response.text.strip(),
                sources=sources,
                confidence=confidence,
                context_used=context
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return RAGResponse(
                answer="I apologize, but I'm unable to generate a response at the moment. Please try again later.",
                sources=[],
                confidence=0.0,
                context_used=""
            )
    
    def _prepare_context(self, context_documents: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        if not context_documents:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(context_documents):
            # Format document content
            doc_content = f"""
Source {i+1}: {doc.get('title', 'Unknown')}
Type: {doc.get('doc_type', 'Unknown')}
Relevance Score: {doc.get('similarity_score', 0):.3f}
Content: {doc.get('content', '')}
---
"""
            
            # Check if adding this document would exceed max context length
            if total_length + len(doc_content) > self.max_context_length:
                break
            
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the complete prompt for the LLM"""
        prompt = f"""{self.system_prompt}

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context above. If the context contains relevant information, use it to answer the question. If not, clearly state that the information is not available in the current knowledge base.

Include specific references to the sources when relevant (e.g., "According to the ICAR report..." or "Based on West Bengal market data...").

ANSWER:"""
        
        return prompt
    
    def _extract_sources(self, context_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context documents"""
        sources = []
        
        for doc in context_documents:
            metadata = doc.get('metadata', {})
            source = {
                'id': doc.get('id', ''),
                'title': doc.get('title', ''),
                'type': doc.get('doc_type', ''),
                'similarity_score': doc.get('similarity_score', 0),
                'source_file': metadata.get('source_file', ''),
                'region': metadata.get('region', ''),
                'data_type': metadata.get('data_type', '')
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, context_documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved context quality"""
        if not context_documents:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        similarities = [doc.get('similarity_score', 0) for doc in context_documents]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost confidence if we have multiple relevant documents
        document_bonus = min(len(context_documents) / 5.0, 0.2)
        
        # Final confidence score (0.0 to 1.0)
        confidence = min(avg_similarity + document_bonus, 1.0)
        
        return confidence
    
    def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response without RAG context"""
        try:
            response = self.model.generate_content(f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            return "I apologize, but I'm unable to respond at the moment. Please try again later."
