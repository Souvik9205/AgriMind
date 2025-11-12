#!/usr/bin/env python3
"""
Test script for the improved RAG system with better fallback handling
"""

import sys
import os
from typing import List, Dict, Any

# Mock classes for testing without external dependencies
class MockRAGResponse:
    def __init__(self, answer: str, sources: List[Dict], confidence: float, context_used: str):
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.context_used = context_used

class MockLLMClient:
    def __init__(self):
        self.model_name = "test-model"
    
    def is_agriculture_related(self, query: str) -> bool:
        """Check if a query is related to agriculture, farming, or crops"""
        agriculture_keywords = [
            # Core agriculture terms
            'crop', 'crops', 'farming', 'farm', 'farmer', 'farmers', 'agriculture', 'agricultural',
            'plant', 'plants', 'grow', 'growing', 'cultivation', 'cultivate', 'harvest', 'harvesting', 'yield',
            
            # Specific crops
            'rice', 'paddy', 'wheat', 'maize', 'corn', 'sugarcane', 'jute', 'potato', 'potatoes',
            'vegetables', 'tomato', 'onion', 'brinjal', 'cabbage', 'cauliflower', 'peas', 'beans',
            'mustard', 'sesame', 'groundnut', 'sunflower', 'coconut', 'banana', 'mango',
            
            # Farming inputs and practices
            'fertilizer', 'fertiliser', 'manure', 'compost', 'pesticide', 'insecticide', 'herbicide',
            'irrigation', 'water', 'soil', 'seed', 'seeds', 'nursery', 'transplant', 'sowing',
            
            # Seasons and timing
            'kharif', 'rabi', 'zaid', 'season', 'seasonal', 'monsoon', 'weather', 'rainfall', 'climate',
            
            # Market and economics
            'market', 'price', 'prices', 'selling', 'buying', 'mandi', 'procurement', 'subsidy',
            
            # Regional terms
            'west bengal', 'bengal', 'kolkata', 'calcutta', 'wb', 'icar', 'advisory', 'agro',
            'district', 'village', 'rural', 'krishi', 'kisan',
            
            # Problems and diseases
            'disease', 'pest', 'insect', 'fungus', 'blight', 'wilt', 'rot', 'damage', 'loss',
            
            # Technology and methods
            'organic', 'hybrid', 'variety', 'breed', 'technique', 'method', 'technology', 'innovation'
        ]
        
        query_lower = query.lower()
        
        # Check for direct keyword matches
        keyword_match = any(keyword in query_lower for keyword in agriculture_keywords)
        
        # Check for agriculture-related phrases
        agriculture_phrases = [
            'how to grow', 'when to plant', 'best time', 'crop rotation', 'soil preparation',
            'pest control', 'disease management', 'water management', 'nutrient management',
            'market rate', 'farming tips', 'agricultural advice', 'cultivation practices'
        ]
        
        phrase_match = any(phrase in query_lower for phrase in agriculture_phrases)
        
        return keyword_match or phrase_match
    
    def generate_fallback_response(self, query: str) -> MockRAGResponse:
        """Generate a mock fallback response"""
        fallback_answer = f"""Based on general agricultural knowledge, here's information about your query: "{query}"

This response is generated using general agricultural principles since specific information wasn't found in our knowledge base. For topics related to crops, farming practices, or agricultural techniques, I can provide general guidance based on common agricultural practices.

For West Bengal specifically, the monsoon season (June-September) is typically the Kharif season when rice, jute, sugarcane, and various vegetables are commonly grown. The region's high humidity and rainfall during this period make it suitable for water-intensive crops.

Please note: This is general agricultural guidance. For specific local recommendations, consult with local agricultural extension officers or ICAR advisories."""
        
        return MockRAGResponse(
            answer=fallback_answer,
            sources=[],
            confidence=0.6,
            context_used="General agricultural knowledge (no specific KB documents found)"
        )

class MockRetriever:
    def __init__(self):
        self.top_k = 5
        self.similarity_threshold = 0.5
    
    def get_diverse_results(self, query: str) -> List[Dict[str, Any]]:
        # Simulate no documents found (empty KB scenario)
        return []
    
    def retrieve_with_reranking(self, query: str, filters=None) -> List[Dict[str, Any]]:
        # Simulate no documents found
        return []

class MockRAGSystem:
    def __init__(self):
        self.llm_client = MockLLMClient()
        self.retriever = MockRetriever()
    
    def query(self, question: str, filters=None, use_reranking: bool = True, diverse_results: bool = False):
        """
        Simulate improved RAG query processing with fallback logic
        """
        print(f"Processing query: {question[:50]}...")
        
        # Simulate document retrieval (returns empty for testing)
        if diverse_results:
            documents = self.retriever.get_diverse_results(question)
        elif use_reranking:
            documents = self.retriever.retrieve_with_reranking(question, filters)
        else:
            documents = []
        
        print(f"Retrieved {len(documents)} documents from knowledge base")
        
        # If no documents found, check if agriculture-related
        if not documents:
            if self.llm_client.is_agriculture_related(question):
                print("→ Query is agriculture-related, using LLM fallback")
                return self.llm_client.generate_fallback_response(question)
            else:
                print("→ Query is not agriculture-related")
                return MockRAGResponse(
                    answer="I'm specialized in agricultural topics, particularly for West Bengal. Your question doesn't seem to be related to agriculture, farming, or crops. Please ask about farming practices, crop cultivation, market prices, or agricultural advisories.",
                    sources=[],
                    confidence=0.0,
                    context_used="Non-agricultural query detected"
                )
        
        # If we had documents, we would process them here
        return MockRAGResponse(
            answer="This would be a regular RAG response with KB documents",
            sources=documents,
            confidence=0.8,
            context_used="Knowledge base documents"
        )

def test_improved_rag():
    """Test the improved RAG system"""
    print("="*60)
    print("Testing Improved AgriMind RAG System with Fallback Logic")
    print("="*60)
    
    rag = MockRAGSystem()
    
    # Test queries
    test_queries = [
        "What are the best crops for monsoon season in West Bengal?",
        "How to grow rice during Kharif season?", 
        "Current market prices of vegetables in Kolkata",
        "Best fertilizers for potato cultivation",
        "What is the capital of France?",  # Non-agriculture
        "How to code in Python?",  # Non-agriculture
        "Pest control methods for rice crops",
        "When to plant jute in Bengal?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 50)
        
        response = rag.query(query, diverse_results=True)
        
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence:.1%}")
        print(f"Sources: {len(response.sources)} documents")
        print(f"Context: {response.context_used}")

if __name__ == "__main__":
    test_improved_rag()
