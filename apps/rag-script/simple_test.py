#!/usr/bin/env python3
"""
Simple test script for AgriMind RAG System
"""

import sys
import os

def test_simple_query():
    """Test the RAG system with a simple query"""
    
    print("🌾 AgriMind RAG System - Simple Test")
    print("=" * 50)
    
    # Simple test questions
    test_questions = [
        "What crops are suitable for West Bengal?",
        "What is the best time to plant rice?", 
        "How much does wheat cost in the market?",
        "What are common potato diseases?",
        "Tell me about farming in West Bengal"
    ]
    
    try:
        # Try to import the RAG system
        from rag_system import RAGSystem
        
        print("✅ RAG System imported successfully!")
        
        # Initialize the system
        print("🚀 Initializing RAG System...")
        rag = RAGSystem()
        
        print("✅ RAG System initialized!")
        
        # Test health check
        print("🏥 Checking system health...")
        health = rag.health_check()
        print(f"Status: {health['status']}")
        
        if health['status'] == 'healthy':
            print("✅ System is healthy!")
            
            # Ask a simple question
            test_query = test_questions[0]  # "What crops are suitable for West Bengal?"
            print(f"\n🔍 Testing query: '{test_query}'")
            
            response = rag.query(test_query, diverse_results=True)
            
            print("\n📝 RESPONSE:")
            print("-" * 30)
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence:.1%}")
            print(f"Sources: {len(response.sources)} documents")
            
            if response.sources:
                print("\n📚 Top sources:")
                for i, source in enumerate(response.sources[:3], 1):
                    print(f"{i}. {source['title']} (Relevance: {source['similarity_score']:.1%})")
                    
        else:
            print("⚠️  System health issues detected:")
            for issue in health.get('issues', []):
                print(f"  - {issue}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed.")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("There might be configuration issues.")
        return False
    
    return True

if __name__ == "__main__":
    success = test_simple_query()
    sys.exit(0 if success else 1)
