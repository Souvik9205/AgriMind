#!/usr/bin/env python3
"""
Test script for the enhanced image+query flow
Demonstrates the improved analysis process
"""

import requests
import json
import sys
from pathlib import Path

def test_enhanced_flow():
    """Test the enhanced analysis flow"""
    
    # API endpoint
    base_url = "http://localhost:8000"
    
    print("🌾 Testing Enhanced AgriMind Flow")
    print("=" * 50)
    
    # Test query
    test_query = "My corn plants have brown spots on leaves and some yellowing. What could be causing this and how should I treat it?"
    
    # Test with a sample image (you can replace this with an actual image path)
    # For now, we'll test with the regular analyze endpoint
    
    try:
        # Test 1: Enhanced Analysis Endpoint
        print("\n1️⃣ Testing Enhanced Analysis Endpoint")
        print("-" * 40)
        
        # Create a dummy image file for testing
        test_image_path = Path("test_sample.jpg")
        if not test_image_path.exists():
            print(f"ℹ️  Please place a test image at {test_image_path} to test the complete flow")
            print("   For now, testing API structure...")
            
            # Test the health endpoint
            response = requests.get(f"{base_url}/api/health")
            if response.status_code == 200:
                print("✅ API is running and healthy")
            else:
                print("❌ API is not responding correctly")
                return
        else:
            # Test with actual image
            with open(test_image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'query': test_query}
                
                response = requests.post(f"{base_url}/api/enhanced-analyze", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Enhanced Analysis Successful!")
                    print("\n🔍 Disease Detection:")
                    print(f"   Crop: {result['metadata']['crop_detected']}")
                    print(f"   Disease: {result['metadata']['disease_detected']}")
                    print(f"   Confidence: {result['metadata']['detection_confidence']:.1%}")
                    
                    print("\n🧠 Query Enhancement:")
                    print(f"   Original: {result['original_query'][:80]}...")
                    print(f"   Enhanced: {'Yes' if result['metadata']['query_enhanced'] else 'No'}")
                    print(f"   Reason: {result['metadata']['enhancement_reason']}")
                    
                    print("\n💡 RAG Response:")
                    print(f"   Answer: {result['rag_response']['answer'][:100]}...")
                    print(f"   Sources: {len(result['rag_response']['sources'])} found")
                    
                    print(f"\n📊 Overall Confidence: {result['overall_confidence']:.1%}")
                    
                else:
                    print(f"❌ Enhanced analysis failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        
        # Test 2: Regular Analysis for comparison
        print("\n\n2️⃣ Testing Regular vs Enhanced Flow")
        print("-" * 40)
        
        # Regular RAG query
        rag_data = {"query": test_query}
        response = requests.post(f"{base_url}/api/rag", json=rag_data)
        
        if response.status_code == 200:
            result = response.json()
            print("📝 Regular RAG Response:")
            print(f"   {result['answer'][:100]}...")
        else:
            print(f"❌ Regular RAG failed: {response.status_code}")
        
        print("\n✨ Key Improvements in Enhanced Flow:")
        print("   • Disease detection provides crop and condition context")
        print("   • Enhanced queries include visual analysis results") 
        print("   • RAG responses are more specific to detected conditions")
        print("   • Confidence scoring considers both image and text analysis")
        print("   • Chat sessions maintain rich context for follow-ups")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("   Please ensure the API is running on http://localhost:8000")
        print("   Run: uvicorn main:app --reload")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

def create_sample_query_variations():
    """Create sample queries to test different scenarios"""
    
    queries = [
        {
            "scenario": "Disease Identification",
            "query": "I see brown spots on my corn leaves. What disease could this be?"
        },
        {
            "scenario": "Treatment Request", 
            "query": "My potato plants have blight symptoms. How should I treat this?"
        },
        {
            "scenario": "Prevention Advice",
            "query": "How can I prevent fungal diseases in my rice crop during monsoon?"
        },
        {
            "scenario": "General Crop Care",
            "query": "What are the best practices for healthy wheat cultivation?"
        }
    ]
    
    print("\n📝 Sample Query Variations for Testing:")
    print("=" * 50)
    
    for i, q in enumerate(queries, 1):
        print(f"\n{i}️⃣ {q['scenario']}:")
        print(f"   Query: {q['query']}")
        print(f"   Expected Enhancement: Image analysis will add crop/disease context")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "samples":
        create_sample_query_variations()
    else:
        test_enhanced_flow()
