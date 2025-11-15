#!/usr/bin/env python3
"""
Test script to verify the frontend-backend integration
"""

import requests
import sys
import os
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on port 8000")
        return False

def test_rag_endpoint():
    """Test the RAG endpoint"""
    try:
        payload = {"query": "What causes yellow leaves in crops?"}
        response = requests.post(f"{API_BASE_URL}/api/rag", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ RAG endpoint test passed")
            print(f"   Answer: {data['answer'][:100]}...")
            return True
        else:
            print(f"❌ RAG endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ RAG endpoint test error: {str(e)}")
        return False

def test_combined_endpoint():
    """Test the combined analysis endpoint"""
    # Create a small test image (1x1 pixel PNG)
    test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
    
    try:
        files = {"image": ("test.png", test_image_data, "image/png")}
        data = {"query": "What could be wrong with my crop leaves?"}
        
        response = requests.post(f"{API_BASE_URL}/api/analyze", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Combined analysis endpoint test passed")
            print(f"   Disease detected: {result['disease_detection']['disease']}")
            print(f"   Detection confidence: {result['disease_detection']['confidence']:.1%}")
            print(f"   Overall confidence: {result['confidence_score']:.2f}")
            print(f"   Combined insights: {result['combined_insights'][:100]}...")
            return True
        else:
            print(f"❌ Combined analysis endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Combined analysis endpoint test error: {str(e)}")
        return False

def main():
    print("🧪 Testing AgriMind Frontend-Backend Integration\n")
    
    # Run tests
    tests = [
        ("API Health Check", test_health_check),
        ("RAG Endpoint", test_rag_endpoint),
        ("Combined Analysis Endpoint", test_combined_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    # Summary
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Frontend-backend integration is working correctly.")
        print("\nTo start the full application:")
        print("1. Make sure the API is running: pnpm dev:api")
        print("2. Start the frontend: pnpm dev:web")
        print("3. Open http://localhost:3000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the API configuration and try again.")
        print("\nTroubleshooting:")
        print("- Make sure the API server is running: pnpm dev:api")
        print("- Check if all dependencies are installed")
        print("- Verify the database is running: pnpm setup-db")
    
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
