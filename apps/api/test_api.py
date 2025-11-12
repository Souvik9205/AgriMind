#!/usr/bin/env python3
"""
Test script for AgriMind API endpoints
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_status_endpoint():
    """Test the status check endpoint"""
    print("Testing status endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_health_endpoint():
    """Test the health check endpoint"""
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_rag_endpoint():
    """Test the RAG query endpoint"""
    print("\nTesting RAG endpoint...")
    try:
        query_data = {
            "query": "What are the best practices for rice cultivation?",
            "max_results": 3
        }
        response = requests.post(
            f"{API_BASE_URL}/api/rag",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_detection_endpoint():
    """Test the disease detection endpoint"""
    print("\nTesting detection endpoint...")
    
    # Create a dummy image file for testing
    test_image_path = Path("test_image.jpg")
    if not test_image_path.exists():
        print("Creating dummy test image...")
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_image_path)
    
    try:
        with open(test_image_path, 'rb') as img_file:
            files = {'image': ('test_image.jpg', img_file, 'image/jpeg')}
            data = {'additional_info': 'Test plant image'}
            
            response = requests.post(
                f"{API_BASE_URL}/api/detect",
                files=files,
                data=data
            )
            
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up test image
        if test_image_path.exists():
            test_image_path.unlink()

def main():
    """Run all tests"""
    print("=" * 50)
    print("AgriMind API Test Suite")
    print("=" * 50)
    
    tests = [
        test_status_endpoint,
        test_health_endpoint,
        test_rag_endpoint,
        test_detection_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()
