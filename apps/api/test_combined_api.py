#!/usr/bin/env python3
"""
Test script for the new combined analysis API endpoints
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_combined_analysis():
    """Test the combined analysis endpoint"""
    print("Testing /api/analyze-and-advise endpoint...")
    
    # You would need to provide a real image file here
    # For testing, create a small test image or use an existing one
    test_image_path = "test_image.jpg"  # Replace with actual image path
    
    if not Path(test_image_path).exists():
        print(f"Test image {test_image_path} not found. Please provide a test image.")
        return
    
    url = f"{BASE_URL}/api/analyze-and-advise"
    
    # Prepare form data
    data = {
        'query': 'How do I treat this plant disease? What are the best organic solutions?',
        'additional_context': 'This is a tomato plant in West Bengal, grown in humid conditions'
    }
    
    files = {
        'image': ('test_plant.jpg', open(test_image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        response = requests.post(url, data=data, files=files)
        files['image'][1].close()  # Close the file
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Combined analysis successful!")
            print(f"   Detected disease: {result['detection_result']['disease']}")
            print(f"   Confidence: {result['detection_result']['confidence']:.2f}")
            print(f"   Processing time: {result['processing_time']} seconds")
            print(f"   RAG answer preview: {result['rag_response']['answer'][:100]}...")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during request: {str(e)}")

def test_simple_diagnosis():
    """Test the simplified diagnosis endpoint"""
    print("\nTesting /api/diagnose-and-treat endpoint...")
    
    test_image_path = "test_image.jpg"  # Replace with actual image path
    
    if not Path(test_image_path).exists():
        print(f"Test image {test_image_path} not found. Please provide a test image.")
        return
    
    url = f"{BASE_URL}/api/diagnose-and-treat"
    
    data = {
        'question': 'What is the best organic treatment for this disease?'
    }
    
    files = {
        'image': ('test_plant.jpg', open(test_image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        response = requests.post(url, data=data, files=files)
        files['image'][1].close()  # Close the file
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple diagnosis successful!")
            print(f"   Detected disease: {result['detected_disease']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing time: {result['processing_time']} seconds")
            print(f"   Treatment advice: {result['treatment_advice'][:150]}...")
            print(f"   Prevention tips count: {len(result['prevention_tips'])}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during request: {str(e)}")

def test_api_status():
    """Test API status endpoint"""
    print("\nTesting API status...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ API Status check successful!")
            print(f"   Status: {result['status']}")
            print(f"   Version: {result['version']}")
            print("   Services:", json.dumps(result['services'], indent=4))
        else:
            print(f"❌ Status check failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {str(e)}")

def main():
    """Main test function"""
    print("🌾 AgriMind Combined API Test Suite")
    print("=" * 50)
    
    # Test API status first
    test_api_status()
    
    # Test the new endpoints
    test_combined_analysis()
    test_simple_diagnosis()
    
    print("\n" + "=" * 50)
    print("📝 Test Notes:")
    print("- Make sure the API server is running on localhost:8000")
    print("- Provide a test plant image as 'test_image.jpg'")
    print("- Ensure both ML inference and RAG services are working")

if __name__ == "__main__":
    main()
