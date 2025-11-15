#!/usr/bin/env python3
"""
Test script to verify the new query flow works correctly
"""

import requests
import json

# Test the new flow
def test_detect_with_query():
    """Test the /api/detect endpoint with a user query"""
    
    # Use the test image
    image_path = "/Users/souvik/Desktop/AgriMind/test/test.jpg"
    user_query = "What are these spots on my plant leaves?"
    
    try:
        # Test 1: Detection with query
        print("🧪 Testing fast detection with user query...")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'query': user_query}
            
            response = requests.post(
                'http://localhost:8001/api/detect',
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection successful!")
            print(f"   Plant: {result.get('plant', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   Disease: {result.get('disease', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Session ID: {result.get('session_id', 'N/A')}")
            print(f"   User Query Stored: {result.get('user_query', 'N/A')}")
            
            # Test 2: Process the stored query
            if result.get('user_query'):
                print("\n🧪 Testing query processing...")
                
                chat_data = {
                    "message": f"Based on the detected {result.get('disease', 'condition')} on {result.get('plant', 'plant')}, user asks: \"{result.get('user_query')}\"",
                    "chat_history": []
                }
                
                chat_response = requests.post(
                    'http://localhost:8001/api/chat',
                    json=chat_data
                )
                
                if chat_response.status_code == 200:
                    chat_result = chat_response.json()
                    print("✅ Query processing successful!")
                    print(f"   Response length: {len(chat_result.get('response', ''))} characters")
                    print(f"   First 200 chars: {chat_result.get('response', '')[:200]}...")
                else:
                    print(f"❌ Query processing failed: {chat_response.text}")
        else:
            print(f"❌ Detection failed: {response.text}")
            
        # Test 3: Detection without query
        print("\n🧪 Testing detection without query...")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            
            response = requests.post(
                'http://localhost:8001/api/detect',
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection without query successful!")
            print(f"   Plant: {result.get('plant', 'N/A')}")
            print(f"   Status: {result.get('status', 'N/A')}")
            print(f"   User Query: {result.get('user_query', 'None - as expected')}")
        else:
            print(f"❌ Detection without query failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Testing the new detection + query flow...")
    print("=" * 50)
    test_detect_with_query()
    print("=" * 50)
    print("✨ Test completed!")
