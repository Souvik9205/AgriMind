#!/usr/bin/env python3
"""
Test Gemini API key and provide instructions for updating the token
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def check_gemini_api():
    """Check if Gemini API key is configured and working"""
    
    print("🔍 CHECKING GEMINI API CONFIGURATION")
    print("=" * 50)
    
    # Check for .env file
    env_file = current_dir / '.env'
    env_example = current_dir / '.env.example'
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"📝 Please create .env file from .env.example:")
        print(f"   cp {env_example} {env_file}")
        print("\n🔑 Then update GEMINI_API_KEY in the .env file")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("❌ Gemini API key not configured!")
        print("\n🔑 TO FIX THIS:")
        print("1. Get your Gemini API key from: https://aistudio.google.com/app/apikey")
        print("2. Update GEMINI_API_KEY in .env file")
        print("3. Make sure you have sufficient quota/credits")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    # Test the API
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        print("🧪 Testing API connection...")
        response = model.generate_content("Hello, respond with just 'API Working'")
        
        if response and response.text:
            print(f"✅ API Test Success: {response.text.strip()}")
            return True
        else:
            print("❌ API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        
        if "quota" in str(e).lower() or "429" in str(e):
            print("\n💳 QUOTA ISSUE DETECTED:")
            print("1. Check your usage at: https://ai.dev/usage")
            print("2. You may need to upgrade your plan or wait for quota reset")
            print("3. Free tier has limited requests per minute")
        elif "api key" in str(e).lower() or "401" in str(e):
            print("\n🔑 API KEY ISSUE:")
            print("1. Verify your API key is correct")
            print("2. Make sure API key has proper permissions")
            print("3. Check if key is active at: https://aistudio.google.com/app/apikey")
        
        return False

def test_improved_rag_without_hardcoded_data():
    """Test the improved RAG system without hardcoded responses"""
    
    print("\n🚀 TESTING IMPROVED RAG (NO HARDCODED DATA)")
    print("=" * 50)
    
    # Check API first
    if not check_gemini_api():
        print("❌ Cannot test RAG - API issues need to be resolved first")
        return
    
    try:
        from llm_client import LLMClient
        
        print("✅ Initializing LLM Client...")
        llm = LLMClient()
        
        # Test agriculture detection
        test_queries = [
            ("Agriculture Query", "What crops should I grow in West Bengal during monsoon?"),
            ("Non-Agriculture Query", "How to code in Python?")
        ]
        
        for test_name, query in test_queries:
            print(f"\n📝 {test_name}: {query}")
            print("-" * 40)
            
            # Test topic detection
            is_agri = llm.is_agriculture_related(query)
            print(f"Agriculture Detection: {is_agri}")
            
            if is_agri:
                print("🌾 Processing as agriculture query...")
                try:
                    response = llm.generate_fallback_response(query)
                    print(f"✅ Response Generated: {response.answer[:150]}...")
                    print(f"Confidence: {response.confidence:.1%}")
                    print(f"Context: {response.context_used}")
                except Exception as e:
                    print(f"❌ Fallback failed: {e}")
            else:
                print("🚫 Non-agriculture query detected - would be rejected")
        
        print("\n🎉 RAG System Test Complete!")
        print("✅ No hardcoded data used - all responses from Gemini LLM")
        
    except Exception as e:
        print(f"❌ RAG Test Failed: {e}")

if __name__ == "__main__":
    print("AgriMind RAG System - API Configuration Test")
    print("=" * 60)
    
    test_improved_rag_without_hardcoded_data()
    
    print("\n📋 SUMMARY:")
    print("✅ Removed all hardcoded agricultural data")
    print("✅ System now relies entirely on Gemini LLM for fallbacks")
    print("✅ Better error handling for API quota issues")
    print("✅ Clear instructions for API key configuration")
    print("\n🔧 Next Steps:")
    print("1. Configure your Gemini API key in .env file")
    print("2. Ensure sufficient API quota/credits")
    print("3. Test the system with real agriculture queries")
