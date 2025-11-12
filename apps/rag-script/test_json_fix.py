#!/usr/bin/env python3
"""
Quick test to demonstrate RAG improvements without database dependency
"""

import sys
import subprocess
import json

def test_rag_with_json():
    """Test RAG system with proper JSON output"""
    
    print("="*70)
    print("TESTING IMPROVED RAG SYSTEM - JSON OUTPUT FIX")
    print("="*70)
    
    test_queries = [
        ("Agriculture Query (Should get fallback)", "What are the best crops for West Bengal monsoon?"),
        ("Rice Query (Should get specific advice)", "How to cultivate rice in Bengal?"),
        ("Non-Agriculture Query (Should be rejected)", "What is machine learning?"),
        ("Market Query (Should get market guidance)", "Vegetable prices in Kolkata today"),
        ("Fertilizer Query (Should get fertilizer advice)", "Best fertilizer for potato")
    ]
    
    for test_name, query in test_queries:
        print(f"\n{test_name}")
        print(f"Query: {query}")
        print("-" * 60)
        
        try:
            # Run the CLI with JSON output
            cmd = [
                "/Users/souvik/Desktop/AgriMind/.venv/bin/python",
                "cli.py",
                "--query", query,
                "--format", "json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/Users/souvik/Desktop/AgriMind/apps/rag-script",
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    # Try to parse as JSON
                    response_data = json.loads(result.stdout.strip())
                    
                    print(f"✅ SUCCESS")
                    print(f"Answer: {response_data['answer'][:150]}...")
                    print(f"Confidence: {response_data['confidence']:.1%}")
                    print(f"Sources: {len(response_data['sources'])} documents")
                    
                    # Analyze the response
                    is_agriculture = any(word in query.lower() for word in ['crop', 'rice', 'fertilizer', 'vegetable', 'bengal', 'monsoon', 'potato'])
                    is_rejected = "specialized in agricultural topics" in response_data['answer']
                    
                    if is_agriculture and not is_rejected:
                        print("🎯 CORRECT: Agriculture query got helpful response")
                    elif not is_agriculture and is_rejected:
                        print("🎯 CORRECT: Non-agriculture query was properly rejected")
                    elif is_agriculture and is_rejected:
                        print("⚠️  UNEXPECTED: Agriculture query was rejected")
                    else:
                        print("🤔 Response needs review")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON PARSE ERROR: {e}")
                    print(f"Raw output: {result.stdout[:200]}...")
            else:
                print(f"❌ COMMAND FAILED: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT: Query took too long")
        except Exception as e:
            print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_rag_with_json()
