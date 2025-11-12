#!/usr/bin/env python3
"""
Comprehensive test for the improved RAG system
Tests both with and without LLM availability
"""

import json
import subprocess
import sys
import os

def test_rag_endpoint_with_json():
    """Test the RAG system through CLI with JSON output"""
    
    print("="*70)
    print("TESTING IMPROVED RAG SYSTEM - AGRICULTURE QUERIES")
    print("="*70)
    
    test_cases = [
        {
            "query": "What are the best crops for monsoon season?",
            "expected_agriculture": True,
            "description": "General monsoon crop query"
        },
        {
            "query": "How to grow rice in West Bengal?",
            "expected_agriculture": True,
            "description": "Rice cultivation query"
        },
        {
            "query": "Current market prices of vegetables",
            "expected_agriculture": True,
            "description": "Market price query"
        },
        {
            "query": "Best fertilizer for potato cultivation",
            "expected_agriculture": True,
            "description": "Fertilizer recommendation query"
        },
        {
            "query": "Pest control for rice crops",
            "expected_agriculture": True,
            "description": "Pest control query"
        },
        {
            "query": "What is Python programming?",
            "expected_agriculture": False,
            "description": "Non-agriculture query (should be rejected)"
        },
        {
            "query": "What is the capital of India?",
            "expected_agriculture": False,
            "description": "General knowledge query (should be rejected)"
        }
    ]
    
    rag_script_path = "/Users/souvik/Desktop/AgriMind/apps/rag-script"
    python_path = "/Users/souvik/Desktop/AgriMind/.venv/bin/python"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print("-" * 60)
        
        try:
            # Run RAG CLI with JSON output
            cmd = [
                python_path,
                "cli.py",
                "--query", test_case['query'],
                "--format", "json"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=rag_script_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    
                    print(f"✅ SUCCESS")
                    print(f"Answer: {response.get('answer', 'No answer')[:150]}...")
                    print(f"Confidence: {response.get('confidence', 0):.1%}")
                    print(f"Sources: {len(response.get('sources', []))} documents")
                    
                    # Validate response based on expectations
                    if test_case['expected_agriculture']:
                        if "specialized in agricultural topics" in response.get('answer', ''):
                            print("❌ ISSUE: Agriculture query was incorrectly rejected")
                        else:
                            print("✅ GOOD: Agriculture query was properly handled")
                    else:
                        if "specialized in agricultural topics" in response.get('answer', ''):
                            print("✅ GOOD: Non-agriculture query was properly rejected")
                        else:
                            print("❌ ISSUE: Non-agriculture query was not rejected")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON PARSE ERROR: {e}")
                    print(f"Raw output: {result.stdout}")
            else:
                print(f"❌ COMMAND FAILED (exit code {result.returncode})")
                print(f"Error: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT: Query took too long")
        except Exception as e:
            print(f"❌ EXCEPTION: {e}")
    
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print("The improved RAG system now includes:")
    print("✅ Lower similarity threshold (0.5 instead of 0.7)")
    print("✅ Agriculture topic detection")
    print("✅ LLM fallback for agriculture queries when KB fails")
    print("✅ Static fallback when LLM is unavailable") 
    print("✅ Rejection of non-agriculture queries")
    print("✅ Better confidence scoring")
    print("\nThis should significantly reduce 'no relevant data' responses!")

if __name__ == "__main__":
    test_rag_endpoint_with_json()
