#!/usr/bin/env python3
"""
Test script to demonstrate the difference between verbose and concise responses
"""

import sys
import subprocess
from pathlib import Path

# Add the rag-script to path
sys.path.append(str(Path(__file__).parent / "apps" / "rag-script"))

def test_responses():
    """Test both verbose and concise responses"""
    
    # Test queries
    test_queries = [
        "what are these dots",
        "what is this disease", 
        "identify these spots on wheat",
        "how to manage brown rust"
    ]
    
    print("="*80)
    print("AGRIMIND RESPONSE MODE COMPARISON")
    print("="*80)
    
    for query in test_queries:
        print(f"\n🔍 QUERY: {query}")
        print("-" * 60)
        
        # Test concise mode
        print("📱 CONCISE MODE (for chat/mobile):")
        try:
            result = subprocess.run([
                sys.executable, 
                "apps/rag-script/cli.py", 
                "--query", query,
                "--concise", "true",
                "--format", "text"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n💻 VERBOSE MODE (for detailed analysis):")
        try:
            result = subprocess.run([
                sys.executable, 
                "apps/rag-script/cli.py", 
                "--query", query,
                "--concise", "false", 
                "--format", "text"
            ], capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # Truncate verbose response to show difference
                output = result.stdout.strip()
                if len(output) > 300:
                    output = output[:300] + "... [TRUNCATED]"
                print(output)
            else:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Error: {e}")
            
        print("\n" + "="*60)

def show_token_savings():
    """Calculate approximate token savings"""
    print("\n💰 TOKEN SAVINGS ANALYSIS")
    print("-" * 40)
    
    # Simulate token counts (rough estimates)
    verbose_tokens = 400  # Your original response
    concise_tokens = 25   # New concise response
    
    savings = verbose_tokens - concise_tokens
    savings_percent = (savings / verbose_tokens) * 100
    
    print(f"📊 Original Response: ~{verbose_tokens} tokens")
    print(f"✅ Concise Response: ~{concise_tokens} tokens")
    print(f"💾 Token Savings: {savings} tokens ({savings_percent:.1f}% reduction)")
    
    # Calculate cost savings (example rates)
    cost_per_1k_tokens = 0.0015  # Example Gemini rate
    original_cost = (verbose_tokens / 1000) * cost_per_1k_tokens
    concise_cost = (concise_tokens / 1000) * cost_per_1k_tokens
    cost_savings = original_cost - concise_cost
    
    print(f"\n💵 Cost Analysis (per query):")
    print(f"   Original: ${original_cost:.6f}")
    print(f"   Concise:  ${concise_cost:.6f}")
    print(f"   Savings:  ${cost_savings:.6f} ({savings_percent:.1f}%)")
    
    # Projections
    daily_queries = 1000
    print(f"\n📈 Daily Projections ({daily_queries} queries):")
    print(f"   Token savings: {savings * daily_queries:,} tokens")
    print(f"   Cost savings: ${cost_savings * daily_queries:.2f}")

if __name__ == "__main__":
    try:
        print("🚀 Testing AgriMind Response Modes...")
        test_responses()
        show_token_savings()
        
        print("\n✅ RECOMMENDATIONS:")
        print("• Use CONCISE mode for chat interfaces and mobile apps")
        print("• Use VERBOSE mode for detailed reports and analysis")
        print("• Auto-detect query type (identification vs advisory)")
        print("• Allow users to toggle between modes")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError running tests: {e}")
