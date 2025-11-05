#!/usr/bin/env python3
"""
Test script for AgriMind RAG System
Run basic tests to verify system functionality
"""

import sys
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemTest:
    """Test suite for RAG System"""
    
    def __init__(self):
        self.rag = None
        self.test_results = []
    
    def setup(self):
        """Setup test environment"""
        try:
            from rag_system import RAGSystem
            self.rag = RAGSystem()
            logger.info("Test setup completed")
            return True
        except Exception as e:
            logger.error(f"Test setup failed: {e}")
            return False
    
    def test_basic_query(self):
        """Test basic query functionality"""
        test_name = "Basic Query Test"
        try:
            response = self.rag.query("What is agriculture?")
            
            success = (
                response.answer and 
                len(response.answer) > 10 and
                response.confidence >= 0.0
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Answer length: {len(response.answer)}, Confidence: {response.confidence:.3f}"
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def test_west_bengal_query(self):
        """Test West Bengal specific query"""
        test_name = "West Bengal Query Test"
        try:
            response = self.rag.query_west_bengal_specific("What crops are grown in West Bengal?")
            
            success = (
                response.answer and 
                len(response.sources) > 0 and
                any('west bengal' in source.get('title', '').lower() or 
                    source.get('metadata', {}).get('region') == 'West Bengal'
                    for source in response.sources)
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Sources: {len(response.sources)}, Answer: {response.answer[:50]}..."
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def test_market_query(self):
        """Test market data query"""
        test_name = "Market Data Query Test"
        try:
            response = self.rag.query_market_data("What are rice prices in Kolkata?")
            
            success = (
                response.answer and 
                ('price' in response.answer.lower() or 
                 'market' in response.answer.lower() or
                 '₹' in response.answer)
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Answer contains price info: {success}"
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def test_agricultural_guide_query(self):
        """Test agricultural guide query"""
        test_name = "Agricultural Guide Query Test"
        try:
            response = self.rag.query_agricultural_guides("How to grow rice?")
            
            success = (
                response.answer and 
                len(response.answer) > 50 and
                ('rice' in response.answer.lower() or 'crop' in response.answer.lower())
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Answer relevance: {success}, Length: {len(response.answer)}"
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def test_system_stats(self):
        """Test system statistics"""
        test_name = "System Statistics Test"
        try:
            stats = self.rag.get_system_stats()
            
            success = (
                'vector_store' in stats and
                'embedding_model' in stats and
                'llm_model' in stats and
                stats.get('vector_store', {}).get('total_documents', 0) > 0
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Documents: {stats.get('vector_store', {}).get('total_documents', 0)}"
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def test_health_check(self):
        """Test system health check"""
        test_name = "Health Check Test"
        try:
            health = self.rag.health_check()
            
            success = (
                health.get('status') in ['healthy', 'warning'] and
                'components' in health
            )
            
            self.test_results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL',
                'details': f"Health status: {health.get('status', 'unknown')}"
            })
            
            return success
            
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*60)
        print("        AgriMind RAG System Tests")
        print("="*60)
        
        if not self.setup():
            print("❌ Test setup failed. Cannot run tests.")
            return False
        
        test_methods = [
            self.test_basic_query,
            self.test_west_bengal_query,
            self.test_market_query,
            self.test_agricultural_guide_query,
            self.test_system_stats,
            self.test_health_check
        ]
        
        print(f"\nRunning {len(test_methods)} tests...\n")
        
        passed = 0
        failed = 0
        errors = 0
        
        for test_method in test_methods:
            test_name = test_method.__doc__.strip()
            print(f"🔄 Running: {test_name}")
            
            try:
                result = test_method()
                if result:
                    print(f"✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_name}: FAILED")
                    failed += 1
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                errors += 1
        
        # Print summary
        print("\n" + "="*60)
        print("        Test Results Summary")
        print("="*60)
        
        total_tests = len(test_methods)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        
        if passed == total_tests:
            print("\n🎉 All tests passed! RAG system is working correctly.")
            return True
        else:
            print(f"\n⚠️  {failed + errors} test(s) failed. Please check the issues.")
            
            # Print detailed results
            print("\nDetailed Results:")
            for result in self.test_results:
                status_symbol = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "💥"
                print(f"{status_symbol} {result['test']}: {result['status']}")
                print(f"   Details: {result['details']}")
            
            return False


def main():
    """Main test function"""
    try:
        tester = RAGSystemTest()
        success = tester.run_all_tests()
        
        if success:
            print("\n🚀 RAG system is ready for use!")
            print("\nNext steps:")
            print("  - Try: python cli.py")
            print("  - Or: python cli.py --query 'Your question here'")
        else:
            print("\n🔧 Please resolve the issues and run tests again.")
            print("  - Check your .env configuration")
            print("  - Ensure database is running")
            print("  - Verify knowledge base is loaded")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
