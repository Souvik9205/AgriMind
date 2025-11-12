#!/usr/bin/env python3
"""
Command-line interface for AgriMind RAG System
Interactive tool for querying the agricultural knowledge base
"""

import argparse
import sys
from typing import Optional
from rag_system import RAGSystem

def interactive_mode(rag: RAGSystem):
    """Run RAG system in interactive mode"""
    
    # Check if we have a proper terminal for interactive mode
    if not sys.stdin.isatty():
        print("Error: Interactive mode requires a terminal. Use --query for non-interactive usage.")
        return
    
    print("="*60)
    print("        AgriMind RAG System - Interactive Mode")
    print("="*60)
    print("Ask questions about West Bengal agriculture, market prices,")
    print("farming practices, and crop recommendations.")
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'stats' - Show system statistics")
    print("  'health' - Show system health")
    print("="*60)
    
    while True:
        try:
            # Get user input
            try:
                query = input("\n🌾 AgriMind> ").strip()
            except EOFError:
                print("\n\nEOF detected. Exiting...")
                break
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            
            if not query:
                continue
            
            # Handle special commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using AgriMind RAG System!")
                break
            
            elif query.lower() == 'stats':
                print("\n📊 System Statistics:")
                print("-" * 30)
                stats = rag.get_system_stats()
                vector_stats = stats.get('vector_store', {})
                print(f"Total Documents: {vector_stats.get('total_documents', 0)}")
                print(f"Documents with Embeddings: {vector_stats.get('documents_with_embeddings', 0)}")
                print(f"Document Types: {vector_stats.get('document_types', {})}")
                print(f"Embedding Model: {stats.get('embedding_model', 'Unknown')}")
                print(f"LLM Model: {stats.get('llm_model', 'Unknown')}")
                continue
            
            elif query.lower() == 'health':
                print("\n🏥 System Health Check:")
                print("-" * 30)
                health = rag.health_check()
                print(f"Overall Status: {health['status'].upper()}")
                for component, info in health['components'].items():
                    status_emoji = "✅" if info['status'] == 'healthy' else "⚠️" if info['status'] == 'warning' else "❌"
                    print(f"{status_emoji} {component}: {info['status']}")
                if health['issues']:
                    print("\nIssues:")
                    for issue in health['issues']:
                        print(f"  ⚠️  {issue}")
                continue
            
            # Process query
            print("\n🔍 Searching knowledge base...")
            response = rag.query(query, diverse_results=True)
            
            # Display answer
            print("\n" + "="*60)
            print("📝 ANSWER:")
            print("="*60)
            print(response.answer)
            
            # Display confidence
            confidence_emoji = "🟢" if response.confidence > 0.8 else "🟡" if response.confidence > 0.6 else "🔴"
            print(f"\n{confidence_emoji} Confidence: {response.confidence:.1%}")
            
            # Display sources
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)} documents):")
                print("-" * 40)
                for i, source in enumerate(response.sources, 1):
                    relevance_emoji = "🔥" if source['similarity_score'] > 0.8 else "⭐" if source['similarity_score'] > 0.6 else "📄"
                    print(f"{relevance_emoji} {i}. {source['title']}")
                    print(f"     Type: {source['type']} | Relevance: {source['similarity_score']:.1%}")
                    if source.get('region'):
                        print(f"     Region: {source['region']}")
                    print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nEOF detected. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")
            # Add a small delay to prevent rapid error loops
            import time
            time.sleep(0.5)

def single_query_mode(rag: RAGSystem, query: str, output_format: str = 'text'):
    """Process a single query and output the result"""
    try:
        response = rag.query(query, diverse_results=True)
        
        if output_format == 'json':
            import json
            result = {
                'query': query,
                'answer': response.answer,
                'confidence': response.confidence,
                'sources': response.sources
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"Query: {query}")
            print(f"\nAnswer: {response.answer}")
            print(f"\nConfidence: {response.confidence:.1%}")
            print(f"Sources: {len(response.sources)} documents")
            
    except Exception as e:
        if output_format == 'json':
            import json
            error_result = {
                'query': query,
                'error': str(e),
                'answer': "I apologize, but I encountered an error while processing your question. Please try again later.",
                'confidence': 0.0,
                'sources': []
            }
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error processing query: {e}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="AgriMind RAG System - Agricultural Knowledge Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cli.py

  # Single query
  python cli.py --query "What crops are best for West Bengal Kharif season?"

  # Query with JSON output
  python cli.py --query "Rice prices in Kolkata" --format json

  # Market-specific query
  python cli.py --query "Current vegetable prices" --type market

  # Regional query
  python cli.py --query "Farming in Murshidabad" --region "Murshidabad"
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        help='Single query to process (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=['market', 'agricultural', 'any'],
        default='any',
        help='Query type filter (default: any)'
    )
    
    parser.add_argument(
        '--region', '-r',
        help='Filter by region (e.g., "West Bengal", "Kolkata", "Murshidabad")'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Perform system health check and exit'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        if args.format != 'json':
            print("🚀 Initializing AgriMind RAG System...")
        rag = RAGSystem()
        
        # Handle special commands
        if args.health_check:
            health = rag.health_check()
            print("System Health Check:")
            print(f"Status: {health['status']}")
            for component, info in health['components'].items():
                print(f"  {component}: {info['status']}")
            if health['issues']:
                print("Issues:")
                for issue in health['issues']:
                    print(f"  - {issue}")
            return
        
        if args.stats:
            stats = rag.get_system_stats()
            print("System Statistics:")
            vector_stats = stats.get('vector_store', {})
            print(f"  Total Documents: {vector_stats.get('total_documents', 0)}")
            print(f"  Document Types: {vector_stats.get('document_types', {})}")
            print(f"  Embedding Model: {stats.get('embedding_model', 'Unknown')}")
            print(f"  LLM Model: {stats.get('llm_model', 'Unknown')}")
            return
        
        # Process query or enter interactive mode
        if args.query:
            # Set up filters based on arguments
            filters = {}
            if args.type == 'market':
                filters['doc_type'] = 'market_summary'
            elif args.type == 'agricultural':
                filters['doc_type'] = 'pdf_content'
            
            if args.region:
                filters['metadata'] = {'region': args.region}
            
            # Process single query
            if filters:
                response = rag.query(args.query, filters=filters)
            else:
                response = rag.query(args.query, diverse_results=True)
            
            # Output result
            if args.format == 'json':
                import json
                result = {
                    'query': args.query,
                    'answer': response.answer,
                    'confidence': response.confidence,
                    'sources': response.sources
                }
                print(json.dumps(result, indent=2))
            else:
                print(f"\nQuery: {args.query}")
                print(f"\nAnswer: {response.answer}")
                print(f"\nConfidence: {response.confidence:.1%}")
                print(f"Sources: {len(response.sources)} documents")
        else:
            # Interactive mode
            interactive_mode(rag)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
