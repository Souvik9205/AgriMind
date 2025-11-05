#!/usr/bin/env python3
"""
RAG Integration Example for AgriMind
Demonstrates how to use the processed West Bengal agricultural data in a RAG system
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriRAGSystem:
    """
    Example RAG system for West Bengal agricultural data
    This is a simplified demonstration - in production, you'd use proper vector databases and embeddings
    """
    
    def __init__(self, processed_data_path: str = "./processed"):
        self.processed_path = Path(processed_data_path)
        self.documents = []
        self.load_documents()
    
    def load_documents(self):
        """Load the processed RAG documents"""
        rag_file = self.processed_path / "rag_ready" / "rag_documents.json"
        
        if not rag_file.exists():
            logger.error("RAG documents file not found. Run the data processing pipeline first.")
            return
        
        with open(rag_file, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        logger.info(f"Loaded {len(self.documents)} documents for RAG system")
    
    def simple_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search (in production, use semantic embeddings)
        """
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            
            # Simple scoring based on keyword matches
            score = 0
            query_words = query_lower.split()
            
            for word in query_words:
                if word in content:
                    score += content.count(word)
                if word in title:
                    score += title.count(word) * 2  # Title matches get higher score
            
            if score > 0:
                results.append({
                    'document': doc,
                    'score': score,
                    'preview': content[:200] + "..." if len(content) > 200 else content
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    def search_by_region(self, region: str) -> List[Dict[str, Any]]:
        """Search for documents specific to a region"""
        results = []
        region_lower = region.lower()
        
        for doc in self.documents:
            metadata = doc.get('metadata', {})
            doc_region = metadata.get('region', '').lower()
            
            if region_lower in doc_region or doc_region in region_lower:
                results.append(doc)
        
        return results
    
    def search_by_commodity(self, commodity: str) -> List[Dict[str, Any]]:
        """Search for documents related to a specific commodity"""
        return self.simple_search(commodity, max_results=10)
    
    def get_market_prices(self, commodity: str = None, district: str = None) -> List[Dict[str, Any]]:
        """Get market price information"""
        results = []
        
        for doc in self.documents:
            if doc.get('type') in ['market_summary', 'district_market_data', 'market_specific_data']:
                content = doc.get('content', '').lower()
                
                match = True
                if commodity and commodity.lower() not in content:
                    match = False
                if district and district.lower() not in content:
                    match = False
                
                if match:
                    results.append(doc)
        
        return results
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Generate an answer to a farming-related question
        This is a simplified version - in production, use LLM with retrieved context
        """
        # Search for relevant documents
        search_results = self.simple_search(question, max_results=3)
        
        if not search_results:
            return {
                'answer': "I don't have specific information about that topic in my West Bengal agricultural database.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Extract relevant information
        context = ""
        sources = []
        
        for result in search_results:
            doc = result['document']
            context += doc.get('content', '') + "\n\n"
            sources.append({
                'title': doc.get('title', 'Unknown'),
                'type': doc.get('type', 'Unknown'),
                'region': doc.get('metadata', {}).get('region', 'Unknown')
            })
        
        # Simple answer generation (in production, use LLM)
        answer = f"Based on the West Bengal agricultural data:\n\n{context[:500]}..."
        
        return {
            'answer': answer,
            'confidence': 0.8,
            'sources': sources,
            'raw_context': context
        }

def demonstrate_rag_queries():
    """Demonstrate various RAG queries"""
    rag = AgriRAGSystem()
    
    if not rag.documents:
        print("❌ No documents loaded. Please run the data processing pipeline first.")
        return
    
    print("🤖 AgriMind RAG System Demonstration")
    print("=" * 50)
    
    # Example queries
    queries = [
        "What are the potato prices in Kolkata?",
        "Rice cultivation in West Bengal",
        "Bankura district agricultural markets",
        "Jute farming information",
        "Vegetable prices in Coochbehar"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)
        
        results = rag.simple_search(query, max_results=2)
        
        if results:
            for i, result in enumerate(results, 1):
                doc = result['document']
                print(f"{i}. {doc.get('title', 'Untitled')}")
                print(f"   Type: {doc.get('type', 'Unknown')}")
                print(f"   Region: {doc.get('metadata', {}).get('region', 'Unknown')}")
                print(f"   Score: {result['score']}")
                print(f"   Preview: {result['preview'][:150]}...")
                print()
        else:
            print("   No relevant documents found.")
    
    # Demonstrate region-specific search
    print("\n🗺️ Kolkata-specific documents:")
    print("-" * 30)
    kolkata_docs = rag.search_by_region("Kolkata")
    for doc in kolkata_docs[:3]:
        print(f"• {doc.get('title', 'Untitled')}")
        print(f"  Type: {doc.get('type', 'Unknown')}")
    
    # Demonstrate commodity search
    print("\n🌾 Rice-related documents:")
    print("-" * 25)
    rice_docs = rag.search_by_commodity("Rice")
    for result in rice_docs[:3]:
        doc = result['document']
        print(f"• {doc.get('title', 'Untitled')}")
        print(f"  Score: {result['score']}")

def create_usage_examples():
    """Create usage examples for developers"""
    examples = """
# 🤖 AGRIMIND RAG SYSTEM USAGE EXAMPLES

## Basic Setup
```python
from agri_rag_system import AgriRAGSystem

# Initialize the RAG system
rag = AgriRAGSystem()

# Check if documents are loaded
print(f"Loaded {len(rag.documents)} documents")
```

## Example Queries

### 1. Search for Agricultural Information
```python
# Search for potato-related information
results = rag.simple_search("potato cultivation West Bengal")

for result in results:
    print(f"Title: {result['document']['title']}")
    print(f"Score: {result['score']}")
    print(f"Preview: {result['preview']}")
```

### 2. Region-Specific Queries
```python
# Get Kolkata-specific agricultural data
kolkata_docs = rag.search_by_region("Kolkata")

# Get Bankura district information
bankura_docs = rag.search_by_region("Bankura")
```

### 3. Commodity Price Information
```python
# Get market prices for rice
rice_prices = rag.get_market_prices(commodity="rice")

# Get all market data for Kolkata
kolkata_markets = rag.get_market_prices(district="Kolkata")
```

### 4. Question Answering
```python
# Ask agricultural questions
answer = rag.answer_question("What crops are grown in West Bengal?")
print(answer['answer'])
print(f"Confidence: {answer['confidence']}")
```

## Production Implementation

For production use, replace the simple search with:

1. **Vector Embeddings**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode([doc['content'] for doc in documents])
```

2. **Vector Database**:
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("agri_docs")
collection.add(
    documents=[doc['content'] for doc in documents],
    metadatas=[doc['metadata'] for doc in documents],
    ids=[doc['id'] for doc in documents]
)
```

3. **LLM Integration**:
```python
import openai

def generate_answer(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an agricultural expert for West Bengal."},
            {"role": "user", "content": f"Question: {question}\nContext: {context}"}
        ]
    )
    return response.choices[0].message.content
```

## Sample Use Cases

1. **Farmer Support**: "What's the current market price for rice in Kolkata?"
2. **Agricultural Planning**: "Which crops are most profitable in Bankura district?"
3. **Market Analysis**: "Compare vegetable prices across West Bengal districts"
4. **Seasonal Advice**: "What are the best crops for kharif season in West Bengal?"
5. **Government Schemes**: "What agricultural schemes are available for West Bengal farmers?"

## Data Categories Available

- Market prices and trends
- District-wise agricultural information
- Commodity-specific data
- Government reports and advisories
- Weather and seasonal information
- Agricultural best practices

## Performance Tips

1. Use specific district names for better results
2. Include commodity names in local languages if available
3. Combine multiple search approaches for comprehensive results
4. Filter results by document type for specific use cases
5. Use metadata for fine-grained searches
"""
    
    # Save examples
    examples_file = Path("processed/reports/usage_examples.md")
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write(examples)
    
    print(f"📝 Usage examples saved to: {examples_file}")

def main():
    """Main demonstration function"""
    # Demonstrate RAG queries
    demonstrate_rag_queries()
    
    # Create usage examples
    create_usage_examples()
    
    print("\n🎉 RAG demonstration completed!")
    print("\n📋 Next Steps:")
    print("1. Integrate with vector database (ChromaDB, Pinecone, etc.)")
    print("2. Add semantic embeddings using sentence-transformers")
    print("3. Connect to LLM for answer generation")
    print("4. Build web interface for farmer queries")
    print("5. Add real-time market data updates")

if __name__ == "__main__":
    main()
