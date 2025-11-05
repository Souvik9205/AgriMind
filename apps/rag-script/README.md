# AgriMind RAG System

A Retrieval-Augmented Generation (RAG) system for agricultural knowledge, specifically focused on West Bengal agriculture and market data.

## Features

- **Vector Database Integration**: Uses PostgreSQL with pgvector for efficient similarity search
- **LLM Integration**: Powered by Google Gemini 2.5 Flash for intelligent responses
- **Agricultural Knowledge Base**: Pre-processed data from ICAR reports, market data, and farming advisories
- **Semantic Search**: Advanced embedding-based retrieval for relevant context
- **Reference Tracking**: Provides source references for all generated responses

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your Gemini API key and database credentials
```

3. Initialize the vector database:

```bash
python setup_db.py
```

4. Load knowledge base into vector database:

```bash
python load_knowledge_base.py
```

## Usage

```python
from rag_system import RAGSystem

# Initialize the system
rag = RAGSystem()

# Query the system
response = rag.query("What are the best crops for West Bengal during Kharif season?")
print(response.answer)
print("Sources:", response.sources)
```

## Components

- `rag_system.py`: Main RAG orchestrator
- `vector_store.py`: Vector database operations
- `embeddings.py`: Text embedding generation
- `llm_client.py`: Gemini LLM integration
- `retriever.py`: Document retrieval logic
- `setup_db.py`: Database initialization
- `load_knowledge_base.py`: Knowledge base ingestion
