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

### From AgriMind root directory:

```bash
# Interactive mode - ask questions interactively
npm run ask-agrimind

# Single query with human-readable output
npm run ask-agrimind -- --query "What are the best crops for West Bengal during Kharif season?"

# Get JSON output for API integration
npm run ask-agrimind -- --query "Rice prices in Kolkata" --format json

# Market-specific queries
npm run ask-agrimind -- --query "Current vegetable prices" --type market --format json

# Regional queries
npm run ask-agrimind -- --query "Farming practices in Murshidabad" --region "Murshidabad"

# System health check
npm run ask-agrimind -- --health-check

# System statistics
npm run ask-agrimind -- --stats
```

### Sample JSON Response:

```json
{
  "query": "What are the best crops for West Bengal during Kharif season?",
  "answer": "For West Bengal during the Kharif season (June-October), the best crops include:\n\n1. **Rice**: The primary Kharif crop, especially high-yielding varieties like IET-4786, Lalat, and Ranjit\n2. **Jute**: West Bengal is the leading jute producer in India\n3. **Sugarcane**: Suitable for areas with adequate water supply\n4. **Cotton**: In some districts of West Bengal\n5. **Maize**: Growing importance as a Kharif crop\n6. **Pulses**: Arhar (Pigeon pea), Moong, and Urad\n\nThe choice depends on soil type, water availability, and local climate conditions.",
  "confidence": 0.92,
  "sources": [
    "ICAR Kharif Agro-Advisories for Farmers 2025",
    "ICAR Annual Report 2023-24"
  ]
}
```

### Interactive Mode Example:

```
🌾 AgriMind> What fertilizers are recommended for rice in West Bengal?

Answer: For rice cultivation in West Bengal, the following fertilizer recommendations are provided:

**Basal Application (at planting):**
- Nitrogen: 40-50 kg/ha
- Phosphorus: 30-40 kg/ha 
- Potassium: 30-40 kg/ha

**Top Dressing:**
- Nitrogen: Split application at tillering and panicle initiation stages
- Apply 40-50 kg N/ha at 20-25 days after transplanting
- Apply remaining 30-40 kg N/ha at panicle initiation

**Micronutrients:**
- Zinc: 25 kg ZnSO4/ha if deficient
- Iron: Foliar spray if iron deficiency symptoms appear

Confidence: 89.5%
Sources: 2 documents

🌾 AgriMind> quit
Thank you for using AgriMind RAG System!
```

### Direct Python usage:

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
