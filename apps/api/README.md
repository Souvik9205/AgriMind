# AgriMind API Server

AI-powered agricultural assistance API built with FastAPI.

## Features

- **Health Check**: Status and health monitoring endpoints
- **RAG Query**: Natural language queries about agriculture using Retrieval-Augmented Generation
- **Disease Detection**: Upload plant images to detect diseases and get recommendations

## API Endpoints

### Status & Health

- `GET /` - Basic status check
- `GET /api/health` - Detailed health check

### RAG (Retrieval-Augmented Generation)

- `POST /api/rag` - Submit agricultural queries
  ```json
  {
    "query": "What are the best practices for rice cultivation?",
    "max_results": 5
  }
  ```

### Disease Detection

- `POST /api/detect` - Upload plant image for disease detection
  - Form data with `image` file and optional `additional_info` text

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
pnpm dev:api
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
# Build and run with docker-compose
pnpm setup-full

# Or build just the API
docker build -t agrimind-api .
docker run -p 8000:8000 agrimind-api
```

### Testing

```bash
# Run API tests
pnpm test:api
# or
python test_api.py
```

## API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://agrimind:agrimind@localhost:5432/agrimind
REDIS_URL=redis://localhost:6379/0
```

## Architecture

The API server integrates with:

- **RAG Script** (`../rag-script/`) - For agricultural knowledge queries
- **ML Inference** (`../ml-inference/`) - For plant disease detection
- **PostgreSQL** - Primary database with vector search (pgvector)
- **Redis** - Caching and session management
