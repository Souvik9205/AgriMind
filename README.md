# AgriMind

An AI-powered agricultural assistance platform combining plant disease detection and intelligent knowledge retrieval for farmers and agricultural professionals.

## 🌟 Features

- **🔬 Plant Disease Detection**: AI-powered disease identification using Vision Transformer models
- **🧠 Agricultural Knowledge Assistant**: RAG-powered Q&A system for farming guidance
- **🌾 Crop-Specific Insights**: Specialized knowledge for Corn, Potato, Rice, and Wheat
- **📊 Market Intelligence**: Real-time market data and price information
- **🗺️ Regional Expertise**: Focused on West Bengal agriculture and practices

## 🚀 Quick Start

### Plant Disease Detection

Detect diseases in plant images with high accuracy:

```bash
# Detect disease in an image (human-readable output)
npm run detect-disease path/to/image.jpg

# Get JSON output for API integration
npm run detect-disease path/to/image.jpg -- --json

# Quiet mode (suppress loading messages)
npm run detect-disease path/to/image.jpg -- --quiet --json
```

**Supported diseases**: 13+ conditions across Corn, Potato, Rice, and Wheat including rusts, blights, spots, and healthy conditions.

### Agricultural Knowledge Assistant

Get intelligent answers to farming questions:

```bash
# Interactive mode - ask questions interactively
npm run ask-agrimind

# Single query with human-readable output
npm run ask-agrimind -- --query "What are the best crops for West Bengal during Kharif season?"

# Get JSON output for API integration
npm run ask-agrimind -- --query "Rice prices in Kolkata" --format json

# Market-specific queries
npm run ask-agrimind -- --query "Current vegetable prices" --type market

# Regional queries with filters
npm run ask-agrimind -- --query "Farming practices in Murshidabad" --region "Murshidabad"
```

**Knowledge base includes**: ICAR reports, market data, farming advisories, weather patterns, and crop recommendations.

## 📋 Requirements

- **Python 3.8+** with virtual environment
- **Node.js 18+** and pnpm
- **Docker** (for database services)
- **PostgreSQL with pgvector** (auto-setup via Docker)

## ⚙️ Setup

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd AgriMind
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pnpm install
   pip install -r apps/ml-inference/requirements.txt
   pip install -r apps/rag-script/requirements.txt
   ```

3. **Setup database services**:
   ```bash
   npm run setup-db
   ```

4. **Initialize RAG system** (one-time setup):
   ```bash
   cd apps/rag-script
   python setup_db.py
   python load_knowledge_base.py
   cd ../..
   ```

5. **Test the systems**:
   ```bash
   # Test disease detection
   npm run detect-disease apps/ml-inference/sample_image.jpg

   # Test RAG system
   npm run ask-agrimind -- --query "Hello, what can you help me with?"
   ```

## 🏗️ Architecture

```
AgriMind/
├── apps/
│   ├── api/              # Backend API server
│   ├── frontend/         # Next.js web interface
│   ├── ml-inference/     # Disease detection service
│   └── rag-script/       # Knowledge retrieval service
├── packages/
│   ├── ui/               # Shared UI components
│   ├── kb/               # Knowledge base processing
│   └── typescript-config/ # Shared TypeScript configs
└── infra/
    └── compose.yml       # Docker services
```

## 🔧 Development

```bash
# Start all services in development mode
npm run dev

# Run linting across all packages
npm run lint

# Build all packages
npm run build

# Check system health
npm run ask-agrimind -- --health-check
npm run ask-agrimind -- --stats
```

## 📖 Documentation

- [Plant Disease Detection](./apps/ml-inference/README.md) - Detailed ML inference documentation
- [RAG System](./apps/rag-script/README.md) - Knowledge retrieval system guide
- [Knowledge Base Processing](./packages/kb/README.md) - Data processing pipeline
- [Frontend](./apps/frontend/README.md) - Web interface documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**AgriMind** - Empowering farmers with AI-driven insights for better agricultural outcomes.
