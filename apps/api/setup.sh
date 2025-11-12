#!/bin/bash

echo "🌾 Starting AgriMind API Server Setup..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "🚀 To start the development server:"
echo "   pnpm dev:api"
echo ""
echo "🐳 To start with Docker:"
echo "   pnpm setup-full"
echo ""
echo "📊 API Documentation will be available at:"
echo "   http://localhost:8000/docs (Swagger)"
echo "   http://localhost:8000/redoc (ReDoc)"
