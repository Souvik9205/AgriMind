#!/usr/bin/env python3
"""
Setup script for AgriMind RAG System
Automated setup and configuration
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment variables"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from template...")
        
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.warning("Please edit .env file with your actual configuration:")
        logger.warning("1. Set your GEMINI_API_KEY")
        logger.warning("2. Verify database connection settings")
        logger.warning("3. Adjust other settings as needed")
        
        return False  # Need manual configuration
    
    elif env_file.exists():
        logger.info(".env file already exists")
        return True
    
    else:
        logger.error("No .env.example file found")
        return False

def check_database_connection():
    """Check if database is accessible"""
    try:
        from dotenv import load_dotenv
        import psycopg2
        
        load_dotenv()
        
        db_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'agrimind'),
            'user': os.getenv('DB_USER', 'agrimind'),
            'password': os.getenv('DB_PASSWORD', 'agrimind')
        }
        
        conn = psycopg2.connect(**db_params)
        conn.close()
        logger.info("Database connection successful")
        return True
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.warning("Make sure PostgreSQL with pgvector is running")
        logger.warning("You can start it with: docker-compose -f ../../infra/compose.yml up -d")
        return False

def setup_database():
    """Initialize database schema"""
    logger.info("Setting up database schema...")
    try:
        from setup_db import setup_database
        setup_database()
        logger.info("Database schema setup completed")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False

def check_knowledge_base():
    """Check if knowledge base is available"""
    kb_path = Path(os.getenv('KNOWLEDGE_BASE_PATH', '../../packages/kb/dist'))
    
    if not kb_path.exists():
        logger.error(f"Knowledge base not found at: {kb_path}")
        logger.warning("Please run the knowledge base processing first:")
        logger.warning("cd ../../packages/kb && python run_pipeline.py")
        return False
    
    # Check for required files
    required_files = [
        'rag_ready/rag_documents.json',
        'west_bengal_market_data.json',
        'pdf_processing_results.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (kb_path / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Some knowledge base files are missing: {missing_files}")
        logger.warning("The system will work with available data")
    else:
        logger.info("Knowledge base files found")
    
    return True

def load_knowledge_base():
    """Load knowledge base into vector database"""
    logger.info("Loading knowledge base...")
    try:
        from load_knowledge_base import KnowledgeBaseLoader
        loader = KnowledgeBaseLoader()
        total_loaded = loader.load_all_documents()
        
        if total_loaded > 0:
            logger.info(f"Successfully loaded {total_loaded} documents")
            return True
        else:
            logger.error("No documents were loaded")
            return False
            
    except Exception as e:
        logger.error(f"Knowledge base loading failed: {e}")
        return False

def run_health_check():
    """Run system health check"""
    logger.info("Running system health check...")
    try:
        from rag_system import RAGSystem
        rag = RAGSystem()
        health = rag.health_check()
        
        logger.info(f"System health: {health['status']}")
        
        for component, info in health['components'].items():
            status_symbol = "✅" if info['status'] == 'healthy' else "⚠️" if info['status'] == 'warning' else "❌"
            logger.info(f"{status_symbol} {component}: {info['status']}")
        
        if health['issues']:
            logger.warning("Issues found:")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        
        return health['status'] in ['healthy', 'warning']
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("        AgriMind RAG System Setup")
    print("="*60)
    
    setup_steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Checking database connection", check_database_connection),
        ("Setting up database schema", setup_database),
        ("Checking knowledge base", check_knowledge_base),
        ("Loading knowledge base", load_knowledge_base),
        ("Running health check", run_health_check)
    ]
    
    failed_steps = []
    
    for step_name, step_func in setup_steps:
        print(f"\n🔄 {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed")
            else:
                print(f"✅ {step_name} completed")
        except Exception as e:
            failed_steps.append(step_name)
            print(f"❌ {step_name} failed: {e}")
    
    print("\n" + "="*60)
    print("        Setup Summary")
    print("="*60)
    
    if not failed_steps:
        print("🎉 Setup completed successfully!")
        print("\nYou can now use the RAG system:")
        print("  - Interactive mode: python cli.py")
        print("  - Single query: python cli.py --query 'Your question'")
        print("  - System stats: python cli.py --stats")
    else:
        print(f"⚠️  Setup completed with {len(failed_steps)} issue(s):")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease resolve these issues before using the system.")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
