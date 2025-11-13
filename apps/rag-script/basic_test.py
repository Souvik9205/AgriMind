#!/usr/bin/env python3
"""
Basic configuration and import test for AgriMind RAG System
"""

import sys
import os

def test_basic_imports():
    """Test basic imports without ML dependencies"""
    
    print("🔍 Testing Basic Configuration and Imports")
    print("=" * 50)
    
    # Test 1: Basic Python imports
    try:
        import psycopg2
        print("✅ psycopg2 imported successfully")
    except ImportError as e:
        print(f"❌ psycopg2 import failed: {e}")
        return False
    
    # Test 2: Config loading
    try:
        from config import Config
        config = Config()
        print("✅ Configuration loaded successfully")
        
        # Validate config
        validation = config.validate()
        print(f"Config validation: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  ❌ {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  ⚠️  {warning}")
                
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 3: Environment variables
    print("\n🔧 Environment Configuration:")
    print("-" * 30)
    
    env_vars = {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', 'Not set'),
        'DB_HOST': os.getenv('DB_HOST', 'localhost'),
        'DB_NAME': os.getenv('DB_NAME', 'agrimind'),
        'DB_USER': os.getenv('DB_USER', 'agrimind'),
    }
    
    for key, value in env_vars.items():
        masked_value = value[:8] + '...' if key == 'GEMINI_API_KEY' and len(value) > 8 else value
        print(f"{key}: {masked_value}")
    
    # Test 4: Database connection (if configured)
    try:
        import psycopg2
        db_config = config.database
        
        print(f"\n🗄️  Testing database connection to {db_config.host}:{db_config.port}/{db_config.name}")
        
        conn = psycopg2.connect(**db_config.connection_params)
        print("✅ Database connection successful")
        conn.close()
        
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("   This is expected if PostgreSQL is not running or configured")
    
    return True

def suggest_setup_steps():
    """Suggest setup steps for the user"""
    print("\n📋 Setup Recommendations:")
    print("-" * 30)
    print("1. Set up environment variables:")
    print("   cp .env.example .env")
    print("   # Edit .env with your API keys")
    print("")
    print("2. Set up PostgreSQL database:")
    print("   # Install PostgreSQL with pgvector extension")
    print("   # Create database and user as configured")
    print("")
    print("3. Load knowledge base:")
    print("   python setup_db.py")
    print("")
    print("4. Test with a simple query:")
    print("   python cli.py --query 'What crops are suitable for West Bengal?'")

if __name__ == "__main__":
    print("🌾 AgriMind RAG System - Basic Configuration Test")
    print("=" * 60)
    
    success = test_basic_imports()
    
    if success:
        print("\n✅ Basic imports and configuration working!")
        suggest_setup_steps()
    else:
        print("\n❌ Basic setup issues detected. Please fix dependencies first.")
    
    sys.exit(0 if success else 1)
