#!/usr/bin/env python3
"""
Configuration module for AgriMind RAG System
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    name: str = os.getenv('DB_NAME', 'agrimind')
    user: str = os.getenv('DB_USER', 'agrimind')
    password: str = os.getenv('DB_PASSWORD', 'agrimind')
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'database': self.name,
            'user': self.user,
            'password': self.password
        }

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    dimension: int = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
    
    @property
    def is_sentence_transformer(self) -> bool:
        return 'sentence-transformers' in self.model_name

@dataclass
class LLMConfig:
    """LLM configuration"""
    api_key: str = os.getenv('GEMINI_API_KEY', '')
    model_name: str = os.getenv('LLM_MODEL', 'gemini-2.5-flash')
    max_context_length: int = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
    temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = int(os.getenv('TOP_K_RESULTS', '5'))
    similarity_threshold: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    max_chunk_size: int = int(os.getenv('MAX_CHUNK_SIZE', '1000'))
    chunk_overlap: int = int(os.getenv('CHUNK_OVERLAP', '100'))
    enable_reranking: bool = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
    diverse_results: bool = os.getenv('DIVERSE_RESULTS', 'true').lower() == 'true'

@dataclass
class SystemConfig:
    """System-wide configuration"""
    knowledge_base_path: Path = Path(os.getenv('KNOWLEDGE_BASE_PATH', '../../packages/kb/dist'))
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    cache_enabled: bool = os.getenv('CACHE_ENABLED', 'false').lower() == 'true'
    cache_ttl: int = int(os.getenv('CACHE_TTL', '3600'))  # 1 hour
    max_query_length: int = int(os.getenv('MAX_QUERY_LENGTH', '500'))
    
    @property
    def rag_ready_path(self) -> Path:
        return self.knowledge_base_path / "rag_ready"
    
    @property
    def has_knowledge_base(self) -> bool:
        return self.knowledge_base_path.exists()

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.embedding = EmbeddingConfig()
        self.llm = LLMConfig()
        self.retrieval = RetrievalConfig()
        self.system = SystemConfig()
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration"""
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required configurations
        if not self.llm.is_configured:
            validation['valid'] = False
            validation['errors'].append("GEMINI_API_KEY is required")
        
        if not self.system.has_knowledge_base:
            validation['warnings'].append(f"Knowledge base not found at {self.system.knowledge_base_path}")
        
        # Check database configuration
        if not all([self.database.host, self.database.name, self.database.user]):
            validation['valid'] = False
            validation['errors'].append("Database configuration incomplete")
        
        # Validate numerical values
        if self.retrieval.top_k <= 0:
            validation['warnings'].append("TOP_K_RESULTS should be > 0")
        
        if not (0.0 <= self.retrieval.similarity_threshold <= 1.0):
            validation['warnings'].append("SIMILARITY_THRESHOLD should be between 0.0 and 1.0")
        
        if self.llm.max_context_length <= 0:
            validation['warnings'].append("MAX_CONTEXT_LENGTH should be > 0")
        
        return validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'name': self.database.name,
                'user': self.database.user
                # Note: password not included for security
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'dimension': self.embedding.dimension,
                'batch_size': self.embedding.batch_size
            },
            'llm': {
                'model_name': self.llm.model_name,
                'max_context_length': self.llm.max_context_length,
                'temperature': self.llm.temperature,
                'configured': self.llm.is_configured
            },
            'retrieval': {
                'top_k': self.retrieval.top_k,
                'similarity_threshold': self.retrieval.similarity_threshold,
                'max_chunk_size': self.retrieval.max_chunk_size,
                'chunk_overlap': self.retrieval.chunk_overlap,
                'enable_reranking': self.retrieval.enable_reranking,
                'diverse_results': self.retrieval.diverse_results
            },
            'system': {
                'knowledge_base_path': str(self.system.knowledge_base_path),
                'log_level': self.system.log_level,
                'cache_enabled': self.system.cache_enabled,
                'cache_ttl': self.system.cache_ttl,
                'max_query_length': self.system.max_query_length,
                'has_knowledge_base': self.system.has_knowledge_base
            }
        }
    
    @classmethod
    def from_env_file(cls, env_file: str = '.env') -> 'Config':
        """Load configuration from specific env file"""
        from dotenv import load_dotenv
        load_dotenv(env_file)
        return cls()

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def reload_config():
    """Reload configuration from environment"""
    global config
    load_dotenv(override=True)
    config = Config()
    return config

def print_config():
    """Print current configuration (for debugging)"""
    import json
    print("Current Configuration:")
    print(json.dumps(config.to_dict(), indent=2))

def validate_config() -> bool:
    """Validate current configuration and print results"""
    validation = config.validate()
    
    if validation['errors']:
        print("❌ Configuration Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("⚠️  Configuration Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['valid'] and not validation['warnings']:
        print("✅ Configuration is valid")
    
    return validation['valid']

if __name__ == "__main__":
    print_config()
    print("\nValidation:")
    validate_config()
