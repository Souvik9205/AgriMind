import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # Database Configuration
    database_url: str = "postgresql://agrimind:agrimind@localhost:5432/agrimind"
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # ML Configuration
    ml_model_path: str = str(Path(__file__).parent.parent.parent / "apps" / "ml-inference" / "models")
    ml_confidence_threshold: float = 0.7
    
    # RAG Configuration
    rag_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_max_results: int = 5
    
    # Security
    secret_key: str = "change-me-in-production"
    cors_origins: str = "*"
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = ["jpg", "jpeg", "png", "bmp", "tiff"]
    
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent
    rag_script_path: Path = project_root / "apps" / "rag-script"
    ml_script_path: Path = project_root / "apps" / "ml-inference"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()
