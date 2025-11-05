#!/usr/bin/env python3
"""
Embeddings module for AgriMind RAG System
Handles text embedding generation using sentence transformers
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EmbeddingGenerator:
    """Generates embeddings for text using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        self.dimension = int(os.getenv('EMBEDDING_DIMENSION', '384'))
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        if not text or not text.strip():
            return np.zeros(self.dimension)
        
        try:
            embedding = self.model.encode(text.strip(), convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.dimension)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise ValueError("Model not loaded")
        
        if not texts:
            return []
        
        # Filter out empty texts
        processed_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(text.strip())
                original_indices.append(i)
        
        if not processed_texts:
            return [np.zeros(self.dimension) for _ in texts]
        
        try:
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.extend(batch_embeddings)
            
            # Map back to original order
            result_embeddings = [np.zeros(self.dimension) for _ in texts]
            for i, original_idx in enumerate(original_indices):
                result_embeddings[original_idx] = all_embeddings[i]
            
            return result_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return [np.zeros(self.dimension) for _ in texts]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension
