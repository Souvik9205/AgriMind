#!/usr/bin/env python3
"""
Utility functions for AgriMind RAG System
"""

import re
import string
from typing import List, Dict, Any, Optional
import numpy as np
import tiktoken

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'₹]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    if not text:
        return []
    
    # Simple keyword extraction based on word frequency
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    clean_text_lower = text.lower().translate(translator)
    
    # Split into words
    words = clean_text_lower.split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    }
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if len(word) > 2 and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, freq in sorted_words[:max_keywords]]
    
    return keywords

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def truncate_text(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to fit within token limit"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    except Exception:
        # Fallback: character-based truncation
        estimated_chars = max_tokens * 4
        return text[:estimated_chars] if len(text) > estimated_chars else text

def format_price(price: float, currency: str = "₹") -> str:
    """Format price with proper currency symbol"""
    if price == 0:
        return f"{currency}0"
    elif price < 1:
        return f"{currency}{price:.2f}"
    elif price < 1000:
        return f"{currency}{price:.0f}"
    elif price < 100000:
        return f"{currency}{price/1000:.1f}K"
    else:
        return f"{currency}{price/100000:.1f}L"

def extract_region_mentions(text: str) -> List[str]:
    """Extract West Bengal region mentions from text"""
    regions = []
    text_lower = text.lower()
    
    # West Bengal districts and regions
    wb_regions = [
        'west bengal', 'kolkata', 'howrah', 'hooghly', 'burdwan', 'purba bardhaman',
        'paschim bardhaman', 'birbhum', 'murshidabad', 'nadia', 'north 24 parganas',
        'south 24 parganas', 'purulia', 'bankura', 'paschim medinipur', 'purba medinipur',
        'jhargram', 'alipurduar', 'jalpaiguri', 'darjeeling', 'kalimpong', 'cooch behar',
        'uttar dinajpur', 'dakshin dinajpur', 'malda'
    ]
    
    for region in wb_regions:
        if region in text_lower:
            regions.append(region.title())
    
    return list(set(regions))  # Remove duplicates

def extract_crop_mentions(text: str) -> List[str]:
    """Extract crop mentions from text"""
    crops = []
    text_lower = text.lower()
    
    # Common crops in West Bengal
    crop_names = [
        'rice', 'paddy', 'wheat', 'maize', 'barley', 'sugarcane', 'jute', 'cotton',
        'potato', 'sweet potato', 'onion', 'garlic', 'ginger', 'turmeric', 'chili',
        'tomato', 'brinjal', 'cauliflower', 'cabbage', 'peas', 'beans', 'okra',
        'cucumber', 'bottle gourd', 'bitter gourd', 'ridge gourd', 'pumpkin',
        'mango', 'banana', 'guava', 'papaya', 'jackfruit', 'coconut', 'areca nut',
        'mustard', 'sesame', 'groundnut', 'sunflower', 'lentil', 'chickpea',
        'black gram', 'green gram', 'field pea', 'lathyrus'
    ]
    
    for crop in crop_names:
        if crop in text_lower:
            crops.append(crop.title())
    
    return list(set(crops))  # Remove duplicates

def calculate_similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    try:
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    except Exception:
        return 0.0

def parse_query_intent(query: str) -> Dict[str, Any]:
    """Parse query to understand user intent"""
    query_lower = query.lower()
    
    intent = {
        'type': 'general',
        'focus': [],
        'regions': extract_region_mentions(query),
        'crops': extract_crop_mentions(query),
        'keywords': extract_keywords(query, max_keywords=5)
    }
    
    # Determine query type
    if any(word in query_lower for word in ['price', 'cost', 'market', 'sell', 'buy', 'rate']):
        intent['type'] = 'market'
        intent['focus'].append('pricing')
    
    if any(word in query_lower for word in ['grow', 'plant', 'cultivation', 'farming', 'crop']):
        intent['type'] = 'agricultural'
        intent['focus'].append('cultivation')
    
    if any(word in query_lower for word in ['weather', 'climate', 'rain', 'temperature', 'season']):
        intent['focus'].append('weather')
    
    if any(word in query_lower for word in ['disease', 'pest', 'problem', 'issue', 'control']):
        intent['focus'].append('disease_management')
    
    if any(word in query_lower for word in ['fertilizer', 'nutrients', 'soil', 'manure']):
        intent['focus'].append('soil_nutrients')
    
    if any(word in query_lower for word in ['kharif', 'rabi', 'zaid', 'monsoon']):
        intent['focus'].append('seasonal')
    
    return intent

def validate_environment_config() -> Dict[str, Any]:
    """Validate environment configuration"""
    import os
    
    config = {
        'valid': True,
        'missing': [],
        'warnings': []
    }
    
    # Required environment variables
    required_vars = [
        'GEMINI_API_KEY',
        'DB_HOST',
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            config['valid'] = False
            config['missing'].append(var)
    
    # Optional but recommended variables
    recommended_vars = [
        'EMBEDDING_MODEL',
        'TOP_K_RESULTS',
        'SIMILARITY_THRESHOLD',
        'KNOWLEDGE_BASE_PATH'
    ]
    
    for var in recommended_vars:
        if not os.getenv(var):
            config['warnings'].append(f"Optional variable {var} not set, using default")
    
    return config

def format_rag_response(response_text: str, sources: List[Dict[str, Any]]) -> str:
    """Format RAG response with proper citations"""
    if not sources:
        return response_text
    
    # Add source citations to the response
    formatted_response = response_text
    
    # Add sources section
    formatted_response += "\n\n📚 **Sources:**\n"
    
    for i, source in enumerate(sources, 1):
        title = source.get('title', 'Unknown Source')
        doc_type = source.get('type', 'Unknown')
        similarity = source.get('similarity_score', 0)
        
        formatted_response += f"\n{i}. **{title}**\n"
        formatted_response += f"   - Type: {doc_type}\n"
        formatted_response += f"   - Relevance: {similarity:.1%}\n"
        
        # Add region if available
        if 'region' in source and source['region']:
            formatted_response += f"   - Region: {source['region']}\n"
    
    return formatted_response
