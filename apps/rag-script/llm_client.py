#!/usr/bin/env python3
"""
LLM client for AgriMind RAG System
Handles communication with Google Gemini 2.5 Flash
"""

import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class RAGResponse:
    """Response from the RAG system"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    context_used: str

class LLMClient:
    """Client for Google Gemini LLM"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '4000'))
        self.model_name = "gemini-2.5-flash"
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # System prompt for agricultural domain
        self.system_prompt = """You are AgriMind, an expert agricultural assistant specializing in West Bengal and Indian agriculture. You are deeply knowledgeable about:

🌾 WEST BENGAL AGRICULTURE:
- Sub-tropical monsoon climate with high humidity (75-85%)
- Alluvial soil in Gangetic plains, laterite in western districts
- Three cropping seasons: Kharif (June-Nov), Rabi (Nov-Apr), Zaid (Apr-Jun)
- Major crops: Rice (aman, aus, boro varieties), jute, tea, sugarcane, potato
- Key districts: Hooghly, Burdwan, Nadia (rice belt), Cooch Behar (jute), Darjeeling (tea)
- Traditional practices: SRI method for rice, organic farming in hills

🇮🇳 INDIAN AGRICULTURAL CONTEXT:
- ICAR recommendations and government schemes (PM-KISAN, soil health cards)
- MSP (Minimum Support Price) for major crops
- Weather patterns: Southwest monsoon (June-Sep), Northeast monsoon (Oct-Dec)
- Regional variations: Eastern India's high rainfall (1200-2000mm annually)
- Market systems: APMC mandis, e-NAM platform, FPOs (Farmer Producer Organizations)

🎯 YOUR EXPERTISE FOCUS:
1. Provide Bengal-specific crop varieties and cultivation timing
2. Consider local soil types, rainfall patterns, and temperature ranges
3. Reference nearby districts for comparative practices
4. Include traditional Bengali farming wisdom alongside modern techniques
5. Mention relevant government schemes and subsidies available in West Bengal
6. Consider local market preferences and cultural food habits
7. Address common regional challenges: flood management, pest issues, soil salinity

Always contextualize advice for West Bengal's unique agro-climatic conditions and Indian agricultural policies."""

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]], concise: bool = False) -> RAGResponse:
        """Generate a response using the LLM with retrieved context"""
        try:
            # Prepare context
            context = self._prepare_context(context_documents)
            
            # Create the prompt
            prompt = self._create_prompt(query, context, concise)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract sources
            sources = self._extract_sources(context_documents)
            
            # Calculate confidence (simple heuristic based on context relevance)
            confidence = self._calculate_confidence(context_documents)
            
            return RAGResponse(
                answer=response.text.strip(),
                sources=sources,
                confidence=confidence,
                context_used=context
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return RAGResponse(
                answer="I apologize, but I'm unable to generate a response at the moment. Please try again later.",
                sources=[],
                confidence=0.0,
                context_used=""
            )
    
    def _prepare_context(self, context_documents: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        if not context_documents:
            return "No relevant context found in the knowledge base."
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(context_documents):
            # Format document content
            doc_content = f"""
Source {i+1}: {doc.get('title', 'Unknown')}
Type: {doc.get('doc_type', 'Unknown')}
Relevance Score: {doc.get('similarity_score', 0):.3f}
Content: {doc.get('content', '')}
---
"""
            
            # Check if adding this document would exceed max context length
            if total_length + len(doc_content) > self.max_context_length:
                break
            
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, concise: bool = False) -> str:
        """Create the complete prompt for the LLM"""
        
        if concise:
            # Detect query type for ultra-concise responses
            query_lower = query.lower()
            is_identification_query = any(word in query_lower for word in 
                                        ['what is', 'what are', 'identify', 'this', 'these', 'spots', 'dots', 'marks'])
            
            if is_identification_query:
                # Ultra-concise for identification queries
                concise_system_prompt = """You are AgriMind. Answer in maximum 30 words. Be direct and specific."""
                
                prompt = f"""{concise_system_prompt}

CONTEXT: {context}

QUESTION: {query}

Answer format: **[Disease/Issue Name]** - [2-3 word description]. [One action needed]."""
            else:
                # Standard concise for other queries  
                concise_system_prompt = """You are AgriMind, an agricultural expert for West Bengal. Provide concise, actionable answers only. Maximum 60 words. Use bullet points."""
                
                prompt = f"""{concise_system_prompt}

CONTEXT: {context}

QUESTION: {query}

Format response as:
**Issue:** [1 line]
**Actions:**
• [Action 1]
• [Action 2] 
**Contact:** [KVK/local expert]"""
        else:
            # Original comprehensive prompt
            prompt = f"""{self.system_prompt}

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context above. If the context contains relevant information, use it to answer the question. If not, clearly state that the information is not available in the current knowledge base.

Include specific references to the sources when relevant (e.g., "According to the ICAR report..." or "Based on West Bengal market data...").

ANSWER:"""
        
        return prompt
    
    def _extract_sources(self, context_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context documents"""
        sources = []
        
        for doc in context_documents:
            metadata = doc.get('metadata', {})
            source = {
                'id': doc.get('id', ''),
                'title': doc.get('title', ''),
                'type': doc.get('doc_type', ''),
                'similarity_score': doc.get('similarity_score', 0),
                'source_file': metadata.get('source_file', ''),
                'region': metadata.get('region', ''),
                'data_type': metadata.get('data_type', '')
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, context_documents: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieved context quality"""
        if not context_documents:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        similarities = [doc.get('similarity_score', 0) for doc in context_documents]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost confidence if we have multiple relevant documents
        document_bonus = min(len(context_documents) / 5.0, 0.2)
        
        # Final confidence score (0.0 to 1.0)
        confidence = min(avg_similarity + document_bonus, 1.0)
        
        return confidence
    
    def is_agriculture_related(self, query: str) -> bool:
        """Check if a query is related to agriculture, farming, or crops"""
        agriculture_keywords = [
            # Core agriculture terms
            'crop', 'crops', 'farming', 'farm', 'farmer', 'farmers', 'agriculture', 'agricultural',
            'plant', 'plants', 'grow', 'growing', 'cultivation', 'cultivate', 'harvest', 'harvesting', 'yield',
            
            # Specific crops
            'rice', 'paddy', 'wheat', 'maize', 'corn', 'sugarcane', 'jute', 'potato', 'potatoes',
            'vegetables', 'tomato', 'onion', 'brinjal', 'cabbage', 'cauliflower', 'peas', 'beans',
            'mustard', 'sesame', 'groundnut', 'sunflower', 'coconut', 'banana', 'mango',
            
            # Farming inputs and practices
            'fertilizer', 'fertiliser', 'manure', 'compost', 'pesticide', 'insecticide', 'herbicide',
            'irrigation', 'water', 'soil', 'seed', 'seeds', 'nursery', 'transplant', 'sowing',
            
            # Seasons and timing
            'kharif', 'rabi', 'zaid', 'season', 'seasonal', 'monsoon', 'weather', 'rainfall', 'climate',
            
            # Market and economics
            'market', 'price', 'prices', 'selling', 'buying', 'mandi', 'procurement', 'subsidy',
            
            # Regional terms
            'west bengal', 'bengal', 'kolkata', 'calcutta', 'wb', 'icar', 'advisory', 'agro',
            'district', 'village', 'rural', 'krishi', 'kisan',
            
            # Problems and diseases
            'disease', 'pest', 'insect', 'fungus', 'blight', 'wilt', 'rot', 'damage', 'loss',
            
            # Technology and methods
            'organic', 'hybrid', 'variety', 'breed', 'technique', 'method', 'technology', 'innovation'
        ]
        
        query_lower = query.lower()
        
        # Check for direct keyword matches
        keyword_match = any(keyword in query_lower for keyword in agriculture_keywords)
        
        # Check for agriculture-related phrases
        agriculture_phrases = [
            'how to grow', 'when to plant', 'best time', 'crop rotation', 'soil preparation',
            'pest control', 'disease management', 'water management', 'nutrient management',
            'market rate', 'farming tips', 'agricultural advice', 'cultivation practices'
        ]
        
        phrase_match = any(phrase in query_lower for phrase in agriculture_phrases)
        
        return keyword_match or phrase_match

    def generate_fallback_response(self, query: str, concise: bool = False) -> RAGResponse:
        """Generate response using LLM's general knowledge when KB fails"""
        try:
            # Create appropriate prompt based on mode
            if concise:
                # Check if it's an identification query
                query_lower = query.lower()
                is_identification_query = any(word in query_lower for word in 
                                            ['what is', 'what are', 'identify', 'this', 'these', 'spots', 'dots', 'marks'])
                
                if is_identification_query:
                    fallback_prompt = f"""You are AgriMind, an agricultural expert. The user asked: "{query}"
                    
Based on general agricultural knowledge, provide a direct 25-word answer in format:
**[Disease/Issue Name]** - [brief description]. Apply [specific fungicide/treatment] immediately."""
                else:
                    fallback_prompt = f"""You are AgriMind, agricultural expert for West Bengal. User asked: "{query}"
                    
Provide concise answer (max 50 words) with format:
**Issue:** [problem]
**Action:** [2 main steps]
**Contact:** Local KVK"""
            else:
                # Full detailed response
                fallback_prompt = f"""You are AgriMind, an expert agricultural assistant specializing in West Bengal and Indian agriculture. 

The user has asked: "{query}"

While I don't have specific data from our knowledge base for this question, please provide helpful agricultural guidance based on your knowledge of:

� WEST BENGAL AGRICULTURE:
- Sub-tropical monsoon climate with distinct seasons (Kharif, Rabi, Zaid)
- Alluvial soil in Gangetic plains, suitable for rice and jute cultivation
- Major crops: Rice (multiple varieties), jute, potato, sugarcane, vegetables
- Traditional farming practices adapted to Bengal's high humidity and monsoon patterns
- Key agricultural districts: Hooghly, Burdwan, Nadia, Cooch Behar, Murshidabad

🇮🇳 INDIAN AGRICULTURAL CONTEXT:
- ICAR guidelines and research recommendations
- Government schemes and subsidies available to farmers
- Market systems including mandis and cooperative societies
- Seasonal patterns and weather considerations for Eastern India

Please provide:
1. Specific advice relevant to West Bengal's climate and soil conditions
2. Mention of appropriate crop varieties if applicable
3. Seasonal timing considerations for Bengal's agricultural calendar
4. References to local agricultural practices where relevant
5. Government resources or extension services that could help

Start your response by clearly stating this is general agricultural knowledge since specific data wasn't found in our knowledge base.

Provide practical, actionable advice that would be valuable for farmers in West Bengal."""
            
            response = self.model.generate_content(fallback_prompt)
            
            return RAGResponse(
                answer=response.text.strip(),
                sources=[],
                confidence=0.6,  # Lower confidence for fallback responses
                context_used="General West Bengal/Indian agricultural knowledge via LLM"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate LLM fallback response: {e}")
            
            # Check if it's a quota/API issue
            if "quota" in str(e).lower() or "429" in str(e):
                return RAGResponse(
                    answer=f"""I apologize, but I'm currently experiencing API limitations and cannot provide a detailed response to your question: "{query}"

This appears to be related to API quota limits. To resolve this:

1. **For the system administrator**: Please check and update the Gemini API key and quota limits
2. **For immediate help**: Please contact your local agricultural extension office or Krishi Vigyan Kendra
3. **Alternative resources**: 
   - West Bengal State Agriculture Department: wb.gov.in
   - ICAR research publications: icar.org.in
   - West Bengal Agricultural University: wbau.ac.in

Your question about "{query}" is important for agricultural planning. Please try again later or consult local agricultural experts for immediate guidance.""",
                    sources=[],
                    confidence=0.0,
                    context_used="API quota exceeded - unable to generate response"
                )
            else:
                return RAGResponse(
                    answer=f"""I apologize, but I'm currently unable to process your agricultural question: "{query}" due to a technical issue.

For immediate agricultural guidance, I recommend:

1. **Contact local experts**: Visit your nearest Krishi Vigyan Kendra or agricultural extension office
2. **Government resources**: Check West Bengal State Agriculture Department website (wb.gov.in)
3. **Research institutions**: West Bengal Agricultural University (Mohanpur) provides expert guidance
4. **ICAR resources**: Visit icar.org.in for research-based agricultural advice

Please try your question again later, or consult these local agricultural resources for immediate assistance.""",
                    sources=[],
                    confidence=0.0,
                    context_used=f"Technical error occurred: {str(e)}"
                )
    


    def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple response without RAG context"""
        try:
            response = self.model.generate_content(f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            return "I apologize, but I'm unable to respond at the moment. Please try again later."
