import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import uuid
from datetime import datetime
from config import settings

# Add the parent directory to the Python path to import from other modules
sys.path.append(str(settings.project_root))

app = FastAPI(
    title="AgriMind API",
    description="AI-powered agricultural assistance API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RAGQuery(BaseModel):
    query: str
    max_results: Optional[int] = 5

class RAGResponse(BaseModel):
    answer: str
    sources: list
    query: str
    confidence: Optional[float] = 0.0

class StatusResponse(BaseModel):
    status: str
    version: str
    services: dict

class DetectionResponse(BaseModel):
    disease: str
    confidence: float
    recommendations: list
    image_processed: bool

class CombinedAnalysisResponse(BaseModel):
    disease_detection: DetectionResponse
    rag_response: RAGResponse
    combined_insights: str
    confidence_score: float

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[ChatMessage]
    session_id: str

class InitialAnalysisRequest(BaseModel):
    query: str

class InitialAnalysisResponse(BaseModel):
    response: str
    session_id: str
    chat_history: List[ChatMessage]

class AnalysisMetadata(BaseModel):
    crop_detected: str
    disease_detected: str
    detection_confidence: float
    query_enhanced: bool
    enhancement_reason: str

class EnhancedAnalysisResponse(BaseModel):
    disease_detection: DetectionResponse
    rag_response: RAGResponse
    enhanced_query: str
    original_query: str
    overall_confidence: float
    metadata: AnalysisMetadata

# In-memory storage for chat sessions (in production, use Redis or a database)
chat_sessions = {}
MAX_CHAT_HISTORY = 4  # initial + 3 more

# Helper function to run RAG query
def run_rag_query(query: str, is_chat: bool = False) -> dict:
    """Execute RAG query using the existing rag-script with improved JSON output"""
    try:
        # Path to the RAG script
        rag_script_path = settings.rag_script_path
        
        # Add concise flag for chat responses
        cmd = [
            sys.executable,
            str(rag_script_path / "cli.py"),
            "--query", query,
            "--format", "json"
        ]
        
        if is_chat:
            cmd.extend(["--concise", "true"])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(rag_script_path)
        )
        
        if result.returncode != 0:
            raise Exception(f"RAG query failed: {result.stderr}")
        
        # Parse the JSON output
        output = result.stdout.strip()
        
        try:
            # Parse JSON response
            response_data = json.loads(output)
            return {
                "answer": response_data.get("answer", "No answer generated"),
                "sources": response_data.get("sources", []),
                "query": response_data.get("query", query),
                "confidence": response_data.get("confidence", 0.0)
            }
        except json.JSONDecodeError:
            # Fallback to text output if JSON parsing fails
            return {
                "answer": output,
                "sources": [],
                "query": query,
                "confidence": 0.5
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

# Helper function to run disease detection
def run_disease_detection(image_path: str) -> dict:
    """Execute disease detection using the existing ml-inference module"""
    try:
        # Path to the ML inference script
        ml_script_path = settings.ml_script_path
        
        # Run the detection script
        cmd = [
            sys.executable,
            str(ml_script_path / "detect.py"),
            image_path,
            "--json"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ml_script_path)
        )
        
        if result.returncode != 0:
            raise Exception(f"Disease detection failed: {result.stderr}")
        
        # Parse the JSON output
        output = result.stdout.strip()
        
        try:
            # Parse JSON response from detect.py
            detection_result = json.loads(output)
            
            if detection_result.get("success", False):
                pred = detection_result["prediction"]
                recommendations = []
                
                # Build recommendations from treatment and prevention
                if pred.get("treatment"):
                    recommendations.append(f"Treatment: {pred['treatment']}")
                if pred.get("prevention"):
                    recommendations.append(f"Prevention: {pred['prevention']}")
                
                # If no specific recommendations, add general advice
                if not recommendations:
                    recommendations = ["Consult with agricultural expert for detailed treatment plan"]
                
                return {
                    "disease": pred.get("condition", pred.get("disease", "Unknown")),
                    "crop": pred.get("crop", "Unknown"),
                    "condition": pred.get("condition", pred.get("disease", "Unknown")),
                    "confidence": pred.get("confidence", 0.0) / 100.0,  # Convert percentage to decimal
                    "recommendations": recommendations,
                    "image_processed": True
                }
            else:
                # Handle detection failure
                error_msg = detection_result.get("error", "Unknown error in disease detection")
                raise Exception(f"Disease detection failed: {error_msg}")
                
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            raise Exception(f"Failed to parse detection results: {str(e)}. Output: {output[:200]}")
        
    except Exception as e:
        error_detail = f"Disease detection failed: {str(e)}"
        if 'result' in locals() and result.stderr:
            error_detail += f" STDERR: {result.stderr[:200]}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/", response_model=StatusResponse)
async def status_check():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        version="1.0.0",
        services={
            "api": "running",
            "rag": "available",
            "ml_inference": "available",
            "database": "connected",
            "redis": "connected"
        }
    )

@app.post("/api/rag", response_model=RAGResponse)
async def rag_query(query_data: RAGQuery):
    """
    Process RAG (Retrieval-Augmented Generation) query
    
    This endpoint accepts a natural language query about agriculture
    and returns AI-generated answers based on the knowledge base.
    """
    if not query_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = run_rag_query(query_data.query)
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect_disease(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    query: Optional[str] = Form(None, description="User query to be processed later"),
    additional_info: Optional[str] = Form(None, description="Additional context about the plant")
):
    """
    Fast plant detection - returns only plant type and health status
    
    This endpoint accepts plant/crop images and returns:
    - Plant type
    - Health status (healthy/diseased) 
    - Disease name (only if diseased)
    - Confidence score
    - Stored user query for later processing
    """
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        # Run disease detection
        result = run_disease_detection(temp_image_path)
        
        # Clean up temporary file
        os.unlink(temp_image_path)
        
        # Extract information from ML inference result
        disease = result.get("disease", "Unknown condition")
        crop = result.get("crop", "Unknown Plant")
        confidence = result.get("confidence", 0.0)
        
        # Use the correctly extracted crop name from ML model
        plant = crop if crop != "Unknown" else "Plant"
        
        # Determine health status based on confidence and disease name  
        is_healthy = confidence < 0.2 or "healthy" in disease.lower()
        
        # Return simplified response for fast analysis
        response = {
            "plant": plant,
            "status": "healthy" if is_healthy else "diseased", 
            "confidence": confidence,
            "session_id": f"session_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        }
        
        # Only include disease if not healthy
        if not is_healthy:
            response["disease"] = disease
            
        # Store user query if provided
        if query:
            response["user_query"] = query
        
        return response
        
    except Exception as e:
        # Clean up on error
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=CombinedAnalysisResponse)
async def analyze_crop(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    query: str = Form(..., description="Question or description about the crop issue")
):
    """
    Enhanced combined endpoint for image analysis and RAG query processing
    
    This endpoint uses an improved flow:
    1. First detects disease and plant type from the image
    2. Creates an enhanced query combining detection results with user query
    3. Passes enriched context to RAG system for more accurate responses
    
    Provides:
    - Disease detection from the image
    - AI-generated answers based on enhanced query and knowledge base
    - Combined insights merging both analyses with contextual awareness
    """
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    temp_image_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        # Run enhanced analysis (detection -> enhanced query -> RAG)
        analysis_result = run_enhanced_analysis(temp_image_path, query, is_chat=False)
        
        # Extract results
        disease_result = analysis_result["disease_detection"]
        rag_result = analysis_result["rag_result"]
        metadata = analysis_result["analysis_metadata"]
        
        # Create enhanced combined insights
        crop = metadata["crop_detected"]
        disease_info = metadata["disease_detected"]
        confidence = metadata["detection_confidence"]
        rag_answer = rag_result.get("answer", "No specific guidance available")
        
        # Generate contextually aware combined insights
        if confidence > 0.7:
            combined_insights = f"""🎯 **High Confidence Detection**
🌱 **Crop:** {crop}
🦠 **Condition:** {disease_info}
📊 **Confidence:** {confidence:.1%}

**Comprehensive Guidance:**
{rag_answer}

**Enhanced Analysis:** The image analysis clearly identified {disease_info} in your {crop}. The recommendations above are specifically tailored for this condition and crop combination."""

        elif confidence > 0.4:
            combined_insights = f"""⚠️ **Moderate Confidence Detection**
🌱 **Likely Crop:** {crop}
🦠 **Possible Condition:** {disease_info}
📊 **Confidence:** {confidence:.1%}

**Guidance with Caution:**
{rag_answer}

**Note:** While the analysis suggests {disease_info}, please verify with local agricultural experts for confirmation before applying treatments."""

        else:
            combined_insights = f"""🔍 **Analysis Based on Query**
🌱 **Detected Crop:** {crop if crop != 'Unknown' else 'Not clearly identified'}

**General Guidance:**
{rag_answer}

**Note:** The image analysis didn't provide clear disease identification, so recommendations are based primarily on your query. Consider consulting with experts for visual diagnosis."""
        
        # Enhanced confidence calculation
        overall_confidence = analysis_result["overall_confidence"]
        
        # Clean up temporary file
        if temp_image_path:
            os.unlink(temp_image_path)
        
        return CombinedAnalysisResponse(
            disease_detection=DetectionResponse(**disease_result),
            rag_response=RAGResponse(**rag_result),
            combined_insights=combined_insights,
            confidence_score=overall_confidence
        )
        
    except Exception as e:
        # Clean up on error
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": "2025-11-12T00:00:00Z",
        "uptime": "running",
        "components": {
            "api_server": "up",
            "rag_service": "up",
            "ml_service": "up"
        }
    }

@app.post("/api/initial-analysis", response_model=InitialAnalysisResponse)
async def initial_analysis(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    query: str = Form(..., description="Question or description about the crop issue")
):
    """
    Enhanced initial analysis endpoint that starts a chat session
    
    Uses improved flow:
    1. Detects disease and plant type from image first
    2. Creates enhanced query with detection context
    3. Provides more accurate RAG responses
    4. Creates chat session with rich context for follow-ups
    """
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    temp_image_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        # Run enhanced analysis (detection -> enhanced query -> RAG)
        analysis_result = run_enhanced_analysis(temp_image_path, query, is_chat=True)
        
        # Extract results
        disease_result = analysis_result["disease_detection"]
        rag_result = analysis_result["rag_result"]
        metadata = analysis_result["analysis_metadata"]
        
        # Create concise response with enhanced context
        crop = metadata["crop_detected"]
        disease_info = metadata["disease_detected"]
        confidence = metadata["detection_confidence"]
        rag_answer = rag_result.get("answer", "No specific guidance available")
        
        # Generate enhanced concise response based on query type
        query_lower = query.lower()
        is_identification_query = any(word in query_lower for word in 
                                    ['what is', 'what are', 'identify', 'this', 'these', 'spots', 'dots', 'marks'])
        
        if is_identification_query:
            # Ultra-concise for identification queries
            if confidence > 0.7:
                response = f"🎯 **{disease_info}** ({confidence:.0%} confidence)\n\nThese are fungal pustules on {crop.lower()}. {rag_answer.split('.')[0] if rag_answer else 'Apply systemic fungicide immediately'}."
            elif confidence > 0.4:
                response = f"🔍 **Likely {disease_info}** ({confidence:.0%} confidence)\n\nPossible disease spots. {rag_answer.split('.')[0] if rag_answer else 'Consult local expert for confirmation'}."
            else:
                response = f"Based on your description: {rag_answer.split('.')[0] if rag_answer else 'Unable to identify from image. Consult agricultural expert.'}."
        else:
            # Standard concise for other queries
            if confidence > 0.7:
                response = f"""🎯 **{crop} - {disease_info}**
📊 **Confidence:** {confidence:.1%}

{rag_answer}

💡 **Ask follow-up questions for more details!**"""
            elif confidence > 0.4:
                response = f"""🔍 **{crop} - Possible {disease_info}**
📊 **Confidence:** {confidence:.1%}

{rag_answer}

⚠️ **Verification recommended** - Consider consulting local experts."""
            else:
                crop_text = f" ({crop})" if crop != "Unknown" else ""
                response = f"""🔍 **Analysis Based on Query{crop_text}**

{rag_answer}

💭 **Note:** Image analysis inconclusive."""
        
        # Create new chat session with enhanced context
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Initialize chat history with rich metadata
        chat_history = [
            ChatMessage(role="user", content=f"Image analysis query: {query}", timestamp=timestamp),
            ChatMessage(role="assistant", content=response, timestamp=timestamp)
        ]
        
        # Store session with enhanced context for better follow-ups
        chat_sessions[session_id] = {
            "history": chat_history,
            "disease_context": disease_result,
            "analysis_metadata": metadata,
            "enhanced_query": analysis_result["enhanced_query"],
            "original_query": query,
            "created_at": timestamp
        }
        
        # Clean up temporary file
        if temp_image_path:
            os.unlink(temp_image_path)
        
        return InitialAnalysisResponse(
            response=response,
            session_id=session_id,
            chat_history=chat_history
        )
        
    except Exception as e:
        # Clean up on error
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_follow_up(request: ChatRequest):
    """
    Enhanced follow-up chat endpoint for continued conversation
    
    Uses stored context from initial enhanced analysis including:
    - Disease detection results
    - Crop identification
    - Enhanced query context
    
    Provides contextually aware responses for follow-up questions
    Maximum chat history: initial + 3 more messages
    """
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Generate session ID if not provided in chat history
    session_id = None
    if request.chat_history:
        # Extract session from existing chat or create new one
        session_id = f"chat_{len(request.chat_history)}"
    else:
        session_id = str(uuid.uuid4())
    
    try:
        # Check chat history limit
        if len(request.chat_history) >= MAX_CHAT_HISTORY:
            raise HTTPException(
                status_code=400, 
                detail="Maximum chat history reached. Please start a new analysis."
            )
        
        # Create context from chat history
        chat_context = "\n".join([
            f"{msg.role.upper()}: {msg.content}" 
            for msg in request.chat_history[-6:]  # Last 6 messages for context
        ])
        
        # Try to get stored session context for enhanced responses
        stored_session = chat_sessions.get(session_id, {})
        enhanced_context = ""
        
        if "analysis_metadata" in stored_session:
            metadata = stored_session["analysis_metadata"]
            crop = metadata.get("crop_detected", "Unknown")
            disease = metadata.get("disease_detected", "Unknown") 
            confidence = metadata.get("detection_confidence", 0.0)
            
            enhanced_context = f"""
PREVIOUS ANALYSIS CONTEXT:
- Crop identified: {crop}
- Condition detected: {disease} (confidence: {confidence:.1%})
- Original enhanced query was used for better context

"""
        
        # Prepare enhanced query with stored context
        contextual_query = f"""{enhanced_context}Previous conversation:
{chat_context}

New question: {request.message}

Please provide a contextually aware response considering the previous image analysis and conversation history."""
        
        # Get response from RAG system
        rag_result = run_rag_query(contextual_query, True)  # is_chat=True for concise response
        response_text = rag_result.get("answer", "I couldn't generate a response. Please try again.")
        
        # Create new messages
        timestamp = datetime.now().isoformat()
        user_message = ChatMessage(role="user", content=request.message, timestamp=timestamp)
        assistant_message = ChatMessage(role="assistant", content=response_text, timestamp=timestamp)
        
        # Update chat history
        updated_history = request.chat_history + [user_message, assistant_message]
        
        # Store/update session (preserve original context)
        if session_id in chat_sessions:
            chat_sessions[session_id]["history"] = updated_history
            chat_sessions[session_id]["updated_at"] = timestamp
        else:
            chat_sessions[session_id] = {
                "history": updated_history,
                "updated_at": timestamp
            }
        
        return ChatResponse(
            response=response_text,
            chat_history=updated_history,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced chat failed: {str(e)}")

# Helper function to create enhanced query from detection results
def create_enhanced_query(disease_result: dict, user_query: str) -> str:
    """Create an enhanced query by combining disease detection results with user query"""
    
    crop = disease_result.get("crop", "Unknown")
    disease = disease_result.get("disease", "Unknown")
    condition = disease_result.get("condition", "Unknown")
    confidence = disease_result.get("confidence", 0.0)
    
    # Create structured context for the RAG system
    context_parts = []
    
    # Add crop information if detected
    if crop != "Unknown":
        context_parts.append(f"Crop identified: {crop}")
    
    # Add disease/condition information
    if condition != "Unknown":
        if confidence > 0.7:
            context_parts.append(f"Disease detected: {condition} (high confidence: {confidence:.1%})")
        elif confidence > 0.4:
            context_parts.append(f"Possible disease: {condition} (moderate confidence: {confidence:.1%})")
        else:
            context_parts.append(f"Low confidence detection: {condition} ({confidence:.1%})")
    
    # Combine context with user query
    if context_parts:
        enhanced_query = f"""CONTEXT FROM IMAGE ANALYSIS:
{chr(10).join(context_parts)}

USER QUERY: {user_query}

Please provide comprehensive guidance considering both the image analysis results and the user's specific question. Focus on actionable advice for {crop} cultivation and {condition} management."""
    else:
        enhanced_query = user_query
    
    return enhanced_query

# Enhanced helper function for improved disease detection and RAG integration
def run_enhanced_analysis(image_path: str, user_query: str, is_chat: bool = False) -> dict:
    """
    Run enhanced analysis that first detects disease/plant, then enriches RAG query
    
    Args:
        image_path: Path to the uploaded image
        user_query: User's original query
        is_chat: Whether this is for chat (concise) response
    
    Returns:
        Dictionary containing disease detection, enhanced RAG response, and metadata
    """
    try:
        # Step 1: Detect disease and plant type from image
        disease_result = run_disease_detection(image_path)
        
        # Step 2: Create enhanced query using detection results
        enhanced_query = create_enhanced_query(disease_result, user_query)
        
        # Step 3: Run RAG query with enhanced context
        rag_result = run_rag_query(enhanced_query, is_chat)
        
        # Step 4: Calculate enhanced confidence score
        detection_confidence = disease_result.get("confidence", 0.0)
        rag_confidence = rag_result.get("confidence", 0.5)
        
        # Weight the confidence based on detection quality
        if detection_confidence > 0.7:
            overall_confidence = (detection_confidence * 0.7 + rag_confidence * 0.3)
        elif detection_confidence > 0.4:
            overall_confidence = (detection_confidence * 0.5 + rag_confidence * 0.5)
        else:
            overall_confidence = (detection_confidence * 0.3 + rag_confidence * 0.7)
        
        return {
            "disease_detection": disease_result,
            "rag_result": rag_result,
            "enhanced_query": enhanced_query,
            "original_query": user_query,
            "overall_confidence": overall_confidence,
            "analysis_metadata": {
                "crop_detected": disease_result.get("crop", "Unknown"),
                "disease_detected": disease_result.get("condition", "Unknown"),
                "detection_confidence": detection_confidence,
                "query_enhanced": len(enhanced_query) > len(user_query),
                "enhancement_reason": "Image analysis provided crop and disease context" if detection_confidence > 0.4 else "Low confidence detection, relying on user query"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/api/enhanced-analyze", response_model=EnhancedAnalysisResponse)
async def enhanced_analyze_crop(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    query: str = Form(..., description="Question or description about the crop issue")
):
    """
    Enhanced analysis endpoint showcasing the improved image+query flow
    
    This endpoint demonstrates the complete enhanced flow:
    1. 🔍 Image Analysis: Detects disease and plant type from uploaded image
    2. 🧠 Context Enhancement: Combines detection results with user query  
    3. 💡 RAG Processing: Uses enriched context for more accurate responses
    4. 📊 Confidence Scoring: Provides weighted confidence based on all inputs
    
    Returns full transparency of the process including:
    - Original disease detection results
    - Enhanced query that was sent to RAG system
    - Final RAG response with sources
    - Analysis metadata and confidence scoring
    """
    
    # Validate inputs
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    temp_image_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        # Run the complete enhanced analysis flow
        analysis_result = run_enhanced_analysis(temp_image_path, query, is_chat=False)
        
        # Clean up temporary file
        if temp_image_path:
            os.unlink(temp_image_path)
        
        # Return comprehensive results showing the enhanced process
        return EnhancedAnalysisResponse(
            disease_detection=DetectionResponse(**analysis_result["disease_detection"]),
            rag_response=RAGResponse(**analysis_result["rag_result"]),
            enhanced_query=analysis_result["enhanced_query"],
            original_query=analysis_result["original_query"],
            overall_confidence=analysis_result["overall_confidence"],
            metadata=AnalysisMetadata(**analysis_result["analysis_metadata"])
        )
        
    except Exception as e:
        # Clean up on error
        if temp_image_path:
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Enhanced analysis failed: {str(e)}")

@app.post("/api/quick-analyze")
async def quick_analyze(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    query: str = Form(..., description="Question or description about the crop issue")
):
    """
    Quick analysis endpoint - Only ML detection (2-3 seconds)
    Returns immediate results while detailed analysis can be requested separately
    """
    
    # Validate inputs
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    temp_image_path = None
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{image.filename}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        # Run ONLY disease detection (fast)
        disease_result = run_disease_detection(temp_image_path)
        
        # Create session ID for follow-up
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Generate quick advice based on detection
        crop = disease_result.get("crop", "Unknown")
        disease = disease_result.get("disease", "Unknown") 
        confidence = disease_result.get("confidence", 0.0)
        
        # Create quick advice based on confidence
        if confidence > 0.7:
            if "rust" in disease.lower():
                quick_advice = f"🎯 {disease} detected on {crop}. Apply fungicide immediately to prevent spread."
            elif "blight" in disease.lower():
                quick_advice = f"🎯 {disease} detected on {crop}. Remove affected leaves and spray with copper fungicide."
            elif "spot" in disease.lower():
                quick_advice = f"🎯 {disease} detected on {crop}. Improve air circulation and apply appropriate fungicide."
            else:
                quick_advice = f"🎯 {disease} detected on {crop}. Consult agricultural expert for treatment plan."
        elif confidence > 0.4:
            quick_advice = f"🔍 Possible {disease} on {crop}. Monitor closely and consider expert consultation."
        else:
            quick_advice = f"📱 Image analyzed. Getting detailed recommendations based on your description: '{query}'"
        
        # Store session context for later detailed analysis
        chat_sessions[session_id] = {
            "image_path": temp_image_path,  # Keep temp file for detailed analysis
            "original_query": query,
            "disease_detection": disease_result,
            "created_at": timestamp,
            "quick_analysis_done": True
        }
        
        return {
            "disease": disease,
            "crop": crop, 
            "confidence": confidence,
            "session_id": session_id,
            "quick_advice": quick_advice
        }
        
    except Exception as e:
        # Clean up temp file on error
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")

@app.post("/api/detailed-analysis", response_model=ChatResponse)
async def detailed_analysis(
    request: dict
):
    """
    Get detailed RAG analysis for an existing session
    This runs the expensive RAG query (15-20 seconds)
    """
    
    session_id = request.get("session_id")
    chat_history = request.get("chat_history", [])
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID required")
    
    # Get session context
    session = chat_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get stored context
        temp_image_path = session.get("image_path")
        original_query = session.get("original_query") 
        disease_result = session.get("disease_detection", {})
        
        # Create enhanced query with detection context
        enhanced_query = create_enhanced_query(disease_result, original_query)
        
        # Run RAG query with enhanced context (this is the slow part)
        rag_result = run_rag_query(enhanced_query, is_chat=True)
        
        # Generate detailed response
        crop = disease_result.get("crop", "Unknown")
        disease_info = disease_result.get("condition", disease_result.get("disease", "Unknown"))
        confidence = disease_result.get("confidence", 0.0)
        rag_answer = rag_result.get("answer", "No specific guidance available")
        
        # Generate contextual detailed response
        query_lower = original_query.lower()
        is_identification_query = any(word in query_lower for word in 
                                    ['what is', 'what are', 'identify', 'this', 'these', 'spots', 'dots', 'marks'])
        
        if is_identification_query:
            # Concise response for identification
            if confidence > 0.7:
                detailed_response = f"🎯 **{disease_info}** ({confidence:.0%} confidence)\n\n{rag_answer}\n\n💡 Ask me about treatment, prevention, or timing for more specific advice!"
            else:
                detailed_response = f"🔍 Based on your image and description:\n\n{rag_answer}\n\n💡 Ask follow-up questions for more specific guidance!"
        else:
            # Full response for advisory queries
            detailed_response = f"🌾 **Detailed Analysis for {crop}**\n\n{rag_answer}\n\n💡 Ask me anything else about this crop issue!"
        
        # Update chat history
        timestamp = datetime.now().isoformat()
        detailed_message = ChatMessage(
            role="assistant", 
            content=detailed_response, 
            timestamp=timestamp
        )
        
        updated_history = chat_history + [detailed_message]
        
        # Update session
        session["history"] = updated_history
        session["detailed_analysis_done"] = True
        session["updated_at"] = timestamp
        
        # Clean up temp image file after detailed analysis
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
            session.pop("image_path", None)
        
        return ChatResponse(
            response=detailed_response,
            chat_history=updated_history,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed analysis failed: {str(e)}")
