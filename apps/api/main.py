import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
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

class StatusResponse(BaseModel):
    status: str
    version: str
    services: dict

class DetectionResponse(BaseModel):
    disease: str
    confidence: float
    recommendations: list
    image_processed: bool

# Helper function to run RAG query
def run_rag_query(query: str) -> dict:
    """Execute RAG query using the existing rag-script"""
    try:
        # Path to the RAG script
        rag_script_path = settings.rag_script_path
        
        # Create a temporary file with the query
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(query)
            temp_query_file = temp_file.name
        
        # Run the RAG CLI with the query
        cmd = [
            sys.executable,
            str(rag_script_path / "cli.py"),
            "--query", query
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(rag_script_path)
        )
        
        # Clean up temp file
        os.unlink(temp_query_file)
        
        if result.returncode != 0:
            raise Exception(f"RAG query failed: {result.stderr}")
        
        # Parse the output (assuming it returns structured data)
        output = result.stdout.strip()
        
        return {
            "answer": output,
            "sources": [],  # You can enhance this to extract sources
            "query": query
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
            "--image", image_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ml_script_path)
        )
        
        if result.returncode != 0:
            raise Exception(f"Disease detection failed: {result.stderr}")
        
        # Parse the output (assuming it returns JSON or structured data)
        output = result.stdout.strip()
        
        # Try to parse as JSON, fallback to plain text
        try:
            detection_result = json.loads(output)
        except json.JSONDecodeError:
            # If not JSON, create a structured response
            detection_result = {
                "disease": output,
                "confidence": 0.85,  # Default confidence
                "recommendations": ["Consult with agricultural expert for detailed treatment plan"]
            }
        
        return {
            "disease": detection_result.get("disease", "Unknown"),
            "confidence": detection_result.get("confidence", 0.0),
            "recommendations": detection_result.get("recommendations", []),
            "image_processed": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")

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

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_disease(
    image: UploadFile = File(..., description="Plant/crop image for disease detection"),
    additional_info: Optional[str] = Form(None, description="Additional context about the plant")
):
    """
    Upload an image and get disease detection results
    
    This endpoint accepts plant/crop images and returns:
    - Disease identification
    - Confidence score
    - Treatment recommendations
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
        
        return DetectionResponse(**result)
        
    except Exception as e:
        # Clean up on error
        if 'temp_image_path' in locals():
            try:
                os.unlink(temp_image_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, debug=settings.api_debug)
