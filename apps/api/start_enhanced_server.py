#!/usr/bin/env python3
"""
Enhanced AgriMind API Server Startup Script
Includes the new image+query enhanced flow
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'python-multipart',
        'Pillow',
        'torch',
        'transformers',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_server():
    """Start the enhanced API server"""
    
    print("🌾 Starting Enhanced AgriMind API Server")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get current directory
    api_dir = Path(__file__).parent
    
    print("✅ Dependencies OK")
    print(f"📂 API Directory: {api_dir}")
    print("🚀 Starting server...")
    print("\n🔗 Available endpoints:")
    print("   • GET  /                    - Health check")
    print("   • POST /api/rag             - Regular RAG query")
    print("   • POST /api/detect          - Image disease detection")
    print("   • POST /api/analyze         - Enhanced combined analysis")
    print("   • POST /api/enhanced-analyze - Full enhanced flow with transparency")
    print("   • POST /api/initial-analysis - Enhanced analysis with chat session")
    print("   • POST /api/chat            - Enhanced chat with context")
    print("   • GET  /api/health          - Detailed health check")
    print("\n" + "=" * 50)
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app",
            "--reload",
            "--host", "0.0.0.0", 
            "--port", "8000"
        ]
        
        subprocess.run(cmd, cwd=api_dir)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
