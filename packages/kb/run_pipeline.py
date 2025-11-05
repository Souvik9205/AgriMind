#!/usr/bin/env python3
"""
AgriMind Data Processing Pipeline
Runs the complete data processing workflow for West Bengal and Kolkata farming data
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_data_processor():
    """Run the main data processor"""
    print("\n🔄 Running data processor...")
    try:
        subprocess.check_call([sys.executable, "data_processor.py"])
        print("✅ Data processing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in data processing: {e}")
        return False

def run_rag_preprocessor():
    """Run the RAG preprocessor"""
    print("\n🔄 Running RAG preprocessor...")
    try:
        subprocess.check_call([sys.executable, "rag_preprocessor.py"])
        print("✅ RAG preprocessing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in RAG preprocessing: {e}")
        return False

def check_input_files():
    """Check if required input files exist"""
    csv_file = Path("others/9ef84268-d588-465a-a308-a864a43d0070.csv")
    pdf_files = list(Path("others").glob("*.pdf"))
    
    print("📋 Checking input files...")
    
    if csv_file.exists():
        print(f"✅ Market data CSV found: {csv_file}")
    else:
        print(f"❌ Market data CSV not found: {csv_file}")
        return False
    
    print(f"📄 Found {len(pdf_files)} PDF documents:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    return True

def create_output_summary():
    """Create a summary of processed outputs"""
    processed_path = Path("processed")
    rag_path = processed_path / "rag_ready"
    
    print("\n📊 Processing Summary:")
    print("=" * 50)
    
    if processed_path.exists():
        files = list(processed_path.glob("*"))
        print(f"✅ Processed files created: {len(files)}")
        for file in files:
            if file.is_file():
                print(f"   - {file.name}")
    
    if rag_path.exists():
        rag_files = list(rag_path.glob("*"))
        print(f"✅ RAG-ready files created: {len(rag_files)}")
        for file in rag_files:
            if file.is_file():
                print(f"   - {file.name}")
    
    # Show districts CSV files
    district_path = processed_path / "districts"
    if district_path.exists():
        district_files = list(district_path.glob("*.csv"))
        print(f"✅ District-wise CSV files created: {len(district_files)}")
        for file in district_files[:5]:  # Show first 5
            print(f"   - {file.name}")
        if len(district_files) > 5:
            print(f"   ... and {len(district_files) - 5} more files")

def main():
    """Main pipeline function"""
    print("🌾 AgriMind Data Processing Pipeline")
    print("🎯 Focus: West Bengal and Kolkata Agricultural Data")
    print("=" * 60)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Working directory: {current_dir}")
    
    # Check input files
    if not check_input_files():
        print("\n❌ Missing required input files. Please ensure:")
        print("   1. CSV file is in 'others/' directory")
        print("   2. PDF files are in 'others/' directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        return
    
    # Run data processor
    if not run_data_processor():
        print("\n❌ Data processing failed. Check the logs above.")
        return
    
    # Run RAG preprocessor
    if not run_rag_preprocessor():
        print("\n❌ RAG preprocessing failed. Check the logs above.")
        return
    
    # Create summary
    create_output_summary()
    
    print("\n🎉 Pipeline completed successfully!")
    print("\n📋 Next Steps for RAG Implementation:")
    print("   1. Use processed/rag_ready/rag_documents.json for embeddings")
    print("   2. Load embeddings_metadata.json for configuration")
    print("   3. Implement vector search using the chunked documents")
    print("   4. Set up your RAG model with West Bengal agricultural context")
    
    print("\n💡 RAG System Recommendations:")
    print("   - Use sentence-transformers/all-MiniLM-L6-v2 for embeddings")
    print("   - Implement semantic search with agricultural terminology")
    print("   - Consider local variety names and regional terms")
    print("   - Include seasonal and weather-based context")

if __name__ == "__main__":
    main()
