#!/usr/bin/env python3
"""
AgriMind Knowledge Base Processor
Processes agricultural data for West Bengal and Kolkata
Usage: python process.py [--clean] [--output-dir DIR] [--input-dir DIR]
"""

import subprocess
import sys
import os
import argparse
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

def check_input_files(input_dir):
    """Check if required input files exist"""
    csv_files = list(input_dir.glob("*.csv"))
    pdf_files = list(input_dir.glob("*.pdf"))
    
    print("📋 Checking input files...")
    
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV file(s):")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
    else:
        print(f"❌ No CSV files found in {input_dir}")
        return False
    
    if pdf_files:
        print(f"📄 Found {len(pdf_files)} PDF document(s):")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
    else:
        print(f"ℹ️ No PDF files found in {input_dir}")
    
    return True

def run_processors(input_dir, output_dir):
    """Run both data processor and RAG preprocessor"""
    print("\n🔄 Running knowledge base processors...")
    
    # Import and run processors
    try:
        # Import the processor modules
        sys.path.insert(0, str(Path(__file__).parent))
        from data_processor import AgriDataProcessor
        from rag_preprocessor import RAGPreprocessor
        
        # Run data processor
        print("Processing agricultural data...")
        processor = AgriDataProcessor(kb_path=input_dir.parent, output_path=output_dir)
        
        # Find CSV files and process them
        csv_files = list(input_dir.glob("*.csv"))
        for csv_file in csv_files:
            processor.process_market_data(str(csv_file))
        
        # Process PDF documents
        processor.process_pdf_documents(pdf_dir=input_dir.name)
        
        # Create knowledge base index
        processor.create_knowledge_base_index()
        
        # Run RAG preprocessor
        print("Creating RAG documents...")
        rag_processor = RAGPreprocessor(processed_data_path=str(output_dir))
        rag_processor.create_rag_documents()
        rag_processor.create_embeddings_metadata()
        
        print("✅ Processing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return False

def create_output_summary(output_dir):
    """Create a summary of processed outputs"""
    print("\n📊 Processing Summary:")
    print("=" * 50)
    
    if output_dir.exists():
        json_files = list(output_dir.glob("*.json"))
        print(f"✅ Processed files: {len(json_files)}")
        for file in json_files:
            print(f"   - {file.name}")
    
    rag_path = output_dir / "rag_ready"
    if rag_path.exists():
        rag_files = list(rag_path.glob("*"))
        print(f"✅ RAG-ready files: {len(rag_files)}")
        for file in rag_files:
            print(f"   - {file.name}")
    
    districts_path = output_dir / "districts"
    if districts_path.exists():
        district_files = list(districts_path.glob("*.csv"))
        print(f"✅ District files: {len(district_files)}")

def main():
    """Main processor function"""
    parser = argparse.ArgumentParser(description='Process AgriMind knowledge base data')
    parser.add_argument('--clean', action='store_true', help='Clean output directory before processing')
    parser.add_argument('--output-dir', default='dist', help='Output directory for processed data')
    parser.add_argument('--input-dir', default='data', help='Input directory containing raw data')
    
    args = parser.parse_args()
    
    print("🌾 AgriMind Knowledge Base Processor")
    print("🎯 Focus: West Bengal and Kolkata Agricultural Data")
    print("=" * 60)
    
    # Resolve paths relative to kb package root
    kb_root = Path(__file__).parent.parent
    output_dir = kb_root / args.output_dir
    input_dir = kb_root / args.input_dir
    
    # Clean if requested
    if args.clean and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"🧹 Cleaned output directory: {output_dir}")
    
    # Create directories
    output_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"📂 Input directory: {input_dir}")
    print(f"📤 Output directory: {output_dir}")
    
    # Check input files
    if not check_input_files(input_dir):
        print(f"\n❌ Please add your data files to: {input_dir}")
        print("   - CSV files with market data")
        print("   - PDF files with agricultural reports")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        return
    
    # Run processors
    if not run_processors(input_dir, output_dir):
        print("\n❌ Processing failed")
        return
    
    # Create summary
    create_output_summary(output_dir)
    
    print("\n🎉 Knowledge base processing completed!")
    print(f"\n📤 Processed data available in: {output_dir}")
    print("\n📋 Next Steps:")
    print("   1. Copy rag_ready/ to apps/api-rag/src/data/")
    print("   2. Set up vector database in api-rag")
    print("   3. Implement RAG endpoints")

if __name__ == "__main__":
    main()
