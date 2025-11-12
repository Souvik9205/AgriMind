#!/usr/bin/env python3
"""
Simple test script to verify the ML inference system is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_model_loading():
    """Test if we can load the pre-trained model."""
    print("🧪 Testing model loading...")
    
    try:
        from inference import PlantDiseasePredictor
        
        predictor = PlantDiseasePredictor()
        info = predictor.get_model_info()
        
        print("✅ Model loaded successfully!")
        print(f"   Architecture: {info['backbone']}")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Parameters: {info['total_parameters']:,}")
        print(f"   Device: {info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_class_mapping():
    """Test if class mapping file is valid."""
    print("\n🧪 Testing class mapping...")
    
    try:
        import json
        from config import CLASS_MAPPING_PATH
        
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        print(f"✅ Class mapping loaded successfully!")
        print(f"   Total classes: {len(class_mapping)}")
        print(f"   Sample classes: {list(class_mapping.keys())[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Class mapping test failed: {e}")
        return False

def test_dummy_prediction():
    """Test prediction with a dummy tensor."""
    print("\n🧪 Testing dummy prediction...")
    
    try:
        from inference import PlantDiseasePredictor
        import torch
        import numpy as np
        from PIL import Image
        
        predictor = PlantDiseasePredictor()
        
        # Create a dummy RGB image (224x224x3)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_image)
        
        # Make prediction
        result = predictor.predict(dummy_pil, top_k=3)
        
        print("✅ Dummy prediction successful!")
        print(f"   Predicted class: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Top predictions: {len(result['top_predictions'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dummy prediction failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "output/crop_best_model.pth",
        "models/class_mapping.json",
        "src/inference.py",
        "src/models.py",
        "src/config.py",
        "tool.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found!")
        return True

def main():
    """Run all tests."""
    print("🔬 AgriMind ML Inference Test Suite")
    print("=" * 40)
    
    tests = [
        test_file_structure,
        test_class_mapping,
        test_model_loading,
        test_dummy_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTry running:")
        print("   python tool.py info")
        print("   python tool.py predict <your_image.jpg>")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
