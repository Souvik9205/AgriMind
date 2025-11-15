#!/usr/bin/env python3
"""
Quick test for the ML inference detect.py script
"""

import sys
import subprocess
import tempfile
from PIL import Image
from pathlib import Path

def create_test_image():
    """Create a simple test image"""
    # Create a small green image (simulating a healthy leaf)
    img = Image.new('RGB', (224, 224), color=(0, 128, 0))
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img.save(tmp_file.name)
        return tmp_file.name

def test_detect_script():
    """Test the detect.py script"""
    
    # Path to detect script
    script_path = Path(__file__).parent.parent / "apps" / "ml-inference" / "detect.py"
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Create test image
    test_image = create_test_image()
    
    try:
        print(f"🧪 Testing detect.py with test image...")
        print(f"   Script: {script_path}")
        print(f"   Test image: {test_image}")
        
        # Run the script with JSON output
        cmd = [sys.executable, str(script_path), test_image, "--json"]
        
        print(f"   Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        print(f"   Return code: {result.returncode}")
        print(f"   STDOUT length: {len(result.stdout)} characters")
        print(f"   STDERR length: {len(result.stderr)} characters")
        
        if result.returncode == 0:
            print("✅ Script executed successfully")
            print(f"   Output preview: {result.stdout[:200]}...")
            
            # Try to parse JSON
            import json
            try:
                data = json.loads(result.stdout)
                print("✅ JSON parsing successful")
                if data.get("success"):
                    pred = data["prediction"]
                    print(f"   Disease: {pred.get('disease', 'N/A')}")
                    print(f"   Confidence: {pred.get('confidence', 'N/A')}%")
                    print(f"   Crop: {pred.get('crop', 'N/A')}")
                else:
                    print(f"   Error in detection: {data.get('error', 'Unknown error')}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"   Raw output: {result.stdout}")
                return False
        else:
            print("❌ Script execution failed")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script execution timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up test image
        try:
            Path(test_image).unlink()
        except:
            pass

if __name__ == "__main__":
    print("🔬 Testing ML Inference Script")
    print("=" * 40)
    
    success = test_detect_script()
    
    if success:
        print("\n🎉 ML inference test passed!")
        print("The detect.py script is working correctly.")
    else:
        print("\n⚠️ ML inference test failed!")
        print("Please check the error messages above.")
        print("\nTroubleshooting tips:")
        print("- Ensure all dependencies are installed: pip install -r apps/ml-inference/requirements.txt")
        print("- Check if the model downloads successfully on first run")
        print("- Verify internet connectivity for model download")
    
    sys.exit(0 if success else 1)
