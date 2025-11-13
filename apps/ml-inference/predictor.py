"""
Smart Plant Disease Predictor
Uses local trained models when available, falls back to Hugging Face model
"""

from huggingface_predictor import HuggingFacePredictor
from local_predictor import LocalModelPredictor
import argparse
from pathlib import Path

def predict_disease(image_path: str, top_k: int = 3):
    """
    Smart prediction function - uses local model if available, otherwise Hugging Face
    
    Args:
        image_path: Path to the image file
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Try local model first
    local_predictor = LocalModelPredictor()
    
    if local_predictor.is_available():
        print("🏠 Using local trained model...")
        result = local_predictor.infer(image_path, top_k)
        
        # Format result to match the old predictor interface
        formatted_result = {
            'predicted_class': result['label'],
            'confidence': result['confidence'],
            'top_predictions': result['top_predictions'],
            'model_type': f"Local_{result['model_info']['architecture']}",
            'model_name': result['model_info']['model_name'],
            'training_accuracy': result['model_info']['training_accuracy'],
            'image_path': image_path
        }
        
    else:
        print("🌐 Using Hugging Face model...")
        # Fallback to HuggingFace predictor
        hf_predictor = HuggingFacePredictor()
        result = hf_predictor.infer(image_path)
        
        # Format result to match the old predictor interface
        formatted_result = {
            'predicted_class': result['label'],
            'confidence': result['confidence'],
            'top_predictions': [{
                'class': result['label'],
                'confidence': result['confidence']
            }],
            'model_type': 'HuggingFace_ViT',
            'image_path': image_path
        }
    
    return formatted_result

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Plant Disease Predictor (HuggingFace)')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    try:
        result = predict_disease(args.image_path, args.top_k)
        
        print(f"\n🔍 Analysis Results for: {Path(args.image_path).name}")
        print("=" * 50)
        print(f"🎯 Predicted Disease: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"🤖 Model: {result['model_type']}")
        
        # Show additional model info for local models
        if 'Local' in result['model_type']:
            print(f"📝 Model Name: {result.get('model_name', 'N/A')}")
            print(f"🎯 Training Accuracy: {result.get('training_accuracy', 0):.1%}")
        
        # Show top predictions if available
        if len(result['top_predictions']) > 1:
            print(f"\n📋 Top {len(result['top_predictions'])} Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence']:.1%}")
        
        # Additional disease information
        disease = result['predicted_class']
        print(f"\n📋 Disease Information:")
        
        if 'Brown_Rust' in disease:
            print("   🌾 Brown Rust (Leaf Rust)")
            print("   🔸 Crop: Wheat")
            print("   🔸 Severity: Moderate to High")
            print("   🔸 Treatment: Fungicide application recommended")
            print("   🔸 Prevention: Resistant varieties, proper spacing")
        elif 'Healthy' in disease:
            print("   ✅ Healthy Plant")
            print("   🔸 No disease detected")
            print("   🔸 Continue regular monitoring")
        elif 'Blight' in disease:
            print("   🍂 Blight Disease")
            print("   🔸 Severity: High")
            print("   🔸 Treatment: Remove affected leaves, fungicide")
        elif 'Spot' in disease:
            print("   🔴 Leaf Spot Disease")
            print("   🔸 Severity: Moderate")
            print("   🔸 Treatment: Improve air circulation, fungicide")
        else:
            print("   🔍 Check agricultural resources for specific treatment")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
