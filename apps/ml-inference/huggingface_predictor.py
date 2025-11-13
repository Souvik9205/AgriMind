"""
Hugging Face Plant Disease Inference Script
Using pre-trained Vision Transformer model from Hugging Face Hub
"""

import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import argparse
from pathlib import Path
import json
from typing import Dict, Union, List
import time

class HuggingFacePredictor:
    """Plant disease predictor using Hugging Face Vision Transformer"""
    
    def __init__(self, repo_id: str = "wambugu71/crop_leaf_diseases_vit"):
        """
        Initialize the Hugging Face predictor
        
        Args:
            repo_id: Hugging Face model repository ID
        """
        self.repo_id = repo_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"🤗 Loading Hugging Face model: {repo_id}")
        print(f"📱 Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModelForImageClassification.from_pretrained(repo_id)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Number of classes: {len(self.model.config.id2label)}")
        
        # Print available classes
        print("🏷️ Available disease classes:")
        for idx, label in self.model.config.id2label.items():
            print(f"   {idx}: {label}")
    
    def infer(self, img_path_or_url: str) -> Dict[str, Union[str, float]]:
        """
        Perform inference on a single image
        
        Args:
            img_path_or_url: Path to local image or URL to image
            
        Returns:
            Dictionary with prediction label and confidence score
        """
        try:
            # Load image
            if img_path_or_url.startswith("http"):
                img = Image.open(requests.get(img_path_or_url, stream=True).raw).convert("RGB")
            else:
                img = Image.open(img_path_or_url).convert("RGB")
            
            # Process image
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get prediction
            pred_id = int(logits.argmax(-1))
            label = self.model.config.id2label[pred_id]
            score = float(logits.softmax(-1).max())
            
            return {
                "label": label, 
                "confidence": round(score, 4),
                "predicted_class": label,  # For compatibility
                "image_path": img_path_or_url
            }
            
        except Exception as e:
            return {
                "label": "Error", 
                "confidence": 0.0,
                "predicted_class": "Error",
                "error": str(e),
                "image_path": img_path_or_url
            }
    
    def predict_batch(self, image_paths: List[str], top_k: int = 3) -> List[Dict]:
        """
        Perform inference on multiple images
        
        Args:
            image_paths: List of image paths or URLs
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            try:
                # Load image
                if img_path.startswith("http"):
                    img = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
                else:
                    img = Image.open(img_path).convert("RGB")
                
                # Process image
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Perform inference
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probabilities = torch.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.model.config.id2label)))
                
                predictions = []
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    class_name = self.model.config.id2label[idx.item()]
                    predictions.append({
                        'class': class_name,
                        'confidence': float(prob)
                    })
                
                result = {
                    'image_path': img_path,
                    'predicted_class': predictions[0]['class'],
                    'confidence': predictions[0]['confidence'],
                    'top_predictions': predictions
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'predicted_class': 'Error',
                    'confidence': 0.0,
                    'error': str(e),
                    'top_predictions': []
                })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "repo_id": self.repo_id,
            "device": self.device,
            "num_classes": len(self.model.config.id2label),
            "classes": list(self.model.config.id2label.values()),
            "model_type": type(self.model).__name__
        }

def test_on_sample_images(predictor: HuggingFacePredictor, test_dir: Path, num_samples: int = 5):
    """Test the predictor on sample images from test directory"""
    test_images = list(test_dir.glob("*.jpg"))[:num_samples]
    
    if not test_images:
        print(f"❌ No test images found in {test_dir}")
        return
    
    print(f"\n🧪 Testing on {len(test_images)} sample images:")
    print("=" * 60)
    
    for img_path in test_images:
        print(f"\n📷 Testing: {img_path.name}")
        
        start_time = time.time()
        result = predictor.infer(str(img_path))
        inference_time = time.time() - start_time
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🎯 Prediction: {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"⏱️ Time: {inference_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='Hugging Face Plant Disease Predictor')
    parser.add_argument('--image', help='Path to image file or URL')
    parser.add_argument('--test-dir', help='Test on sample images from directory')
    parser.add_argument('--repo-id', default="wambugu71/crop_leaf_diseases_vit", 
                       help='Hugging Face model repository ID')
    parser.add_argument('--batch', nargs='+', help='Batch inference on multiple images')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = HuggingFacePredictor(repo_id=args.repo_id)
        
        if args.image:
            # Single image inference
            print(f"\n🔍 Analyzing: {Path(args.image).name if not args.image.startswith('http') else args.image}")
            print("=" * 50)
            
            result = predictor.infer(args.image)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🎯 Predicted Disease: {result['label']}")
                print(f"📊 Confidence: {result['confidence']:.1%}")
        
        elif args.batch:
            # Batch inference
            print(f"\n🔍 Batch Analysis of {len(args.batch)} images:")
            print("=" * 50)
            
            results = predictor.predict_batch(args.batch, args.top_k)
            
            for result in results:
                img_name = Path(result['image_path']).name if not result['image_path'].startswith('http') else result['image_path']
                print(f"\n📷 {img_name}")
                
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   🎯 Prediction: {result['predicted_class']}")
                    print(f"   📊 Confidence: {result['confidence']:.1%}")
                    
                    if len(result['top_predictions']) > 1:
                        print(f"   📈 Top {len(result['top_predictions'])} predictions:")
                        for i, pred in enumerate(result['top_predictions'], 1):
                            print(f"      {i}. {pred['class']}: {pred['confidence']:.1%}")
        
        elif args.test_dir:
            # Test on sample images
            test_dir = Path(args.test_dir)
            test_on_sample_images(predictor, test_dir)
        
        else:
            # Show model info and test on default test images
            print("\n📋 Model Information:")
            info = predictor.get_model_info()
            print(f"   Repository: {info['repo_id']}")
            print(f"   Device: {info['device']}")
            print(f"   Classes: {info['num_classes']}")
            
            # Test on default test images if available
            test_dir = Path(__file__).parent / "test_images"
            if test_dir.exists():
                test_on_sample_images(predictor, test_dir)
            else:
                print(f"\n💡 Usage examples:")
                print(f"   python {Path(__file__).name} --image path/to/image.jpg")
                print(f"   python {Path(__file__).name} --test-dir test_images/")
                print(f"   python {Path(__file__).name} --batch img1.jpg img2.jpg img3.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
