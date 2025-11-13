"""
Local Model Predictor
Uses locally trained custom models for plant disease prediction
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlantDiseaseModel(nn.Module):
    """Custom plant disease classification model based on ResNet"""
    
    def __init__(self, num_classes: int):
        super(PlantDiseaseModel, self).__init__()
        
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class LocalModelPredictor:
    """Predictor using locally trained models"""
    
    def __init__(self, model_root: str = "models"):
        self.model_root = Path(model_root)
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.model_info = {}
        self.transform = None
        
        # Try to load the best available model
        self._load_best_model()
        
    def _get_available_models(self) -> List[Dict]:
        """Get list of available trained models"""
        if not self.model_root.exists():
            return []
            
        models = []
        
        # Look for model info files
        for info_file in self.model_root.glob("*_info.json"):
            try:
                with open(info_file, 'r') as f:
                    model_info = json.load(f)
                    
                # Check if model files exist
                model_name = model_info["model_name"]
                best_model_file = self.model_root / f"{model_name}_best.pth"
                
                if best_model_file.exists():
                    model_info["model_path"] = best_model_file
                    model_info["info_path"] = info_file
                    models.append(model_info)
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not load model info from {info_file}: {e}")
                continue
                
        # Sort by validation accuracy (best first)
        models.sort(key=lambda x: x.get("best_val_accuracy", 0), reverse=True)
        
        return models
        
    def _load_best_model(self) -> bool:
        """Load the best available model"""
        available_models = self._get_available_models()
        
        if not available_models:
            print("📭 No trained models found in models directory")
            return False
            
        # Use the best model
        best_model_info = available_models[0]
        model_path = best_model_info["model_path"]
        
        try:
            print(f"🔄 Loading local model: {best_model_info['model_name']}")
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract model info
            self.num_classes = checkpoint["num_classes"]
            self.class_names = checkpoint["class_names"]
            self.model_info = best_model_info
            
            # Initialize model
            self.model = PlantDiseaseModel(num_classes=self.num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(device)
            self.model.eval()
            
            # Setup transforms
            self._setup_transforms()
            
            print(f"✅ Local model loaded successfully!")
            print(f"   📊 Classes: {len(self.class_names)}")
            print(f"   🏆 Validation accuracy: {best_model_info.get('best_val_accuracy', 'N/A'):.1%}")
            print(f"   📅 Trained: {best_model_info.get('training_date', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {model_path}: {e}")
            self.model = None
            return False
            
    def _setup_transforms(self):
        """Setup image transforms for inference"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def is_available(self) -> bool:
        """Check if a local model is available"""
        return self.model is not None
        
    def infer(self, image_path: str, top_k: int = 3) -> Dict:
        """
        Predict plant disease using local model
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_available():
            raise RuntimeError("No local model available")
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
            # Get top predictions
            top_prob, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
            
            top_predictions = []
            for i in range(len(top_prob)):
                class_name = self.class_names[top_indices[i]]
                confidence = float(top_prob[i])
                top_predictions.append({
                    "class": class_name,
                    "confidence": confidence
                })
                
            # Format result
            result = {
                "label": top_predictions[0]["class"],
                "confidence": top_predictions[0]["confidence"],
                "top_predictions": top_predictions,
                "model_info": {
                    "type": "local_trained",
                    "architecture": "ResNet50",
                    "model_name": self.model_info.get("model_name", "unknown"),
                    "training_accuracy": self.model_info.get("best_val_accuracy", 0),
                    "num_classes": self.num_classes,
                    "classes": self.class_names
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
            
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_available():
            return {"available": False}
            
        return {
            "available": True,
            "model_name": self.model_info.get("model_name", "unknown"),
            "architecture": "ResNet50",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "training_accuracy": self.model_info.get("best_val_accuracy", 0),
            "training_date": self.model_info.get("training_date", "unknown"),
            "model_path": str(self.model_info.get("model_path", ""))
        }
        
    def list_available_models(self) -> List[Dict]:
        """List all available trained models"""
        return self._get_available_models()

# For backward compatibility and easy import
def load_local_predictor(model_root: str = "models") -> Optional[LocalModelPredictor]:
    """
    Load local predictor if available
    
    Args:
        model_root: Directory containing trained models
        
    Returns:
        LocalModelPredictor instance if available, None otherwise
    """
    try:
        predictor = LocalModelPredictor(model_root)
        return predictor if predictor.is_available() else None
    except Exception as e:
        print(f"⚠️  Warning: Could not load local predictor: {e}")
        return None

def main():
    """Test the local predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test local model predictor')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--model-root', default='models', help='Models directory')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = LocalModelPredictor(args.model_root)
        
        if not predictor.is_available():
            print("❌ No local models available")
            print("   Train a model first using: python train_model.py")
            return 1
            
        # Get model info
        model_info = predictor.get_model_info()
        print(f"\n🤖 Using local model: {model_info['model_name']}")
        print(f"🏗️  Architecture: {model_info['architecture']}")
        print(f"📊 Classes: {model_info['num_classes']}")
        print(f"🎯 Training accuracy: {model_info['training_accuracy']:.1%}")
        
        # Predict
        result = predictor.infer(args.image_path, args.top_k)
        
        print(f"\n🔍 Prediction Results:")
        print(f"🎯 Predicted Disease: {result['label']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        if len(result['top_predictions']) > 1:
            print(f"\n📋 Top {len(result['top_predictions'])} Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence']:.1%}")
                
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
