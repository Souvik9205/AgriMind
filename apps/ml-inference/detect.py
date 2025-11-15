#!/usr/bin/env python3
"""
AgriMind Plant Disease Detection
Single command-line tool for detecting plant diseases from images
"""

import json
import sys
import argparse
from pathlib import Path
from PIL import Image
import requests
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

class AgriMindPredictor:
    """Plant disease predictor for AgriMind"""
    
    def __init__(self):
        self.repo_id = "wambugu71/crop_leaf_diseases_vit"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.repo_id)
        self.model.eval()
        self.model.to(self.device)
    
    def predict(self, image_path: str) -> dict:
        """
        Predict plant disease from image
        
        Args:
            image_path: Path to image file or URL
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load image
            if image_path.startswith("http"):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                from io import BytesIO
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")
            
            # Process image
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction
            pred_id = int(logits.argmax(-1))
            label = self.model.config.id2label[pred_id]
            confidence = float(probabilities.max())
            
            # Parse disease information
            disease_info = self._parse_disease_info(label)
            
            return {
                "success": True,
                "prediction": {
                    "disease": label,
                    "confidence": round(confidence * 100, 2),
                    "crop": disease_info["crop"],
                    "condition": disease_info["condition"],
                    "severity": disease_info["severity"],
                    "treatment": disease_info["treatment"],
                    "prevention": disease_info["prevention"]
                },
                "model_info": {
                    "model": "Vision Transformer",
                    "version": "wambugu71/crop_leaf_diseases_vit",
                    "device": self.device
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prediction": None
            }
    
    def _parse_disease_info(self, label: str) -> dict:
        """Parse disease label and return structured information"""
        
        # Default values
        info = {
            "crop": "Unknown",
            "condition": "Unknown Disease",
            "severity": "Medium",
            "treatment": "Consult agricultural expert",
            "prevention": "Regular monitoring and good farming practices"
        }
        
        # Parse crop type from model labels (format: Crop___Disease)
        if "Corn___" in label or "Corn" in label:
            info["crop"] = "Corn/Maize"
        elif "Potato___" in label or "Potato" in label:
            info["crop"] = "Potato"
        elif "Rice___" in label or "Rice" in label:
            info["crop"] = "Rice"
        elif "Wheat___" in label or "Wheat" in label:
            info["crop"] = "Wheat"
        else:
            # Try to extract crop from the first part of the label if it follows Crop___Disease format
            if "___" in label:
                crop_part = label.split("___")[0]
                if crop_part and crop_part != "Invalid":
                    info["crop"] = crop_part.replace("_", " ").title()
            else:
                # Fallback: try to identify common crop patterns
                label_lower = label.lower()
                if any(crop in label_lower for crop in ["corn", "maize"]):
                    info["crop"] = "Corn/Maize"
                elif any(crop in label_lower for crop in ["potato", "tomato"]):
                    info["crop"] = "Potato"
                elif "rice" in label_lower:
                    info["crop"] = "Rice"
                elif "wheat" in label_lower:
                    info["crop"] = "Wheat"
                elif any(crop in label_lower for crop in ["bean", "soybean"]):
                    info["crop"] = "Bean"
                elif any(crop in label_lower for crop in ["cotton"]):
                    info["crop"] = "Cotton"
        
        # Parse condition from model labels
        if "Healthy" in label:
            info.update({
                "condition": "Healthy Plant", 
                "severity": "None",
                "treatment": "No treatment needed - continue regular care",
                "prevention": "Maintain current good practices"
            })
        elif "Brown_Rust" in label or "Brown Rust" in label:
            info.update({
                "condition": "Brown Rust (Leaf Rust)",
                "severity": "High",
                "treatment": "Apply fungicide immediately. Remove infected leaves. Improve air circulation.",
                "prevention": "Use resistant varieties, proper plant spacing, avoid overhead watering"
            })
        elif "Yellow_Rust" in label or "Yellow Rust" in label:
            info.update({
                "condition": "Yellow Rust",
                "severity": "High", 
                "treatment": "Apply systemic fungicide, monitor weather conditions",
                "prevention": "Use resistant cultivars, timely planting"
            })
        elif "Common_Rust" in label or "Common Rust" in label:
            info.update({
                "condition": "Common Rust",
                "severity": "Medium",
                "treatment": "Apply fungicide if severe, monitor crop closely",
                "prevention": "Plant resistant hybrids, proper field spacing"
            })
        elif "Gray_Leaf_Spot" in label or "Gray Leaf Spot" in label:
            info.update({
                "condition": "Gray Leaf Spot",
                "severity": "Medium",
                "treatment": "Apply fungicide, improve field drainage", 
                "prevention": "Crop rotation, resistant varieties, proper field sanitation"
            })
        elif "Brown_Spot" in label or "Brown Spot" in label:
            info.update({
                "condition": "Brown Spot",
                "severity": "Medium",
                "treatment": "Apply appropriate fungicide, improve field management",
                "prevention": "Proper water management, balanced nutrition, resistant varieties"
            })
        elif "Leaf_Blast" in label or "Leaf Blast" in label:
            info.update({
                "condition": "Leaf Blast",
                "severity": "High",
                "treatment": "Apply systemic fungicide immediately, remove infected plants",
                "prevention": "Use resistant varieties, proper nitrogen management, avoid excessive moisture"
            })
        elif "Early_Blight" in label:
            info.update({
                "condition": "Early Blight",
                "severity": "Medium-High",
                "treatment": "Remove infected plant parts, apply copper-based fungicide",
                "prevention": "Crop rotation, proper spacing, avoid wetting leaves during watering"
            })
        elif "Late_Blight" in label:
            info.update({
                "condition": "Late Blight",
                "severity": "Very High",
                "treatment": "Immediate fungicide application, remove all infected plants",
                "prevention": "Use certified disease-free seeds, ensure good drainage"
            })
        elif "Leaf_Spot" in label or "Spot" in label:
            info.update({
                "condition": "Leaf Spot Disease",
                "severity": "Medium",
                "treatment": "Remove infected leaves, apply appropriate fungicide",
                "prevention": "Improve air circulation, avoid overhead irrigation"
            })
        elif "Gray_Leaf_Spot" in label:
            info.update({
                "condition": "Gray Leaf Spot",
                "severity": "Medium",
                "treatment": "Apply fungicide, improve field drainage",
                "prevention": "Crop rotation, resistant varieties, proper field sanitation"
            })
        elif "Yellow_Rust" in label:
            info.update({
                "condition": "Yellow Rust",
                "severity": "High",
                "treatment": "Apply systemic fungicide, monitor weather conditions",
                "prevention": "Use resistant cultivars, timely planting"
            })
        elif "Common_Rust" in label:
            info.update({
                "condition": "Common Rust",
                "severity": "Medium",
                "treatment": "Apply fungicide if severe, monitor crop closely",
                "prevention": "Plant resistant hybrids, proper field spacing"
            })
        else:
            # Fallback: try to extract disease from the label format
            if "___" in label and not "Invalid" in label:
                disease_part = label.split("___")[1] if len(label.split("___")) > 1 else label
                # Clean up the disease name
                condition_name = disease_part.replace("_", " ").title()
                info["condition"] = condition_name
                
                # Set default severity based on disease type patterns
                if any(keyword in condition_name.lower() for keyword in ["rust", "blight", "blast"]):
                    info["severity"] = "High"
                elif any(keyword in condition_name.lower() for keyword in ["spot", "mold"]):
                    info["severity"] = "Medium"
        
        return info

def main():
    parser = argparse.ArgumentParser(description='AgriMind Plant Disease Detection')
    parser.add_argument('image_path', help='Path to image file or image URL')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--quiet', action='store_true', help='Suppress loading messages')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🌾 AgriMind Plant Disease Detection", file=sys.stderr)
        print("Loading model...", file=sys.stderr)
    
    try:
        # Initialize predictor
        predictor = AgriMindPredictor()
        
        if not args.quiet:
            print("✅ Model loaded successfully", file=sys.stderr)
            image_name = Path(args.image_path).name if not args.image_path.startswith('http') else args.image_path.split('/')[-1]
            print(f"🔍 Analyzing: {image_name}", file=sys.stderr)
        
        # Make prediction
        result = predictor.predict(args.image_path)
        
        if args.json:
            # Output JSON for programmatic use
            print(json.dumps(result, indent=2))
        else:
            # Output human-readable format
            if result["success"]:
                pred = result["prediction"]
                print(f"\n🎯 Disease Detection Results")
                print(f"=" * 40)
                print(f"🌱 Crop: {pred['crop']}")
                print(f"🔬 Condition: {pred['condition']}")
                print(f"📊 Confidence: {pred['confidence']}%")
                print(f"⚠️ Severity: {pred['severity']}")
                print(f"\n💊 Treatment:")
                print(f"   {pred['treatment']}")
                print(f"\n🛡️ Prevention:")
                print(f"   {pred['prevention']}")
                print(f"\n🤖 Model: {result['model_info']['model']}")
            else:
                print(f"❌ Error: {result['error']}")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "prediction": None
        }
        
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            print(f"❌ Error: {e}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
