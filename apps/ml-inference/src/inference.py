"""
Inference module for plant disease detection using pre-trained models.
This module provides functionality to load pre-trained models and make predictions on new images.
"""

import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from torchvision import transforms

from models import PlantDiseaseClassifier
from config import BASE_DIR, MODELS_DIR, CLASS_MAPPING_PATH, AUGMENTATION_CONFIG

logger = logging.getLogger(__name__)

class PlantDiseasePredictor:
    """
    Plant disease predictor that uses pre-trained models for inference.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        class_mapping_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the pre-trained model (.pth file)
            class_mapping_path: Path to class mapping JSON file
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default paths
        if model_path is None:
            from config import PRETRAINED_MODEL_PATH
            model_path = PRETRAINED_MODEL_PATH
        if class_mapping_path is None:
            class_mapping_path = CLASS_MAPPING_PATH
            
        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        self.num_classes = len(self.class_mapping)
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Load model
        self.model = self._load_model()
        
        # Setup image preprocessing
        self.transform = self._get_transform()
        
        logger.info(f"Predictor initialized with {self.num_classes} classes on {self.device}")
    
    def _load_class_mapping(self) -> Dict[str, int]:
        """Load class mapping from JSON file."""
        try:
            with open(self.class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            logger.info(f"Loaded class mapping with {len(class_mapping)} classes")
            return class_mapping
        except FileNotFoundError:
            logger.error(f"Class mapping file not found: {self.class_mapping_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing class mapping JSON: {e}")
            raise
    
    def _load_model(self) -> torch.nn.Module:
        """Load the pre-trained model."""
        try:
            # Create model architecture (using rexnet_150 as mentioned by user)
            model = PlantDiseaseClassifier(
                num_classes=self.num_classes,
                backbone="rexnet_150",  # Updated to rexnet_150 as specified
                pretrained=False  # We're loading pre-trained weights
            )
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=AUGMENTATION_CONFIG.mean,
                std=AUGMENTATION_CONFIG.std
            )
        ])
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probabilities: bool = False,
        top_k: int = 5
    ) -> Dict:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes), dim=1)
        
        # Convert to CPU and numpy
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Prepare results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.idx_to_class[idx]
            predictions.append({
                'class': class_name,
                'confidence': float(prob),
                'class_id': int(idx)
            })
        
        result = {
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions
        }
        
        if return_probabilities:
            all_probs = probabilities.cpu().numpy()[0]
            result['all_probabilities'] = {
                self.idx_to_class[i]: float(prob) 
                for i, prob in enumerate(all_probs)
            }
        
        return result
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for image in batch_images:
                try:
                    tensor = self.preprocess_image(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Failed to process image {i}: {e}")
                    results.append({'error': str(e)})
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Get top prediction for each image
                top_probs, top_indices = torch.topk(probabilities, 1, dim=1)
                
                for j in range(len(batch_tensors)):
                    class_idx = top_indices[j, 0].item()
                    confidence = top_probs[j, 0].item()
                    class_name = self.idx_to_class[class_idx]
                    
                    results.append({
                        'predicted_class': class_name,
                        'confidence': confidence,
                        'class_id': class_idx
                    })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': str(self.model_path),
            'backbone': getattr(self.model, 'backbone_name', 'unknown'),
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }


def load_predictor(
    model_path: Optional[str] = None,
    class_mapping_path: Optional[str] = None
) -> PlantDiseasePredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_path: Path to model file
        class_mapping_path: Path to class mapping file
        
    Returns:
        Initialized predictor
    """
    return PlantDiseasePredictor(
        model_path=model_path,
        class_mapping_path=class_mapping_path
    )


def predict_image(
    image_path: str,
    model_path: Optional[str] = None,
    top_k: int = 3
) -> Dict:
    """
    Simple function to predict a single image.
    
    Args:
        image_path: Path to image file
        model_path: Path to model file (optional)
        top_k: Number of top predictions to return
        
    Returns:
        Prediction results
    """
    predictor = load_predictor(model_path=model_path)
    return predictor.predict(image_path, top_k=top_k)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--model", help="Path to model file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        result = predict_image(args.image_path, args.model, args.top_k)
        
        print(f"\nPrediction for: {args.image_path}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        print(f"\nTop {args.top_k} predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['confidence']:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
