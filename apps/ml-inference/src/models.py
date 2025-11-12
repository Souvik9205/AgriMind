"""
Neural network model architectures for plant disease detection.
Supports multiple backbone architectures with transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
from typing import Optional, Dict, Any

class PlantDiseaseClassifier(nn.Module):
    """
    Plant disease classifier with configurable backbone architecture.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_attention = use_attention
        
        # Create backbone
        if backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(
                backbone, 
                pretrained=pretrained, 
                num_classes=0  # Remove classifier
            )
            self.feature_dim = self.backbone.num_features
            
        elif backbone.startswith('resnet'):
            if backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
            else:
                self.backbone = models.resnet34(pretrained=pretrained)
            
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
            
        elif backbone.startswith('densenet'):
            if backbone == 'densenet121':
                self.backbone = models.densenet121(pretrained=pretrained)
            else:
                self.backbone = models.densenet161(pretrained=pretrained)
                
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone.startswith('vit'):
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
            
        elif backbone.startswith('rexnet'):
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Classifier head
        self.classifier = self._build_classifier(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_classifier(self, dropout_rate: float) -> nn.Module:
        """Build the classification head."""
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention (assuming 2D features)
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
            
            attended_features, _ = self.attention(features, features, features)
            features = attended_features.squeeze(1)  # Remove sequence dimension
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps for visualization."""
        return self.backbone(x)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Weighted average
        weighted_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weighted_output += self.weights[i] * output
        
        return weighted_output


class MultiScaleModel(nn.Module):
    """
    Multi-scale model that processes images at different resolutions.
    """
    
    def __init__(
        self, 
        num_classes: int,
        backbone: str = "efficientnet_b3",
        scales: list = [224, 299, 384],
        pretrained: bool = True
    ):
        super().__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create a model for each scale
        self.scale_models = nn.ModuleList([
            PlantDiseaseClassifier(
                num_classes=0,  # Feature extractor only
                backbone=backbone,
                pretrained=pretrained
            ) for _ in scales
        ])
        
        # Get feature dimension
        dummy_model = PlantDiseaseClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False
        )
        self.feature_dim = dummy_model.feature_dim
        
        # Fusion layer
        self.fusion = nn.Linear(self.feature_dim * self.num_scales, self.feature_dim)
        
        # Final classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        scale_features = []
        
        for i, (model, scale) in enumerate(zip(self.scale_models, self.scales)):
            # Resize input to target scale
            if x.size(-1) != scale:
                scaled_x = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Extract features
            features = model(scaled_x)
            scale_features.append(features)
        
        # Concatenate features from all scales
        fused_features = torch.cat(scale_features, dim=1)
        
        # Apply fusion layer
        fused_features = F.relu(self.fusion(fused_features))
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits


def create_model(
    model_config: Dict[str, Any], 
    num_classes: int
) -> nn.Module:
    """
    Factory function to create models based on configuration.
    """
    
    backbone = model_config.get('backbone', 'efficientnet_b3')
    pretrained = model_config.get('pretrained', True)
    dropout_rate = model_config.get('dropout_rate', 0.3)
    use_attention = model_config.get('use_attention', False)
    
    model = PlantDiseaseClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )
    
    return model


def load_pretrained_model(model_path: str, num_classes: int, backbone: str = "efficientnet_b3") -> nn.Module:
    """
    Load a pre-trained model from checkpoint.
    """
    model = PlantDiseaseClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False
    )
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    return model


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params
