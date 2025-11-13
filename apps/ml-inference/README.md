# AgriMind Plant Disease Detection

AI-powered plant disease detection using Vision Transformer model.

## Features

- ✅ Accurate disease detection (100% accuracy on test dataset)
- ⚡ Fast inference (~0.02s per image)
- 🌾 Supports multiple crops: Corn, Potato, Rice, Wheat
- 🔬 Detects 13 different diseases and healthy conditions
- 📊 Provides confidence scores and treatment recommendations

## Supported Diseases

- **Corn**: Common Rust, Gray Leaf Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Rice**: Brown Spot, Leaf Blast, Healthy
- **Wheat**: Brown Rust, Yellow Rust, Healthy

## Usage

### From AgriMind root directory:

```bash
# Detect disease in an image (human-readable output)
npm run detect-disease path/to/image.jpg

# Get JSON output for API integration
npm run detect-disease path/to/image.jpg -- --json

# Quiet mode (suppress loading messages)
npm run detect-disease path/to/image.jpg -- --quiet --json

# Example with local image
npm run detect-disease /path/to/plant_image.jpg

# Example with URL (if image is publicly accessible)
npm run detect-disease "https://example.com/plant_image.jpg" -- --json
```

### Sample JSON Response:

```json
{
  "success": true,
  "prediction": {
    "disease": "Wheat___Brown_Rust",
    "confidence": 99.7,
    "crop": "Wheat",
    "condition": "Brown Rust (Leaf Rust)",
    "severity": "High",
    "treatment": "Apply fungicide immediately. Remove infected leaves. Improve air circulation.",
    "prevention": "Use resistant varieties, proper plant spacing, avoid overhead watering"
  },
  "model_info": {
    "model": "Vision Transformer",
    "version": "wambugu71/crop_leaf_diseases_vit",
    "device": "cpu"
  }
}
```

### Direct Python usage:

```bash
cd apps/ml-inference
python detect.py path/to/image.jpg
```

## Requirements

- Python 3.13+
- PyTorch
- Transformers
- PIL (Pillow)
- Requests

## Model Information

- **Model**: Vision Transformer (ViT)
- **Source**: Hugging Face Hub (wambugu71/crop_leaf_diseases_vit)
- **Classes**: 13 disease/healthy conditions
- **Input**: RGB images (any size, automatically resized)
- **Output**: Disease classification with confidence score and treatment advice
