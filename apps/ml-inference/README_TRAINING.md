# Plant Disease Detection ML Inference

This directory contains the machine learning inference system for plant disease detection with support for both custom trained models and pre-trained Hugging Face models.

## 🚀 Quick Start

### Option 1: Using Pre-trained Models (Default)

```bash
# Install dependencies
pip install -r requirements.txt

# Predict disease from image (uses Hugging Face model)
python predictor.py path/to/image.jpg
```

### Option 2: Train Your Own Custom Model

```bash
# 1. Add your training data
mkdir -p data/raw/Healthy
mkdir -p data/raw/Brown_Rust
# Copy your images to respective class folders

# 2. Preprocess the data
python preprocess_data.py

# 3. Train your model
python train_model.py

# 4. Use your trained model (automatically detected)
python predictor.py path/to/image.jpg
```

## 🎯 Model Priority System

The system automatically uses the best available model:

1. **🏠 Local Trained Model** (if available) - Your custom trained model
2. **🌐 Hugging Face Model** (fallback) - Pre-trained model

### Example Output with Local Model

```
🏠 Using local trained model...

🔍 Analysis Results for: wheat_leaf.jpg
==================================================
🎯 Predicted Disease: Brown_Rust
📊 Confidence: 94.2%
🤖 Model: Local_ResNet50
📝 Model Name: plant_disease_model_20241112_143022
🎯 Training Accuracy: 96.8%

📋 Top 3 Predictions:
   1. Brown_Rust: 94.2%
   2. Healthy: 4.1%
   3. Leaf_Spot: 1.7%
```

## 📁 Project Structure

```
apps/ml-inference/
├── predictor.py              # Smart predictor (local → HF fallback)
├── local_predictor.py        # Custom model predictor
├── huggingface_predictor.py  # HuggingFace model predictor
├── preprocess_data.py        # Data preprocessing pipeline
├── train_model.py           # Model training pipeline
├── model_comparison.py      # Compare model performance
├── data/                    # Training data directory
│   ├── raw/                # Your raw images (organized by class)
│   ├── train/              # Training set (auto-generated)
│   ├── validation/         # Validation set (auto-generated)
│   └── test/               # Test set (auto-generated)
├── models/                 # Trained models directory
│   ├── *.pth              # Model weights
│   ├── *_info.json        # Model metadata
│   └── *_history.json     # Training history
└── requirements.txt
```

## 🛠️ Training Your Own Model

### Step 1: Prepare Your Data

```bash
# Create class directories in data/raw/
mkdir -p data/raw/Healthy
mkdir -p data/raw/Brown_Rust
mkdir -p data/raw/Late_Blight
mkdir -p data/raw/Early_Blight
# Add more classes as needed

# Add your images (JPG, PNG, etc.)
# Minimum 20 images per class recommended
# 100+ images per class for better results
```

### Step 2: Preprocess Data

```bash
# Run preprocessing pipeline
python preprocess_data.py

# With custom ratios
python preprocess_data.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

**Preprocessing Features:**

- Image resizing and normalization
- Data augmentation preparation
- Automatic train/validation/test splitting
- Image quality enhancement (CLAHE, sharpening)
- Metadata generation

### Step 3: Train Your Model

```bash
# Train with default settings
python train_model.py

# Custom training settings
python train_model.py --epochs 100 --batch-size 64 --learning-rate 0.0005
```

**Training Features:**

- ResNet50 backbone with transfer learning
- Data augmentation (rotation, flip, color jitter)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing (saves best model)
- Training visualization and metrics

### Step 4: Use Your Trained Model

Once training is complete, the system automatically uses your trained model:

```bash
# Will automatically use your local model if available
python predictor.py path/to/test/image.jpg

# Compare local vs HuggingFace model performance
python model_comparison.py path/to/test/images/*.jpg
```

## 🔍 Model Comparison

Compare performance between your local model and the HuggingFace model:

```bash
# Compare models on test images
python model_comparison.py test_images/*.jpg --output comparison_results.csv
```

**Comparison Metrics:**

- Prediction accuracy agreement
- Inference speed comparison
- Confidence score analysis
- Per-class performance breakdown

## 📊 Features

### Smart Model Selection

- **Automatic Detection**: Uses local model if available, falls back to HuggingFace
- **Performance Tracking**: Monitors model accuracy and inference time
- **Model Metadata**: Tracks training date, accuracy, and model details

### Data Processing

- **Image Enhancement**: CLAHE, sharpening, normalization
- **Flexible Splitting**: Customizable train/val/test ratios
- **Format Support**: JPG, PNG, BMP, TIFF
- **Quality Validation**: Checks image integrity and minimum dataset requirements

### Training Pipeline

- **Transfer Learning**: ResNet50 pre-trained on ImageNet
- **Data Augmentation**: Rotation, flipping, color jittering
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Visualization**: Training plots and metrics tracking

### Disease Information

Provides treatment recommendations for detected diseases:

- Crop-specific information
- Severity assessment
- Treatment recommendations
- Prevention strategies

## 🎯 Supported Disease Classes

### Pre-trained Model (HuggingFace)

38+ plant diseases including:

- Brown Rust (Wheat)
- Late Blight (Potato/Tomato)
- Early Blight (Potato/Tomato)
- Powdery Mildew (various crops)
- Bacterial Spot (Tomato/Pepper)
- Healthy plants

### Custom Models

Support any classes you train on:

- Define your own disease categories
- Multi-crop disease detection
- Regional disease variants
- Custom healthy/unhealthy classifications

## ⚡ Performance

### Inference Speed

- **Local Model**: 0.1-0.5s per image (GPU) / 1-3s (CPU)
- **HuggingFace Model**: 0.5-2.0s per image (GPU) / 2-5s (CPU)

### Accuracy

- **Pre-trained Model**: 90%+ on standard benchmarks
- **Custom Models**: Depends on your data quality and quantity
- **Best Practices**: 100+ images per class for 85%+ accuracy

### Resource Usage

- **Memory**: 2-4GB RAM (training), <2GB (inference)
- **Storage**: ~100MB per trained model
- **GPU**: Optional but recommended for training

## 🔌 API Integration

### Python Script Integration

```python
from predictor import predict_disease

# Will use local model if available, otherwise HuggingFace
result = predict_disease('plant_image.jpg')

print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Model Type: {result['model_type']}")

# Check if using local model
if 'Local' in result['model_type']:
    print(f"Model Name: {result['model_name']}")
    print(f"Training Accuracy: {result['training_accuracy']:.1%}")
```

### Flask API Example

```python
from flask import Flask, request, jsonify
from predictor import predict_disease
import tempfile
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)

        try:
            result = predict_disease(tmp.name)
            return jsonify(result)
        finally:
            os.unlink(tmp.name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 🛠️ Advanced Usage

### Custom Training Parameters

```bash
# Fine-tune training parameters
python train_model.py \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 0.001 \
    --patience 10 \
    --model-name my_custom_model
```

### Multi-Model Management

```python
from local_predictor import LocalModelPredictor

# Load specific model
predictor = LocalModelPredictor()

# List available models
models = predictor.list_available_models()
for model in models:
    print(f"Model: {model['model_name']}")
    print(f"Accuracy: {model['best_val_accuracy']:.1%}")
    print(f"Classes: {model['num_classes']}")
```

### Batch Processing

```bash
# Process multiple images
python model_comparison.py dataset/*.jpg --output batch_results.csv
```

## 🚨 Troubleshooting

### Common Issues

1. **No local model found**

   ```
   📭 No trained models found in models directory
   ```

   **Solution**: Train a model using `python train_model.py`

2. **Preprocessing fails**

   ```
   ❌ No valid image classes found in raw data directory
   ```

   **Solution**: Add images to `data/raw/` organized by class folders

3. **Training fails with small dataset**

   ```
   ⚠️ Warning: These classes have < 20 images
   ```

   **Solution**: Add more images or reduce the number of classes

4. **CUDA memory errors during training**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size: `python train_model.py --batch-size 16`

### Performance Optimization

- **GPU Training**: Ensure CUDA-compatible PyTorch installation
- **Data Quality**: Use high-resolution, well-lit images
- **Data Quantity**: 100+ images per class for best results
- **Class Balance**: Try to have similar numbers of images per class

## 📚 References

### Pre-trained Model

- **HuggingFace Model**: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`
- **Architecture**: MobileNetV2 optimized for plant diseases
- **Dataset**: PlantVillage and custom agricultural datasets

### Custom Training

- **Backbone**: ResNet50 pre-trained on ImageNet
- **Optimization**: Adam optimizer with learning rate scheduling
- **Augmentation**: Albumentations library for advanced transforms

## 🤝 Contributing

1. **Adding New Features**: Update both local and HuggingFace predictors
2. **New Model Architectures**: Extend the training pipeline
3. **Data Processing**: Improve preprocessing for specific crop types
4. **Documentation**: Keep README updated with new features

## 📋 Dependencies

Key dependencies (see `requirements.txt` for complete list):

- **PyTorch** >= 2.0.0 (Deep learning framework)
- **Torchvision** >= 0.15.0 (Computer vision utilities)
- **Transformers** >= 4.35.0 (HuggingFace models)
- **OpenCV** >= 4.8.0 (Image processing)
- **Scikit-learn** >= 1.3.0 (ML utilities)
- **Matplotlib** >= 3.7.0 (Plotting and visualization)
