# AgriMind ML Inference

A comprehensive machine learning system for plant disease detection and diagnosis using computer vision.

## 🌱 Overview

This ML inference application processes plant images to detect and classify various diseases across multiple crop types. It supports datasets from PlantVillage, PlantDoc, Bangladeshi crops, Rice leaf diseases, and custom Roboflow datasets.

## 🚀 Features

- **Multi-Dataset Support**: Integrates 5 different plant disease datasets
- **Advanced Model Architecture**: EfficientNet-based models with attention mechanisms
- **Comprehensive Training Pipeline**: Automated data preprocessing, training, and evaluation
- **REST API**: FastAPI-based inference service
- **Batch Processing**: Support for multiple image predictions
- **Extensive Evaluation**: Detailed metrics, confusion matrices, and performance analysis

## 📁 Project Structure

```
ml-inference/
├── src/
│   ├── config.py              # Configuration management
│   ├── data_preprocessing.py   # Dataset processing and unification
│   ├── data_loader.py         # Data loading and augmentation
│   ├── models.py              # Neural network architectures
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   ├── api.py                 # FastAPI inference service
│   └── main.py                # Main application entry point
├── scripts/
│   └── download_datasets.py   # Dataset download automation
├── data/                      # Dataset storage
├── models/                    # Saved model checkpoints
├── logs/                      # Training and application logs
├── results/                   # Evaluation results and plots
├── requirements.txt           # Python dependencies
└── package.json              # Project metadata
```

## 🛠 Installation

1. **Clone and Navigate**:

   ```bash
   cd apps/ml-inference
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Kaggle API** (for dataset downloads):

   ```bash
   # Install kaggle
   pip install kaggle

   # Set up credentials (get from kaggle.com/account)
   mkdir -p ~/.kaggle
   # Place kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 📊 Supported Datasets

The system integrates the following datasets:

1. **PlantVillage Dataset** (`emmarex/plantdisease`)
   - Comprehensive plant disease images
   - Multiple crop types and disease conditions

2. **PlantDoc Dataset** (`pratik1120/plantdoc-dataset`)
   - Real-world plant disease images
   - 27 classes across 13 plant species

3. **Bangladeshi Crops Disease** (`nafishamoin/bangladeshi-crops-disease-dataset`)
   - Region-specific crop diseases
   - Local agricultural conditions

4. **Rice Leaf Diseases** (`vbookshelf/rice-leaf-diseases`)
   - Specialized rice disease detection
   - Multiple rice disease types

5. **Indian Trees Dataset** (Roboflow export)
   - Custom dataset for Indian tree species
   - Object detection format support

## 🔄 Usage

### Quick Start (Full Pipeline)

```bash
# Run complete ML pipeline
python src/main.py pipeline
```

### Step-by-Step Execution

1. **Setup Project Structure**:

   ```bash
   python src/main.py setup
   ```

2. **Download Datasets**:

   ```bash
   python src/main.py download
   # Or manually run:
   python scripts/download_datasets.py
   ```

3. **Preprocess Data**:

   ```bash
   python src/main.py preprocess
   ```

4. **Train Model**:

   ```bash
   python src/main.py train
   ```

5. **Evaluate Model**:

   ```bash
   python src/main.py evaluate
   ```

6. **Start API Server**:
   ```bash
   python src/main.py serve
   # Or with custom settings:
   python src/main.py serve --host 0.0.0.0 --port 8080 --workers 4
   ```

### Using the API

Once the server is running, you can make predictions:

```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/plant_image.jpg"

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

## ⚙️ Configuration

The system is highly configurable through `src/config.py`:

### Model Configuration

```python
MODEL_CONFIG = ModelConfig(
    backbone="efficientnet-b3",
    batch_size=32,
    learning_rate=1e-4,
    epochs=100,
    # ... other parameters
)
```

### Data Augmentation

```python
AUGMENTATION_CONFIG = AugmentationConfig(
    horizontal_flip=0.5,
    rotation_limit=20,
    brightness_limit=0.2,
    # ... other augmentations
)
```

## 📈 Model Architecture

The system supports multiple neural network architectures:

- **EfficientNet**: B0-B7 variants with transfer learning
- **ResNet**: ResNet34, ResNet50, ResNet101
- **DenseNet**: DenseNet121, DenseNet161
- **Vision Transformers**: Various ViT models
- **Ensemble Models**: Combine multiple architectures
- **Multi-Scale Models**: Process images at different resolutions

### Key Features:

- Transfer learning from ImageNet
- Attention mechanisms
- Dropout and batch normalization
- Advanced data augmentation (Mixup, CutMix)
- Class balancing for imbalanced datasets

## 🎯 Training Features

- **Automated Data Processing**: Unified dataset creation from multiple sources
- **Advanced Augmentation**: Albumentations integration
- **Smart Training**: Learning rate scheduling, early stopping
- **Monitoring**: Weights & Biases integration
- **Checkpointing**: Automatic model saving and resuming
- **Validation**: Comprehensive evaluation metrics

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Top-1 and Top-5 accuracy
- **Per-Class Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Visual classification analysis
- **Misclassification Analysis**: Common error patterns
- **ROC Curves**: Performance visualization

## 🔧 API Endpoints

### Core Endpoints:

- `GET /`: API information and status
- `GET /health`: Health check and model status
- `GET /info`: Model and configuration details
- `POST /predict`: Single image prediction
- `POST /predict/batch`: Batch image prediction

### Response Format:

```json
{
  "predictions": [
    {
      "class": "Tomato_Late_Blight",
      "confidence": 0.95,
      "percentage": 95.0
    }
  ],
  "top_prediction": {
    "class": "Tomato_Late_Blight",
    "confidence": 0.95,
    "percentage": 95.0
  },
  "filename": "plant_image.jpg"
}
```

## 🚀 Enhanced Download Progress Features

The dataset downloader now includes advanced progress tracking:

### 📊 **Real-time Progress Monitoring**

- **File Size Tracking**: Shows downloaded size in real-time (MB/GB format)
- **Progress Bars**: Visual progress indicators for each dataset
- **ETA Calculations**: Estimated time remaining for downloads
- **Speed Monitoring**: Current download speed tracking

### 📈 **Multi-Stage Progress Tracking**

- **Stage 1**: Kaggle credential verification
- **Stage 2**: Dataset information fetching
- **Stage 3**: Actual file downloads with size tracking
- **Stage 4**: Extraction and validation

### 📋 **Enhanced Download Information**

```bash
📦 Dataset 1/4: PlantVillage Dataset
🔗 Kaggle ID: emmarex/plantdisease
📏 Estimated size: 870 MB
📊 Monitoring download progress...
   📦 Downloaded: 245.8 MB
✅ PlantVillage Dataset completed successfully!
   📦 Final size: 870.2 MB
   ⏱️  Time taken: 125.3 seconds
   📄 Files downloaded: 54,305
```

### 🎛️ **Progress Features**

- **Resume Support**: Skip already downloaded datasets
- **Error Recovery**: Detailed error messages and retry suggestions
- **System Monitoring**: Disk space and system requirements check
- **Summary Reports**: Complete download statistics

## 🚀 Getting Started

Ready to detect plant diseases? Run the complete pipeline:

```bash
# Quick start - everything automated
python src/main.py pipeline

# Then start the API server
python src/main.py serve
```

Your plant disease detection system is now ready! 🌿🔬
