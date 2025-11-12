# AgriMind ML Inference Tool

A standalone plant disease detection tool powered by deep learning. This tool can analyze crop images and identify diseases using pre-trained neural networks.

## 🚀 Quick Start

The tool comes with a pre-trained model (`crop_best_model.pth`) trained on rexnet_150 architecture that can detect 35 different plant diseases across various crops.

### Predict a Single Image

```bash
# Basic prediction
python tool.py predict path/to/your/image.jpg

# With custom model and top 5 predictions
python tool.py predict image.jpg --model custom_model.pth --top-k 5

# Save results to JSON
python tool.py predict image.jpg --output results.json
```

### Batch Process Multiple Images

```bash
# Process all images in a directory
python tool.py batch path/to/images/ --output results.csv

# With custom model
python tool.py batch images/ --model custom_model.pth --output results.csv
```

### Model Information

```bash
# View model details and supported classes
python tool.py info
```

## 📋 Supported Diseases

The pre-trained model can detect these 35 plant diseases:

- **Rice**: Brown Spot, Healthy, Leaf Blast, Neck Blast, Bacterial leaf blight, Leaf smut
- **Corn**: Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight
- **Potato**: Early Blight, Healthy, Late Blight
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy
- **Wheat**: Brown Rust, Healthy, Yellow Rust
- **Pepper**: Bacterial spot, Healthy
- **Sugarcane**: Bacterial Blight, Healthy, Red Rot

## 🔧 Installation & Setup

### Prerequisites

```bash
# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check if the tool works
python tool.py info
```

## 📊 Usage Examples

### Example 1: Quick Disease Check

```bash
python tool.py predict rice_leaf.jpg
```

Output:

```
🔍 Analysis Results for: rice_leaf.jpg
==================================================
🎯 Predicted Disease: Rice___Leaf_Blast
📊 Confidence: 94.2%

📈 Top 3 Predictions:
   1. Rice___Leaf_Blast: 94.2%
   2. Rice___Brown_Spot: 4.1%
   3. Rice___Healthy: 1.7%
```

### Example 2: Batch Processing

```bash
python tool.py batch crop_images/ --output disease_analysis.csv
```

This creates a CSV file with columns:

- `image_name`: Name of the image file
- `predicted_class`: Detected disease
- `confidence`: Prediction confidence (0.0-1.0)
- `status`: Processing status

### Example 3: Using Custom Model

```bash
python tool.py predict image.jpg --model my_trained_model.pth
```

## 🏋️ Training Your Own Model

If you want to train a custom model with your own data:

```bash
# Train with default settings
python tool.py train --data path/to/dataset/

# Custom training parameters
python tool.py train --data dataset/ --epochs 50 --batch-size 32 --learning-rate 0.001
```

### Dataset Structure

Organize your training data like this:

```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── class3/
    └── ...
```

The training process will:

1. Automatically detect classes from folder names
2. Split data into train/validation/test sets
3. Apply data augmentation
4. Save the best model as `crop_best_model.pth`
5. Update `class_mapping.json` with new classes

## 🛠️ Advanced Usage

### Using the Python API

```python
from src.inference import PlantDiseasePredictor

# Load predictor
predictor = PlantDiseasePredictor()

# Predict single image
result = predictor.predict("image.jpg", top_k=5)
print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = predictor.predict_batch(images)
```

### Command Line Interface (Alternative)

You can also use the main CLI:

```bash
# Traditional CLI commands
python src/main.py predict image.jpg --top-k 5
python src/main.py batch-predict images/ --output results.csv
python src/main.py train --config custom_config.json
```

## 🎯 Model Performance

The pre-trained model achieves:

- **Architecture**: ReXNet-150
- **Training Classes**: 35 plant diseases
- **Input Size**: 224x224 pixels
- **Supported Formats**: JPG, PNG, BMP, TIFF, WebP

## 📝 Output Formats

### Single Prediction JSON

```json
{
  "predicted_class": "Rice___Leaf_Blast",
  "confidence": 0.942,
  "top_predictions": [
    { "class": "Rice___Leaf_Blast", "confidence": 0.942, "class_id": 17 },
    { "class": "Rice___Brown_Spot", "confidence": 0.041, "class_id": 15 },
    { "class": "Rice___Healthy", "confidence": 0.017, "class_id": 16 }
  ]
}
```

### Batch Results CSV

```csv
image_name,image_path,predicted_class,confidence,status
rice1.jpg,rice1.jpg,Rice___Leaf_Blast,0.942,success
corn2.jpg,corn2.jpg,Corn___Healthy,0.889,success
tomato3.jpg,tomato3.jpg,Tomato_Early_blight,0.756,success
```

## 🔍 Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Ensure `output/crop_best_model.pth` exists
   - Or specify custom model with `--model path/to/model.pth`

2. **"Class mapping file not found"**
   - Ensure `models/class_mapping.json` exists
   - This file maps class names to numbers

3. **CUDA/GPU Issues**
   - The tool automatically uses CPU if GPU is not available
   - For GPU training, ensure PyTorch CUDA version matches your GPU

4. **Memory Issues**
   - Reduce batch size for batch processing
   - Use smaller images (tool auto-resizes to 224x224)

### Getting Help

```bash
# See all available commands
python tool.py --help

# Get help for specific command
python tool.py predict --help
python tool.py batch --help
python tool.py train --help
```

## 📁 File Structure

```
ml-inference/
├── tool.py              # Main standalone tool
├── output/
│   └── crop_best_model.pth    # Pre-trained model
├── models/
│   └── class_mapping.json     # Class definitions
├── src/                 # Source code
│   ├── inference.py     # Inference logic
│   ├── models.py        # Neural network architectures
│   ├── train.py         # Training pipeline
│   ├── config.py        # Configuration
│   └── ...
└── requirements.txt     # Dependencies
```

## 🔄 Integration

This tool is designed to work independently but can be integrated into larger systems:

- **APIs**: The `apps/api` folder will contain REST endpoints
- **RAG System**: Can be combined with the knowledge base in `apps/rag-script`
- **Web Interface**: Can be integrated with the frontend in `apps/frontend`

## 🚫 Note on API Endpoints

This ML inference module is a **standalone tool**, not a web service. For API endpoints, see the `apps/api` directory which will contain REST APIs for both ML inference and RAG functionality.

## 🔧 Development Commands

For development and advanced usage:

```bash
# Setup project structure
python src/main.py setup

# Download datasets (if training from scratch)
python src/main.py download

# Preprocess data
python src/main.py preprocess

# Train model
python src/main.py train --resume checkpoint.pth

# Evaluate model
python src/main.py evaluate --model best_model.pth

# Run full pipeline
python src/main.py pipeline --skip-download
```

## 📦 Dependencies

Core dependencies include:

- PyTorch & torchvision
- timm (for model architectures)
- PIL/Pillow (image processing)
- NumPy & pandas
- tqdm (progress bars)

Optional:

- wandb (experiment tracking)
- albumentations (data augmentation)

Install all dependencies:

```bash
pip install -r requirements.txt
```
