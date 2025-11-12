# AgriMind ML Tool - Quick Start Guide

## 🚀 Ready to Use!

The AgriMind ML inference tool is now set up and ready for plant disease detection.

### What You Have:

- ✅ Pre-trained model (`crop_best_model.pth`) with ReXNet-150 architecture
- ✅ 35 disease classes across Rice, Corn, Potato, Tomato, Wheat, Pepper, and Sugarcane
- ✅ Standalone tool - no API setup needed
- ✅ Support for single image and batch processing

### Quick Commands:

```bash
# See model info
python3 tool.py info

# Predict single image
python3 tool.py predict path/to/your/image.jpg

# Process multiple images
python3 tool.py batch path/to/images/ --output results.csv

# Train your own model (if you have data)
python3 tool.py train --data path/to/dataset/
```

### Usage Examples:

```bash
# Basic prediction
python3 tool.py predict rice_leaf.jpg

# Get top 5 predictions with confidence scores
python3 tool.py predict tomato_leaf.jpg --top-k 5

# Save results to JSON file
python3 tool.py predict leaf.jpg --output results.json

# Process all images in a folder
python3 tool.py batch crop_photos/ --output analysis.csv

# Use custom trained model
python3 tool.py predict image.jpg --model my_model.pth
```

### Output Format:

The tool provides clear, formatted output with disease predictions and confidence scores.

### Integration:

This is a **standalone tool** - just input images and get predictions back. Perfect for:

- Research and analysis
- Farm management systems
- Agricultural apps
- Educational purposes

For web APIs and integration endpoints, see the `apps/api` directory in the main project.

## 🎯 What's Next?

- The tool is ready to use with the pre-trained model
- You can train custom models if you have your own data
- For API endpoints, check out the main `apps/api` folder
- For RAG-based agricultural advice, see `apps/rag-script`
