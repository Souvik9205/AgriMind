#!/bin/bash

# Batch test script for the new HuggingFace model
# This script tests all images in the test_images directory

echo "🧪 Batch Testing Plant Disease Detection"
echo "========================================"

cd /Users/souvik/Desktop/AgriMind/apps/ml-inference

# Create output directory for results
mkdir -p batch_results

# Test individual images
echo "📷 Testing individual images:"
echo ""

for image in test_images/*.jpg; do
    if [ -f "$image" ]; then
        echo "Testing: $(basename "$image")"
        /Users/souvik/Desktop/AgriMind/.venv/bin/python predictor.py "$image" >> "batch_results/$(basename "$image" .jpg)_result.txt" 2>&1
        echo "✅ Result saved to batch_results/$(basename "$image" .jpg)_result.txt"
        echo ""
    fi
done

echo "🎯 Batch testing completed!"
echo "📁 Results saved in batch_results/ directory"

# Generate summary
echo ""
echo "📊 Quick Summary:"
for result in batch_results/*_result.txt; do
    if [ -f "$result" ]; then
        filename=$(basename "$result" _result.txt)
        prediction=$(grep "🎯 Predicted Disease:" "$result" | cut -d: -f2 | xargs)
        confidence=$(grep "📊 Confidence:" "$result" | cut -d: -f2 | xargs)
        echo "   $filename: $prediction ($confidence)"
    fi
done
