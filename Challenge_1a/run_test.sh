#!/bin/bash

# This script demonstrates how to build and run the PDF outline extractor

echo "Building Docker image..."
docker build --platform linux/amd64 -t pdf-outline-extractor .

echo ""
echo "Testing with sample data..."

# Create test directories
mkdir -p test_input test_output

# Copy a sample PDF to test input
cp sample_dataset/pdfs/file01.pdf test_input/

echo "Running container..."
docker run --rm \
  -v $(pwd)/test_input:/app/input:ro \
  -v $(pwd)/test_output:/app/output \
  --network none \
  pdf-outline-extractor

echo ""
echo "Output should be in test_output/file01.json"
echo "Contents:"
cat test_output/file01.json
