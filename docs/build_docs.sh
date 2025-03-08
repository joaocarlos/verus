#!/bin/bash
set -e

echo "Generating API documentation..."
python generate_api_docs.py

echo "Building Sphinx documentation..."
cd "$(dirname "$0")"
make clean
make html

echo "Documentation built successfully!"
echo "Open ./build/html/index.html in your browser to view it."
