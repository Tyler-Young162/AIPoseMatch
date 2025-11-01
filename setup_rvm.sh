#!/bin/bash
# Setup script for RVM (Robust Video Matting)
# This script clones the official RVM repository and sets it up

echo "Setting up RVM (Robust Video Matting)"
echo "====================================="

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Clone the RVM repository if it doesn't exist
if [ ! -d "RobustVideoMatting" ]; then
    echo "Cloning RVM repository..."
    git clone https://github.com/PeterL1n/RobustVideoMatting.git
    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone RVM repository"
        exit 1
    fi
    echo "✓ Repository cloned successfully"
else
    echo "RVM repository already exists, skipping clone"
fi

# Create models directory
mkdir -p models

# Check if model files exist
echo ""
echo "Checking for model files..."
if [ ! -f "models/rvm_mobilenetv3.pth" ]; then
    echo "⚠ rvm_mobilenetv3.pth not found"
    echo "Please download it from:"
    echo "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth"
    echo ""
    echo "Or run the download script:"
    echo "python download_rvm_model.py"
else
    echo "✓ rvm_mobilenetv3.pth found"
fi

if [ ! -f "models/rvm_resnet50.pth" ]; then
    echo "⚠ rvm_resnet50.pth not found"
    echo "Please download it from:"
    echo "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_resnet50.pth"
else
    echo "✓ rvm_resnet50.pth found"
fi

echo ""
echo "Setup complete!"
echo "If you need to use the official RVM, you can import it from RobustVideoMatting directory"

