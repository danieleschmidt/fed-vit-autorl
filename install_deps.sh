#!/bin/bash
# Generation 1: Simple dependency installation script

source fed_vit_env/bin/activate

# Install core dependencies
pip install --upgrade pip setuptools wheel

# Core ML dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy pillow pyyaml tqdm tensorboard cryptography

# Development dependencies
pip install pytest pytest-cov black ruff mypy

# Install package in development mode
pip install -e .

echo "Generation 1 dependencies installed successfully!"