#!/bin/bash
set -e

echo "🚀 Setting up Fed-ViT-AutoRL development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    wget \
    curl \
    unzip \
    htop \
    tmux \
    tree

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
pip install -e ".[dev,simulation,edge]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Set up Jupyter Lab
echo "📊 Configuring Jupyter Lab..."
pip install jupyterlab jupyterlab-git
jupyter lab --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_origin = '*'" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.disable_check_xsrf = True" >> ~/.jupyter/jupyter_lab_config.py

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p ~/.cache/torch
mkdir -p ~/.cache/huggingface
mkdir -p ~/.cache/fed-vit-autorl

# Set up Git configuration
echo "🔧 Configuring Git..."
git config --global --add safe.directory /workspaces/fed-vit-autorl
git config --global pull.rebase false

# Create development directories
echo "📁 Creating development directories..."
mkdir -p {data,logs,models,experiments}

# Download sample data (if available)
echo "📥 Setting up sample data..."
mkdir -p data/samples
# Note: Add actual data download commands here when available

# Set up environment variables
echo "🔧 Setting up environment variables..."
cat << 'EOF' >> ~/.bashrc
# Fed-ViT-AutoRL Development Environment
export PYTHONPATH="/workspaces/fed-vit-autorl:$PYTHONPATH"
export FED_VIT_AUTORL_DATA_DIR="/workspaces/fed-vit-autorl/data"
export FED_VIT_AUTORL_LOGS_DIR="/workspaces/fed-vit-autorl/logs"
export FED_VIT_AUTORL_MODELS_DIR="/workspaces/fed-vit-autorl/models"

# Development shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias pytest='python -m pytest'
alias mypy='python -m mypy'
alias black='python -m black'
alias ruff='python -m ruff'

# Quick development commands
alias dev-test='pytest tests/ -v'
alias dev-lint='ruff check . && mypy fed_vit_autorl/'
alias dev-format='black . && ruff check --fix .'
alias dev-clean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true'
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start:"
echo "   - Run tests: dev-test"
echo "   - Format code: dev-format"
echo "   - Start Jupyter: jupyter lab --ip=0.0.0.0 --port=8888"
echo "   - Start TensorBoard: tensorboard --logdir=logs --host=0.0.0.0"
echo ""
echo "📚 See docs/guides/development-setup.md for detailed information"