# Multi-stage build for Fed-ViT-AutoRL
# Stage 1: Base dependencies
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 fedvit && \
    useradd -r -u 1000 -g fedvit -m -d /home/fedvit -s /bin/bash fedvit

# Set working directory
WORKDIR /app

# Stage 2: Development dependencies
FROM base AS dev

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install -e ".[dev,simulation]"

# Copy source code
COPY --chown=fedvit:fedvit . .

# Switch to non-root user
USER fedvit

# Default command for development
CMD ["python", "-m", "fed_vit_autorl.server", "--config", "configs/federated_training.yaml"]

# Stage 3: Production dependencies
FROM base AS prod

# Install only production dependencies
COPY pyproject.toml .
RUN pip install . && \
    pip cache purge

# Copy source code
COPY --chown=fedvit:fedvit fed_vit_autorl/ ./fed_vit_autorl/
COPY --chown=fedvit:fedvit configs/ ./configs/

# Switch to non-root user
USER fedvit

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fed_vit_autorl; print('healthy')" || exit 1

# Default command for production
CMD ["python", "-m", "fed_vit_autorl.server", "--config", "configs/production.yaml"]

# Stage 4: Edge deployment (optimized for resource-constrained devices)
FROM python:3.11-slim-bullseye AS edge

# Minimal dependencies for edge deployment
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Install edge-optimized dependencies
COPY pyproject.toml .
RUN pip install .[edge] && \
    pip cache purge

# Copy only necessary files
COPY --chown=1000:1000 fed_vit_autorl/ ./fed_vit_autorl/
COPY --chown=1000:1000 configs/edge_deployment.yaml ./config.yaml

# Create non-root user
RUN useradd -r -u 1000 -m -d /home/fedvit fedvit
USER fedvit

# Expose edge API port
EXPOSE 8080

# Health check for edge deployment
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command for edge deployment
CMD ["python", "-m", "fed_vit_autorl.edge.server", "--config", "config.yaml", "--port", "8080"]

# Stage 5: Simulation environment (includes CARLA dependencies)
FROM base AS simulation

# Install additional simulation dependencies
RUN apt-get update && apt-get install -y \
    xvfb \
    x11vnc \
    fluxbox \
    wmctrl \
    && rm -rf /var/lib/apt/lists/*

# Install simulation dependencies
COPY pyproject.toml .
RUN pip install ".[dev,simulation]"

# Copy source code
COPY --chown=fedvit:fedvit . .

# Switch to non-root user
USER fedvit

# Set up virtual display
ENV DISPLAY=:1.0

# Default command for simulation
CMD ["python", "-m", "fed_vit_autorl.simulation.server", "--headless"]