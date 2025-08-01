# Development Guide

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/terragon-labs/fed-vit-autorl.git
   cd fed-vit-autorl
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Check code quality**:
   ```bash
   ruff check .
   black .
   mypy fed_vit_autorl/
   ```

## Architecture Overview

```
fed_vit_autorl/
├── models/           # Vision Transformer models
├── federated/        # Federated learning algorithms
├── reinforcement/    # RL algorithms for driving
├── edge/             # Edge deployment tools
├── simulation/       # CARLA integration
└── evaluation/       # Metrics and evaluation
```

## Key Components

### Vision Transformer Models
- **ViTPerception**: Base ViT backbone for perception
- **DetectionHead**: Object detection for vehicles/pedestrians
- **SegmentationHead**: Lane and free space segmentation

### Federated Learning
- **FederatedTrainer**: Coordinate distributed training
- **PrivacyMechanisms**: Differential privacy and secure aggregation
- **Aggregation**: FedAvg, FedProx, and custom algorithms

### Edge Deployment
- **ModelOptimization**: Pruning, quantization, distillation
- **EdgeInference**: Real-time inference on vehicle hardware
- **ResourceManager**: Memory and compute allocation

## Testing

- **Unit Tests**: `pytest tests/`
- **Integration Tests**: `pytest tests/integration/`
- **Performance Tests**: `pytest tests/performance/`
- **Coverage**: Maintain >80% coverage

## Development Workflow

1. Create feature branch from `main`
2. Make changes with tests
3. Run full test suite
4. Create pull request
5. Code review and merge

## Common Tasks

### Adding New Models
1. Create model class in `fed_vit_autorl/models/`
2. Add to `__init__.py`
3. Write tests in `tests/test_models.py`
4. Update documentation

### Running Experiments
```bash
# Local simulation
python scripts/train_federated.py --config configs/local.yaml

# Multi-node training
python scripts/distributed_train.py --nodes 4 --gpus 2
```

### Performance Profiling
```bash
# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# Compute profiling
python -m cProfile scripts/profile_compute.py
```