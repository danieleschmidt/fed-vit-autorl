# Fed-ViT-AutoRL Makefile
# Provides standardized commands for development workflow

.PHONY: help install install-dev test lint format clean build docs serve-docs docker-build docker-run

# Default target
help: ## Show this help message
	@echo "Fed-ViT-AutoRL Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install package for production use
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,simulation,edge]"
	pre-commit install

install-all: ## Install all dependencies including optional ones
	pip install -e ".[dev,simulation,edge]"
	pre-commit install

# Development
dev-setup: install-dev ## Set up development environment
	mkdir -p {data,logs,models,experiments,cache}
	echo "Development environment ready!"

# Testing
test: ## Run all tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --cov=fed_vit_autorl --cov-report=html --cov-report=term

test-fast: ## Run tests excluding slow ones
	python -m pytest tests/ -v -m "not slow"

test-watch: ## Run tests continuously on file changes
	python -m pytest-watch tests/

# Code Quality
lint: ## Run all linters
	python -m ruff check .
	python -m mypy fed_vit_autorl/
	python -m bandit -r fed_vit_autorl/

format: ## Format code with black and ruff
	python -m black .
	python -m ruff check --fix .

format-check: ## Check if code formatting is correct
	python -m black --check .
	python -m ruff check .

type-check: ## Run type checking
	python -m mypy fed_vit_autorl/

security-check: ## Run security checks
	python -m bandit -r fed_vit_autorl/
	python -m safety check

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Cleaning
clean: ## Clean up build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

clean-data: ## Clean up data and model directories (use with caution)
	@read -p "This will delete all data, models, and logs. Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/* models/* logs/* cache/*; \
		echo "Data directories cleaned."; \
	fi

# Building
build: ## Build package
	python -m build

build-wheel: ## Build wheel package
	python -m build --wheel

build-sdist: ## Build source distribution
	python -m build --sdist

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && make livehtml

docs-clean: ## Clean documentation build
	cd docs && make clean

# Docker
docker-build: ## Build Docker image
	docker build -t fed-vit-autorl:latest .

docker-build-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t fed-vit-autorl:dev .

docker-run: ## Run Docker container
	docker run -it --rm --gpus all -v $(PWD):/workspace fed-vit-autorl:latest

docker-run-dev: ## Run development Docker container
	docker run -it --rm --gpus all -v $(PWD):/workspace -p 8888:8888 -p 6006:6006 fed-vit-autorl:dev

# Simulation
carla-setup: ## Set up CARLA simulator
	@echo "Setting up CARLA simulator..."
	@echo "Please ensure CARLA is installed and configured properly"
	@echo "See docs/guides/carla-integration.md for details"

# Training and Experiments
train-local: ## Run local training example
	python scripts/train_local.py --config configs/local_training.yaml

train-federated: ## Run federated training example
	python scripts/train_federated.py --config configs/federated_training.yaml

experiment: ## Run a complete experiment
	python scripts/run_experiment.py --config configs/experiment.yaml

# Benchmarking
benchmark: ## Run performance benchmarks
	python -m pytest tests/benchmarks/ -v --benchmark-only

benchmark-save: ## Run benchmarks and save results
	python -m pytest tests/benchmarks/ -v --benchmark-only --benchmark-save=benchmark

profile: ## Profile the application
	python -m cProfile -o profile.stats scripts/profile_app.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Deployment
deploy-edge: ## Deploy to edge device
	@echo "Deploying to edge device..."
	@echo "See docs/guides/edge-deployment.md for details"

deploy-cloud: ## Deploy to cloud
	@echo "Deploying to cloud..."
	@echo "See docs/guides/cloud-deployment.md for details"

# Monitoring
tensorboard: ## Start TensorBoard
	tensorboard --logdir=logs --host=0.0.0.0 --port=6006

jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Release
release-check: ## Check if ready for release
	python -m build
	python -m twine check dist/*

release-test: ## Upload to test PyPI
	python -m twine upload --repository testpypi dist/*

release: ## Upload to PyPI (production)
	python -m twine upload dist/*

# Environment
env-info: ## Show environment information
	@echo "Python version: $$(python --version)"
	@echo "Pip version: $$(pip --version)"
	@echo "Installed packages:"
	@pip list | grep -E "(torch|transformers|fed-vit-autorl)"

# Git hooks
pre-push: lint test ## Run checks before pushing
	@echo "All checks passed! Ready to push."

# Development shortcuts
dev: install-dev format test ## Full development cycle
quick-check: format-check lint test-fast ## Quick development checks

# CI/CD helpers
ci-install: ## Install dependencies for CI
	pip install -e ".[dev]"

ci-test: ## Run tests for CI
	python -m pytest tests/ -v --cov=fed_vit_autorl --cov-report=xml

ci-lint: ## Run linting for CI
	python -m ruff check .
	python -m mypy fed_vit_autorl/
	python -m black --check .

# Database (if applicable)
db-init: ## Initialize database
	python scripts/init_db.py

db-migrate: ## Run database migrations
	python scripts/migrate_db.py

db-reset: ## Reset database (use with caution)
	@read -p "This will reset the database. Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		python scripts/reset_db.py; \
		echo "Database reset."; \
	fi