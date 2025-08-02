# Fed-ViT-AutoRL Build Guide

This guide covers building, containerizing, and deploying the Fed-ViT-AutoRL framework across different environments.

## Quick Start

```bash
# Build all container variants
make docker-build

# Start development environment
make docker-dev

# Start full federated system
make docker-up
```

## Build Targets

### Production Build
Optimized for production federated learning servers:

```bash
docker build --target prod -t fed-vit-autorl:prod .
```

**Features:**
- Minimal dependencies
- Security hardening
- Health checks
- Non-root user

### Development Build
Full development environment with all tools:

```bash
docker build --target dev -t fed-vit-autorl:dev .
```

**Features:**
- Development dependencies
- Jupyter Lab
- TensorBoard
- Debugging tools

### Edge Build
Optimized for resource-constrained edge devices:

```bash
docker build --target edge -t fed-vit-autorl:edge .
```

**Features:**
- Minimal footprint (~500MB)
- Edge-optimized libraries
- ARM64 support
- Low memory usage

### Simulation Build
Environment for CARLA and multi-agent simulation:

```bash
docker build --target simulation -t fed-vit-autorl:simulation .
```

**Features:**
- CARLA dependencies
- Virtual display support
- Multi-vehicle simulation
- Headless operation

## Build Configuration

### Environment Variables

```bash
# Build arguments
DOCKER_BUILDKIT=1 docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg TORCH_VERSION=2.0.0 \
  --build-arg CUDA_VERSION=11.8 \
  --target prod \
  -t fed-vit-autorl:latest .
```

### Multi-platform Builds

```bash
# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target prod \
  -t fed-vit-autorl:multiarch .
```

## Deployment Scenarios

### Local Development

```bash
# Start development stack
docker-compose up fed-dev postgres redis

# Access Jupyter Lab
open http://localhost:8888

# Access TensorBoard
open http://localhost:6007
```

### Federated Learning Server

```bash
# Production server deployment
docker-compose up -d fed-server prometheus grafana

# Monitor system
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

### Edge Vehicle Deployment

```bash
# Deploy edge nodes
docker-compose up -d fed-edge-1 fed-edge-2

# Check edge node status
curl http://localhost:8081/health
curl http://localhost:8082/health
```

### Full Simulation Environment

```bash
# Start simulation stack
docker-compose up simulation

# Run federated simulation
docker exec -it fed-vit-simulation python scripts/run_simulation.py
```

## Security Considerations

### Container Security

1. **Non-root User**: All containers run as non-root user `fedvit`
2. **Read-only Filesystems**: Production containers use read-only root filesystem
3. **Resource Limits**: Memory and CPU limits enforced
4. **Health Checks**: Continuous health monitoring

### Secrets Management

```bash
# Use Docker secrets for production
echo "your-secret-key" | docker secret create fed_vit_key -

# Reference in compose file
docker-compose -f docker-compose.prod.yml up -d
```

## Performance Optimization

### Build Cache Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use registry cache
docker build \
  --cache-from fed-vit-autorl:cache \
  --cache-to type=registry,ref=fed-vit-autorl:cache \
  -t fed-vit-autorl:latest .
```

### Resource Allocation

```yaml
# Recommended resource limits
services:
  fed-server:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

## Monitoring and Logging

### Container Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f fed-server

# Export logs
docker-compose logs --no-color > fed-vit-logs.txt
```

### Health Monitoring

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Detailed health check
docker inspect --format='{{.State.Health.Status}}' fed-vit-server
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase shared memory
   docker run --shm-size=2g fed-vit-autorl:latest
   ```

2. **Permission Errors**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data ./logs ./models
   ```

3. **Network Issues**
   ```bash
   # Recreate network
   docker network rm fed-vit-network
   docker-compose up -d
   ```

### Debug Mode

```bash
# Run container in debug mode
docker run -it --rm \
  -v $(pwd):/app \
  fed-vit-autorl:dev \
  bash

# Enable debug logging
docker-compose -f docker-compose.yml \
  -f docker-compose.debug.yml up
```

## CI/CD Integration

### GitHub Actions Build

```yaml
# .github/workflows/docker.yml
name: Docker Build
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build --target prod -t fed-vit-autorl:prod .
          docker build --target edge -t fed-vit-autorl:edge .
```

### Registry Push

```bash
# Tag and push to registry
docker tag fed-vit-autorl:prod your-registry.com/fed-vit-autorl:latest
docker push your-registry.com/fed-vit-autorl:latest
```

## Maintenance

### Image Cleanup

```bash
# Remove unused images
docker image prune -f

# Remove all fed-vit images
docker images | grep fed-vit | awk '{print $3}' | xargs docker rmi
```

### Volume Backup

```bash
# Backup volumes
docker run --rm -v postgres-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v postgres-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres-backup.tar.gz -C /data
```

## Best Practices

1. **Layer Caching**: Order Dockerfile commands by change frequency
2. **Multi-stage Builds**: Use appropriate target for each use case
3. **Resource Limits**: Always set memory and CPU limits
4. **Health Checks**: Implement comprehensive health checks
5. **Secrets**: Never bake secrets into images
6. **Logs**: Use structured logging with appropriate levels
7. **Monitoring**: Implement metrics collection and alerting

## Advanced Topics

### Custom Base Images

```dockerfile
# Custom PyTorch base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel AS pytorch-base
# ... custom configuration
```

### ARM64 Support

```bash
# Build for ARM64 (for Jetson devices)
docker buildx build \
  --platform linux/arm64 \
  --target edge \
  -t fed-vit-autorl:arm64 .
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fed-vit-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fed-vit-server
  template:
    spec:
      containers:
      - name: fed-vit-server
        image: fed-vit-autorl:prod
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
```