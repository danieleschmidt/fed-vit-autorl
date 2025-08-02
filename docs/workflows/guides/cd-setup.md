# Continuous Deployment Setup Guide

This guide covers setting up automated deployment workflows for Fed-ViT-AutoRL.

## Overview

The CD pipeline handles:
- Automated releases and versioning
- Package publishing to PyPI
- Docker image builds and pushes
- Documentation deployment
- Cloud environment deployments

## Prerequisites

### Required Secrets
Configure these secrets in GitHub repository settings:

```bash
# Package publishing
PYPI_API_TOKEN=pypi-xxxxx
TEST_PYPI_API_TOKEN=pypi-xxxxx

# Container registry
DOCKER_HUB_USERNAME=your-username
DOCKER_HUB_ACCESS_TOKEN=your-token

# Cloud deployment (optional)
AWS_ACCESS_KEY_ID=AKIAXXXXX
AWS_SECRET_ACCESS_KEY=xxxxx
AWS_REGION=us-west-2

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/xxxxx
```

### Branch Protection
Ensure these branch protection rules are configured:
- Require pull request reviews
- Require status checks to pass
- Include administrators in restrictions

## Workflow Configuration

### 1. Release Workflow

Create `.github/workflows/cd.yml`:

```yaml
name: Continuous Deployment

on:
  push:
    tags:
      - 'v*'
  release:
    types: [published]

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install build twine
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          pytest tests/ -v
          ruff check .
          mypy fed_vit_autorl/
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
      
      - name: Build Docker images
        run: |
          docker build --target prod -t fed-vit-autorl:${{ github.ref_name }} .
          docker build --target edge -t fed-vit-autorl:${{ github.ref_name }}-edge .
      
      - name: Push Docker images
        env:
          DOCKER_USERNAME: ${{ secrets.DOCKER_HUB_USERNAME }}
          DOCKER_PASSWORD: ${{ secrets.DOCKER_HUB_ACCESS_TOKEN }}
        run: |
          echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin
          docker push fed-vit-autorl:${{ github.ref_name }}
          docker push fed-vit-autorl:${{ github.ref_name }}-edge
```

### 2. Documentation Deployment

Add documentation deployment to CD workflow:

```yaml
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install sphinx sphinx-rtd-theme
      
      - name: Build documentation
        run: |
          cd docs
          make html
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_build/html
```

### 3. Staging Deployment

For automatic staging deployments:

```yaml
  staging:
    runs-on: ubuntu-latest
    environment: staging
    if: github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to staging
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          # Build and deploy to staging environment
          docker build --target prod -t fed-vit-autorl:staging .
          
          # Deploy to ECS/EKS/etc
          aws ecs update-service \
            --cluster staging \
            --service fed-vit-autorl \
            --force-new-deployment
```

## Release Process

### Semantic Versioning

Fed-ViT-AutoRL follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Creating Releases

#### 1. Prepare Release
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit changes
git add .
git commit -m "chore: prepare release v1.2.3"
git push origin main
```

#### 2. Create Tag
```bash
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin v1.2.3
```

#### 3. Create GitHub Release
- Go to Releases in GitHub
- Click "Create a new release"
- Select the tag
- Generate release notes
- Publish release

### Automated Release Notes

Configure automatic release notes in `.github/release.yml`:

```yaml
changelog:
  exclude:
    labels:
      - ignore-for-release
  categories:
    - title: Breaking Changes 🛠
      labels:
        - breaking-change
    - title: New Features 🎉
      labels:
        - enhancement
        - feature
    - title: Bug Fixes 🐛
      labels:
        - bug
    - title: Documentation 📚
      labels:
        - documentation
    - title: Other Changes
      labels:
        - "*"
```

## Deployment Environments

### Staging Environment

**Purpose**: Test releases before production
**Trigger**: Pushes to `develop` branch
**Configuration**:
```yaml
environment: staging
variables:
  - FED_VIT_AUTORL_ENV=staging
  - LOG_LEVEL=DEBUG
  - PRIVACY_EPSILON=2.0  # More relaxed for testing
```

### Production Environment

**Purpose**: Live federated learning system
**Trigger**: Release tags
**Configuration**:
```yaml
environment: production
variables:
  - FED_VIT_AUTORL_ENV=production
  - LOG_LEVEL=INFO
  - PRIVACY_EPSILON=1.0
```

### Edge Deployment

**Purpose**: Deploy to edge devices
**Trigger**: Manual or scheduled
**Configuration**:
```yaml
  edge-deploy:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        device: [jetson-xavier, jetson-nano, raspberry-pi]
    steps:
      - name: Build for ${{ matrix.device }}
        run: |
          docker buildx build \
            --platform linux/arm64 \
            --target edge \
            -t fed-vit-autorl:${{ matrix.device }} .
      
      - name: Deploy to device
        run: |
          # Deploy to specific device type
          ansible-playbook -i inventory/${{ matrix.device }} deploy.yml
```

## Rollback Procedures

### Automatic Rollback

Configure automatic rollback on deployment failure:

```yaml
  rollback:
    runs-on: ubuntu-latest
    if: failure()
    needs: [deploy]
    steps:
      - name: Rollback deployment
        run: |
          # Rollback to previous version
          kubectl rollout undo deployment/fed-vit-server
          
          # Or for ECS
          aws ecs update-service \
            --cluster production \
            --service fed-vit-autorl \
            --task-definition fed-vit-autorl:previous
```

### Manual Rollback

```bash
# List recent deployments
kubectl get deployments
kubectl rollout history deployment/fed-vit-server

# Rollback to specific revision
kubectl rollout undo deployment/fed-vit-server --to-revision=2

# Verify rollback
kubectl rollout status deployment/fed-vit-server
```

## Monitoring Deployments

### Deployment Health Checks

```yaml
  health-check:
    runs-on: ubuntu-latest
    needs: [deploy]
    steps:
      - name: Wait for deployment
        run: sleep 60
      
      - name: Check service health
        run: |
          curl -f https://fed-vit-autorl.example.com/health
          
          # Check key metrics
          curl -s https://fed-vit-autorl.example.com/metrics | \
            grep "fed_vit_autorl_active_clients" | \
            awk '{if ($2 > 0) exit 0; else exit 1}'
      
      - name: Run smoke tests
        run: |
          python scripts/smoke_tests.py --environment production
```

### Deployment Notifications

```yaml
  notify:
    runs-on: ubuntu-latest
    needs: [deploy, health-check]
    if: always()
    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          text: |
            Deployment ${{ job.status }}: Fed-ViT-AutoRL ${{ github.ref_name }}
            Environment: production
            Commit: ${{ github.sha }}
```

## Security Considerations

### Supply Chain Security

1. **Pin Dependencies**: Use exact versions in requirements.txt
2. **Verify Signatures**: Check package signatures when possible
3. **Scan Images**: Use container scanning tools
4. **Sign Artifacts**: Sign releases and container images

### Access Control

1. **Environment Protection**: Use GitHub environment protection rules
2. **Required Reviewers**: Require approval for production deployments
3. **Time Delays**: Add delays for production deployments
4. **Restricted Branches**: Limit who can push to protected branches

### Secret Management

```yaml
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-west-2
      
      - name: Get secrets from AWS Secrets Manager
        run: |
          SECRET=$(aws secretsmanager get-secret-value \
            --secret-id fed-vit-autorl/production \
            --query SecretString --output text)
          echo "::add-mask::$SECRET"
```

## Troubleshooting

### Common Issues

1. **PyPI Upload Fails**
   ```bash
   # Check token permissions
   # Verify package name availability
   # Test with test PyPI first
   ```

2. **Docker Push Fails**
   ```bash
   # Check registry credentials
   # Verify image names and tags
   # Check repository permissions
   ```

3. **Deployment Timeout**
   ```bash
   # Check resource limits
   # Verify health check endpoints
   # Review container logs
   ```

### Debugging Deployments

```yaml
  debug:
    runs-on: ubuntu-latest
    if: failure()
    steps:
      - name: Collect debug info
        run: |
          kubectl describe deployment/fed-vit-server
          kubectl logs deployment/fed-vit-server --tail=100
          kubectl get events --sort-by='.lastTimestamp'
      
      - name: Upload debug artifacts
        uses: actions/upload-artifact@v3
        with:
          name: debug-logs
          path: debug/
```

## Best Practices

1. **Gradual Rollouts**: Use blue-green or canary deployments
2. **Feature Flags**: Enable gradual feature rollout
3. **Monitoring**: Monitor key metrics during deployment
4. **Testing**: Run comprehensive tests before deployment
5. **Documentation**: Keep deployment docs up to date
6. **Automation**: Automate as much as possible
7. **Recovery**: Have rollback procedures ready

## Integration with Monitoring

### Deployment Tracking

```python
# In your application
from prometheus_client import Counter

DEPLOYMENTS = Counter(
    'fed_vit_autorl_deployments_total',
    'Total number of deployments',
    ['environment', 'version', 'status']
)

def track_deployment(environment, version, status):
    DEPLOYMENTS.labels(
        environment=environment,
        version=version,
        status=status
    ).inc()
```

### Alerting on Deployment Issues

```yaml
# In alert_rules.yml
- alert: DeploymentFailed
  expr: increase(fed_vit_autorl_deployments_total{status="failed"}[1h]) > 0
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: "Deployment failed"
    description: "Deployment failed in {{ $labels.environment }}"
```