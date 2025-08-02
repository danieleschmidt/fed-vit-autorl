# Continuous Integration Setup Guide

This guide provides step-by-step instructions for setting up the CI workflow for Fed-ViT-AutoRL.

## Prerequisites

- Repository admin access
- GitHub Actions enabled
- Required secrets configured

## Step 1: Copy Workflow File

1. Navigate to your repository root
2. Create the `.github/workflows/` directory if it doesn't exist:
   ```bash
   mkdir -p .github/workflows
   ```
3. Copy the CI workflow:
   ```bash
   cp docs/workflows/examples/ci.yml .github/workflows/ci.yml
   ```

## Step 2: Configure Repository Secrets

Navigate to **Repository Settings > Secrets and Variables > Actions** and add:

### Required Secrets

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `CODECOV_TOKEN` | Code coverage reporting | `12345678-1234-1234-1234-123456789012` |
| `GITGUARDIAN_API_KEY` | Secret scanning | `ggshield_api_key_here` |

### Optional Secrets

| Secret Name | Description | When Needed |
|-------------|-------------|-------------|
| `SLACK_WEBHOOK_URL` | Notifications | If using Slack notifications |
| `SONAR_TOKEN` | SonarCloud integration | If using SonarCloud |

## Step 3: Set Up Branch Protection Rules

1. Go to **Settings > Branches**
2. Click **Add rule** for the main branch
3. Configure the following settings:

### Branch Protection Configuration

```yaml
Branch name pattern: main
Settings:
  ✅ Require a pull request before merging
    ✅ Require approvals: 2
    ✅ Dismiss stale PR approvals when new commits are pushed
    ✅ Require review from CODEOWNERS
  ✅ Require status checks to pass before merging
    ✅ Require branches to be up to date before merging
    Required status checks:
      - lint-and-format
      - test (ubuntu-latest, 3.11)
      - security
      - docs
      - build
  ✅ Require conversation resolution before merging
  ✅ Require linear history
  ✅ Restrict pushes that create files
  ✅ Do not allow bypassing the above settings
```

## Step 4: Configure Code Coverage

### Codecov Setup

1. Visit [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Add your repository
4. Copy the repository token
5. Add `CODECOV_TOKEN` to repository secrets

### Coverage Configuration

Create `.codecov.yml` in repository root:

```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 2%
    patch:
      default:
        target: 80%
        threshold: 2%

comment:
  layout: "reach,diff,flags,tree"
  behavior: default
  require_changes: false
```

## Step 5: Configure Security Scanning

### GitGuardian Setup

1. Visit [gitguardian.com](https://www.gitguardian.com)
2. Create an account and get API key
3. Add `GITGUARDIAN_API_KEY` to repository secrets

### Bandit Configuration

Create `.bandit` file in repository root:

```ini
[bandit]
exclude_dirs = tests,docs,scripts
skips = B101,B601
```

## Step 6: Test the CI Setup

1. Create a test branch:
   ```bash
   git checkout -b test-ci-setup
   ```

2. Make a small change (e.g., update README)

3. Commit and push:
   ```bash
   git add .
   git commit -m "test: verify CI setup"
   git push -u origin test-ci-setup
   ```

4. Create a pull request

5. Verify all CI checks pass:
   - ✅ Lint and format checks
   - ✅ Tests on multiple Python versions
   - ✅ Security scans
   - ✅ Documentation build
   - ✅ Package build
   - ✅ Performance benchmarks (if applicable)

## Step 7: Troubleshooting

### Common Issues

#### Tests Failing
```bash
# Check test dependencies
pip install -e ".[dev]"
pytest tests/ -v
```

#### Linting Errors
```bash
# Run pre-commit hooks locally
pre-commit run --all-files
```

#### Security Scan Failures
- Review and fix any identified security issues
- Update `.bandit` configuration if false positives
- Rotate any exposed secrets

#### Permission Errors
- Verify repository secrets are set correctly
- Check branch protection rules
- Ensure GitHub Actions has necessary permissions

### Debugging Steps

1. **Check workflow syntax**:
   - Use GitHub's workflow editor to validate YAML
   - Verify all required fields are present

2. **Review action logs**:
   - Click on failed step in Actions tab
   - Look for specific error messages
   - Check environment variables and secrets

3. **Test locally**:
   ```bash
   # Install act for local testing
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # Run CI locally
   act push
   ```

## Step 8: Optimization

### Performance Optimization

1. **Enable caching**:
   - Python dependencies are cached by default
   - Consider caching test data or models

2. **Parallel execution**:
   - Matrix builds run in parallel
   - Independent jobs execute concurrently

3. **Conditional execution**:
   - Skip expensive steps for documentation changes
   - Use path filters for targeted testing

### Cost Optimization

1. **Reduce matrix size**:
   - Test on fewer OS/Python combinations for PRs
   - Full matrix only for main branch

2. **Skip redundant steps**:
   - Use `if` conditions to skip unnecessary jobs
   - Fast-fail strategy for critical errors

## Step 9: Maintenance

### Regular Tasks

1. **Update dependencies**:
   - Keep action versions current
   - Update Python versions in matrix
   - Review security tool versions

2. **Monitor performance**:
   - Check CI execution times
   - Optimize slow steps
   - Review resource usage

3. **Review security**:
   - Update security scanning rules
   - Rotate secrets regularly
   - Monitor vulnerability reports

### Quarterly Review

- Analyze CI metrics and trends
- Update branch protection rules
- Review and update test coverage targets
- Evaluate new tools and integrations

## Advanced Configuration

### Custom Test Environments

```yaml
# Add custom test environments
test-environments:
  - name: "edge-simulation"
    python-version: "3.11"
    extra-packages: "carla>=0.9.13"
    environment-variables:
      CARLA_HOST: "localhost"
      CARLA_PORT: "2000"
```

### Integration Testing

```yaml
# Extended integration tests
integration-extended:
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - name: Run extended integration tests
      run: pytest tests/integration/ -v --slow
```

### Performance Regression Detection

```yaml
# Automated performance regression detection
performance-check:
  runs-on: ubuntu-latest
  steps:
    - name: Compare performance
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        fail-on-alert: true
        comment-on-alert: true
```

## Support and Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Code Coverage Best Practices](https://codecov.io/docs)

For project-specific support, open an issue in the repository.