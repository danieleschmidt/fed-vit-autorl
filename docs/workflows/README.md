# GitHub Workflows Documentation

**⚠️ MANUAL SETUP REQUIRED**: Due to GitHub App permission limitations, the workflow files in this directory must be manually created by repository maintainers.

## Overview

This directory contains documentation and templates for GitHub Actions workflows that should be implemented for the Fed-ViT-AutoRL project. These workflows provide automated CI/CD, security scanning, and deployment capabilities.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)
- **Purpose**: Validate pull requests and commits
- **Triggers**: Pull requests, pushes to main/develop
- **Actions**: Linting, testing, security scanning, type checking

### 2. Continuous Deployment (`cd.yml`)
- **Purpose**: Automated deployment and releases
- **Triggers**: Tags, main branch pushes
- **Actions**: Build packages, deploy documentation, create releases

### 3. Security Scanning (`security-scan.yml`)
- **Purpose**: Comprehensive security assessment
- **Triggers**: Schedule (daily), pull requests
- **Actions**: Dependency scanning, secret detection, SAST analysis

### 4. Dependency Management (`dependency-update.yml`)
- **Purpose**: Automated dependency updates
- **Triggers**: Schedule (weekly)
- **Actions**: Update dependencies, create PRs, run tests

### 5. Performance Monitoring (`performance.yml`)
- **Purpose**: Track performance regressions
- **Triggers**: Pull requests, releases
- **Actions**: Benchmark tests, performance reporting

## Setup Instructions

1. **Create `.github/workflows/` directory** in the repository root
2. **Copy workflow files** from `docs/workflows/examples/` to `.github/workflows/`
3. **Configure secrets** as documented in each workflow
4. **Set up branch protection** rules as specified in security documentation
5. **Test workflows** with a test PR or push

## Branch Protection Requirements

The following branch protection rules should be configured:

### Main Branch Protection
- Require pull request reviews (minimum 2)
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to administrators only
- Require linear history

### Required Status Checks
- `ci / lint-and-test`
- `ci / security-scan`
- `ci / type-check`
- `performance / benchmark`

## Secrets Configuration

The following secrets must be configured in repository settings:

### Required Secrets
- `PYPI_API_TOKEN`: For publishing packages
- `DOCKER_HUB_USERNAME`: For Docker registry
- `DOCKER_HUB_ACCESS_TOKEN`: For Docker registry
- `CODECOV_TOKEN`: For code coverage reporting

### Optional Secrets
- `SLACK_WEBHOOK_URL`: For deployment notifications
- `AWS_ACCESS_KEY_ID`: For cloud deployments
- `AWS_SECRET_ACCESS_KEY`: For cloud deployments

## Workflow Examples

See the `examples/` directory for complete workflow implementations:

- [`examples/ci.yml`](examples/ci.yml) - Continuous Integration
- [`examples/cd.yml`](examples/cd.yml) - Continuous Deployment  
- [`examples/security-scan.yml`](examples/security-scan.yml) - Security Scanning
- [`examples/dependency-update.yml`](examples/dependency-update.yml) - Dependency Updates
- [`examples/performance.yml`](examples/performance.yml) - Performance Monitoring

## Workflow Guides

See the `guides/` directory for detailed setup instructions:

- [`guides/ci-setup.md`](guides/ci-setup.md) - CI Configuration Guide
- [`guides/cd-setup.md`](guides/cd-setup.md) - CD Configuration Guide
- [`guides/security-setup.md`](guides/security-setup.md) - Security Configuration
- [`guides/monitoring-setup.md`](guides/monitoring-setup.md) - Monitoring Setup

## Validation

After setting up workflows, validate the configuration:

1. **Create a test PR** to trigger CI workflows
2. **Check all status checks** pass successfully
3. **Verify security scans** complete without critical issues
4. **Test deployment** workflows with a test release
5. **Monitor performance** benchmarks for regressions

## Troubleshooting

Common issues and solutions:

### Workflow Not Triggering
- Check file placement in `.github/workflows/`
- Verify YAML syntax with GitHub's workflow editor
- Ensure triggers are correctly configured

### Permission Errors
- Verify repository secrets are configured
- Check GitHub App permissions
- Ensure branch protection rules allow workflows

### Test Failures
- Review test logs in Actions tab
- Check dependencies and environment setup
- Verify test data and fixtures are available

## Support

For workflow setup assistance:
1. Review the detailed guides in `guides/`
2. Check example implementations in `examples/`
3. Consult GitHub Actions documentation
4. Open an issue for project-specific questions

---

**Last Updated**: January 2025  
**Next Review**: Quarterly workflow optimization review