# Manual Setup Required

This document outlines the manual setup steps required to complete the Fed-ViT-AutoRL SDLC implementation due to GitHub App permission limitations.

## GitHub Actions Workflows

⚠️ **Repository maintainers must manually create GitHub Actions workflows** as the automated setup has limited permissions.

### Required Actions

1. **Create `.github/workflows/` directory** in the repository root
2. **Copy workflow files** from the templates in `docs/workflows/examples/` to `.github/workflows/`
3. **Configure repository secrets** as documented below
4. **Set up branch protection rules** as specified

### Workflow Files to Create

Copy these files from `docs/workflows/examples/` to `.github/workflows/`:

- `ci.yml` - Continuous Integration
- `cd.yml` - Continuous Deployment
- `security-scan.yml` - Security Scanning
- `dependency-update.yml` - Dependency Updates
- `performance.yml` - Performance Monitoring

### Repository Secrets Configuration

Configure these secrets in **Settings → Secrets and variables → Actions**:

#### Required Secrets
```bash
# Package publishing
PYPI_API_TOKEN=pypi-xxxxx
TEST_PYPI_API_TOKEN=pypi-xxxxx

# Container registry
DOCKER_HUB_USERNAME=your-username
DOCKER_HUB_ACCESS_TOKEN=your-token

# Code coverage and quality
CODECOV_TOKEN=your-codecov-token
SONAR_TOKEN=your-sonarcloud-token

# Security scanning
SNYK_TOKEN=your-snyk-token
```

#### Optional Secrets
```bash
# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/xxxxx

# Cloud deployment
AWS_ACCESS_KEY_ID=AKIAXXXXX
AWS_SECRET_ACCESS_KEY=xxxxx
AWS_REGION=us-west-2

# Monitoring
GRAFANA_API_KEY=your-grafana-key
PROMETHEUS_URL=https://prometheus.example.com
```

## Branch Protection Rules

Configure branch protection in **Settings → Branches**:

### Main Branch Protection
- [x] Restrict pushes that create files larger than 100MB
- [x] Require a pull request before merging
  - [x] Require approvals: **2**
  - [x] Dismiss stale reviews when new commits are pushed
  - [x] Require review from code owners
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Required status checks:
    - `ci / lint-and-test`
    - `ci / security-scan`
    - `ci / type-check`
    - `performance / benchmark`
- [x] Require linear history
- [x] Include administrators

### Develop Branch Protection
- [x] Require a pull request before merging
  - [x] Require approvals: **1**
- [x] Require status checks to pass before merging
  - Required status checks:
    - `ci / lint-and-test`
    - `ci / security-scan`

## GitHub Security Features

Enable these security features in **Settings → Security & analysis**:

### Dependency Graph
- [x] Enable dependency graph

### Dependabot
- [x] Enable Dependabot alerts
- [x] Enable Dependabot security updates
- [x] Enable Dependabot version updates

### Code Scanning
- [x] Enable CodeQL analysis
- [x] Enable third-party code scanning tools

### Secret Scanning
- [x] Enable secret scanning
- [x] Enable push protection

## Repository Settings

Configure these repository settings:

### General Settings
- **Description**: "Federated reinforcement learning framework for Vision Transformer based autonomous vehicles"
- **Topics**: `federated-learning`, `reinforcement-learning`, `vision-transformer`, `autonomous-vehicles`, `pytorch`, `privacy-preserving`
- **Website**: https://fed-vit-autorl.readthedocs.io (when available)

### Features
- [x] Wikis
- [x] Issues
- [x] Sponsorships
- [x] Preserve this repository
- [x] Projects
- [x] Discussions

### Pull Requests
- [x] Allow merge commits
- [x] Allow squash merging
- [x] Allow rebase merging
- [x] Always suggest updating pull request branches
- [x] Automatically delete head branches

## Issue and PR Templates

Create these template files in `.github/`:

### Issue Templates
1. `.github/ISSUE_TEMPLATE/bug_report.yml`
2. `.github/ISSUE_TEMPLATE/feature_request.yml`
3. `.github/ISSUE_TEMPLATE/security_vulnerability.yml`

### Pull Request Template
1. `.github/PULL_REQUEST_TEMPLATE.md`

Copy templates from `docs/workflows/templates/` directory.

## Integrations Setup

### External Services

#### Code Quality
1. **SonarCloud**: Connect repository for code quality analysis
2. **Codecov**: Setup for test coverage reporting
3. **Snyk**: Configure for security vulnerability scanning

#### Monitoring
1. **Grafana**: Import dashboards from `monitoring/grafana-dashboards/`
2. **Prometheus**: Deploy configuration from `monitoring/prometheus.yml`

#### Documentation
1. **ReadTheDocs**: Connect repository for documentation hosting
2. **GitHub Pages**: Configure for additional documentation

## Verification Checklist

After completing manual setup, verify:

- [ ] All GitHub Actions workflows are running successfully
- [ ] Branch protection rules are enforced
- [ ] Dependabot is creating update PRs
- [ ] Security scanning is detecting and reporting issues
- [ ] Code quality checks are passing
- [ ] Documentation is building and deploying
- [ ] Monitoring dashboards are functional
- [ ] Container builds are working
- [ ] Package publishing is configured (test with pre-release)

## Support and Troubleshooting

### Common Issues

1. **Workflow Permission Errors**
   - Ensure GITHUB_TOKEN has required permissions
   - Check repository settings for Actions permissions

2. **Secret Access Issues**
   - Verify secrets are configured correctly
   - Check secret names match workflow references

3. **Branch Protection Conflicts**
   - Ensure required status checks exist before enabling
   - Use branch protection preview to verify rules

### Getting Help

1. Review detailed setup guides in `docs/workflows/guides/`
2. Check example workflow implementations
3. Consult GitHub Actions documentation
4. Open an issue for project-specific questions

## Implementation Timeline

Recommended implementation order:

1. **Week 1**: Basic CI/CD workflows and branch protection
2. **Week 2**: Security scanning and dependency management
3. **Week 3**: Monitoring and observability setup
4. **Week 4**: Documentation and final integrations

## Maintenance

Regular maintenance tasks:

- **Monthly**: Review and update workflow configurations
- **Quarterly**: Audit security settings and permissions
- **Annually**: Review and update branch protection rules

---

**Last Updated**: January 2025  
**Next Review**: Quarterly  
**Document Owner**: Repository Maintainers