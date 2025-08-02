# Security Workflow Setup Guide

This guide covers implementing comprehensive security workflows for Fed-ViT-AutoRL, including vulnerability scanning, dependency management, and compliance checking.

## Overview

Security workflows provide:
- Automated vulnerability scanning
- Dependency security analysis
- Secret detection and prevention
- SAST (Static Application Security Testing)
- DAST (Dynamic Application Security Testing)
- Container security scanning
- Supply chain security validation

## Prerequisites

### Required Tools and Tokens

```bash
# GitHub Security Features
# - Dependabot (built-in)
# - CodeQL (built-in)
# - Secret scanning (built-in)

# External Integrations
SNYK_TOKEN=your-snyk-token
SONAR_TOKEN=your-sonarcloud-token
CODECOV_TOKEN=your-codecov-token

# Container Scanning
DOCKER_HUB_USERNAME=your-username
DOCKER_HUB_ACCESS_TOKEN=your-token
```

### Repository Settings

Enable these GitHub security features:
1. **Dependency Graph**: Repository → Settings → Security & analysis
2. **Dependabot Alerts**: Enable automatic dependency vulnerability alerts
3. **Dependabot Security Updates**: Enable automatic security updates
4. **Secret Scanning**: Enable for public/private repos
5. **Code Scanning**: Enable CodeQL analysis

## Security Workflows

### 1. Comprehensive Security Scan

Create `.github/workflows/security-scan.yml`:

```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install safety bandit semgrep
          pip install -e ".[dev]"
      
      - name: Run Safety (dependency vulnerabilities)
        run: |
          safety check --json --output safety-report.json || true
          safety check
      
      - name: Run Bandit (Python security linter)
        run: |
          bandit -r fed_vit_autorl/ -f json -o bandit-report.json || true
          bandit -r fed_vit_autorl/
      
      - name: Run Semgrep (SAST)
        run: |
          semgrep --config=auto fed_vit_autorl/ --json --output=semgrep-report.json || true
          semgrep --config=auto fed_vit_autorl/
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json
            semgrep-report.json

  codeql-analysis:
    name: CodeQL Security Analysis
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python
          queries: security-and-quality
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build container images
        run: |
          docker build --target prod -t fed-vit-autorl:security-scan .
          docker build --target edge -t fed-vit-autorl:edge-security-scan .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'fed-vit-autorl:security-scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Docker Scout (if available)
        run: |
          # Install Docker Scout
          curl -sSfL https://raw.githubusercontent.com/docker/scout-cli/main/install.sh | sh -s --
          
          # Scan images
          docker scout cves fed-vit-autorl:security-scan
          docker scout cves fed-vit-autorl:edge-security-scan

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better detection
      
      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pip-licenses licensecheck
          pip install -e .
      
      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=plain-vertical
          
          # Check for problematic licenses
          licensecheck --zero
      
      - name: Upload license report
        uses: actions/upload-artifact@v3
        with:
          name: license-report
          path: licenses.json
```

### 2. Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    commit-message:
      prefix: "chore"
      include: "scope"
    
    # Group updates for related packages
    groups:
      pytorch:
        patterns:
          - "torch*"
          - "torchvision"
      testing:
        patterns:
          - "pytest*"
          - "coverage"
      linting:
        patterns:
          - "black"
          - "ruff"
          - "mypy"
    
    # Security-only updates for production dependencies
    versioning-strategy: "increase-if-necessary"
    
    # Ignore specific packages if needed
    ignore:
      - dependency-name: "numpy"
        versions: ["1.24.0"]  # Known issue with this version

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"
    labels:
      - "docker"
      - "dependencies"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"
    labels:
      - "github-actions"
      - "dependencies"
```

### 3. Supply Chain Security

Create `.github/workflows/supply-chain.yml`:

```yaml
name: Supply Chain Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate Python SBOM
        uses: anchore/sbom-action@v0
        with:
          path: ./
          format: spdx-json
          output-file: sbom.spdx.json
      
      - name: Generate Container SBOM
        run: |
          docker build --target prod -t fed-vit-autorl:sbom .
          syft fed-vit-autorl:sbom -o spdx-json=container-sbom.spdx.json
      
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom-reports
          path: |
            sbom.spdx.json
            container-sbom.spdx.json

  provenance:
    name: Generate Provenance
    runs-on: ubuntu-latest
    permissions:
      actions: read
      id-token: write
      contents: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
        with:
          base64-subjects: |
            ${{ hashFiles('fed_vit_autorl/**/*.py') }}
          provenance-name: "fed-vit-autorl-provenance.intoto.jsonl"
      
      - name: Upload provenance
        uses: actions/upload-artifact@v3
        with:
          name: provenance
          path: fed-vit-autorl-provenance.intoto.jsonl

  verify-signatures:
    name: Verify Package Signatures
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install cosign
        uses: sigstore/cosign-installer@v3
      
      - name: Verify Python package signatures
        run: |
          # Verify critical dependencies
          pip download torch --no-deps
          # Note: Not all packages are signed yet, this is aspirational
          # cosign verify-blob torch-*.whl --certificate-identity=...
      
      - name: Verify container base images
        run: |
          # Verify official Python image
          cosign verify python:3.11-slim-bullseye \
            --certificate-identity-regexp=".*" \
            --certificate-oidc-issuer-regexp=".*"

  malware-scan:
    name: Malware Scanning
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: ClamAV scan
        run: |
          sudo apt-get update
          sudo apt-get install -y clamav clamav-daemon
          sudo freshclam
          
          # Scan repository
          clamscan -r . --exclude-dir=.git || true
          
          # Scan for specific patterns
          grep -r "eval\|exec\|import os" . --exclude-dir=.git || true
```

## Privacy and Compliance

### GDPR Compliance Check

```yaml
  gdpr-compliance:
    name: GDPR Compliance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check for personal data handling
        run: |
          # Look for potential personal data processing
          grep -r "email\|phone\|address\|personal" fed_vit_autorl/ || true
          
          # Verify privacy controls are in place
          python scripts/check_privacy_controls.py
      
      - name: Data flow analysis
        run: |
          # Analyze data flows for privacy compliance
          python scripts/analyze_data_flows.py --check-gdpr
```

### Differential Privacy Validation

```yaml
  privacy-validation:
    name: Privacy Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Validate privacy mechanisms
        run: |
          python -m pytest tests/privacy/ -v
          
          # Check privacy budget calculations
          python scripts/validate_privacy_budget.py
          
          # Verify noise injection is working
          python scripts/test_differential_privacy.py
```

## Incident Response Integration

### Security Incident Workflow

Create `.github/workflows/security-incident.yml`:

```yaml
name: Security Incident Response

on:
  repository_dispatch:
    types: [security-incident]
  issues:
    types: [labeled]

jobs:
  incident-response:
    name: Security Incident Response
    runs-on: ubuntu-latest
    if: contains(github.event.label.name, 'security-incident')
    steps:
      - name: Notify security team
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              "text": "🚨 Security Incident Reported",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Security Incident Reported*\n\nIssue: ${{ github.event.issue.html_url }}\nReporter: ${{ github.event.issue.user.login }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SECURITY_SLACK_WEBHOOK }}
      
      - name: Create incident tracking issue
        uses: actions/github-script@v6
        with:
          script: |
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `[SECURITY INCIDENT] ${new Date().toISOString().split('T')[0]}`,
              body: `
                ## Security Incident Response
                
                **Original Issue**: ${{ github.event.issue.html_url }}
                **Reporter**: ${{ github.event.issue.user.login }}
                **Date**: ${new Date().toISOString()}
                
                ## Response Checklist
                
                - [ ] Acknowledge incident
                - [ ] Assess severity
                - [ ] Contain threat
                - [ ] Investigate root cause
                - [ ] Implement fix
                - [ ] Verify resolution
                - [ ] Document lessons learned
                
                ## Communication
                
                - [ ] Notify stakeholders
                - [ ] Update status page
                - [ ] Prepare public communication (if needed)
              `,
              labels: ['security-incident-response', 'priority-high']
            });
```

## Automated Security Fixes

### Auto-merge Security Updates

Create `.github/workflows/auto-merge-security.yml`:

```yaml
name: Auto-merge Security Updates

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  auto-merge:
    name: Auto-merge Security Updates
    runs-on: ubuntu-latest
    if: |
      github.actor == 'dependabot[bot]' &&
      contains(github.event.pull_request.labels.*.name, 'security')
    steps:
      - name: Check if security update
        id: check
        run: |
          if [[ "${{ github.event.pull_request.title }}" == *"security"* ]]; then
            echo "is_security=true" >> $GITHUB_OUTPUT
          fi
      
      - name: Wait for status checks
        if: steps.check.outputs.is_security == 'true'
        uses: lewagon/wait-on-check-action@v1.3.1
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          check-name: 'ci'
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          wait-interval: 10
      
      - name: Auto-merge
        if: steps.check.outputs.is_security == 'true'
        uses: pascalgn/merge-action@v0.15.6
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          merge_method: squash
          merge_commit_message: "Automatically merged security update"
```

## Security Monitoring

### Continuous Security Monitoring

```yaml
  security-monitoring:
    name: Security Monitoring
    runs-on: ubuntu-latest
    steps:
      - name: Check for new vulnerabilities
        run: |
          # Check CVE databases for new vulnerabilities
          curl -s "https://services.nvd.nist.gov/rest/json/cves/1.0/" | \
            jq '.result.CVE_Items[] | select(.cve.affects.vendor.vendor_data[].product.product_data[].product_name | contains("python"))'
      
      - name: Monitor security advisories
        run: |
          # Check GitHub security advisories
          gh api graphql -f query='
            query {
              securityAdvisories(first: 10, orderBy: {field: PUBLISHED_AT, direction: DESC}) {
                nodes {
                  summary
                  severity
                  publishedAt
                  vulnerabilities(first: 5) {
                    nodes {
                      package {
                        name
                        ecosystem
                      }
                    }
                  }
                }
              }
            }'
```

## Reporting and Dashboards

### Security Metrics Collection

```python
# scripts/collect_security_metrics.py
import json
import requests
from datetime import datetime

def collect_security_metrics():
    metrics = {
        'timestamp': datetime.utcnow().isoformat(),
        'vulnerabilities': {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        },
        'dependencies': {
            'total': 0,
            'outdated': 0,
            'vulnerable': 0
        },
        'code_quality': {
            'security_hotspots': 0,
            'security_issues': 0
        }
    }
    
    # Collect from various sources
    # - GitHub Security API
    # - Snyk API
    # - SonarCloud API
    
    return metrics

if __name__ == "__main__":
    metrics = collect_security_metrics()
    print(json.dumps(metrics, indent=2))
```

### Security Dashboard

```yaml
  security-dashboard:
    name: Update Security Dashboard
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Collect security metrics
        run: |
          python scripts/collect_security_metrics.py > security-metrics.json
      
      - name: Update dashboard
        run: |
          # Update Grafana dashboard
          curl -X POST \
            -H "Authorization: Bearer ${{ secrets.GRAFANA_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d @security-dashboard.json \
            https://grafana.example.com/api/dashboards/db
```

## Best Practices

### 1. Security by Design
- Implement security controls early in development
- Use secure coding practices
- Regular security training for developers

### 2. Zero Trust Approach
- Verify all dependencies and inputs
- Implement least privilege access
- Continuous monitoring and validation

### 3. Automated Security
- Automate security scans and tests
- Implement security gates in CI/CD
- Auto-remediation where possible

### 4. Incident Preparedness
- Have incident response procedures
- Regular security drills
- Clear communication channels

### 5. Compliance Monitoring
- Regular compliance audits
- Automated compliance checking
- Documentation of security controls

## Troubleshooting

### Common Issues

1. **False Positives in Security Scans**
   - Review and whitelist known safe patterns
   - Use specific tool configurations
   - Implement manual review process

2. **Slow Security Scans**
   - Optimize scan scope and frequency
   - Use incremental scanning
   - Parallelize scan jobs

3. **Dependabot PR Conflicts**
   - Configure grouping for related dependencies
   - Set appropriate update intervals
   - Review and merge promptly

### Security Tool Configuration

```yaml
# .bandit
exclude_dirs:
  - tests/
  - docs/
skips:
  - B101  # Test assertions
  - B603  # Subprocess without shell check

# .semgrepignore
tests/
docs/
*.md
```