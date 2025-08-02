# Monitoring Workflow Setup Guide

This guide covers setting up automated monitoring workflows for Fed-ViT-AutoRL, including performance tracking, health monitoring, and alerting integration.

## Overview

Monitoring workflows provide:
- Automated performance benchmarking
- Health check validation
- Metrics collection and reporting
- Integration with monitoring systems
- Alert management and escalation
- Performance regression detection

## Prerequisites

### Required Integrations

```bash
# Monitoring Services
PROMETHEUS_URL=https://prometheus.example.com
GRAFANA_API_KEY=your-grafana-api-key
DATADOG_API_KEY=your-datadog-key (optional)

# Notification Channels
SLACK_WEBHOOK_URL=https://hooks.slack.com/xxxxx
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key
EMAIL_SMTP_PASSWORD=your-smtp-password

# Cloud Monitoring
AWS_CLOUDWATCH_ROLE=arn:aws:iam::account:role/cloudwatch
GCP_MONITORING_SA=monitoring@project.iam.gserviceaccount.com
```

### Monitoring Stack Setup

Ensure your monitoring infrastructure is configured:
1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Visualization and dashboards
3. **AlertManager**: Alert routing and management
4. **Node Exporter**: System metrics collection

## Core Monitoring Workflows

### 1. Performance Monitoring

Create `.github/workflows/performance.yml`:

```yaml
name: Performance Monitoring

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Run performance tests daily
    - cron: '0 4 * * *'

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
        test-type: ['unit', 'integration', 'e2e']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,simulation]"
          pip install pytest-benchmark pytest-xdist
      
      - name: Run performance benchmarks
        run: |
          pytest tests/benchmarks/ \
            --benchmark-only \
            --benchmark-json=benchmark-${{ matrix.python-version }}-${{ matrix.test-type }}.json \
            --benchmark-save=baseline \
            --benchmark-compare-fail=min:10% \
            -v
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-*.json
      
      - name: Compare with baseline
        if: github.event_name == 'pull_request'
        run: |
          pytest tests/benchmarks/ \
            --benchmark-only \
            --benchmark-compare=baseline \
            --benchmark-compare-fail=min:10%

  memory-profiling:
    name: Memory Profiling
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
          pip install memory-profiler py-spy
      
      - name: Profile memory usage
        run: |
          # Profile federated training
          mprof run python scripts/profile_federated_training.py
          mprof plot -o memory-profile.png
          
          # Generate memory report
          python scripts/generate_memory_report.py > memory-report.txt
      
      - name: Upload profiling results
        uses: actions/upload-artifact@v3
        with:
          name: memory-profiles
          path: |
            memory-profile.png
            memory-report.txt

  load-testing:
    name: Load Testing
    runs-on: ubuntu-latest
    services:
      fed-server:
        image: fed-vit-autorl:latest
        ports:
          - 8000:8000
        env:
          FED_VIT_AUTORL_ENV: testing
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install k6
        run: |
          sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
      
      - name: Run load tests
        run: |
          # Wait for service to be ready
          timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
          
          # Run load test scenarios
          k6 run tests/load/federated_training_load.js \
            --out json=load-test-results.json
          
          k6 run tests/load/inference_load.js \
            --out json=inference-load-results.json
      
      - name: Analyze load test results
        run: |
          python scripts/analyze_load_test_results.py \
            load-test-results.json \
            inference-load-results.json
      
      - name: Upload load test results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: |
            load-test-results.json
            inference-load-results.json
            load-test-analysis.html
```

### 2. Health Check Automation

Create `.github/workflows/health-checks.yml`:

```yaml
name: Health Checks

on:
  schedule:
    # Run every 15 minutes
    - cron: '*/15 * * * *'
  workflow_dispatch:

jobs:
  endpoint-health:
    name: Endpoint Health Checks
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Check service endpoints
        run: |
          # Define endpoints for each environment
          case "${{ matrix.environment }}" in
            staging)
              BASE_URL="https://staging.fed-vit-autorl.example.com"
              ;;
            production)
              BASE_URL="https://fed-vit-autorl.example.com"
              ;;
          esac
          
          # Health check endpoints
          endpoints=(
            "/health"
            "/health/detailed"
            "/metrics"
            "/api/status"
            "/api/clients/count"
          )
          
          failed_checks=()
          
          for endpoint in "${endpoints[@]}"; do
            echo "Checking ${BASE_URL}${endpoint}"
            
            if ! curl -f -s --max-time 30 "${BASE_URL}${endpoint}" > /dev/null; then
              failed_checks+=("${endpoint}")
              echo "❌ FAILED: ${endpoint}"
            else
              echo "✅ OK: ${endpoint}"
            fi
          done
          
          if [ ${#failed_checks[@]} -gt 0 ]; then
            echo "Failed health checks: ${failed_checks[*]}"
            exit 1
          fi
      
      - name: Notify on failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          text: |
            🚨 Health Check Failed
            Environment: ${{ matrix.environment }}
            Failed endpoints detected
            Time: $(date -u)

  deep-health-check:
    name: Deep Health Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install requests psutil
      
      - name: Run comprehensive health check
        run: |
          python scripts/comprehensive_health_check.py \
            --environment production \
            --output health-report.json
      
      - name: Validate system metrics
        run: |
          # Check if metrics are within acceptable ranges
          python scripts/validate_system_metrics.py \
            --metrics-url "${{ secrets.PROMETHEUS_URL }}" \
            --alert-thresholds config/alert-thresholds.yaml
      
      - name: Upload health report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: health-reports
          path: health-report.json
```

### 3. Metrics Collection and Reporting

Create `.github/workflows/metrics-collection.yml`:

```yaml
name: Metrics Collection

on:
  schedule:
    # Collect metrics every hour
    - cron: '0 * * * *'
  workflow_dispatch:
    inputs:
      metric_type:
        description: 'Type of metrics to collect'
        required: true
        default: 'all'
        type: choice
        options:
          - all
          - performance
          - business
          - security

jobs:
  collect-metrics:
    name: Collect System Metrics
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install prometheus-client requests pandas
      
      - name: Collect performance metrics
        if: github.event.inputs.metric_type == 'all' || github.event.inputs.metric_type == 'performance'
        run: |
          python scripts/collect_performance_metrics.py \
            --prometheus-url "${{ secrets.PROMETHEUS_URL }}" \
            --output performance-metrics.json \
            --time-range 1h
      
      - name: Collect business metrics
        if: github.event.inputs.metric_type == 'all' || github.event.inputs.metric_type == 'business'
        run: |
          python scripts/collect_business_metrics.py \
            --api-base-url "${{ secrets.FED_VIT_API_URL }}" \
            --output business-metrics.json
      
      - name: Collect security metrics
        if: github.event.inputs.metric_type == 'all' || github.event.inputs.metric_type == 'security'
        run: |
          python scripts/collect_security_metrics.py \
            --output security-metrics.json
      
      - name: Generate metrics report
        run: |
          python scripts/generate_metrics_report.py \
            --performance performance-metrics.json \
            --business business-metrics.json \
            --security security-metrics.json \
            --output metrics-report.html
      
      - name: Upload to monitoring storage
        run: |
          # Upload to time-series database
          python scripts/upload_metrics.py \
            --influxdb-url "${{ secrets.INFLUXDB_URL }}" \
            --influxdb-token "${{ secrets.INFLUXDB_TOKEN }}" \
            --metrics performance-metrics.json business-metrics.json security-metrics.json
      
      - name: Update Grafana annotations
        run: |
          # Add annotation for metrics collection
          curl -X POST \
            -H "Authorization: Bearer ${{ secrets.GRAFANA_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{
              "time": '$(date +%s000)',
              "text": "Automated metrics collection completed",
              "tags": ["automation", "metrics"]
            }' \
            "${{ secrets.GRAFANA_URL }}/api/annotations"

  federated-learning-metrics:
    name: Collect FL-specific Metrics
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Collect federated learning metrics
        run: |
          # Training progress metrics
          curl -s "${{ secrets.FED_VIT_API_URL }}/api/training/progress" | \
            jq '{
              current_round: .current_round,
              total_rounds: .total_rounds,
              active_clients: .active_clients,
              global_accuracy: .global_accuracy,
              convergence_rate: .convergence_rate
            }' > fl-progress.json
          
          # Client participation metrics
          curl -s "${{ secrets.FED_VIT_API_URL }}/api/clients/participation" | \
            jq '{
              total_clients: .total_clients,
              active_clients: .active_clients,
              participation_rate: .participation_rate,
              avg_data_quality: .avg_data_quality
            }' > fl-participation.json
          
          # Privacy metrics
          curl -s "${{ secrets.FED_VIT_API_URL }}/api/privacy/status" | \
            jq '{
              epsilon_remaining: .epsilon_remaining,
              delta_used: .delta_used,
              privacy_budget_utilization: .privacy_budget_utilization
            }' > fl-privacy.json
      
      - name: Analyze training convergence
        run: |
          python scripts/analyze_convergence.py \
            --progress-file fl-progress.json \
            --output convergence-analysis.json
      
      - name: Upload FL metrics
        uses: actions/upload-artifact@v3
        with:
          name: fl-metrics
          path: |
            fl-progress.json
            fl-participation.json
            fl-privacy.json
            convergence-analysis.json
```

### 4. Alert Integration

Create `.github/workflows/alert-management.yml`:

```yaml
name: Alert Management

on:
  repository_dispatch:
    types: [alert-triggered, alert-resolved]
  issues:
    types: [labeled]

jobs:
  process-alert:
    name: Process Monitoring Alert
    runs-on: ubuntu-latest
    if: github.event.action == 'alert-triggered'
    steps:
      - uses: actions/checkout@v4
      
      - name: Parse alert payload
        id: parse_alert
        run: |
          echo "Alert received: ${{ github.event.client_payload.alert_name }}"
          echo "Severity: ${{ github.event.client_payload.severity }}"
          echo "Description: ${{ github.event.client_payload.description }}"
          
          # Set outputs for subsequent steps
          echo "alert_name=${{ github.event.client_payload.alert_name }}" >> $GITHUB_OUTPUT
          echo "severity=${{ github.event.client_payload.severity }}" >> $GITHUB_OUTPUT
      
      - name: Create incident issue
        if: steps.parse_alert.outputs.severity == 'critical' || steps.parse_alert.outputs.severity == 'high'
        uses: actions/github-script@v6
        with:
          script: |
            const alertName = '${{ steps.parse_alert.outputs.alert_name }}';
            const severity = '${{ steps.parse_alert.outputs.severity }}';
            const description = '${{ github.event.client_payload.description }}';
            
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `🚨 ${severity.toUpperCase()}: ${alertName}`,
              body: `
                ## Alert Details
                
                **Alert Name**: ${alertName}
                **Severity**: ${severity}
                **Description**: ${description}
                **Triggered**: ${new Date().toISOString()}
                
                ## Response Actions
                
                - [ ] Acknowledge alert
                - [ ] Investigate issue
                - [ ] Implement fix
                - [ ] Verify resolution
                - [ ] Update documentation
                
                ## Monitoring Links
                
                - [Grafana Dashboard](${{ secrets.GRAFANA_URL }}/dashboard)
                - [Prometheus Alerts](${{ secrets.PROMETHEUS_URL }}/alerts)
                - [Logs](${{ secrets.LOGS_URL }})
              `,
              labels: ['alert', `severity-${severity}`, 'monitoring'],
              assignees: ['on-call-engineer']
            });
            
            console.log(`Created issue #${issue.data.number}`);
      
      - name: Notify on-call team
        if: steps.parse_alert.outputs.severity == 'critical'
        run: |
          # Send PagerDuty alert for critical issues
          curl -X POST \
            -H "Content-Type: application/json" \
            -H "Authorization: Token token=${{ secrets.PAGERDUTY_INTEGRATION_KEY }}" \
            -d '{
              "event_action": "trigger",
              "routing_key": "${{ secrets.PAGERDUTY_INTEGRATION_KEY }}",
              "payload": {
                "summary": "${{ steps.parse_alert.outputs.alert_name }}",
                "severity": "critical",
                "source": "Fed-ViT-AutoRL Monitoring",
                "component": "federated-learning-system",
                "group": "infrastructure",
                "class": "monitoring"
              }
            }' \
            https://events.pagerduty.com/v2/enqueue

  update-dashboards:
    name: Update Monitoring Dashboards
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Update Grafana dashboards
        run: |
          # Update all dashboards in monitoring/grafana-dashboards/
          for dashboard in monitoring/grafana-dashboards/*.json; do
            echo "Updating dashboard: $dashboard"
            
            curl -X POST \
              -H "Authorization: Bearer ${{ secrets.GRAFANA_API_KEY }}" \
              -H "Content-Type: application/json" \
              -d @"$dashboard" \
              "${{ secrets.GRAFANA_URL }}/api/dashboards/db"
          done
      
      - name: Update Prometheus rules
        run: |
          # Validate and reload Prometheus configuration
          curl -X POST \
            -H "Authorization: Bearer ${{ secrets.PROMETHEUS_API_TOKEN }}" \
            "${{ secrets.PROMETHEUS_URL }}/-/reload"
```

## Performance Regression Detection

### Automated Performance Analysis

```yaml
  performance-regression:
    name: Performance Regression Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need history for comparison
      
      - name: Run performance comparison
        run: |
          # Compare current performance with baseline
          python scripts/performance_regression_detector.py \
            --baseline-branch main \
            --current-branch ${{ github.head_ref || github.ref_name }} \
            --metrics-config config/performance-metrics.yaml \
            --threshold-config config/regression-thresholds.yaml \
            --output regression-report.json
      
      - name: Check for regressions
        run: |
          # Analyze regression report
          if python scripts/check_regressions.py regression-report.json; then
            echo "No performance regressions detected"
          else
            echo "Performance regression detected!"
            cat regression-report.json
            exit 1
          fi
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('regression-report.json', 'utf8'));
            
            let comment = '## Performance Analysis\n\n';
            
            if (report.regressions.length === 0) {
              comment += '✅ No performance regressions detected.\n\n';
            } else {
              comment += '⚠️ Performance regressions detected:\n\n';
              report.regressions.forEach(regression => {
                comment += `- **${regression.metric}**: ${regression.change}% change (threshold: ${regression.threshold}%)\n`;
              });
              comment += '\n';
            }
            
            comment += '### Performance Summary\n\n';
            comment += `| Metric | Current | Baseline | Change |\n`;
            comment += `|--------|---------|----------|--------|\n`;
            
            report.metrics.forEach(metric => {
              comment += `| ${metric.name} | ${metric.current} | ${metric.baseline} | ${metric.change}% |\n`;
            });
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

## Custom Monitoring Scripts

### Comprehensive Health Check Script

```python
# scripts/comprehensive_health_check.py
import json
import requests
import psutil
import time
from typing import Dict, Any

def check_api_endpoints(base_url: str) -> Dict[str, Any]:
    """Check API endpoint health."""
    endpoints = [
        '/health',
        '/health/detailed',
        '/api/status',
        '/metrics'
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            results[endpoint] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            results[endpoint] = {
                'status': 'error',
                'error': str(e)
            }
    
    return results

def check_system_resources() -> Dict[str, Any]:
    """Check system resource utilization."""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }

def check_federated_learning_metrics(api_url: str) -> Dict[str, Any]:
    """Check federated learning specific metrics."""
    try:
        response = requests.get(f"{api_url}/api/training/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {'status': 'error', 'message': 'Failed to fetch FL metrics'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def main():
    health_report = {
        'timestamp': time.time(),
        'api_health': check_api_endpoints('http://localhost:8000'),
        'system_resources': check_system_resources(),
        'federated_learning': check_federated_learning_metrics('http://localhost:8000')
    }
    
    # Determine overall health
    api_healthy = all(
        result.get('status') == 'healthy' 
        for result in health_report['api_health'].values()
    )
    
    resources_healthy = (
        health_report['system_resources']['cpu_percent'] < 80 and
        health_report['system_resources']['memory_percent'] < 85 and
        health_report['system_resources']['disk_percent'] < 90
    )
    
    health_report['overall_status'] = 'healthy' if api_healthy and resources_healthy else 'unhealthy'
    
    print(json.dumps(health_report, indent=2))

if __name__ == '__main__':
    main()
```

## Integration with External Monitoring

### DataDog Integration

```yaml
  datadog-integration:
    name: DataDog Metrics Upload
    runs-on: ubuntu-latest
    steps:
      - name: Send custom metrics to DataDog
        run: |
          # Send federated learning metrics
          curl -X POST \
            -H "Content-Type: application/json" \
            -H "DD-API-KEY: ${{ secrets.DATADOG_API_KEY }}" \
            -d '{
              "series": [
                {
                  "metric": "fed_vit_autorl.active_clients",
                  "points": [['$(date +%s)', 150]],
                  "tags": ["environment:production"]
                },
                {
                  "metric": "fed_vit_autorl.training_accuracy",
                  "points": [['$(date +%s)', 0.95]],
                  "tags": ["environment:production"]
                }
              ]
            }' \
            https://api.datadoghq.com/api/v1/series
```

### AWS CloudWatch Integration

```yaml
  cloudwatch-integration:
    name: CloudWatch Metrics
    runs-on: ubuntu-latest
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: ${{ secrets.AWS_CLOUDWATCH_ROLE }}
          aws-region: us-west-2
      
      - name: Send metrics to CloudWatch
        run: |
          # Send custom metrics
          aws cloudwatch put-metric-data \
            --namespace "FedViTAutoRL" \
            --metric-data \
              MetricName=ActiveClients,Value=150,Unit=Count \
              MetricName=TrainingAccuracy,Value=0.95,Unit=Percent \
              MetricName=InferenceLatency,Value=45,Unit=Milliseconds
```

## Best Practices

1. **Monitoring Strategy**
   - Monitor what matters to users
   - Use SLI/SLO framework
   - Implement monitoring at multiple layers

2. **Alert Design**
   - Minimize false positives
   - Include actionable information
   - Implement alert fatigue prevention

3. **Performance Tracking**
   - Establish baselines
   - Track trends over time
   - Automate regression detection

4. **Documentation**
   - Document all monitoring configurations
   - Maintain runbooks for alerts
   - Keep dashboard documentation updated

5. **Continuous Improvement**
   - Regular monitoring review
   - Update thresholds based on data
   - Incorporate feedback from incidents