# Fed-ViT-AutoRL Observability Guide

This guide covers monitoring, logging, and observability for the Fed-ViT-AutoRL federated learning system.

## Overview

Fed-ViT-AutoRL includes comprehensive observability features:

- **Metrics Collection**: Prometheus-based metrics for all components
- **Alerting**: Intelligent alerts for system health and ML-specific issues  
- **Dashboards**: Grafana dashboards for visualization
- **Distributed Tracing**: Request tracing across federated components
- **Structured Logging**: Centralized log aggregation and analysis
- **Health Checks**: Continuous health monitoring

## Quick Start

### Starting the Monitoring Stack

```bash
# Start full monitoring stack
docker-compose up -d prometheus grafana

# Access dashboards
open http://localhost:3000  # Grafana (admin/fedvit123)
open http://localhost:9090  # Prometheus
```

### Key Dashboards

1. **Fed-ViT Overview**: System health and training progress
2. **Edge Devices**: Edge device monitoring and performance
3. **Infrastructure**: Docker containers, databases, networking
4. **Privacy Metrics**: Differential privacy and security monitoring

## Metrics Categories

### System Metrics

```prometheus
# Service availability
up{job="fed-server"}

# Resource utilization
process_cpu_seconds_total
process_memory_bytes
container_memory_usage_bytes

# Network metrics
http_requests_total
http_request_duration_seconds
```

### Federated Learning Metrics

```prometheus
# Training progress
fed_vit_autorl_training_rounds_total
fed_vit_autorl_training_loss
fed_vit_autorl_model_accuracy

# Client participation
fed_vit_autorl_active_clients
fed_vit_autorl_client_updates_total
fed_vit_autorl_client_dropped_total

# Aggregation metrics
fed_vit_autorl_aggregation_duration_seconds
fed_vit_autorl_model_size_bytes
fed_vit_autorl_convergence_rate
```

### Edge Device Metrics

```prometheus
# Device health
fed_vit_autorl_edge_device_info
fed_vit_autorl_edge_battery_level
fed_vit_autorl_edge_temperature

# Inference performance
fed_vit_autorl_inference_duration_seconds
fed_vit_autorl_inference_requests_total
fed_vit_autorl_model_load_time_seconds

# Resource constraints
fed_vit_autorl_edge_memory_usage_bytes
fed_vit_autorl_edge_cpu_usage_percent
```

### Privacy and Security Metrics

```prometheus
# Differential privacy
fed_vit_autorl_privacy_epsilon_remaining
fed_vit_autorl_privacy_delta_used
fed_vit_autorl_noise_scale

# Security events
fed_vit_autorl_unauthorized_requests_total
fed_vit_autorl_encryption_failures_total
fed_vit_autorl_certificate_expiry_days
```

## Custom Metric Implementation

### Adding New Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, Info

# Define metrics
TRAINING_ROUNDS = Counter(
    'fed_vit_autorl_training_rounds_total',
    'Total number of training rounds completed',
    ['client_id', 'model_version']
)

INFERENCE_LATENCY = Histogram(
    'fed_vit_autorl_inference_duration_seconds',
    'Time spent on inference',
    ['device_type', 'model_size'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

ACTIVE_CLIENTS = Gauge(
    'fed_vit_autorl_active_clients',
    'Number of currently active clients'
)

DEVICE_INFO = Info(
    'fed_vit_autorl_edge_device_info',
    'Edge device information',
    ['device_id', 'device_type', 'location']
)

# Usage in code
def train_round(client_id, model_version):
    with INFERENCE_LATENCY.labels(
        device_type="jetson", 
        model_size="base"
    ).time():
        # Training logic here
        pass
    
    TRAINING_ROUNDS.labels(
        client_id=client_id,
        model_version=model_version
    ).inc()
```

### Exposing Metrics Endpoint

```python
from prometheus_client import start_http_server, generate_latest
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(
        generate_latest(),
        mimetype='text/plain'
    )

# Start metrics server
start_http_server(8080)
```

## Alerting Rules

### Critical Alerts

```yaml
# Service down
- alert: FedServerDown
  expr: up{job="fed-server"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Federated learning server is down"

# Privacy budget exhausted
- alert: PrivacyBudgetExhausted
  expr: fed_vit_autorl_privacy_epsilon_remaining < 0.1
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Privacy budget nearly exhausted"
```

### Warning Alerts

```yaml
# Low client participation
- alert: LowClientParticipation
  expr: fed_vit_autorl_active_clients < 50
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Low client participation in federated learning"

# High inference latency
- alert: HighInferenceLatency
  expr: fed_vit_autorl_inference_duration_seconds{quantile="0.95"} > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High inference latency on edge devices"
```

## Log Management

### Structured Logging Configuration

```python
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'client_id'):
            log_entry['client_id'] = record.client_id
        if hasattr(record, 'round_id'):
            log_entry['round_id'] = record.round_id
            
        return json.dumps(log_entry)

# Configure logger
logger = logging.getLogger('fed_vit_autorl')
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info(
    "Training round completed",
    extra={
        'client_id': 'edge_001',
        'round_id': 42,
        'accuracy': 0.95,
        'loss': 0.23
    }
)
```

### Log Aggregation with ELK Stack

```yaml
# docker-compose.yml addition
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in federated learning
@tracer.start_as_current_span("federated_training_round")
def training_round(round_id):
    with tracer.start_as_current_span("client_selection") as span:
        span.set_attribute("round_id", round_id)
        clients = select_clients()
        span.set_attribute("selected_clients", len(clients))
    
    with tracer.start_as_current_span("model_aggregation"):
        aggregated_model = aggregate_models(client_updates)
    
    return aggregated_model
```

## Performance Monitoring

### Custom Performance Metrics

```python
import time
import psutil
from prometheus_client import Gauge, Histogram

# System performance
CPU_USAGE = Gauge('fed_vit_autorl_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('fed_vit_autorl_memory_usage_bytes', 'Memory usage in bytes')
GPU_UTILIZATION = Gauge('fed_vit_autorl_gpu_utilization_percent', 'GPU utilization')

# ML performance
MODEL_ACCURACY = Gauge('fed_vit_autorl_model_accuracy', 'Current model accuracy')
CONVERGENCE_RATE = Gauge('fed_vit_autorl_convergence_rate', 'Model convergence rate')
DATA_QUALITY = Gauge('fed_vit_autorl_data_quality_score', 'Data quality score')

def collect_system_metrics():
    """Collect and update system performance metrics."""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().used)
    
    # GPU metrics (if available)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        GPU_UTILIZATION.set(info.gpu)
    except:
        pass
```

## Dashboard Configuration

### Grafana Dashboard Import

```bash
# Import Fed-ViT overview dashboard
curl -X POST \
  http://admin:fedvit123@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboards/fed-vit-overview.json
```

### Custom Dashboard Panels

```json
{
  "title": "Training Progress",
  "type": "graph",
  "targets": [
    {
      "expr": "fed_vit_autorl_training_loss",
      "legendFormat": "Training Loss"
    },
    {
      "expr": "fed_vit_autorl_model_accuracy",
      "legendFormat": "Model Accuracy"
    }
  ],
  "yAxes": [
    {"label": "Loss", "min": 0},
    {"label": "Accuracy", "min": 0, "max": 1}
  ]
}
```

## Health Checks

### Application Health Endpoints

```python
from flask import Flask, jsonify
import psutil
import torch

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/health/detailed')
def detailed_health_check():
    """Detailed health check with component status."""
    status = {
        'status': 'healthy',
        'timestamp': time.time(),
        'components': {
            'model_loaded': check_model_loaded(),
            'gpu_available': torch.cuda.is_available(),
            'memory_ok': psutil.virtual_memory().percent < 90,
            'disk_ok': psutil.disk_usage('/').percent < 85,
            'database_connected': check_database_connection(),
        }
    }
    
    # Overall health based on components
    if not all(status['components'].values()):
        status['status'] = 'unhealthy'
    
    return jsonify(status)

@app.route('/metrics/health')
def health_metrics():
    """Health metrics for Prometheus."""
    # Return metrics in Prometheus format
    pass
```

## Troubleshooting Guide

### Common Monitoring Issues

1. **Missing Metrics**
   ```bash
   # Check metric endpoint
   curl http://localhost:8000/metrics
   
   # Verify Prometheus configuration
   docker exec prometheus promtool check config /etc/prometheus/prometheus.yml
   ```

2. **High Memory Usage**
   ```bash
   # Check container stats
   docker stats
   
   # Analyze memory usage patterns
   docker exec grafana cat /proc/meminfo
   ```

3. **Alert Not Firing**
   ```bash
   # Check alert rules
   docker exec prometheus promtool check rules /etc/prometheus/alert_rules.yml
   
   # Verify alert evaluation
   curl http://localhost:9090/api/v1/rules
   ```

### Performance Optimization

1. **Reduce Metric Cardinality**
   ```python
   # Bad: Too many labels
   REQUESTS = Counter('requests_total', 'Total requests', 
                     ['method', 'endpoint', 'user_id', 'session_id'])
   
   # Good: Essential labels only
   REQUESTS = Counter('requests_total', 'Total requests', 
                     ['method', 'status_code'])
   ```

2. **Optimize Scrape Intervals**
   ```yaml
   # High-frequency for critical metrics
   - job_name: 'fed-server'
     scrape_interval: 10s
   
   # Lower frequency for edge devices
   - job_name: 'fed-edge'
     scrape_interval: 30s
   ```

## Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Label Consistency**: Keep labels consistent across related metrics
3. **Alert Fatigue**: Avoid too many low-priority alerts
4. **Dashboard Organization**: Group related metrics logically
5. **Documentation**: Document custom metrics and alerts
6. **Testing**: Test alerts and dashboards regularly
7. **Retention**: Configure appropriate data retention policies
8. **Security**: Secure monitoring endpoints appropriately