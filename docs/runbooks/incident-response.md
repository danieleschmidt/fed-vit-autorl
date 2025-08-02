# Fed-ViT-AutoRL Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in the Fed-ViT-AutoRL federated learning system.

## Incident Classification

### Severity Levels

- **Critical (P0)**: Complete system outage, privacy breach, safety-critical failure
- **High (P1)**: Major functionality impaired, significant performance degradation
- **Medium (P2)**: Minor functionality impaired, some performance impact
- **Low (P3)**: Minor issues, no immediate impact

### Response Times

- **P0**: Immediate (< 15 minutes)
- **P1**: Within 1 hour
- **P2**: Within 4 hours
- **P3**: Within 24 hours

## Common Incidents

### 1. Federated Server Down

**Symptoms:**
- Alert: `ServiceDown` for fed-server
- Clients cannot connect to aggregation server
- Training rounds stopped

**Investigation Steps:**
```bash
# Check container status
docker ps | grep fed-server
docker logs fed-server --tail 100

# Check resource usage
docker stats fed-server

# Check system resources
free -h
df -h
```

**Resolution:**
```bash
# Restart service
docker-compose restart fed-server

# If persistent, check configuration
docker exec fed-server cat /app/config.yaml

# Check database connectivity
docker exec fed-server python -c "import fed_vit_autorl.db; fed_vit_autorl.db.test_connection()"
```

**Escalation:** If restart doesn't resolve within 15 minutes

### 2. High Training Loss

**Symptoms:**
- Alert: `HighTrainingLoss` 
- Model accuracy decreasing
- Training convergence stalled

**Investigation Steps:**
```bash
# Check recent training metrics
curl -s "http://localhost:9090/api/v1/query?query=fed_vit_autorl_training_loss" | jq

# Review client data quality
curl -s "http://localhost:8000/api/clients/data-quality" | jq

# Check for data drift
curl -s "http://localhost:8000/api/monitoring/data-drift" | jq
```

**Resolution:**
```bash
# Check client participation
curl -s "http://localhost:8000/api/clients/status" | jq

# Restart training with validated clients only
curl -X POST "http://localhost:8000/api/training/restart" \
  -H "Content-Type: application/json" \
  -d '{"client_selection": "validated_only"}'

# If data drift detected, update the global model
curl -X POST "http://localhost:8000/api/model/retrain" \
  -d '{"strategy": "full_retrain"}'
```

**Escalation:** If loss doesn't improve after 2 training rounds

### 3. Edge Device Offline

**Symptoms:**
- Alert: `EdgeDeviceOffline`
- Device not responding to health checks
- Missing inference metrics

**Investigation Steps:**
```bash
# Check device connectivity
ping <edge-device-ip>

# Check device logs (if accessible)
ssh edge-device "journalctl -u fed-vit-client --since '1 hour ago'"

# Check network connectivity from server
docker exec fed-server ping <edge-device-ip>
```

**Resolution:**
```bash
# Remove device from active client list
curl -X DELETE "http://localhost:8000/api/clients/<device-id>"

# If device comes back online, re-register
curl -X POST "http://localhost:8000/api/clients/register" \
  -H "Content-Type: application/json" \
  -d '{"device_id": "<device-id>", "capabilities": {...}}'
```

**Escalation:** If multiple devices offline (>10%)

### 4. Privacy Budget Exhausted

**Symptoms:**
- Alert: `PrivacyBudgetExhausted`
- Privacy epsilon below threshold
- Training may need to stop

**Investigation Steps:**
```bash
# Check current privacy budget
curl -s "http://localhost:8000/api/privacy/budget" | jq

# Review privacy spending history
curl -s "http://localhost:8000/api/privacy/history?hours=24" | jq

# Check if budget calculation is correct
docker exec fed-server python -c "
from fed_vit_autorl.privacy import PrivacyAccountant
accountant = PrivacyAccountant()
print(f'Remaining epsilon: {accountant.get_remaining_epsilon()}')
"
```

**Resolution:**
```bash
# Stop training immediately
curl -X POST "http://localhost:8000/api/training/stop"

# Review and potentially reset privacy budget (with approval)
# This requires careful consideration and stakeholder approval
curl -X POST "http://localhost:8000/api/privacy/reset-budget" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"new_epsilon": 1.0, "justification": "..."}'
```

**Escalation:** Immediate escalation to privacy officer

### 5. High Inference Latency

**Symptoms:**
- Alert: `EdgeHighInferenceLatency`
- 95th percentile latency > 100ms
- User complaints about slow response

**Investigation Steps:**
```bash
# Check current latency metrics
curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,fed_vit_autorl_inference_duration_seconds_bucket)" | jq

# Check edge device resources
for device in edge-1 edge-2; do
  echo "=== $device ==="
  curl -s "http://localhost:808${device#edge-}/health/detailed" | jq
done

# Check model size and complexity
curl -s "http://localhost:8000/api/model/info" | jq '.size, .complexity'
```

**Resolution:**
```bash
# Enable model optimization for edge devices
curl -X POST "http://localhost:8000/api/edge/optimize" \
  -d '{"strategy": "quantization", "target_latency": 50}'

# If needed, switch to lighter model variant
curl -X POST "http://localhost:8000/api/model/switch" \
  -d '{"variant": "edge_optimized"}'

# Scale up edge devices if needed
docker-compose scale fed-edge=5
```

**Escalation:** If latency doesn't improve within 30 minutes

### 6. Network Partition

**Symptoms:**
- Alert: `NetworkPartition`
- Large number of clients disconnected
- Asymmetric communication issues

**Investigation Steps:**
```bash
# Check client connectivity
curl -s "http://localhost:8000/api/clients/connectivity" | jq

# Test network paths
docker exec fed-server traceroute <client-subnet>

# Check for network policy changes
kubectl get networkpolicies  # if using k8s
```

**Resolution:**
```bash
# Switch to asynchronous aggregation mode
curl -X POST "http://localhost:8000/api/aggregation/mode" \
  -d '{"mode": "asynchronous", "staleness_tolerance": 5}'

# Reduce required client participation
curl -X POST "http://localhost:8000/api/training/config" \
  -d '{"min_clients": 10, "wait_timeout": 300}'

# Enable backup communication channels
curl -X POST "http://localhost:8000/api/communication/enable-backup"
```

**Escalation:** If partition persists > 1 hour

## Emergency Procedures

### Complete System Shutdown

```bash
# Graceful shutdown
curl -X POST "http://localhost:8000/api/system/shutdown" \
  -d '{"mode": "graceful", "save_state": true}'

# Force shutdown if needed
docker-compose down

# Save current state
docker run --rm -v fed-vit-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/emergency-backup-$(date +%Y%m%d-%H%M%S).tar.gz /data
```

### Privacy Breach Response

```bash
# Immediate actions
1. Stop all training immediately
2. Isolate affected components
3. Notify privacy officer and legal team
4. Begin forensic data collection

# Technical steps
curl -X POST "http://localhost:8000/api/emergency/privacy-breach" \
  -H "Authorization: Bearer <emergency-token>" \
  -d '{
    "incident_id": "PB-$(date +%Y%m%d-%H%M%S)",
    "affected_clients": ["client1", "client2"],
    "breach_type": "data_exposure",
    "immediate_actions": ["stop_training", "isolate_data"]
  }'
```

## Recovery Procedures

### Restore from Backup

```bash
# Stop services
docker-compose down

# Restore data
docker run --rm -v fed-vit-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/backup-file.tar.gz -C /data

# Restart services
docker-compose up -d

# Verify integrity
curl -s "http://localhost:8000/api/system/health" | jq
```

### Model Rollback

```bash
# List available model versions
curl -s "http://localhost:8000/api/model/versions" | jq

# Rollback to previous version
curl -X POST "http://localhost:8000/api/model/rollback" \
  -d '{"version": "v1.2.3", "force": true}'

# Verify rollback
curl -s "http://localhost:8000/api/model/current" | jq
```

## Post-Incident

### Incident Documentation

1. **Timeline**: Record all actions taken with timestamps
2. **Root Cause**: Document the underlying cause
3. **Impact**: Assess business and technical impact
4. **Lessons Learned**: Identify improvements
5. **Action Items**: Create follow-up tasks

### Template

```markdown
# Incident Report: [INC-YYYY-MMDD-XXX]

## Summary
- **Date**: 
- **Duration**: 
- **Severity**: 
- **Root Cause**: 

## Timeline
- HH:MM - Issue detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Resolution implemented
- HH:MM - Service restored

## Impact
- **Clients Affected**: 
- **Data Loss**: 
- **Privacy Impact**: 
- **Business Impact**: 

## Root Cause Analysis
[Detailed analysis]

## Resolution
[Steps taken to resolve]

## Prevention
[Measures to prevent recurrence]

## Action Items
- [ ] Update monitoring alerts
- [ ] Improve documentation
- [ ] Add automated recovery
```

## Contact Information

### Escalation Matrix

| Role | Contact | Phone | Email |
|------|---------|-------|-------|
| On-Call Engineer | [Name] | [Phone] | [Email] |
| Team Lead | [Name] | [Phone] | [Email] |
| Privacy Officer | [Name] | [Phone] | [Email] |
| Security Team | [Name] | [Phone] | [Email] |

### Emergency Contacts

- **24/7 Hotline**: [Number]
- **Incident Commander**: [Contact]
- **Executive Escalation**: [Contact]

## Tools and Resources

### Monitoring URLs
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Fed-ViT API: http://localhost:8000

### Useful Commands
```bash
# Quick health check
make health-check

# Emergency backup
make emergency-backup

# System diagnostics
make diagnostics

# View all logs
make logs
```

### Documentation Links
- [Architecture Overview](../ARCHITECTURE.md)
- [Deployment Guide](../deployment/build-guide.md)
- [API Documentation](../api/README.md)
- [Security Procedures](../security/README.md)