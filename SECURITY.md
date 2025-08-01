# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **Do NOT** create a public GitHub issue
2. Email us privately at security@terragon.ai
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Security Considerations

### Federated Learning Security
- All model updates use differential privacy
- Secure aggregation protocols prevent model inversion
- Communication channels are encrypted

### Edge Deployment Security
- Models are signed and verified before deployment
- Runtime integrity checks prevent tampering
- Secure boot and attestation support

### Data Privacy
- No raw sensor data leaves vehicles
- Local differential privacy applied at source
- Zero-knowledge aggregation protocols

## Security Response

- We will acknowledge receipt within 24 hours
- Initial assessment within 72 hours
- Security patches released ASAP for critical issues
- Public disclosure after fix is available

## Security Tools Used

- Bandit for static security analysis
- Safety for dependency vulnerability scanning
- Trivy for container security scanning
- CodeQL for semantic code analysis

## Responsible Disclosure

We appreciate responsible disclosure and will:
- Credit security researchers (if desired)
- Provide reasonable time for fix development
- Coordinate disclosure timing
- Consider bug bounty rewards for significant findings