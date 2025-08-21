"""Security management and hardening for Fed-ViT-AutoRL."""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import base64
import os

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    severity: SecurityLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class SecurityManager:
    """Comprehensive security manager for federated learning."""

    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize security manager.

        Args:
            master_key: Master encryption key (generated if not provided)
        """
        self.master_key = master_key or Fernet.generate_key()
        self.cipher = Fernet(self.master_key)

        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.failed_attempts: Dict[str, List[float]] = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        self.max_requests_per_minute = 60

        # Thread safety
        self._lock = threading.RLock()

        logger.info("Security manager initialized")

    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        try:
            encrypted = self.cipher.encrypt(data)
            self._log_security_event(
                "data_encryption",
                SecurityLevel.LOW,
                "Data encrypted successfully"
            )
            return encrypted
        except Exception as e:
            self._log_security_event(
                "encryption_error",
                SecurityLevel.HIGH,
                f"Encryption failed: {str(e)}"
            )
            raise

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt encrypted data.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            self._log_security_event(
                "data_decryption",
                SecurityLevel.LOW,
                "Data decrypted successfully"
            )
            return decrypted
        except Exception as e:
            self._log_security_event(
                "decryption_error",
                SecurityLevel.HIGH,
                f"Decryption failed: {str(e)}"
            )
            raise

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token.

        Args:
            length: Token length in bytes

        Returns:
            Secure token as hex string
        """
        token = secrets.token_hex(length)
        self._log_security_event(
            "token_generation",
            SecurityLevel.LOW,
            "Secure token generated"
        )
        return token

    def validate_input(self, input_data: Any, input_type: str) -> bool:
        """Validate input data for security threats.

        Args:
            input_data: Data to validate
            input_type: Type of input for validation rules

        Returns:
            True if input is safe, False otherwise
        """
        try:
            # Convert to string for analysis
            input_str = str(input_data)

            # Check for common injection patterns
            dangerous_patterns = [
                r'<script.*?>',
                r'javascript:',
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__',
                r'subprocess',
                r'os\.system',
                r'rm\s+-rf',
                r'DROP\s+TABLE',
                r'DELETE\s+FROM',
            ]

            import re
            for pattern in dangerous_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    self._log_security_event(
                        "malicious_input_detected",
                        SecurityLevel.CRITICAL,
                        f"Dangerous pattern detected in {input_type}: {pattern}"
                    )
                    return False

            # Check length limits
            max_lengths = {
                'username': 64,
                'password': 256,
                'email': 254,
                'message': 10000,
                'filename': 255,
                'default': 1000
            }

            max_length = max_lengths.get(input_type, max_lengths['default'])
            if len(input_str) > max_length:
                self._log_security_event(
                    "input_length_violation",
                    SecurityLevel.MEDIUM,
                    f"Input exceeds maximum length for {input_type}"
                )
                return False

            return True

        except Exception as e:
            self._log_security_event(
                "input_validation_error",
                SecurityLevel.HIGH,
                f"Input validation failed: {str(e)}"
            )
            return False

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits.

        Args:
            identifier: Unique identifier (IP, user ID, etc.)

        Returns:
            True if within limits, False if rate limited
        """
        with self._lock:
            current_time = time.time()

            # Clean old entries
            if identifier in self.rate_limits:
                self.rate_limits[identifier] = [
                    timestamp for timestamp in self.rate_limits[identifier]
                    if current_time - timestamp < 60  # Keep last minute
                ]
            else:
                self.rate_limits[identifier] = []

            # Check rate limit
            if len(self.rate_limits[identifier]) >= self.max_requests_per_minute:
                self._log_security_event(
                    "rate_limit_exceeded",
                    SecurityLevel.MEDIUM,
                    f"Rate limit exceeded for {identifier}"
                )
                return False

            # Add current request
            self.rate_limits[identifier].append(current_time)
            return True

    def authenticate_request(self, token: str, expected_hash: str) -> bool:
        """Authenticate API request using HMAC.

        Args:
            token: Request token
            expected_hash: Expected HMAC hash

        Returns:
            True if authenticated, False otherwise
        """
        try:
            # Generate expected hash
            computed_hash = hmac.new(
                self.master_key,
                token.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Compare hashes securely
            if hmac.compare_digest(computed_hash, expected_hash):
                self._log_security_event(
                    "authentication_success",
                    SecurityLevel.LOW,
                    "Request authenticated successfully"
                )
                return True
            else:
                self._log_security_event(
                    "authentication_failure",
                    SecurityLevel.HIGH,
                    "Authentication failed - invalid hash"
                )
                return False

        except Exception as e:
            self._log_security_event(
                "authentication_error",
                SecurityLevel.HIGH,
                f"Authentication error: {str(e)}"
            )
            return False

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path components
        sanitized = os.path.basename(filename)

        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')

        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip('. ')

        # Ensure reasonable length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:250] + ext

        # Prevent empty filename
        if not sanitized:
            sanitized = "sanitized_file"

        return sanitized

    def scan_for_secrets(self, content: str) -> List[str]:
        """Scan content for potential secrets.

        Args:
            content: Content to scan

        Returns:
            List of potential secret patterns found
        """
        import re

        secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'private_key': r'-----BEGIN PRIVATE KEY-----',
            'password_field': r'password\s*=\s*["\'][^"\']+["\']',
            'api_key': r'api[_-]?key\s*[=:]\s*["\'][a-zA-Z0-9_-]{20,}["\']',
            'token': r'token\s*[=:]\s*["\'][a-zA-Z0-9_-]{20,}["\']',
            'secret': r'secret\s*[=:]\s*["\'][a-zA-Z0-9_-]{10,}["\']',
        }

        found_secrets = []
        for secret_type, pattern in secret_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                found_secrets.append(f"{secret_type}: {match.group()}")
                self._log_security_event(
                    "secret_detected",
                    SecurityLevel.CRITICAL,
                    f"Potential {secret_type} found in content"
                )

        return found_secrets

    def _log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        description: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event.

        Args:
            event_type: Type of security event
            severity: Event severity level
            description: Event description
            additional_data: Additional event data
        """
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            description=description,
            additional_data=additional_data
        )

        with self._lock:
            self.security_events.append(event)

            # Keep only recent events (last 1000)
            if len(self.security_events) > 1000:
                self.security_events = self.security_events[-1000:]

        # Log to system logger
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)

        logger.log(log_level, f"[SECURITY] {event_type}: {description}")

    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report.

        Returns:
            Security report with recent events and metrics
        """
        with self._lock:
            recent_events = self.security_events[-50:]  # Last 50 events

            # Count events by severity
            severity_counts = {}
            for event in recent_events:
                severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1

            # Count events by type
            type_counts = {}
            for event in recent_events:
                type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1

            return {
                'total_events': len(self.security_events),
                'recent_events': len(recent_events),
                'severity_distribution': severity_counts,
                'event_type_distribution': type_counts,
                'rate_limited_identifiers': len(self.rate_limits),
                'events': [
                    {
                        'timestamp': event.timestamp,
                        'type': event.event_type,
                        'severity': event.severity.value,
                        'description': event.description
                    }
                    for event in recent_events
                ]
            }
