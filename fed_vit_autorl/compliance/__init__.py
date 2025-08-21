"""Global compliance and regulatory framework for Fed-ViT-AutoRL."""

import time
import logging
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    SOX = "sox"    # Sarbanes-Oxley Act (US)
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework


class DataCategory(Enum):
    """Data categories for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "spi"
    BIOMETRIC = "biometric"
    LOCATION = "location"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    ANONYMOUS = "anonymous"


@dataclass
class ComplianceRule:
    """Individual compliance rule."""
    rule_id: str
    framework: ComplianceFramework
    description: str
    data_categories: Set[DataCategory]
    required: bool = True
    validation_function: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    description: str
    severity: str  # "low", "medium", "high", "critical"
    data_affected: List[str]
    timestamp: float
    remediated: bool = False
    remediation_notes: Optional[str] = None


class GlobalComplianceManager:
    """Comprehensive global compliance management system."""

    def __init__(self, enabled_frameworks: Optional[List[ComplianceFramework]] = None):
        """Initialize compliance manager.

        Args:
            enabled_frameworks: List of compliance frameworks to enforce
        """
        self.enabled_frameworks = enabled_frameworks or [
            ComplianceFramework.GDPR,
            ComplianceFramework.CCPA,
            ComplianceFramework.ISO27001
        ]

        self.rules: Dict[str, ComplianceRule] = {}
        self.violations: List[ComplianceViolation] = []
        self.audit_log: List[Dict[str, Any]] = []

        # Data processing records for GDPR/CCPA
        self.data_processing_records: List[Dict[str, Any]] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_retention_policies: Dict[DataCategory, int] = {}  # retention days

        self._initialize_compliance_rules()
        self._setup_data_retention_policies()

        logger.info(f"Initialized compliance manager with frameworks: {[f.value for f in self.enabled_frameworks]}")

    def _initialize_compliance_rules(self):
        """Initialize compliance rules for enabled frameworks."""

        # GDPR Rules
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            self._add_gdpr_rules()

        # CCPA Rules
        if ComplianceFramework.CCPA in self.enabled_frameworks:
            self._add_ccpa_rules()

        # ISO 27001 Rules
        if ComplianceFramework.ISO27001 in self.enabled_frameworks:
            self._add_iso27001_rules()

        # PDPA Rules
        if ComplianceFramework.PDPA in self.enabled_frameworks:
            self._add_pdpa_rules()

    def _add_gdpr_rules(self):
        """Add GDPR compliance rules."""
        rules = [
            ComplianceRule(
                rule_id="GDPR-001",
                framework=ComplianceFramework.GDPR,
                description="Lawful basis required for personal data processing",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL},
                validation_function="validate_lawful_basis"
            ),
            ComplianceRule(
                rule_id="GDPR-002",
                framework=ComplianceFramework.GDPR,
                description="Explicit consent required for sensitive personal data",
                data_categories={DataCategory.SENSITIVE_PERSONAL, DataCategory.BIOMETRIC},
                validation_function="validate_explicit_consent"
            ),
            ComplianceRule(
                rule_id="GDPR-003",
                framework=ComplianceFramework.GDPR,
                description="Data subject rights must be implementable",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_data_subject_rights"
            ),
            ComplianceRule(
                rule_id="GDPR-004",
                framework=ComplianceFramework.GDPR,
                description="Data minimization principle compliance",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE, DataCategory.SENSITIVE_PERSONAL},
                validation_function="validate_data_minimization"
            ),
            ComplianceRule(
                rule_id="GDPR-005",
                framework=ComplianceFramework.GDPR,
                description="Cross-border transfer protections",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_cross_border_transfer"
            )
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    def _add_ccpa_rules(self):
        """Add CCPA compliance rules."""
        rules = [
            ComplianceRule(
                rule_id="CCPA-001",
                framework=ComplianceFramework.CCPA,
                description="Consumer right to know about personal information collection",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_transparency_disclosure"
            ),
            ComplianceRule(
                rule_id="CCPA-002",
                framework=ComplianceFramework.CCPA,
                description="Consumer right to delete personal information",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_deletion_capability"
            ),
            ComplianceRule(
                rule_id="CCPA-003",
                framework=ComplianceFramework.CCPA,
                description="Consumer right to opt-out of sale",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_opt_out_mechanism"
            )
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    def _add_iso27001_rules(self):
        """Add ISO 27001 compliance rules."""
        rules = [
            ComplianceRule(
                rule_id="ISO27001-001",
                framework=ComplianceFramework.ISO27001,
                description="Information security policy implementation",
                data_categories={DataCategory.TECHNICAL},
                validation_function="validate_security_policy"
            ),
            ComplianceRule(
                rule_id="ISO27001-002",
                framework=ComplianceFramework.ISO27001,
                description="Access control measures",
                data_categories={DataCategory.TECHNICAL, DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_access_controls"
            ),
            ComplianceRule(
                rule_id="ISO27001-003",
                framework=ComplianceFramework.ISO27001,
                description="Incident response procedures",
                data_categories={DataCategory.TECHNICAL},
                validation_function="validate_incident_response"
            )
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    def _add_pdpa_rules(self):
        """Add PDPA compliance rules."""
        rules = [
            ComplianceRule(
                rule_id="PDPA-001",
                framework=ComplianceFramework.PDPA,
                description="Notification of collection purposes",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_collection_notification"
            ),
            ComplianceRule(
                rule_id="PDPA-002",
                framework=ComplianceFramework.PDPA,
                description="Individual access and correction rights",
                data_categories={DataCategory.PERSONAL_IDENTIFIABLE},
                validation_function="validate_individual_rights"
            )
        ]

        for rule in rules:
            self.rules[rule.rule_id] = rule

    def _setup_data_retention_policies(self):
        """Setup data retention policies based on compliance frameworks."""
        # Default retention periods (in days)
        self.data_retention_policies = {
            DataCategory.PERSONAL_IDENTIFIABLE: 2555,  # 7 years
            DataCategory.SENSITIVE_PERSONAL: 1095,     # 3 years
            DataCategory.BIOMETRIC: 730,               # 2 years
            DataCategory.LOCATION: 365,                # 1 year
            DataCategory.BEHAVIORAL: 1095,             # 3 years
            DataCategory.TECHNICAL: 2555,              # 7 years (audit logs)
            DataCategory.ANONYMOUS: -1,                # No retention limit
        }

    def validate_data_processing(self,
                                data_type: DataCategory,
                                purpose: str,
                                lawful_basis: Optional[str] = None,
                                consent_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate data processing for compliance.

        Args:
            data_type: Category of data being processed
            purpose: Purpose of data processing
            lawful_basis: Legal basis for processing (GDPR)
            consent_id: Consent record ID if applicable

        Returns:
            Validation result with compliance status
        """
        validation_result = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "processing_id": hashlib.md5(f"{data_type.value}{purpose}{time.time()}".encode()).hexdigest()[:12]
        }

        # Check applicable rules
        applicable_rules = [
            rule for rule in self.rules.values()
            if data_type in rule.data_categories
        ]

        for rule in applicable_rules:
            if rule.validation_function:
                violation = self._execute_validation(rule, {
                    "data_type": data_type,
                    "purpose": purpose,
                    "lawful_basis": lawful_basis,
                    "consent_id": consent_id
                })

                if violation:
                    validation_result["violations"].append(violation)
                    validation_result["compliant"] = False

        # Record data processing activity
        processing_record = {
            "processing_id": validation_result["processing_id"],
            "timestamp": time.time(),
            "data_type": data_type.value,
            "purpose": purpose,
            "lawful_basis": lawful_basis,
            "consent_id": consent_id,
            "compliant": validation_result["compliant"],
            "violations": len(validation_result["violations"])
        }

        self.data_processing_records.append(processing_record)
        self._log_audit_event("data_processing_validated", processing_record)

        return validation_result

    def _execute_validation(self, rule: ComplianceRule, context: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Execute validation function for a rule."""
        try:
            # Simplified validation logic - in production would call actual validation functions
            if rule.validation_function == "validate_lawful_basis":
                if not context.get("lawful_basis") and context["data_type"] == DataCategory.PERSONAL_IDENTIFIABLE:
                    return ComplianceViolation(
                        violation_id=f"V-{time.time()}",
                        rule_id=rule.rule_id,
                        framework=rule.framework,
                        description="No lawful basis specified for personal data processing",
                        severity="high",
                        data_affected=[context["data_type"].value],
                        timestamp=time.time()
                    )

            elif rule.validation_function == "validate_explicit_consent":
                if not context.get("consent_id") and context["data_type"] in [DataCategory.SENSITIVE_PERSONAL, DataCategory.BIOMETRIC]:
                    return ComplianceViolation(
                        violation_id=f"V-{time.time()}",
                        rule_id=rule.rule_id,
                        framework=rule.framework,
                        description="No explicit consent for sensitive data processing",
                        severity="critical",
                        data_affected=[context["data_type"].value],
                        timestamp=time.time()
                    )

            # Additional validation functions would be implemented here

        except Exception as e:
            logger.error(f"Validation error for rule {rule.rule_id}: {e}")
            return ComplianceViolation(
                violation_id=f"V-{time.time()}",
                rule_id=rule.rule_id,
                framework=rule.framework,
                description=f"Validation error: {str(e)}",
                severity="medium",
                data_affected=[],
                timestamp=time.time()
            )

        return None

    def record_consent(self, individual_id: str, data_categories: List[DataCategory],
                      purposes: List[str], expiry_date: Optional[float] = None) -> str:
        """Record consent for data processing.

        Args:
            individual_id: Unique identifier for the individual
            data_categories: Categories of data consent applies to
            purposes: Purposes consent is given for
            expiry_date: Optional expiry timestamp

        Returns:
            Consent record ID
        """
        consent_id = hashlib.md5(f"{individual_id}{time.time()}".encode()).hexdigest()[:12]

        consent_record = {
            "consent_id": consent_id,
            "individual_id": individual_id,
            "data_categories": [cat.value for cat in data_categories],
            "purposes": purposes,
            "granted_timestamp": time.time(),
            "expiry_date": expiry_date,
            "active": True,
            "withdrawal_timestamp": None
        }

        self.consent_records[consent_id] = consent_record
        self._log_audit_event("consent_recorded", consent_record)

        logger.info(f"Recorded consent {consent_id} for individual {individual_id}")
        return consent_id

    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw previously given consent.

        Args:
            consent_id: ID of consent to withdraw

        Returns:
            True if consent was successfully withdrawn
        """
        if consent_id in self.consent_records:
            consent_record = self.consent_records[consent_id]
            consent_record["active"] = False
            consent_record["withdrawal_timestamp"] = time.time()

            self._log_audit_event("consent_withdrawn", {"consent_id": consent_id})

            logger.info(f"Withdrew consent {consent_id}")
            return True

        logger.warning(f"Consent {consent_id} not found for withdrawal")
        return False

    def check_data_retention(self, data_category: DataCategory, creation_timestamp: float) -> Dict[str, Any]:
        """Check if data meets retention policy requirements.

        Args:
            data_category: Category of data to check
            creation_timestamp: When the data was created

        Returns:
            Retention check result
        """
        retention_period = self.data_retention_policies.get(data_category, 365)

        if retention_period == -1:  # No retention limit
            return {
                "should_delete": False,
                "days_remaining": -1,
                "retention_period": -1
            }

        current_time = time.time()
        data_age_days = (current_time - creation_timestamp) / 86400  # Convert to days
        days_remaining = retention_period - data_age_days

        return {
            "should_delete": days_remaining <= 0,
            "days_remaining": max(0, days_remaining),
            "retention_period": retention_period,
            "data_age_days": data_age_days
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        current_time = time.time()

        # Analyze violations
        violation_summary = {}
        for violation in self.violations:
            framework = violation.framework.value
            if framework not in violation_summary:
                violation_summary[framework] = {"total": 0, "by_severity": {}}

            violation_summary[framework]["total"] += 1
            severity = violation.severity
            if severity not in violation_summary[framework]["by_severity"]:
                violation_summary[framework]["by_severity"][severity] = 0
            violation_summary[framework]["by_severity"][severity] += 1

        # Check consent status
        active_consents = sum(1 for consent in self.consent_records.values() if consent["active"])
        expired_consents = sum(
            1 for consent in self.consent_records.values()
            if consent.get("expiry_date") and consent["expiry_date"] < current_time
        )

        # Data retention analysis
        retention_alerts = []
        # This would analyze actual data in production

        report = {
            "generated_timestamp": current_time,
            "enabled_frameworks": [f.value for f in self.enabled_frameworks],
            "total_rules": len(self.rules),
            "violations": {
                "total": len(self.violations),
                "by_framework": violation_summary,
                "recent": [
                    {
                        "id": v.violation_id,
                        "rule": v.rule_id,
                        "framework": v.framework.value,
                        "severity": v.severity,
                        "description": v.description
                    }
                    for v in sorted(self.violations, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            },
            "consent_management": {
                "total_consents": len(self.consent_records),
                "active_consents": active_consents,
                "expired_consents": expired_consents
            },
            "data_processing": {
                "total_activities": len(self.data_processing_records),
                "compliant_activities": sum(1 for record in self.data_processing_records if record["compliant"]),
                "recent_activities": self.data_processing_records[-10:] if self.data_processing_records else []
            },
            "retention_policies": {
                category.value: days for category, days in self.data_retention_policies.items()
            },
            "audit_trail": {
                "total_events": len(self.audit_log),
                "recent_events": self.audit_log[-10:] if self.audit_log else []
            }
        }

        return report

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event for compliance tracking."""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }

        self.audit_log.append(audit_entry)

        # Keep only last 10000 audit entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]

    def get_framework_status(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Get status for specific compliance framework."""
        framework_rules = [rule for rule in self.rules.values() if rule.framework == framework]
        framework_violations = [v for v in self.violations if v.framework == framework]

        return {
            "framework": framework.value,
            "enabled": framework in self.enabled_frameworks,
            "total_rules": len(framework_rules),
            "violations": len(framework_violations),
            "compliance_score": max(0, 100 - (len(framework_violations) * 10)),  # Simple scoring
            "last_violation": max((v.timestamp for v in framework_violations), default=0)
        }


# Global compliance manager instance
global_compliance_manager = GlobalComplianceManager()

def validate_processing(data_type: DataCategory, purpose: str, **kwargs) -> Dict[str, Any]:
    """Global function to validate data processing."""
    return global_compliance_manager.validate_data_processing(data_type, purpose, **kwargs)

def record_user_consent(individual_id: str, data_categories: List[DataCategory], purposes: List[str]) -> str:
    """Global function to record user consent."""
    return global_compliance_manager.record_consent(individual_id, data_categories, purposes)

def check_compliance_status() -> Dict[str, Any]:
    """Global function to check overall compliance status."""
    return global_compliance_manager.generate_compliance_report()
