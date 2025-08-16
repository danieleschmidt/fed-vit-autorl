"""GDPR compliance framework for federated learning systems."""

import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import threading

from ..error_handling import with_error_handling, ErrorCategory, ErrorSeverity
from ..i18n.translator import t

logger = logging.getLogger(__name__)


class DataSubjectRights(Enum):
    """GDPR data subject rights."""
    ACCESS = "access"                    # Right to access personal data
    RECTIFICATION = "rectification"      # Right to rectify inaccurate data
    ERASURE = "erasure"                  # Right to erasure ("right to be forgotten")
    RESTRICT_PROCESSING = "restrict"     # Right to restrict processing
    DATA_PORTABILITY = "portability"     # Right to data portability
    OBJECT = "object"                    # Right to object to processing
    WITHDRAW_CONSENT = "withdraw"        # Right to withdraw consent


class ProcessingPurpose(Enum):
    """Legal purposes for data processing under GDPR."""
    CONSENT = "consent"                  # Based on consent
    CONTRACT = "contract"                # Necessary for contract performance
    LEGAL_OBLIGATION = "legal"           # Required by law
    VITAL_INTERESTS = "vital"            # To protect vital interests
    PUBLIC_TASK = "public"               # Public task or official authority
    LEGITIMATE_INTERESTS = "legitimate"   # Legitimate interests


class DataCategory(Enum):
    """Categories of personal data."""
    IDENTIFICATION = "identification"    # Names, IDs, etc.
    CONTACT = "contact"                 # Email, phone, address
    DEMOGRAPHIC = "demographic"         # Age, gender, etc.
    BEHAVIORAL = "behavioral"           # Usage patterns, preferences
    TECHNICAL = "technical"             # Device info, IP addresses
    BIOMETRIC = "biometric"             # Biometric identifiers
    HEALTH = "health"                   # Health-related data
    LOCATION = "location"               # Geographic location data


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    data_subject_id: str = ""
    data_categories: List[DataCategory] = field(default_factory=list)
    processing_purpose: ProcessingPurpose = ProcessingPurpose.CONSENT
    legal_basis: str = ""
    retention_period: Optional[int] = None  # Days
    processor_id: str = ""
    controller_id: str = ""
    transfers_to_third_countries: List[str] = field(default_factory=list)
    consent_timestamp: Optional[float] = None
    consent_withdrawn: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'data_subject_id': self.data_subject_id,
            'data_categories': [cat.value for cat in self.data_categories],
            'processing_purpose': self.processing_purpose.value,
            'legal_basis': self.legal_basis,
            'retention_period': self.retention_period,
            'processor_id': self.processor_id,
            'controller_id': self.controller_id,
            'transfers_to_third_countries': self.transfers_to_third_countries,
            'consent_timestamp': self.consent_timestamp,
            'consent_withdrawn': self.consent_withdrawn,
        }


@dataclass
class ConsentRecord:
    """Record of data subject consent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_subject_id: str = ""
    purposes: List[ProcessingPurpose] = field(default_factory=list)
    data_categories: List[DataCategory] = field(default_factory=list)
    consent_timestamp: float = field(default_factory=time.time)
    withdrawal_timestamp: Optional[float] = None
    consent_text: str = ""
    consent_version: str = "1.0"
    is_withdrawn: bool = False
    ip_address: str = ""
    user_agent: str = ""
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.is_withdrawn:
            return False
        
        # Consent expires after 2 years under GDPR
        expiry_date = self.consent_timestamp + (2 * 365 * 24 * 60 * 60)
        return time.time() < expiry_date


@dataclass
class DataBreachIncident:
    """Data breach incident record."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    discovered_timestamp: float = field(default_factory=time.time)
    incident_type: str = ""
    affected_data_subjects: int = 0
    data_categories_affected: List[DataCategory] = field(default_factory=list)
    description: str = ""
    risk_assessment: str = ""
    measures_taken: List[str] = field(default_factory=list)
    notification_required: bool = False
    authority_notified: bool = False
    subjects_notified: bool = False
    notification_timestamp: Optional[float] = None


class GDPRComplianceManager:
    """GDPR compliance management for federated learning systems."""
    
    def __init__(
        self,
        controller_name: str,
        controller_contact: str,
        dpo_contact: Optional[str] = None,
        data_retention_days: int = 1095,  # 3 years default
    ):
        """Initialize GDPR compliance manager.
        
        Args:
            controller_name: Name of the data controller
            controller_contact: Contact information for the controller
            dpo_contact: Data Protection Officer contact (if applicable)
            data_retention_days: Default data retention period in days
        """
        self.controller_name = controller_name
        self.controller_contact = controller_contact
        self.dpo_contact = dpo_contact
        self.data_retention_days = data_retention_days
        
        # Storage for compliance records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.breach_incidents: Dict[str, DataBreachIncident] = {}
        self.data_subject_requests: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info("GDPR compliance manager initialized")
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def record_consent(
        self,
        data_subject_id: str,
        purposes: List[ProcessingPurpose],
        data_categories: List[DataCategory],
        consent_text: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> str:
        """Record data subject consent.
        
        Args:
            data_subject_id: Unique identifier for the data subject
            purposes: List of processing purposes
            data_categories: List of data categories
            consent_text: Text of the consent given
            ip_address: IP address of the consent
            user_agent: User agent string
            
        Returns:
            Consent record ID
        """
        with self._lock:
            consent = ConsentRecord(
                data_subject_id=data_subject_id,
                purposes=purposes,
                data_categories=data_categories,
                consent_text=consent_text,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            self.consent_records[consent.id] = consent
            
            # Log the consent
            self._add_audit_entry(
                action="consent_recorded",
                data_subject_id=data_subject_id,
                details={
                    'consent_id': consent.id,
                    'purposes': [p.value for p in purposes],
                    'categories': [c.value for c in data_categories],
                }
            )
            
            logger.info(f"Consent recorded for data subject {data_subject_id}")
            return consent.id
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def withdraw_consent(self, data_subject_id: str, consent_id: Optional[str] = None) -> bool:
        """Withdraw consent for a data subject.
        
        Args:
            data_subject_id: Data subject identifier
            consent_id: Specific consent to withdraw (optional)
            
        Returns:
            True if successful
        """
        with self._lock:
            if consent_id:
                # Withdraw specific consent
                if consent_id in self.consent_records:
                    consent = self.consent_records[consent_id]
                    if consent.data_subject_id == data_subject_id:
                        consent.is_withdrawn = True
                        consent.withdrawal_timestamp = time.time()
                        
                        self._add_audit_entry(
                            action="consent_withdrawn",
                            data_subject_id=data_subject_id,
                            details={'consent_id': consent_id}
                        )
                        
                        # Stop any ongoing processing
                        self._stop_processing_for_subject(data_subject_id)
                        
                        logger.info(f"Consent {consent_id} withdrawn for subject {data_subject_id}")
                        return True
            else:
                # Withdraw all consents for the subject
                withdrawn_any = False
                for consent in self.consent_records.values():
                    if consent.data_subject_id == data_subject_id and not consent.is_withdrawn:
                        consent.is_withdrawn = True
                        consent.withdrawal_timestamp = time.time()
                        withdrawn_any = True
                
                if withdrawn_any:
                    self._add_audit_entry(
                        action="all_consent_withdrawn",
                        data_subject_id=data_subject_id,
                        details={}
                    )
                    
                    self._stop_processing_for_subject(data_subject_id)
                    logger.info(f"All consent withdrawn for subject {data_subject_id}")
                    return True
            
            return False
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def record_processing_activity(
        self,
        data_subject_id: str,
        data_categories: List[DataCategory],
        purpose: ProcessingPurpose,
        legal_basis: str,
        processor_id: str = "",
    ) -> str:
        """Record a data processing activity.
        
        Args:
            data_subject_id: Data subject identifier
            data_categories: Categories of data being processed
            purpose: Purpose of processing
            legal_basis: Legal basis for processing
            processor_id: ID of the processor
            
        Returns:
            Processing record ID
        """
        with self._lock:
            # Check if we have valid consent for this processing
            if purpose == ProcessingPurpose.CONSENT:
                if not self._has_valid_consent(data_subject_id, purpose, data_categories):
                    raise ValueError(f"No valid consent for processing data subject {data_subject_id}")
            
            record = DataProcessingRecord(
                data_subject_id=data_subject_id,
                data_categories=data_categories,
                processing_purpose=purpose,
                legal_basis=legal_basis,
                processor_id=processor_id,
                controller_id=self.controller_name,
                retention_period=self.data_retention_days,
            )
            
            self.processing_records[record.id] = record
            
            self._add_audit_entry(
                action="processing_recorded",
                data_subject_id=data_subject_id,
                details={
                    'record_id': record.id,
                    'purpose': purpose.value,
                    'categories': [c.value for c in data_categories],
                }
            )
            
            return record.id
    
    @with_error_handling(max_retries=1, auto_recover=True)
    def handle_data_subject_request(
        self,
        data_subject_id: str,
        request_type: DataSubjectRights,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Handle a data subject rights request.
        
        Args:
            data_subject_id: Data subject identifier
            request_type: Type of rights request
            details: Additional request details
            
        Returns:
            Response data
        """
        with self._lock:
            request_id = str(uuid.uuid4())
            request_data = {
                'id': request_id,
                'data_subject_id': data_subject_id,
                'request_type': request_type.value,
                'timestamp': time.time(),
                'details': details or {},
                'status': 'processing',
                'response': None,
            }
            
            self.data_subject_requests[request_id] = request_data
            
            try:
                if request_type == DataSubjectRights.ACCESS:
                    response = self._handle_access_request(data_subject_id)
                elif request_type == DataSubjectRights.ERASURE:
                    response = self._handle_erasure_request(data_subject_id)
                elif request_type == DataSubjectRights.RECTIFICATION:
                    response = self._handle_rectification_request(data_subject_id, details)
                elif request_type == DataSubjectRights.RESTRICT_PROCESSING:
                    response = self._handle_restriction_request(data_subject_id)
                elif request_type == DataSubjectRights.DATA_PORTABILITY:
                    response = self._handle_portability_request(data_subject_id)
                elif request_type == DataSubjectRights.OBJECT:
                    response = self._handle_objection_request(data_subject_id)
                elif request_type == DataSubjectRights.WITHDRAW_CONSENT:
                    response = {'success': self.withdraw_consent(data_subject_id)}
                else:
                    response = {'error': 'Unsupported request type'}
                
                request_data['status'] = 'completed'
                request_data['response'] = response
                
                self._add_audit_entry(
                    action="data_subject_request",
                    data_subject_id=data_subject_id,
                    details={
                        'request_id': request_id,
                        'request_type': request_type.value,
                        'status': 'completed',
                    }
                )
                
                return response
                
            except Exception as e:
                request_data['status'] = 'failed'
                request_data['error'] = str(e)
                
                logger.error(f"Failed to handle data subject request: {e}")
                return {'error': str(e)}
    
    def _handle_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data access request."""
        # Collect all data for the subject
        subject_data = {
            'data_subject_id': data_subject_id,
            'processing_records': [],
            'consent_records': [],
            'data_categories': set(),
            'processing_purposes': set(),
        }
        
        # Find processing records
        for record in self.processing_records.values():
            if record.data_subject_id == data_subject_id:
                subject_data['processing_records'].append(record.to_dict())
                subject_data['data_categories'].update([c.value for c in record.data_categories])
                subject_data['processing_purposes'].add(record.processing_purpose.value)
        
        # Find consent records
        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id:
                subject_data['consent_records'].append({
                    'id': consent.id,
                    'purposes': [p.value for p in consent.purposes],
                    'data_categories': [c.value for c in consent.data_categories],
                    'consent_timestamp': consent.consent_timestamp,
                    'is_withdrawn': consent.is_withdrawn,
                    'withdrawal_timestamp': consent.withdrawal_timestamp,
                })
        
        # Convert sets to lists for JSON serialization
        subject_data['data_categories'] = list(subject_data['data_categories'])
        subject_data['processing_purposes'] = list(subject_data['processing_purposes'])
        
        return subject_data
    
    def _handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data erasure request (right to be forgotten)."""
        # Check if we can legally erase the data
        cannot_erase = []
        
        for record in self.processing_records.values():
            if record.data_subject_id == data_subject_id:
                if record.processing_purpose in [
                    ProcessingPurpose.LEGAL_OBLIGATION,
                    ProcessingPurpose.PUBLIC_TASK,
                    ProcessingPurpose.VITAL_INTERESTS,
                ]:
                    cannot_erase.append(record.id)
        
        if cannot_erase:
            return {
                'success': False,
                'reason': 'Cannot erase data due to legal obligations',
                'affected_records': cannot_erase,
            }
        
        # Perform erasure
        erased_records = []
        
        # Remove processing records
        to_remove = []
        for record_id, record in self.processing_records.items():
            if record.data_subject_id == data_subject_id:
                to_remove.append(record_id)
                erased_records.append(record_id)
        
        for record_id in to_remove:
            del self.processing_records[record_id]
        
        # Mark consent as withdrawn
        for consent in self.consent_records.values():
            if consent.data_subject_id == data_subject_id:
                consent.is_withdrawn = True
                consent.withdrawal_timestamp = time.time()
        
        self._add_audit_entry(
            action="data_erased",
            data_subject_id=data_subject_id,
            details={'erased_records': erased_records}
        )
        
        return {
            'success': True,
            'erased_records': erased_records,
        }
    
    def _handle_rectification_request(
        self,
        data_subject_id: str,
        details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle data rectification request."""
        # This would require integration with the actual data storage system
        # For now, we log the request
        
        self._add_audit_entry(
            action="rectification_requested",
            data_subject_id=data_subject_id,
            details=details or {}
        )
        
        return {
            'success': True,
            'message': 'Rectification request logged for manual processing',
        }
    
    def _handle_restriction_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle processing restriction request."""
        # Mark processing as restricted
        restricted_records = []
        
        for record in self.processing_records.values():
            if record.data_subject_id == data_subject_id:
                # Add restriction flag (would need to modify the record structure)
                restricted_records.append(record.id)
        
        self._add_audit_entry(
            action="processing_restricted",
            data_subject_id=data_subject_id,
            details={'restricted_records': restricted_records}
        )
        
        return {
            'success': True,
            'restricted_records': restricted_records,
        }
    
    def _handle_portability_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Get data in structured format
        access_data = self._handle_access_request(data_subject_id)
        
        # Format for portability (JSON format)
        portable_data = {
            'data_subject_id': data_subject_id,
            'export_timestamp': time.time(),
            'export_format': 'JSON',
            'data': access_data,
        }
        
        return {
            'success': True,
            'portable_data': portable_data,
        }
    
    def _handle_objection_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle objection to processing request."""
        # Check which processing can be stopped
        can_stop = []
        cannot_stop = []
        
        for record in self.processing_records.values():
            if record.data_subject_id == data_subject_id:
                if record.processing_purpose == ProcessingPurpose.LEGITIMATE_INTERESTS:
                    can_stop.append(record.id)
                else:
                    cannot_stop.append(record.id)
        
        # Stop processing where possible
        for record_id in can_stop:
            # Mark as stopped (would need additional record structure)
            pass
        
        self._add_audit_entry(
            action="objection_processed",
            data_subject_id=data_subject_id,
            details={
                'stopped_processing': can_stop,
                'continuing_processing': cannot_stop,
            }
        )
        
        return {
            'success': True,
            'stopped_processing': can_stop,
            'continuing_processing': cannot_stop,
        }
    
    def _has_valid_consent(
        self,
        data_subject_id: str,
        purpose: ProcessingPurpose,
        data_categories: List[DataCategory]
    ) -> bool:
        """Check if there's valid consent for the specified processing."""
        for consent in self.consent_records.values():
            if (consent.data_subject_id == data_subject_id and
                consent.is_valid() and
                purpose in consent.purposes):
                
                # Check if all data categories are covered
                consent_categories = set(consent.data_categories)
                required_categories = set(data_categories)
                
                if required_categories.issubset(consent_categories):
                    return True
        
        return False
    
    def _stop_processing_for_subject(self, data_subject_id: str) -> None:
        """Stop all consent-based processing for a data subject."""
        # This would integrate with the actual ML pipeline to stop processing
        # For now, we just log it
        
        self._add_audit_entry(
            action="processing_stopped",
            data_subject_id=data_subject_id,
            details={}
        )
        
        logger.info(f"Stopped processing for data subject {data_subject_id}")
    
    def _add_audit_entry(
        self,
        action: str,
        data_subject_id: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add entry to audit log."""
        entry = {
            'timestamp': time.time(),
            'action': action,
            'data_subject_id': data_subject_id,
            'details': details or {},
            'controller': self.controller_name,
        }
        
        self.audit_log.append(entry)
        
        # Keep audit log size manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-8000:]  # Keep last 8000 entries
    
    def report_data_breach(
        self,
        incident_type: str,
        affected_subjects: int,
        data_categories: List[DataCategory],
        description: str,
        risk_assessment: str,
    ) -> str:
        """Report a data breach incident.
        
        Args:
            incident_type: Type of breach
            affected_subjects: Number of affected data subjects
            data_categories: Categories of data affected
            description: Description of the incident
            risk_assessment: Risk assessment
            
        Returns:
            Incident ID
        """
        incident = DataBreachIncident(
            incident_type=incident_type,
            affected_data_subjects=affected_subjects,
            data_categories_affected=data_categories,
            description=description,
            risk_assessment=risk_assessment,
            notification_required=affected_subjects > 250 or any(
                cat in [DataCategory.BIOMETRIC, DataCategory.HEALTH] 
                for cat in data_categories
            ),
        )
        
        self.breach_incidents[incident.id] = incident
        
        self._add_audit_entry(
            action="data_breach_reported",
            details={
                'incident_id': incident.id,
                'affected_subjects': affected_subjects,
                'notification_required': incident.notification_required,
            }
        )
        
        if incident.notification_required:
            logger.critical(f"Data breach requires notification: {incident.id}")
            # In real implementation, this would trigger notification workflows
        
        return incident.id
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report."""
        with self._lock:
            # Count records by status
            total_subjects = len(set(
                record.data_subject_id for record in self.processing_records.values()
            ))
            
            active_consents = sum(
                1 for consent in self.consent_records.values() 
                if consent.is_valid()
            )
            
            withdrawn_consents = sum(
                1 for consent in self.consent_records.values() 
                if consent.is_withdrawn
            )
            
            pending_requests = sum(
                1 for req in self.data_subject_requests.values()
                if req['status'] == 'processing'
            )
            
            # Check for overdue deletions
            overdue_deletions = []
            current_time = time.time()
            
            for record in self.processing_records.values():
                if record.retention_period:
                    expiry_time = record.timestamp + (record.retention_period * 24 * 60 * 60)
                    if current_time > expiry_time:
                        overdue_deletions.append(record.id)
            
            return {
                'controller': self.controller_name,
                'dpo_contact': self.dpo_contact,
                'report_timestamp': current_time,
                'statistics': {
                    'total_data_subjects': total_subjects,
                    'active_consents': active_consents,
                    'withdrawn_consents': withdrawn_consents,
                    'pending_requests': pending_requests,
                    'total_processing_records': len(self.processing_records),
                    'data_breaches': len(self.breach_incidents),
                    'overdue_deletions': len(overdue_deletions),
                },
                'compliance_status': {
                    'consent_management': 'compliant' if active_consents > 0 else 'attention_needed',
                    'data_retention': 'compliant' if len(overdue_deletions) == 0 else 'non_compliant',
                    'breach_management': 'compliant',
                    'rights_management': 'compliant' if pending_requests < 10 else 'attention_needed',
                },
                'recommendations': self._generate_compliance_recommendations(),
            }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Check consent expiry
        expiring_soon = 0
        thirty_days = 30 * 24 * 60 * 60
        current_time = time.time()
        
        for consent in self.consent_records.values():
            if consent.is_valid():
                expiry_time = consent.consent_timestamp + (2 * 365 * 24 * 60 * 60)
                if expiry_time - current_time < thirty_days:
                    expiring_soon += 1
        
        if expiring_soon > 0:
            recommendations.append(f"{expiring_soon} consent records expiring within 30 days")
        
        # Check data retention
        overdue = sum(
            1 for record in self.processing_records.values()
            if record.retention_period and 
            current_time > record.timestamp + (record.retention_period * 24 * 60 * 60)
        )
        
        if overdue > 0:
            recommendations.append(f"{overdue} records past retention period - schedule deletion")
        
        # Check pending requests
        pending = sum(
            1 for req in self.data_subject_requests.values()
            if req['status'] == 'processing'
        )
        
        if pending > 5:
            recommendations.append(f"{pending} pending data subject requests - review response times")
        
        return recommendations if recommendations else ["All compliance checks passed"]