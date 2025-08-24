"""Generation 2: Robust Federated ViT-AutoRL Training Implementation

Adds comprehensive error handling, validation, logging, monitoring, 
health checks, security measures, and input sanitization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import hashlib
import json
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fed_vit_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingState(Enum):
    """Training state enumeration for robust state management."""
    INITIALIZING = "initializing"
    TRAINING = "training" 
    EVALUATING = "evaluating"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


class SecurityLevel(Enum):
    """Security level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking."""
    round_id: int
    client_id: int
    loss: float
    accuracy: float
    training_time: float
    memory_usage: float
    gradient_norm: float
    convergence_rate: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    severity: SecurityLevel
    message: str
    client_id: Optional[int]
    timestamp: datetime
    resolved: bool = False


class InputValidator:
    """Robust input validation for federated learning components."""
    
    @staticmethod
    def validate_tensor_input(tensor: torch.Tensor, expected_shape: Tuple[int, ...] = None,
                             name: str = "tensor") -> bool:
        """Validate tensor input with comprehensive checks."""
        try:
            # Check tensor validity
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
            
            # Check for NaN/Inf values
            if torch.isnan(tensor).any():
                raise ValueError(f"{name} contains NaN values")
            
            if torch.isinf(tensor).any():
                raise ValueError(f"{name} contains infinite values")
            
            # Check shape if specified
            if expected_shape is not None and tensor.shape != expected_shape:
                raise ValueError(f"{name} expected shape {expected_shape}, got {tensor.shape}")
            
            # Check value ranges for image data
            if len(tensor.shape) == 4 and tensor.shape[1] == 3:  # Assume image data
                if tensor.min() < -10 or tensor.max() > 10:
                    logger.warning(f"{name} has unusual value range: [{tensor.min():.2f}, {tensor.max():.2f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"Tensor validation failed for {name}: {str(e)}")
            return False
    
    @staticmethod
    def validate_model_parameters(model: nn.Module, name: str = "model") -> bool:
        """Validate model parameters for anomalies."""
        try:
            param_count = 0
            nan_params = 0
            inf_params = 0
            zero_params = 0
            
            for param in model.parameters():
                param_count += param.numel()
                
                if torch.isnan(param).any():
                    nan_params += torch.isnan(param).sum().item()
                
                if torch.isinf(param).any():
                    inf_params += torch.isinf(param).sum().item()
                
                if (param == 0).all():
                    zero_params += 1
            
            # Log parameter health
            logger.info(f"{name} parameter health: {param_count} total, "
                       f"{nan_params} NaN, {inf_params} Inf, {zero_params} zero layers")
            
            # Check for critical issues
            if nan_params > 0:
                raise ValueError(f"{name} has {nan_params} NaN parameters")
            
            if inf_params > 0:
                raise ValueError(f"{name} has {inf_params} infinite parameters")
            
            if zero_params == len(list(model.parameters())):
                raise ValueError(f"{name} has all zero parameters")
            
            return True
            
        except Exception as e:
            logger.error(f"Model parameter validation failed for {name}: {str(e)}")
            return False
    
    @staticmethod
    def validate_client_data(client_id: int, data_size: int, loss_value: float) -> bool:
        """Validate client-specific data for anomalies."""
        try:
            # Validate client ID
            if not isinstance(client_id, int) or client_id < 0:
                raise ValueError(f"Invalid client_id: {client_id}")
            
            # Validate data size
            if not isinstance(data_size, int) or data_size <= 0:
                raise ValueError(f"Invalid data_size: {data_size}")
            
            # Validate loss value
            if not isinstance(loss_value, (int, float)):
                raise ValueError(f"Invalid loss_value type: {type(loss_value)}")
            
            if np.isnan(loss_value) or np.isinf(loss_value):
                raise ValueError(f"Invalid loss_value: {loss_value}")
            
            if loss_value < 0:
                logger.warning(f"Unusual negative loss for client {client_id}: {loss_value}")
            
            if loss_value > 100:
                logger.warning(f"Unusually high loss for client {client_id}: {loss_value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Client data validation failed: {str(e)}")
            return False


class SecurityManager:
    """Comprehensive security management for federated learning."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.alerts: List[SecurityAlert] = []
        self.client_reputation: Dict[int, float] = {}
        self.blocked_clients: Set[int] = set()
        
    def create_alert(self, severity: SecurityLevel, message: str, client_id: Optional[int] = None):
        """Create a security alert."""
        alert = SecurityAlert(
            alert_id=hashlib.md5(f"{message}{time.time()}".encode()).hexdigest()[:8],
            severity=severity,
            message=message,
            client_id=client_id,
            timestamp=datetime.now()
        )
        self.alerts.append(alert)
        
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            logger.critical(f"SECURITY ALERT: {message} (Client: {client_id})")
        
        return alert
    
    def validate_client_update(self, client_id: int, update_data: Dict[str, Any]) -> bool:
        """Validate client update for security threats."""
        try:
            # Check if client is blocked
            if client_id in self.blocked_clients:
                self.create_alert(SecurityLevel.HIGH, f"Blocked client {client_id} attempted update")
                return False
            
            # Check update size (potential DoS attack)
            update_size = len(str(update_data))
            if update_size > 50_000_000:  # 50MB limit
                self.create_alert(SecurityLevel.HIGH, 
                                f"Client {client_id} sent oversized update: {update_size} bytes")
                return False
            
            # Check for suspicious parameter values
            if 'param_update' in update_data:
                for param_name, param_tensor in update_data['param_update'].items():
                    if isinstance(param_tensor, torch.Tensor):
                        if torch.abs(param_tensor).max() > 1000:
                            self.create_alert(SecurityLevel.MEDIUM,
                                            f"Client {client_id} has large parameter values in {param_name}")
                            self._decrease_client_reputation(client_id, 0.1)
            
            # Update client reputation
            self._increase_client_reputation(client_id, 0.01)
            
            return True
            
        except Exception as e:
            self.create_alert(SecurityLevel.HIGH, f"Security validation error: {str(e)}")
            return False
    
    def _increase_client_reputation(self, client_id: int, amount: float):
        """Increase client reputation score."""
        current_rep = self.client_reputation.get(client_id, 0.5)
        self.client_reputation[client_id] = min(1.0, current_rep + amount)
    
    def _decrease_client_reputation(self, client_id: int, amount: float):
        """Decrease client reputation score."""
        current_rep = self.client_reputation.get(client_id, 0.5)
        new_rep = max(0.0, current_rep - amount)
        self.client_reputation[client_id] = new_rep
        
        # Block clients with very low reputation
        if new_rep < 0.1:
            self.blocked_clients.add(client_id)
            self.create_alert(SecurityLevel.CRITICAL, f"Client {client_id} blocked due to low reputation")


class HealthMonitor:
    """System health monitoring for federated learning."""
    
    def __init__(self):
        self.metrics_history: List[TrainingMetrics] = []
        self.system_alerts: List[str] = []
        self.last_health_check = time.time()
        
    def record_metrics(self, metrics: TrainingMetrics):
        """Record training metrics for monitoring."""
        self.metrics_history.append(metrics)
        
        # Detect anomalies
        self._detect_anomalies(metrics)
        
        # Keep only recent metrics (last 1000 records)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _detect_anomalies(self, metrics: TrainingMetrics):
        """Detect training anomalies."""
        # Check for sudden loss spikes
        if len(self.metrics_history) > 5:
            recent_losses = [m.loss for m in self.metrics_history[-5:]]
            avg_recent_loss = np.mean(recent_losses)
            
            if metrics.loss > avg_recent_loss * 2:
                alert = f"Loss spike detected for client {metrics.client_id}: {metrics.loss:.4f}"
                self.system_alerts.append(alert)
                logger.warning(alert)
        
        # Check for memory usage spikes
        if metrics.memory_usage > 8000:  # 8GB threshold
            alert = f"High memory usage for client {metrics.client_id}: {metrics.memory_usage:.1f}MB"
            self.system_alerts.append(alert)
            logger.warning(alert)
        
        # Check for slow training
        if metrics.training_time > 300:  # 5 minute threshold
            alert = f"Slow training for client {metrics.client_id}: {metrics.training_time:.1f}s"
            self.system_alerts.append(alert)
            logger.warning(alert)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.metrics_history:
            return {"status": "no_data", "alerts": self.system_alerts}
        
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_loss = np.mean([m.loss for m in recent_metrics])
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        avg_training_time = np.mean([m.training_time for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        # Determine overall health status
        health_score = 1.0
        
        if avg_loss > 10:
            health_score -= 0.3
        if avg_accuracy < 0.1:
            health_score -= 0.3
        if avg_training_time > 180:
            health_score -= 0.2
        if len(self.system_alerts) > 10:
            health_score -= 0.2
        
        status = "healthy" if health_score > 0.7 else ("warning" if health_score > 0.4 else "critical")
        
        return {
            "status": status,
            "health_score": health_score,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "avg_training_time": avg_training_time,
            "avg_memory_usage": avg_memory,
            "total_alerts": len(self.system_alerts),
            "recent_alerts": self.system_alerts[-5:],
            "metrics_count": len(self.metrics_history)
        }


class RobustFederatedSystem:
    """Generation 2: Robust federated learning system with comprehensive error handling."""
    
    def __init__(self, num_clients: int = 5, embed_dim: int = 768, 
                 security_level: SecurityLevel = SecurityLevel.MEDIUM):
        
        self.num_clients = num_clients
        self.embed_dim = embed_dim
        self.state = TrainingState.INITIALIZING
        self.start_time = time.time()
        
        # Initialize components with error handling
        try:
            self.validator = InputValidator()
            self.security_manager = SecurityManager(security_level)
            self.health_monitor = HealthMonitor()
            
            # Initialize models with validation
            self.global_model = self._create_model()
            self.client_models = []
            self.client_optimizers = []
            
            for i in range(num_clients):
                client_model = self._create_model()
                client_model.load_state_dict(self.global_model.state_dict())
                
                # Validate initial model
                if not self.validator.validate_model_parameters(client_model, f"client_{i}"):
                    raise RuntimeError(f"Client {i} model validation failed during initialization")
                
                client_optimizer = optim.Adam(client_model.parameters(), lr=1e-4)
                
                self.client_models.append(client_model)
                self.client_optimizers.append(client_optimizer)
            
            # Initialize client reputation
            for i in range(num_clients):
                self.security_manager.client_reputation[i] = 0.8  # Good starting reputation
            
            self.state = TrainingState.TRAINING
            logger.info(f"Robust federated system initialized successfully with {num_clients} clients")
            
        except Exception as e:
            self.state = TrainingState.FAILED
            logger.error(f"Failed to initialize robust federated system: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_model(self) -> nn.Module:
        """Create model with proper initialization and validation."""
        try:
            from .simple_training import SimpleViTPerception
            model = SimpleViTPerception(embed_dim=self.embed_dim)
            
            # Validate model after creation
            if not self.validator.validate_model_parameters(model, "new_model"):
                raise RuntimeError("Model validation failed after creation")
            
            return model
            
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            raise
    
    def client_local_training_robust(
        self, 
        client_id: int, 
        data: torch.Tensor, 
        labels: torch.Tensor, 
        epochs: int = 1
    ) -> Optional[Dict[str, Any]]:
        """Robust client local training with comprehensive error handling."""
        
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            # Input validation
            if not self.validator.validate_tensor_input(data, name=f"client_{client_id}_data"):
                raise ValueError(f"Invalid data for client {client_id}")
            
            if not self.validator.validate_tensor_input(labels, name=f"client_{client_id}_labels"):
                raise ValueError(f"Invalid labels for client {client_id}")
            
            if client_id < 0 or client_id >= self.num_clients:
                raise ValueError(f"Invalid client_id: {client_id}")
            
            if epochs <= 0 or epochs > 100:
                raise ValueError(f"Invalid epochs: {epochs}")
            
            # Check if client is blocked by security
            if client_id in self.security_manager.blocked_clients:
                logger.warning(f"Training blocked for client {client_id} due to security concerns")
                return None
            
            client_model = self.client_models[client_id]
            optimizer = self.client_optimizers[client_id]
            criterion = nn.CrossEntropyLoss()
            
            # Pre-training model validation
            if not self.validator.validate_model_parameters(client_model, f"client_{client_id}_pre_train"):
                raise RuntimeError(f"Pre-training model validation failed for client {client_id}")
            
            client_model.train()
            epoch_losses = []
            
            # Training loop with robust error handling
            for epoch in range(epochs):
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass with gradient computation check
                    outputs = client_model(data)
                    
                    if not self.validator.validate_tensor_input(outputs, name=f"client_{client_id}_outputs"):
                        raise RuntimeError(f"Invalid outputs from client {client_id} model")
                    
                    loss = criterion(outputs, labels)
                    
                    # Validate loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise RuntimeError(f"Invalid loss value for client {client_id}: {loss.item()}")
                    
                    # Backward pass with gradient explosion check
                    loss.backward()
                    
                    # Check gradient norms
                    total_norm = 0
                    for p in client_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    if total_norm > 1000:  # Gradient explosion detection
                        logger.warning(f"Large gradient norm for client {client_id}: {total_norm:.2f}")
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_losses.append(loss.item())
                    
                except Exception as epoch_error:
                    logger.error(f"Epoch {epoch} failed for client {client_id}: {str(epoch_error)}")
                    # Continue with next epoch rather than failing completely
                    epoch_losses.append(float('inf'))
                    continue
            
            # Post-training validation
            if not self.validator.validate_model_parameters(client_model, f"client_{client_id}_post_train"):
                raise RuntimeError(f"Post-training model validation failed for client {client_id}")
            
            # Calculate metrics
            training_time = time.time() - start_time
            final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_usage = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            avg_loss = np.mean([l for l in epoch_losses if not np.isinf(l)])
            if np.isnan(avg_loss):
                avg_loss = float('inf')
            
            # Evaluate accuracy
            with torch.no_grad():
                client_model.eval()
                outputs = client_model(data)
                pred = outputs.argmax(dim=1)
                correct = pred.eq(labels).sum().item()
                accuracy = correct / len(labels)
                client_model.train()
            
            # Record metrics for monitoring
            metrics = TrainingMetrics(
                round_id=0,  # Will be set by caller
                client_id=client_id,
                loss=avg_loss,
                accuracy=accuracy,
                training_time=training_time,
                memory_usage=memory_usage,
                gradient_norm=total_norm,
                convergence_rate=0.0,  # Could be calculated based on loss history
                timestamp=datetime.now()
            )
            
            self.health_monitor.record_metrics(metrics)
            
            # Validate result data
            result = {
                "loss": avg_loss,
                "accuracy": accuracy,
                "client_id": client_id,
                "training_time": training_time,
                "memory_usage": memory_usage,
                "gradient_norm": total_norm,
                "successful_epochs": len([l for l in epoch_losses if not np.isinf(l)]),
                "total_epochs": epochs
            }
            
            if not self.validator.validate_client_data(client_id, len(data), avg_loss):
                logger.warning(f"Result validation failed for client {client_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Client {client_id} training failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Record failure metrics
            failure_metrics = TrainingMetrics(
                round_id=0,
                client_id=client_id,
                loss=float('inf'),
                accuracy=0.0,
                training_time=time.time() - start_time,
                memory_usage=0.0,
                gradient_norm=0.0,
                convergence_rate=0.0,
                timestamp=datetime.now()
            )
            self.health_monitor.record_metrics(failure_metrics)
            
            # Decrease client reputation for failures
            self.security_manager._decrease_client_reputation(client_id, 0.2)
            
            return None
    
    def federated_averaging_robust(self) -> bool:
        """Robust federated averaging with security and validation checks."""
        
        try:
            self.state = TrainingState.AGGREGATING
            
            # Validate all client models before aggregation
            valid_clients = []
            for client_id in range(self.num_clients):
                if self.validator.validate_model_parameters(self.client_models[client_id], f"client_{client_id}_pre_agg"):
                    if client_id not in self.security_manager.blocked_clients:
                        valid_clients.append(client_id)
                    else:
                        logger.warning(f"Excluding blocked client {client_id} from aggregation")
                else:
                    logger.error(f"Client {client_id} model validation failed, excluding from aggregation")
            
            if len(valid_clients) == 0:
                raise RuntimeError("No valid clients available for aggregation")
            
            if len(valid_clients) < self.num_clients * 0.5:
                logger.warning(f"Only {len(valid_clients)}/{self.num_clients} clients valid for aggregation")
            
            # Perform weighted aggregation based on reputation
            global_dict = self.global_model.state_dict()
            
            # Initialize aggregated parameters
            for key in global_dict.keys():
                global_dict[key] = torch.zeros_like(global_dict[key])
            
            # Weighted averaging based on client reputation
            total_weight = 0.0
            for client_id in valid_clients:
                client_dict = self.client_models[client_id].state_dict()
                client_weight = self.security_manager.client_reputation.get(client_id, 0.1)
                
                for key in global_dict.keys():
                    global_dict[key] += client_dict[key] * client_weight
                
                total_weight += client_weight
            
            # Normalize by total weight
            if total_weight > 0:
                for key in global_dict.keys():
                    global_dict[key] /= total_weight
            else:
                logger.error("Zero total weight in aggregation")
                return False
            
            # Validate aggregated model
            self.global_model.load_state_dict(global_dict)
            if not self.validator.validate_model_parameters(self.global_model, "aggregated_global"):
                raise RuntimeError("Aggregated global model validation failed")
            
            # Update all valid client models
            for client_id in valid_clients:
                try:
                    self.client_models[client_id].load_state_dict(global_dict)
                    if not self.validator.validate_model_parameters(self.client_models[client_id], f"client_{client_id}_post_agg"):
                        logger.error(f"Client {client_id} model invalid after aggregation update")
                except Exception as e:
                    logger.error(f"Failed to update client {client_id} after aggregation: {str(e)}")
            
            self.state = TrainingState.TRAINING
            logger.info(f"Robust federated averaging completed with {len(valid_clients)} clients")
            
            return True
            
        except Exception as e:
            self.state = TrainingState.FAILED
            logger.error(f"Robust federated averaging failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def evaluate_global_model_robust(
        self, 
        test_data: torch.Tensor, 
        test_labels: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """Robust global model evaluation with comprehensive checks."""
        
        try:
            self.state = TrainingState.EVALUATING
            
            # Input validation
            if not self.validator.validate_tensor_input(test_data, name="test_data"):
                raise ValueError("Invalid test data")
            
            if not self.validator.validate_tensor_input(test_labels, name="test_labels"):
                raise ValueError("Invalid test labels")
            
            # Model validation
            if not self.validator.validate_model_parameters(self.global_model, "global_eval"):
                raise RuntimeError("Global model validation failed before evaluation")
            
            self.global_model.eval()
            
            with torch.no_grad():
                outputs = self.global_model(test_data)
                
                if not self.validator.validate_tensor_input(outputs, name="eval_outputs"):
                    raise RuntimeError("Invalid outputs during evaluation")
                
                loss = nn.CrossEntropyLoss()(outputs, test_labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"Invalid loss during evaluation: {loss.item()}")
                
                # Calculate multiple metrics
                pred = outputs.argmax(dim=1)
                correct = pred.eq(test_labels).sum().item()
                total = len(test_labels)
                accuracy = correct / total if total > 0 else 0.0
                
                # Calculate confidence scores
                probs = torch.softmax(outputs, dim=1)
                max_probs, _ = probs.max(dim=1)
                avg_confidence = max_probs.mean().item()
                min_confidence = max_probs.min().item()
                
                # Calculate per-class accuracy
                per_class_correct = {}
                per_class_total = {}
                for i in range(outputs.size(1)):  # num_classes
                    class_mask = test_labels == i
                    if class_mask.any():
                        class_pred = pred[class_mask]
                        class_labels = test_labels[class_mask]
                        per_class_correct[i] = class_pred.eq(class_labels).sum().item()
                        per_class_total[i] = len(class_labels)
                
            self.state = TrainingState.TRAINING
            
            result = {
                "loss": loss.item(),
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "per_class_accuracy": {
                    k: per_class_correct[k] / per_class_total[k] 
                    for k in per_class_correct.keys()
                }
            }
            
            logger.info(f"Global model evaluation: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")
            
            return result
            
        except Exception as e:
            self.state = TrainingState.FAILED
            logger.error(f"Global model evaluation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        
        uptime = time.time() - self.start_time
        health_report = self.health_monitor.get_health_report()
        
        security_alerts = len([a for a in self.security_manager.alerts if not a.resolved])
        blocked_clients = len(self.security_manager.blocked_clients)
        
        return {
            "state": self.state.value,
            "uptime_seconds": uptime,
            "num_clients": self.num_clients,
            "blocked_clients": blocked_clients,
            "security_alerts": security_alerts,
            "health": health_report,
            "client_reputations": dict(self.security_manager.client_reputation),
            "memory_usage_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }


def run_robust_federated_training(
    num_rounds: int = 3, 
    num_clients: int = 5,
    security_level: SecurityLevel = SecurityLevel.MEDIUM
) -> bool:
    """Run robust federated training with comprehensive error handling and monitoring."""
    
    logger.info("🛡️ Starting Generation 2: Robust Federated Training")
    
    try:
        # Initialize robust system
        fed_system = RobustFederatedSystem(
            num_clients=num_clients, 
            security_level=security_level
        )
        
        # Generate validated mock data with error handling
        try:
            from .simple_training import generate_mock_data
            client_data, client_labels = generate_mock_data(batch_size=16, num_clients=num_clients)
            test_data, test_labels = generate_mock_data(batch_size=8, num_clients=1)
            test_data, test_labels = test_data[0], test_labels[0]
            
            # Validate generated data
            for i, (data, labels) in enumerate(zip(client_data, client_labels)):
                if not fed_system.validator.validate_tensor_input(data, name=f"client_{i}_data"):
                    raise ValueError(f"Generated data validation failed for client {i}")
                if not fed_system.validator.validate_tensor_input(labels, name=f"client_{i}_labels"):
                    raise ValueError(f"Generated labels validation failed for client {i}")
            
            logger.info("✅ Data generation and validation completed")
            
        except Exception as e:
            logger.error(f"Data generation failed: {str(e)}")
            return False
        
        # Robust training loop
        for round_idx in range(num_rounds):
            logger.info(f"--- Round {round_idx + 1}/{num_rounds} ---")
            
            round_results = []
            successful_clients = 0
            
            # Each client trains with robust error handling
            for client_id in range(num_clients):
                
                # Skip blocked clients
                if client_id in fed_system.security_manager.blocked_clients:
                    logger.warning(f"Skipping blocked client {client_id}")
                    continue
                
                result = fed_system.client_local_training_robust(
                    client_id=client_id,
                    data=client_data[client_id],
                    labels=client_labels[client_id],
                    epochs=2
                )
                
                if result is not None:
                    round_results.append(result)
                    successful_clients += 1
                    logger.info(f"Client {client_id}: Loss={result['loss']:.4f}, "
                              f"Accuracy={result['accuracy']:.4f}, "
                              f"Time={result['training_time']:.2f}s")
                else:
                    logger.error(f"Client {client_id} training failed completely")
            
            # Check if enough clients succeeded
            if successful_clients < num_clients * 0.5:
                logger.error(f"Too few successful clients: {successful_clients}/{num_clients}")
                fed_system.state = TrainingState.FAILED
                return False
            
            # Robust federated averaging
            if not fed_system.federated_averaging_robust():
                logger.error("Federated averaging failed")
                return False
            
            # Robust evaluation
            eval_results = fed_system.evaluate_global_model_robust(test_data, test_labels)
            if eval_results is None:
                logger.error("Global model evaluation failed")
                return False
            
            logger.info(f"Round {round_idx + 1} Global Model: "
                       f"Loss={eval_results['loss']:.4f}, "
                       f"Accuracy={eval_results['accuracy']:.4f}")
            
            # System health check
            system_status = fed_system.get_system_status()
            logger.info(f"System Health: {system_status['health']['status']} "
                       f"(Score: {system_status['health']['health_score']:.2f})")
            
            if system_status['health']['status'] == 'critical':
                logger.error("System health critical, stopping training")
                return False
        
        # Final system report
        final_status = fed_system.get_system_status()
        logger.info("🎉 Robust federated training completed successfully!")
        logger.info(f"Final System Status: {json.dumps(final_status, indent=2, default=str)}")
        
        # Save robust training report
        report = {
            "training_completed": True,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
            "final_status": final_status,
            "security_level": security_level.value,
            "timestamp": datetime.now().isoformat()
        }
        
        with open('robust_training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("💾 Robust training report saved")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Robust federated training failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_robust_federated_training()
    if success:
        print("✅ Generation 2 robust training completed!")
    else:
        print("❌ Generation 2 robust training failed!")