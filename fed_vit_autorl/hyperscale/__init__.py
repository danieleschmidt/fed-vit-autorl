"""Hyperscale federated learning capabilities for massive deployment."""

from typing import Optional

try:
    from .global_orchestrator import GlobalOrchestrator
    from .mega_scale_aggregator import MegaScaleAggregator
    from .planetary_federation import PlanetaryFederation
    from .quantum_secure_aggregation import QuantumSecureAggregator
    from .autonomous_model_evolution import AutonomousModelEvolution
    _HYPERSCALE_AVAILABLE = True
except ImportError:
    _HYPERSCALE_AVAILABLE = False
    
    class _MissingHyperscale:
        def __init__(self, *args, **kwargs):
            raise ImportError("Hyperscale components require advanced dependencies")
    
    GlobalOrchestrator = _MissingHyperscale
    MegaScaleAggregator = _MissingHyperscale
    PlanetaryFederation = _MissingHyperscale
    QuantumSecureAggregator = _MissingHyperscale
    AutonomousModelEvolution = _MissingHyperscale

__all__ = [
    "GlobalOrchestrator",
    "MegaScaleAggregator", 
    "PlanetaryFederation",
    "QuantumSecureAggregator",
    "AutonomousModelEvolution",
]