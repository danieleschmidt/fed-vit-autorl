"""Fed-ViT-AutoRL: Federated Vision Transformers for Autonomous Driving.

A federated reinforcement learning framework where edge vehicles jointly
fine-tune Vision Transformer based perception stacks while respecting
latency and privacy constraints.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0-dev"

# Conditional imports for optional dependencies
try:
    from .models import ViTPerception
    from .federated.client import FederatedClient, VehicleClient
    from .federated.server import FederatedServer
    from .federated.aggregation import FedAvgAggregator
    from .reinforcement.ppo_federated import FederatedPPO
    from .edge.optimization import EdgeOptimizer
    from .simulation.carla_env import CARLAFederatedEnv
    _TORCH_AVAILABLE = True
except ImportError as e:
    # Core dependencies not available
    _TORCH_AVAILABLE = False
    import warnings
    warnings.warn(f"Some components unavailable due to missing dependencies: {e}")
    
    # Define placeholder classes
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError("This component requires additional dependencies. Install with: pip install fed-vit-autorl[full]")
    
    ViTPerception = _MissingDependency
    FederatedClient = _MissingDependency
    VehicleClient = _MissingDependency
    FederatedServer = _MissingDependency
    FedAvgAggregator = _MissingDependency
    FederatedPPO = _MissingDependency
    EdgeOptimizer = _MissingDependency
    CARLAFederatedEnv = _MissingDependency

# Main API class
if _TORCH_AVAILABLE:
    class FederatedVehicleRL:
        """Main API for Federated Vehicle Reinforcement Learning."""
        
        def __init__(self, model, num_vehicles=100, aggregation="fedavg", 
                     privacy_mechanism="differential_privacy", epsilon=1.0):
            self.model = model
            self.num_vehicles = num_vehicles
            self.aggregation = aggregation
            self.privacy_mechanism = privacy_mechanism
            self.epsilon = epsilon
            
            # Initialize federated server
            self.server = FederatedServer(
                global_model=model,
                num_clients=num_vehicles,
                aggregation_method=aggregation
            )
            
            # Store client models
            self.clients = {}
        
        def get_vehicle_model(self, vehicle_id):
            """Get model for specific vehicle."""
            if vehicle_id not in self.clients:
                import torch.optim as optim
                import copy
                model_copy = copy.deepcopy(self.model)
                optimizer = optim.Adam(model_copy.parameters(), lr=1e-4)
                
                self.clients[vehicle_id] = VehicleClient(
                    vehicle_id=str(vehicle_id),
                    model=model_copy,
                    optimizer=optimizer,
                    privacy_budget=self.epsilon
                )
            return self.clients[vehicle_id]
        
        def aggregate_updates(self, local_updates, weighted_by="uniform"):
            """Aggregate updates from vehicles."""
            # Process updates for server
            processed_updates = []
            for update in local_updates:
                if hasattr(update, 'get_model_update'):
                    processed_updates.append(update.get_model_update())
                else:
                    processed_updates.append(update)
            
            # Perform aggregation
            global_update = self.server.aggregate_updates(processed_updates)
            
            # Update all client models
            for client in self.clients.values():
                client.set_global_model(global_update)
            
            return global_update
        
        def evaluate_global_model(self, test_scenarios):
            """Evaluate global model performance."""
            return {
                'mAP': 0.85,  # Placeholder
                'latency': 45.0  # Placeholder
            }
else:
    FederatedVehicleRL = _MissingDependency

__all__ = [
    "__version__",
    "ViTPerception", 
    "FederatedClient",
    "VehicleClient",
    "FederatedServer",
    "FedAvgAggregator",
    "FederatedPPO",
    "EdgeOptimizer",
    "CARLAFederatedEnv",
    "FederatedVehicleRL"
]