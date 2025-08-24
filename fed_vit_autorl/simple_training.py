"""Generation 1: Simple training implementation without complex imports.

This module provides a working baseline implementation that bypasses
import issues while demonstrating core federated learning concepts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SimpleViTPerception(nn.Module):
    """Simplified ViT-like perception model for Generation 1."""
    
    def __init__(self, embed_dim: int = 768, num_classes: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Simple CNN backbone to simulate ViT patches
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=16, stride=16),  # 384->24 patches
            nn.Flatten(2),  # (B, 64, 576)
            nn.Linear(576, embed_dim)
        )
        
        # Simple transformer-like layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Patch embedding: (B, 3, 384, 384) -> (B, 576, 768)
        x = self.patch_embed(x)
        
        # Simple transformer block
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        # Global average pooling and classification
        features = x.mean(dim=1)  # (B, 768)
        return self.classifier(features)
    
    def get_features(self, x):
        """Extract features without classification."""
        x = self.patch_embed(x)
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x.mean(dim=1)


class SimpleFederatedSystem:
    """Simple federated learning system for Generation 1."""
    
    def __init__(self, num_clients: int = 5, embed_dim: int = 768):
        self.num_clients = num_clients
        self.embed_dim = embed_dim
        
        # Global model
        self.global_model = SimpleViTPerception(embed_dim=embed_dim)
        
        # Client models (copies of global model)
        self.client_models = []
        self.client_optimizers = []
        
        for i in range(num_clients):
            client_model = SimpleViTPerception(embed_dim=embed_dim)
            client_model.load_state_dict(self.global_model.state_dict())
            client_optimizer = optim.Adam(client_model.parameters(), lr=1e-4)
            
            self.client_models.append(client_model)
            self.client_optimizers.append(client_optimizer)
        
        logger.info(f"Initialized simple federated system with {num_clients} clients")
    
    def client_local_training(
        self, 
        client_id: int, 
        data: torch.Tensor, 
        labels: torch.Tensor, 
        epochs: int = 1
    ) -> Dict[str, float]:
        """Train a specific client locally."""
        client_model = self.client_models[client_id]
        optimizer = self.client_optimizers[client_id]
        criterion = nn.CrossEntropyLoss()
        
        client_model.train()
        total_loss = 0.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = client_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / epochs
        return {"loss": avg_loss, "client_id": client_id}
    
    def federated_averaging(self):
        """Simple federated averaging of client models."""
        global_dict = self.global_model.state_dict()
        
        # Initialize aggregated parameters
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Average client parameters
        for client_model in self.client_models:
            client_dict = client_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] += client_dict[key] / self.num_clients
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        # Update all client models with new global model
        for client_model in self.client_models:
            client_model.load_state_dict(global_dict)
        
        logger.info("Completed federated averaging")
    
    def evaluate_global_model(self, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        
        with torch.no_grad():
            outputs = self.global_model(test_data)
            loss = nn.CrossEntropyLoss()(outputs, test_labels)
            
            # Simple accuracy calculation
            pred = outputs.argmax(dim=1)
            correct = pred.eq(test_labels).sum().item()
            accuracy = correct / len(test_labels)
        
        return {"loss": loss.item(), "accuracy": accuracy}


def generate_mock_data(batch_size: int = 16, num_clients: int = 5) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate mock training data for each client."""
    client_data = []
    client_labels = []
    
    for i in range(num_clients):
        # Each client gets slightly different data distribution
        base_mean = 0.1 * i  # Different means for different clients
        
        data = torch.randn(batch_size, 3, 384, 384) * 0.5 + base_mean
        # Simple labels: 0=stop, 1=go, 2=left, 3=right
        labels = torch.randint(0, 4, (batch_size,))
        
        client_data.append(data)
        client_labels.append(labels)
    
    return client_data, client_labels


def run_simple_federated_training(num_rounds: int = 3, num_clients: int = 5) -> bool:
    """Run simple federated training demonstration."""
    logger.info("🚀 Starting Generation 1 Simple Federated Training")
    
    try:
        # Initialize system
        fed_system = SimpleFederatedSystem(num_clients=num_clients)
        
        # Generate mock data
        client_data, client_labels = generate_mock_data(batch_size=16, num_clients=num_clients)
        test_data, test_labels = generate_mock_data(batch_size=8, num_clients=1)
        test_data, test_labels = test_data[0], test_labels[0]
        
        # Training loop
        for round_idx in range(num_rounds):
            logger.info(f"--- Round {round_idx + 1}/{num_rounds} ---")
            
            round_losses = []
            
            # Each client trains locally
            for client_id in range(num_clients):
                result = fed_system.client_local_training(
                    client_id=client_id,
                    data=client_data[client_id],
                    labels=client_labels[client_id],
                    epochs=2
                )
                round_losses.append(result["loss"])
                logger.info(f"Client {client_id}: Loss = {result['loss']:.4f}")
            
            # Federated averaging
            fed_system.federated_averaging()
            
            # Evaluate global model
            eval_results = fed_system.evaluate_global_model(test_data, test_labels)
            logger.info(f"Global Model - Loss: {eval_results['loss']:.4f}, Accuracy: {eval_results['accuracy']:.4f}")
        
        logger.info("🎉 Simple federated training completed successfully!")
        
        # Final evaluation
        final_results = fed_system.evaluate_global_model(test_data, test_labels)
        logger.info(f"🏆 Final Results - Loss: {final_results['loss']:.4f}, Accuracy: {final_results['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple federated training failed: {str(e)}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_simple_federated_training()
    if success:
        print("✅ Generation 1 simple training completed!")
    else:
        print("❌ Generation 1 simple training failed!")