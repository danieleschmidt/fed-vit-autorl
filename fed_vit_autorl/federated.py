"""
Federated learning components for ViT:
- FederatedViTClient: Local training with top-k gradient sparsification
- FederatedViTServer: FedAvg aggregation with momentum
"""

from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def topk_sparsify(tensor: torch.Tensor, k: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k sparsification: keep only the top k% of gradient values by magnitude.

    Returns:
        values: Non-zero values (top-k)
        indices: Flat indices of kept values
    """
    flat = tensor.view(-1)
    n_keep = max(1, int(len(flat) * k))
    _, top_indices = torch.topk(flat.abs(), n_keep)
    values = flat[top_indices]
    return values, top_indices


def reconstruct_sparse(values: torch.Tensor, indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reconstruct a dense tensor from sparse top-k representation."""
    flat = torch.zeros(shape.numel(), device=values.device, dtype=values.dtype)
    flat[indices] = values
    return flat.view(shape)


class SparseGradient:
    """Container for a sparse gradient update from a client."""

    def __init__(self, param_name: str, values: torch.Tensor, indices: torch.Tensor, shape: torch.Size):
        self.param_name = param_name
        self.values = values
        self.indices = indices
        self.shape = shape

    def to_dense(self) -> torch.Tensor:
        return reconstruct_sparse(self.values, self.indices, self.shape)


class FederatedViTClient:
    """
    Federated learning client that:
    1. Receives global model weights from server
    2. Trains locally for E epochs
    3. Computes gradient delta (local - global)
    4. Applies top-k sparsification before uploading
    """

    def __init__(
        self,
        client_id: int,
        dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        sparsity_k: float = 0.1,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.device = torch.device(device)
        self.sparsity_k = sparsity_k
        self.model: Optional[nn.Module] = None

    def set_model(self, model: nn.Module) -> None:
        """Load global model (deep copy to avoid aliasing)."""
        self.model = deepcopy(model).to(self.device)

    def train(
        self,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 3,
    ) -> Dict[str, float]:
        """
        Train locally and return metrics.

        Returns:
            dict with 'loss' and 'accuracy'
        """
        if self.model is None:
            raise RuntimeError("Call set_model() before train()")

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        n_batches = max(1, total)
        return {
            "loss": total_loss / n_batches,
            "accuracy": correct / n_batches,
        }

    def get_sparse_updates(self, global_model: nn.Module) -> List[SparseGradient]:
        """
        Compute delta = local_weights - global_weights, then sparsify.

        Returns list of SparseGradient objects (one per parameter).
        """
        if self.model is None:
            raise RuntimeError("Call train() before get_sparse_updates()")

        updates = []
        for (name, local_param), (_, global_param) in zip(
            self.model.named_parameters(), global_model.named_parameters()
        ):
            delta = (local_param.data - global_param.data.to(self.device)).cpu()
            values, indices = topk_sparsify(delta, self.sparsity_k)
            updates.append(SparseGradient(name, values, indices, delta.shape))

        return updates

    def evaluate(self, test_dataset: torch.utils.data.Dataset, batch_size: int = 64) -> float:
        """Evaluate local model on a test dataset. Returns accuracy."""
        if self.model is None:
            raise RuntimeError("No model to evaluate")

        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return correct / max(1, total)


class FederatedViTServer:
    """
    Federated server that:
    1. Distributes global model to clients
    2. Collects sparse gradient updates
    3. Aggregates via FedAvg with optional server-side momentum
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        momentum: float = 0.9,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.momentum = momentum
        # Server-side momentum buffers
        self._velocity: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
        }
        self.round = 0

    def get_global_model(self) -> nn.Module:
        """Return current global model (clients should deepcopy this)."""
        return self.model

    def aggregate(
        self,
        client_updates: List[List[SparseGradient]],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """
        FedAvg aggregation with server momentum.

        Args:
            client_updates: List of sparse gradient update lists (one per client)
            client_weights: Relative weights for each client (e.g., dataset sizes).
                            Defaults to uniform weighting.
        """
        n_clients = len(client_updates)
        if n_clients == 0:
            return

        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        # Accumulate weighted dense deltas per parameter
        param_names = [name for name, _ in self.model.named_parameters()]
        avg_deltas: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data, device="cpu")
            for name, param in self.model.named_parameters()
        }

        for updates, weight in zip(client_updates, client_weights):
            for sg in updates:
                avg_deltas[sg.param_name] += weight * sg.to_dense()

        # Apply with momentum
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                delta = avg_deltas[name].to(self.device)
                if self.momentum > 0:
                    self._velocity[name] = self.momentum * self._velocity[name] + delta
                    param.data += self._velocity[name]
                else:
                    param.data += delta

        self.round += 1

    def evaluate(self, test_dataset: torch.utils.data.Dataset, batch_size: int = 64) -> float:
        """Evaluate global model on test dataset. Returns accuracy."""
        from torch.utils.data import DataLoader
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)

        return correct / max(1, total)
