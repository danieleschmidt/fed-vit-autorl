"""Tests for FederatedViTClient and FederatedViTServer."""

import copy
import pytest
import torch
from fed_vit_autorl.models import TinyViT
from fed_vit_autorl.data import SyntheticImageDataset
from fed_vit_autorl.federated import (
    FederatedViTClient,
    FederatedViTServer,
    topk_sparsify,
    reconstruct_sparse,
    SparseGradient,
)


# --- Sparse gradient utilities ---

class TestTopkSparsify:
    def test_keeps_k_percent(self):
        t = torch.randn(100)
        values, indices = topk_sparsify(t, k=0.1)
        assert len(values) == 10

    def test_top_values_are_largest_magnitude(self):
        t = torch.zeros(100)
        t[0] = 10.0  # largest by far
        values, indices = topk_sparsify(t, k=0.1)
        assert 0 in indices

    def test_reconstruct_roundtrip(self):
        t = torch.randn(50)
        values, indices = topk_sparsify(t, k=0.2)
        reconstructed = reconstruct_sparse(values, indices, t.shape)
        assert reconstructed.shape == t.shape
        # Top-k values should be preserved
        for val, idx in zip(values, indices):
            assert abs(reconstructed[idx].item() - val.item()) < 1e-5


# --- FederatedViTClient ---

@pytest.fixture
def small_model():
    return TinyViT(img_size=32, patch_size=8, num_classes=25, embed_dim=32, depth=1, num_heads=2)


@pytest.fixture
def small_dataset():
    return SyntheticImageDataset(num_samples=30, seed=42)


class TestFederatedViTClient:
    def test_set_model_and_train(self, small_model, small_dataset):
        client = FederatedViTClient(client_id=0, dataset=small_dataset)
        client.set_model(small_model)
        metrics = client.train(lr=1e-3, batch_size=16, epochs=1)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["loss"] >= 0.0

    def test_train_without_model_raises(self, small_dataset):
        client = FederatedViTClient(client_id=0, dataset=small_dataset)
        with pytest.raises(RuntimeError):
            client.train()

    def test_sparse_updates_shape(self, small_model, small_dataset):
        client = FederatedViTClient(client_id=0, dataset=small_dataset, sparsity_k=0.1)
        client.set_model(small_model)
        client.train(lr=1e-3, batch_size=16, epochs=1)
        updates = client.get_sparse_updates(small_model)
        # One SparseGradient per parameter
        param_names = [n for n, _ in small_model.named_parameters()]
        assert len(updates) == len(param_names)

    def test_sparse_updates_names_match(self, small_model, small_dataset):
        client = FederatedViTClient(client_id=0, dataset=small_dataset, sparsity_k=0.1)
        client.set_model(small_model)
        client.train(lr=1e-3, batch_size=16, epochs=1)
        updates = client.get_sparse_updates(small_model)
        param_names = {n for n, _ in small_model.named_parameters()}
        update_names = {u.param_name for u in updates}
        assert update_names == param_names

    def test_model_is_deep_copied(self, small_model, small_dataset):
        """Training should not modify the original model."""
        original_params = {n: p.clone() for n, p in small_model.named_parameters()}
        client = FederatedViTClient(client_id=0, dataset=small_dataset)
        client.set_model(small_model)
        client.train(lr=1e-2, batch_size=16, epochs=2)
        for name, param in small_model.named_parameters():
            assert torch.allclose(param, original_params[name]), \
                f"Parameter {name} was modified in original model"

    def test_evaluate(self, small_model, small_dataset):
        client = FederatedViTClient(client_id=0, dataset=small_dataset)
        client.set_model(small_model)
        acc = client.evaluate(small_dataset)
        assert 0.0 <= acc <= 1.0


# --- FederatedViTServer ---

class TestFederatedViTServer:
    def test_init(self, small_model):
        server = FederatedViTServer(small_model)
        assert server.round == 0

    def test_get_global_model(self, small_model):
        server = FederatedViTServer(small_model)
        m = server.get_global_model()
        assert m is not None

    def test_aggregate_increments_round(self, small_model, small_dataset):
        server = FederatedViTServer(small_model)
        clients = [FederatedViTClient(i, small_dataset) for i in range(2)]

        global_model = server.get_global_model()
        all_updates = []
        for c in clients:
            c.set_model(global_model)
            c.train(lr=1e-3, batch_size=16, epochs=1)
            all_updates.append(c.get_sparse_updates(global_model))

        server.aggregate(all_updates)
        assert server.round == 1

    def test_aggregate_changes_weights(self, small_model, small_dataset):
        """After aggregation, global model weights should differ from initial."""
        server = FederatedViTServer(small_model, momentum=0.0)
        initial_params = {n: p.clone() for n, p in server.model.named_parameters()}

        global_model = server.get_global_model()
        clients = [FederatedViTClient(0, small_dataset)]
        clients[0].set_model(global_model)
        clients[0].train(lr=1e-2, batch_size=16, epochs=2)
        updates = [clients[0].get_sparse_updates(global_model)]
        server.aggregate(updates)

        changed = any(
            not torch.allclose(p, initial_params[n])
            for n, p in server.model.named_parameters()
        )
        assert changed, "Server weights should change after aggregation"

    def test_evaluate(self, small_model, small_dataset):
        server = FederatedViTServer(small_model)
        acc = server.evaluate(small_dataset)
        assert 0.0 <= acc <= 1.0

    def test_empty_aggregate(self, small_model):
        """Aggregating empty updates should not raise."""
        server = FederatedViTServer(small_model)
        server.aggregate([])  # should not raise
        assert server.round == 0  # round not incremented for empty
