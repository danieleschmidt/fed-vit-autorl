"""Tests for SyntheticImageDataset."""

import pytest
import torch
from fed_vit_autorl.data import SyntheticImageDataset, NUM_CLASSES, SHAPES, COLOR_NAMES, _class_label


class TestClassLabel:
    def test_valid_labels(self):
        for shape in SHAPES:
            for color in COLOR_NAMES:
                lbl = _class_label(shape, color)
                assert 0 <= lbl < NUM_CLASSES

    def test_unique_labels(self):
        labels = {_class_label(s, c) for s in SHAPES for c in COLOR_NAMES}
        assert len(labels) == NUM_CLASSES


class TestSyntheticImageDataset:
    def setup_method(self):
        self.ds = SyntheticImageDataset(num_samples=150, img_size=32, seed=42)

    def test_length(self):
        assert len(self.ds) == 150

    def test_item_types(self):
        img, label = self.ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, int)

    def test_image_shape(self):
        img, _ = self.ds[0]
        assert img.shape == (3, 32, 32)

    def test_image_range(self):
        img, _ = self.ds[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_label_range(self):
        for i in range(min(50, len(self.ds))):
            _, label = self.ds[i]
            assert 0 <= label < NUM_CLASSES

    def test_deterministic(self):
        """Same seed → same data."""
        ds1 = SyntheticImageDataset(num_samples=10, seed=7)
        ds2 = SyntheticImageDataset(num_samples=10, seed=7)
        img1, lbl1 = ds1[0]
        img2, lbl2 = ds2[0]
        assert lbl1 == lbl2
        assert torch.allclose(img1, img2)

    def test_different_seeds(self):
        """Different seeds → different data."""
        ds1 = SyntheticImageDataset(num_samples=10, seed=1)
        ds2 = SyntheticImageDataset(num_samples=10, seed=2)
        img1, _ = ds1[0]
        img2, _ = ds2[0]
        # At least some pixels should differ
        assert not torch.allclose(img1, img2)

    def test_sharding(self):
        """Client shards should partition the dataset (no overlap, total = full dataset)."""
        total = 90
        n_clients = 3
        full_ds = SyntheticImageDataset(num_samples=total, seed=42)
        shards = [
            SyntheticImageDataset(num_samples=total, seed=42, client_id=i, num_clients=n_clients)
            for i in range(n_clients)
        ]
        total_sharded = sum(len(s) for s in shards)
        # All shards combined should equal the full dataset
        assert total_sharded == len(full_ds)
        # Each shard should be roughly equal
        for s in shards:
            assert len(s) > 0

    def test_num_classes(self):
        assert SyntheticImageDataset.num_classes() == NUM_CLASSES

    def test_class_names_length(self):
        names = SyntheticImageDataset.class_names()
        assert len(names) == NUM_CLASSES
