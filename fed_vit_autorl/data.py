"""
SyntheticImageDataset: Generates synthetic image classification data.

Classes: 5 shapes × 3 colors = 15 classes
  Shapes: circle, square, triangle, diamond, cross
  Colors: red, green, blue, yellow, purple

Images: 32×32 RGB, normalized to [0,1]
"""

import math
import random
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset


# Color definitions: (R, G, B) in [0, 255]
COLORS = {
    "red":    (220, 50,  50),
    "green":  (50,  200, 50),
    "blue":   (50,  50,  220),
    "yellow": (220, 200, 50),
    "purple": (150, 50,  200),
}

SHAPES = ["circle", "square", "triangle", "diamond", "cross"]
COLOR_NAMES = list(COLORS.keys())

# Class label = shape_idx * num_colors + color_idx
NUM_CLASSES = len(SHAPES) * len(COLOR_NAMES)  # 25 (5 shapes × 5 colors)


def _class_label(shape: str, color: str) -> int:
    return SHAPES.index(shape) * len(COLOR_NAMES) + COLOR_NAMES.index(color)


def _draw_shape(
    size: int,
    shape: str,
    color: Tuple[int, int, int],
    bg_noise: float = 0.05,
    torch_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Draw a single shape on a canvas of given size.
    Returns: float tensor (3, size, size) in [0, 1]
    """
    gen = torch.Generator()
    if torch_seed is not None:
        gen.manual_seed(torch_seed)

    img = torch.zeros(3, size, size)

    # Add slight background noise
    img = img + torch.randn(3, size, size, generator=gen) * bg_noise

    cr, cg, cb = [c / 255.0 for c in color]
    cx = size / 2.0
    cy = size / 2.0
    # Shape occupies ~50-70% of canvas
    radius = size * random.uniform(0.22, 0.35)

    ys, xs = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )

    if shape == "circle":
        mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (radius ** 2)

    elif shape == "square":
        mask = ((xs - cx).abs() <= radius) & ((ys - cy).abs() <= radius)

    elif shape == "triangle":
        # Equilateral triangle pointing up
        top_y = cy - radius
        base_y = cy + radius / 2.0
        half_base = radius * math.sqrt(3) / 2.0

        in_height = (ys >= top_y) & (ys <= base_y)
        t = (ys - top_y) / (base_y - top_y + 1e-8)
        half_w = t * half_base
        in_width = (xs - cx).abs() <= half_w
        mask = in_height & in_width

    elif shape == "diamond":
        mask = ((xs - cx).abs() + (ys - cy).abs()) <= radius

    elif shape == "cross":
        arm = radius * 0.35
        horiz = ((xs - cx).abs() <= radius) & ((ys - cy).abs() <= arm)
        vert = ((ys - cy).abs() <= radius) & ((xs - cx).abs() <= arm)
        mask = horiz | vert

    else:
        raise ValueError(f"Unknown shape: {shape}")

    # Fill shape with color + small texture noise
    texture = torch.randn(3, size, size, generator=gen) * 0.03
    img[0][mask] = cr + texture[0][mask]
    img[1][mask] = cg + texture[1][mask]
    img[2][mask] = cb + texture[2][mask]

    return img.clamp(0.0, 1.0)


class SyntheticImageDataset(Dataset):
    """
    Synthetic dataset of colored geometric shapes.

    Args:
        num_samples: Total number of samples
        img_size: Image size (square)
        seed: Random seed for reproducibility
        client_id: If set, shard dataset for federated learning (0-indexed)
        num_clients: Total number of clients for sharding
    """

    def __init__(
        self,
        num_samples: int = 1500,
        img_size: int = 32,
        seed: int = 42,
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None,
    ):
        self.img_size = img_size
        rng = random.Random(seed)

        # Generate balanced labels
        per_class = max(1, num_samples // NUM_CLASSES)
        labels = []
        for s_idx, shape in enumerate(SHAPES):
            for c_idx, color in enumerate(COLOR_NAMES):
                labels.extend([_class_label(shape, color)] * per_class)
        # Trim / pad to num_samples
        rng.shuffle(labels)
        labels = labels[:num_samples]

        self.metadata = []
        for lbl in labels:
            shape = SHAPES[lbl // len(COLOR_NAMES)]
            color = COLOR_NAMES[lbl % len(COLOR_NAMES)]
            self.metadata.append((shape, color, lbl))

        # Shard for federated clients
        if client_id is not None and num_clients is not None:
            self.metadata = [m for i, m in enumerate(self.metadata) if i % num_clients == client_id]

        self._seed = seed

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        shape, color, label = self.metadata[idx]
        # Seed per-sample for fully deterministic generation
        sample_seed = self._seed + idx * 137
        rng_state = random.getstate()
        random.seed(sample_seed)
        img = _draw_shape(self.img_size, shape, COLORS[color], torch_seed=sample_seed)
        random.setstate(rng_state)
        return img, label

    @staticmethod
    def num_classes() -> int:
        return NUM_CLASSES

    @staticmethod
    def class_names() -> list:
        return [f"{s}_{c}" for s in SHAPES for c in COLOR_NAMES]
