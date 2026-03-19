"""
TinyViT: A small Vision Transformer for federated learning experiments.

Architecture:
- Patch embedding (8x8 or 16x16 patches → 64-dim tokens)
- Positional embeddings
- 2-3 transformer blocks (multi-head self-attention + MLP)
- CLS token classification head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Converts image into sequence of patch tokens."""

    def __init__(self, img_size: int = 32, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """
    Tiny Vision Transformer suitable for federated learning.

    Parameters:
        img_size: Input image size (square)
        patch_size: Patch size (8 or 16)
        in_channels: Number of input channels
        num_classes: Number of output classes
        embed_dim: Token embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: Dropout rate
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 25,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                           # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                    # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)                                # (B, N+1, D)
        x = self.norm(x)

        cls_out = x[:, 0]                                 # (B, D) — CLS token
        return self.head(cls_out)                         # (B, num_classes)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
