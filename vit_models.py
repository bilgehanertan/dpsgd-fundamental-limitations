# vit_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import flax.linen as nn
import jax.numpy as jnp


@dataclass
class ViTConfig:
    num_classes: int
    image_size: int = 32  # CIFAR/SVHN = 32
    patch_size: int = 4  # 32/4 = 8 -> 64 patches
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    layer_norm_eps: float = 1e-6


def _pair(x: int) -> tuple[int, int]:
    return (x, x)


class MLPBlock(nn.Module):
    d_model: int
    d_ff: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Dense(self.d_ff, use_bias=True, name="fc1")(x)
        x = nn.gelu(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        x = nn.Dense(self.d_model, use_bias=True, name="fc2")(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        return x


class TransformerEncoderBlock(nn.Module):
    cfg: ViTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        # Pre-LN attention
        y = nn.LayerNorm(epsilon=self.cfg.layer_norm_eps, name="ln1")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.n_heads,
            qkv_features=self.cfg.d_model,
            dropout_rate=self.cfg.attn_dropout,
            name="mha",
        )(y, y, deterministic=not train)
        if self.cfg.dropout > 0.0:
            y = nn.Dropout(rate=self.cfg.dropout)(y, deterministic=not train)
        x = x + y

        # Pre-LN MLP
        y = nn.LayerNorm(epsilon=self.cfg.layer_norm_eps, name="ln2")(x)
        d_ff = int(self.cfg.d_model * self.cfg.mlp_ratio)
        y = MLPBlock(
            d_model=self.cfg.d_model,
            d_ff=d_ff,
            dropout=self.cfg.dropout,
            name="mlp",
        )(y, train=train)
        x = x + y
        return x


class ViTClassifier(nn.Module):
    """
    Minimal ViT for CIFAR/SVHN-style 32x32 images.
    NHWC input expected: [B, H, W, C]
    """

    cfg: ViTConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if x.ndim != 4:
            raise ValueError(f"expected NHWC input [B,H,W,C], got shape {x.shape}")

        B, H, W, C = x.shape
        if H != self.cfg.image_size or W != self.cfg.image_size:
            raise ValueError(
                f"Expected image_size={self.cfg.image_size}, got H={H}, W={W}"
            )
        if self.cfg.image_size % self.cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        ph = pw = self.cfg.patch_size
        gh = gw = self.cfg.image_size // self.cfg.patch_size
        n_patches = gh * gw

        # Patch embedding via Conv with stride=patch_size
        # output: [B, gh, gw, d_model]
        x = nn.Conv(
            features=self.cfg.d_model,
            kernel_size=_pair(self.cfg.patch_size),
            strides=_pair(self.cfg.patch_size),
            padding="VALID",
            use_bias=True,
            name="patch_embed",
        )(x)

        # flatten patches: [B, n_patches, d_model]
        x = x.reshape((B, n_patches, self.cfg.d_model))

        # CLS token: [B, 1, d_model]
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, self.cfg.d_model))
        cls = jnp.tile(cls, (B, 1, 1))

        # Concat: [B, 1 + n_patches, d_model]
        x = jnp.concatenate([cls, x], axis=1)

        # Positional embedding: [1, 1+n_patches, d_model]
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (1, 1 + n_patches, self.cfg.d_model),
        )
        x = x + pos

        if self.cfg.dropout > 0.0:
            x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=not train)

        # Encoder
        for i in range(self.cfg.n_layers):
            x = TransformerEncoderBlock(self.cfg, name=f"block_{i}")(x, train=train)

        x = nn.LayerNorm(epsilon=self.cfg.layer_norm_eps, name="ln_final")(x)

        # CLS pooling
        cls_out = x[:, 0, :]  # [B, d_model]
        logits = nn.Dense(self.cfg.num_classes, use_bias=True, name="head")(cls_out)
        return logits


VIT_PRESETS: Dict[str, Dict[str, int | float]] = {
    "vit_tiny_cifar": dict(
        patch_size=4, d_model=192, n_heads=3, n_layers=12, mlp_ratio=4.0
    ),
    "vit_small_cifar": dict(
        patch_size=4, d_model=384, n_heads=6, n_layers=12, mlp_ratio=4.0
    ),
    "vit_base_cifar": dict(
        patch_size=4, d_model=768, n_heads=12, n_layers=12, mlp_ratio=4.0
    ),
}
