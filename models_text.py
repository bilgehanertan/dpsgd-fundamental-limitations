# models_text.py
from __future__ import annotations
from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp

TX_PRESETS = {
    "tiny_128": dict(
        max_len=128, vocab_size=30000, d_model=256, n_heads=4, n_layers=4, d_ff=1024
    ),
    "small_128": dict(
        max_len=128, vocab_size=30000, d_model=512, n_heads=8, n_layers=6, d_ff=2048
    ),
    "base_256": dict(
        max_len=256, vocab_size=30000, d_model=768, n_heads=12, n_layers=12, d_ff=3072
    ),
}


@dataclass
class TransformerConfig:
    vocab_size: int
    num_classes: int = 4
    max_len: int = 256
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.0  # keep 0 for DP
    norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    """RMSNorm (no mean subtraction). DP-friendly (no batch stats)."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        d = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (d,))
        # rms over last dim
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale


class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) * silu(xW2) then project back."""

    d_ff: int
    d_model: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        w1 = nn.Dense(self.d_ff, use_bias=True, name="w1")
        w2 = nn.Dense(self.d_ff, use_bias=True, name="w2")
        w3 = nn.Dense(self.d_model, use_bias=True, name="w3")

        y = w1(x) * nn.silu(w2(x))
        if self.dropout > 0.0:
            y = nn.Dropout(self.dropout)(y, deterministic=not train)
        y = w3(y)
        return y


class TransformerBlock(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, attn_mask: jnp.ndarray, deterministic: bool
    ) -> jnp.ndarray:
        # Pre-norm (RMSNorm)
        y = RMSNorm(eps=self.cfg.norm_eps, name="rms1")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.n_heads,
            qkv_features=self.cfg.d_model,
            dropout_rate=self.cfg.dropout,
            name="mha",
        )(y, y, mask=attn_mask, deterministic=deterministic)
        if self.cfg.dropout > 0.0:
            y = nn.Dropout(self.cfg.dropout)(y, deterministic=deterministic)
        x = x + y

        y = RMSNorm(eps=self.cfg.norm_eps, name="rms2")(x)
        y = SwiGLU(
            self.cfg.d_ff, self.cfg.d_model, dropout=self.cfg.dropout, name="ffn"
        )(y, train=not deterministic)
        if self.cfg.dropout > 0.0:
            y = nn.Dropout(self.cfg.dropout)(y, deterministic=deterministic)
        x = x + y
        return x


class TransformerClassifier(nn.Module):
    cfg: TransformerConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, attention_mask: jnp.ndarray, train: bool):
        """
        tokens: [B, T] int32
        attention_mask: [B, T] 1 for real tokens, 0 for padding
        """
        deterministic = not train

        B, T = tokens.shape
        assert T == self.cfg.max_len, f"Expected max_len={self.cfg.max_len}, got {T}"

        tok_emb = nn.Embed(self.cfg.vocab_size, self.cfg.d_model, name="tok_emb")(
            tokens
        )
        pos = jnp.arange(T)[None, :]  # [1, T]
        pos_emb = nn.Embed(self.cfg.max_len, self.cfg.d_model, name="pos_emb")(pos)
        x = tok_emb + pos_emb

        if self.cfg.dropout > 0.0:
            x = nn.Dropout(self.cfg.dropout)(x, deterministic=not train)

        # attention mask: [B,T] -> [B,1,1,T] boolean keep-mask
        attn_mask = attention_mask[:, None, None, :].astype(bool)

        def block_apply(module, x, attn_mask, deterministic):
            return module(x, attn_mask=attn_mask, deterministic=deterministic)

        block_apply_ckpt = nn.remat(block_apply, static_argnums=(3,))
        for i in range(self.cfg.n_layers):
            blk = TransformerBlock(self.cfg, name=f"block_{i}")
            x = block_apply_ckpt(blk, x, attn_mask, deterministic)

        x = RMSNorm(eps=self.cfg.norm_eps, name="rms_final")(x)

        # CLS pooling: token 0
        cls = x[:, 0, :]
        logits = nn.Dense(self.cfg.num_classes, name="head")(cls)
        return logits
