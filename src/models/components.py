"""
Core Architectural Components for Custom BERT & GPT-2.

Implements the three required modern components:
1. Rotary Positional Embeddings (RoPE) — Su et al., 2021
2. Grouped Query Attention (GQA) — Ainslie et al., 2023
3. RMSNorm — Zhang and Sennrich, 2019

These replace standard positional embeddings, multi-head attention,
and LayerNorm in our custom Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Instead of adding positional information to token embeddings,
    RoPE encodes position by rotating query and key vectors.
    This naturally captures relative positions through the dot product.

    Key advantages over sinusoidal/learned positional embeddings:
    - Naturally encodes relative position information
    - Decays attention with distance (desirable inductive bias)
    - Generalizes better to unseen sequence lengths

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", 2021.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            dim: Head dimension (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for the frequency computation
        """
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even dimension, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the frequency bands
        # theta_i = base^(-2i/d) for i = 0, 1, ..., d/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        """Precompute cos/sin tables for efficiency."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for pairs: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.

        Given x = [x1, x2, x3, x4, ...], returns [-x_{d/2+1}, ..., x1, x2, ...]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            offset: Position offset (for incremental decoding)

        Returns:
            q_rotated, k_rotated: Same shapes as input
        """
        seq_len = q.shape[2]

        # Extend cache if needed
        if offset + seq_len > self.max_seq_len:
            self._precompute_cache(offset + seq_len)

        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike standard LayerNorm, RMSNorm does not center the activations
    (no mean subtraction). It only rescales by the RMS value.

    Advantages:
    - Simpler and ~10-15% faster than LayerNorm
    - Comparable or better performance in practice
    - Used in LLaMA, Gemma, and other modern LLMs

    Reference: Zhang and Sennrich, "Root Mean Square Layer Normalization", 2019.

    Formula: y = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Feature dimension to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma (learnable scale)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) — input tensor

        Returns:
            Normalized tensor of same shape
        """
        # Cast to float32 for numerical stability, then back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    GQA is a middle ground between Multi-Head Attention (MHA) and
    Multi-Query Attention (MQA):
    - MHA: Each head has its own Q, K, V projections
    - MQA: All heads share a single K, V projection
    - GQA: Groups of Q heads share K, V projections

    Example with 12 Q-heads and 4 KV-groups:
    - Q heads [0,1,2] share KV group 0
    - Q heads [3,4,5] share KV group 1
    - Q heads [6,7,8] share KV group 2
    - Q heads [9,10,11] share KV group 3

    Benefits:
    - Significantly reduces KV cache memory (important for inference)
    - Minimal quality loss compared to full MHA
    - 3x fewer KV parameters than MHA in this config

    Reference: Ainslie et al., "GQA: Training Generalized Multi-Query
    Transformer Models from Multi-Head Checkpoints", 2023.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int = 12,
        num_kv_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            num_q_heads: Number of query heads
            num_kv_heads: Number of key/value heads (must divide num_q_heads)
            head_dim: Per-head dimension (default: hidden_dim // num_q_heads)
            dropout: Attention dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()

        assert num_q_heads % num_kv_heads == 0, (
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads  # Q heads per KV group
        self.head_dim = head_dim or (hidden_dim // num_q_heads)
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        rope: Optional[RoPE] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: Optional attention mask (batch, 1, seq_len, seq_len) or
                  (batch, 1, 1, seq_len) for cross-attention
            is_causal: If True, apply causal (autoregressive) mask
            rope: Optional RoPE module for positional encoding
            kv_cache: Optional tuple of (cached_k, cached_v) for incremental decoding

        Returns:
            output: (batch, seq_len, hidden_dim)
            new_kv_cache: Updated KV cache (if kv_cache was provided)
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, num_q_heads * head_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)

        # Reshape to multi-head format
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope is not None:
            offset = kv_cache[0].shape[2] if kv_cache is not None else 0
            q, k = rope(q, k, offset=offset)

        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match Q heads (GQA core operation)
        # k, v: (batch, num_kv_heads, kv_len, head_dim)
        # Need: (batch, num_q_heads, kv_len, head_dim)
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(
                batch_size, self.num_kv_heads, self.num_groups, k.shape[2], self.head_dim
            ).reshape(batch_size, self.num_q_heads, k.shape[2], self.head_dim)

            v = v.unsqueeze(2).expand(
                batch_size, self.num_kv_heads, self.num_groups, v.shape[2], self.head_dim
            ).reshape(batch_size, self.num_q_heads, v.shape[2], self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=x.device),
                diagonal=k.shape[2] - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Apply provided mask (e.g., padding mask)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, num_q_heads, seq_len, head_dim) -> (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, self.num_q_heads * self.head_dim
        )

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_kv_cache


class CrossAttention(nn.Module):
    """
    Cross-Attention module for encoder-decoder architecture.

    Used in Part II to allow GPT-2 decoder to attend to BERT encoder outputs.
    Uses GQA-style grouped heads for efficiency.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int = 12,
        num_kv_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert num_q_heads % num_kv_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = head_dim or (hidden_dim // num_q_heads)
        self.scale = self.head_dim ** -0.5

        # Q comes from decoder, K/V come from encoder
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden: (batch, tgt_len, hidden_dim) — from decoder
            encoder_output: (batch, src_len, hidden_dim) — from encoder
            encoder_mask: (batch, 1, 1, src_len) — padding mask for encoder

        Returns:
            output: (batch, tgt_len, hidden_dim)
        """
        batch_size, tgt_len, _ = decoder_hidden.shape
        src_len = encoder_output.shape[1]

        # Project
        q = self.q_proj(decoder_hidden)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        # Reshape
        q = q.view(batch_size, tgt_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV for GQA
        if self.num_groups > 1:
            k = k.unsqueeze(2).expand(
                batch_size, self.num_kv_heads, self.num_groups, src_len, self.head_dim
            ).reshape(batch_size, self.num_q_heads, src_len, self.head_dim)
            v = v.unsqueeze(2).expand(
                batch_size, self.num_kv_heads, self.num_groups, src_len, self.head_dim
            ).reshape(batch_size, self.num_q_heads, src_len, self.head_dim)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if encoder_mask is not None:
            attn_weights = attn_weights.masked_fill(encoder_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, tgt_len, self.num_q_heads * self.head_dim
        )

        return self.o_proj(attn_output)
