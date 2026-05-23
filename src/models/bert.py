"""
Custom BERT Architecture for Hindi/Marathi NMT (Part II).

~110M parameter BERT-like model implemented from scratch.
Incorporates modern components: RoPE, GQA, and RMSNorm.
Uses Masked Language Modeling (MLM) objective for pretraining.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .components import RoPE, RMSNorm, GroupedQueryAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(hidden_dim)
        
        # MLP / FeedForward
        inner_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        rope: Optional[RoPE] = None
    ) -> torch.Tensor:
        # Pre-norm architecture
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, mask=mask, rope=rope)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x


class CustomBERT(nn.Module):
    """
    Custom BERT (~110M params).
    Configured similarly to BERT-base (12 layers, 768 hidden).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = 4, # GQA
        mlp_ratio: float = 4.65, # Tuned to hit ~110M with GQA
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        # Note: No absolute position embeddings, we use RoPE
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_norm = RMSNorm(hidden_dim)
        
        # RoPE
        head_dim = hidden_dim // num_heads
        self.rope = RoPE(dim=head_dim, max_seq_len=max_seq_len)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(hidden_dim)
        
        # MLM Head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Weight tying for input/output embeddings
        self.mlm_head[-1].weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert (batch, seq_len) to (batch, 1, 1, seq_len) for self-attention.
        """
        # attention_mask is 1 for real tokens, 0 for padding
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # We need float mask for attention (0.0 for keep, -inf for mask)
        # But our GQA implementation takes boolean/int mask and handles it
        return extended_mask

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_embeddings: If True, skip MLM head and return hidden states
        """
        x = self.token_embedding(input_ids)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()
            
        extended_mask = self.get_extended_attention_mask(attention_mask)
        
        for layer in self.layers:
            x = layer(x, mask=extended_mask, rope=self.rope)
            
        x = self.final_norm(x)
        
        if return_embeddings:
            return x
            
        # For MLM pretraining
        logits = self.mlm_head(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
