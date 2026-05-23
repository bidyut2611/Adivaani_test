"""
Custom GPT-2 Architecture for Hindi/Marathi NMT (Part II).

~124M parameter autoregressive model implemented from scratch.
Incorporates modern components: RoPE, GQA, and RMSNorm.
Uses causal masking for next-token prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .components import RoPE, RMSNorm, GroupedQueryAttention, CrossAttention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        add_cross_attention: bool = False,
    ):
        super().__init__()
        self.add_cross_attention = add_cross_attention
        
        # Self Attention
        self.norm1 = RMSNorm(hidden_dim)
        self.self_attn = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_q_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        
        # Cross Attention (Optional, used when adapting GPT for translation)
        if add_cross_attention:
            self.norm_cross = RMSNorm(hidden_dim)
            self.cross_attn = CrossAttention(
                hidden_dim=hidden_dim,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
            )
            
        # MLP / FeedForward
        self.norm2 = RMSNorm(hidden_dim)
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
        is_causal: bool = True,
        rope: Optional[RoPE] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self Attention (Causal)
        norm_x = self.norm1(x)
        attn_out, new_kv_cache = self.self_attn(
            norm_x, is_causal=is_causal, rope=rope, kv_cache=kv_cache
        )
        x = x + attn_out
        
        # Cross Attention (if encoder output provided)
        if self.add_cross_attention and encoder_output is not None:
            norm_x = self.norm_cross(x)
            cross_out = self.cross_attn(
                norm_x, encoder_output, encoder_mask=encoder_mask
            )
            x = x + cross_out
            
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, new_kv_cache


class CustomGPT2(nn.Module):
    """
    Custom GPT-2 (~124M params).
    Configured similarly to GPT-2 Small (12 layers, 768 hidden).
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = 4, # GQA
        mlp_ratio: float = 6.0, # Tuned to hit ~124M with GQA
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        add_cross_attention: bool = False, # Set True for Part II translation system
    ):
        super().__init__()
        
        self.pad_id = pad_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        # Note: No absolute position embeddings, we use RoPE
        self.embed_dropout = nn.Dropout(dropout)
        
        # RoPE
        head_dim = hidden_dim // num_heads
        self.rope = RoPE(dim=head_dim, max_seq_len=max_seq_len)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                add_cross_attention=add_cross_attention
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(hidden_dim)
        
        # LM Head (Language Modeling)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        input_ids: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: (batch, seq_len)
            encoder_output: (batch, src_len, hidden_dim) - from BERT
            encoder_mask: (batch, 1, 1, src_len) - mask for BERT output
            kv_caches: Cached key/values for fast generation
            return_embeddings: If True, return hidden states instead of logits
            
        Returns:
            logits (or hidden states), updated kv_caches
        """
        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)
        
        new_kv_caches = []
        
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(
                x, 
                is_causal=True, 
                rope=self.rope, 
                kv_cache=layer_cache,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask
            )
            new_kv_caches.append(new_cache)
            
        x = self.final_norm(x)
        
        if return_embeddings:
            return x, new_kv_caches
            
        logits = self.lm_head(x)
        return logits, new_kv_caches

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
