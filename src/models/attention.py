"""
Attention Mechanisms for Seq2Seq NMT.

Implements Bahdanau (additive) attention as the primary mechanism.
Also includes Luong (multiplicative) attention for potential experimentation.

Reference:
    Bahdanau et al., "Neural Machine Translation by Jointly Learning to
    Align and Translate", ICLR 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.

    Computes attention weights using a learned alignment model:
        e_ij = v^T * tanh(W_s * s_{i-1} + W_h * h_j)
        a_ij = softmax(e_ij)
        c_i = sum_j(a_ij * h_j)

    This is the primary attention mechanism for our Seq2Seq model because:
    1. It's the most well-studied attention for RNN-based NMT
    2. It works well with bidirectional encoders
    3. The additive formulation is more expressive than dot-product
       for lower-dimensional hidden states (unlike Transformers)
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        """
        Args:
            encoder_dim: Dimensionality of encoder hidden states
            decoder_dim: Dimensionality of decoder hidden state
            attention_dim: Dimensionality of the attention alignment layer
        """
        super().__init__()

        # Linear layers for the alignment model
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.

        Args:
            decoder_hidden: (batch_size, decoder_dim) - current decoder hidden state
            encoder_outputs: (batch_size, src_len, encoder_dim) - all encoder outputs
            encoder_mask: (batch_size, src_len) - mask for padding (True for real tokens)

        Returns:
            context: (batch_size, encoder_dim) - attention-weighted context
            attention_weights: (batch_size, src_len) - attention distribution
        """
        # Project encoder outputs and decoder hidden state
        # encoder_proj: (batch_size, src_len, attention_dim)
        encoder_proj = self.W_encoder(encoder_outputs)

        # decoder_proj: (batch_size, 1, attention_dim) -> broadcast over src_len
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Compute energy scores
        # energy: (batch_size, src_len, 1) -> squeeze to (batch_size, src_len)
        energy = self.v(torch.tanh(encoder_proj + decoder_proj)).squeeze(-1)

        # Apply mask: set padding positions to -inf before softmax
        if encoder_mask is not None:
            energy = energy.masked_fill(~encoder_mask, float("-inf"))

        # Compute attention weights
        attention_weights = F.softmax(energy, dim=-1)

        # Compute context vector as weighted sum of encoder outputs
        # context: (batch_size, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative/General) Attention.

    Computes attention as:
        e_ij = s_i^T * W * h_j  (general)
        a_ij = softmax(e_ij)
        c_i = sum_j(a_ij * h_j)

    Provided as an alternative for experimentation.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        """
        Args:
            encoder_dim: Dimensionality of encoder hidden states
            decoder_dim: Dimensionality of decoder hidden state
        """
        super().__init__()
        self.W = nn.Linear(encoder_dim, decoder_dim, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: (batch_size, decoder_dim)
            encoder_outputs: (batch_size, src_len, encoder_dim)
            encoder_mask: (batch_size, src_len)

        Returns:
            context: (batch_size, encoder_dim)
            attention_weights: (batch_size, src_len)
        """
        # Project encoder outputs: (batch_size, src_len, decoder_dim)
        encoder_proj = self.W(encoder_outputs)

        # Dot product: (batch_size, src_len)
        energy = torch.bmm(
            encoder_proj, decoder_hidden.unsqueeze(-1)
        ).squeeze(-1)

        if encoder_mask is not None:
            energy = energy.masked_fill(~encoder_mask, float("-inf"))

        attention_weights = F.softmax(energy, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
