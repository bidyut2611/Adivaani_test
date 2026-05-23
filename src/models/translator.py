"""
Translation Framework for Part II.

Encoder: Pretrained Custom BERT (Bidirectional, Hindi)
Decoder: Pretrained Custom GPT-2 (Autoregressive, Marathi)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from .bert import CustomBERT
from .gpt2 import CustomGPT2


class BERTGPT2Translator(nn.Module):
    """
    Encoder-Decoder model leveraging pretrained BERT and GPT-2.
    
    BERT acts as the encoder for the source language (Hindi).
    GPT-2 acts as the decoder for the target language (Marathi),
    with added cross-attention layers to attend to BERT's output.
    """
    def __init__(
        self,
        encoder: CustomBERT,
        decoder: CustomGPT2,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.tgt_vocab_size = encoder.vocab_size
        self.src_vocab_size = encoder.vocab_size
        
        # Verify decoder has cross attention enabled
        assert any(layer.add_cross_attention for layer in self.decoder.layers), \
            "Decoder must be initialized with add_cross_attention=True"

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor, # Added for API compatibility with Seq2SeqTrainer
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0, # Kept for API compatibility, but GPT-2 is fully parallel
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for training.
        
        Args:
            src_ids: (batch, src_len)
            src_mask: (batch, src_len)
            tgt_ids: (batch, tgt_len)
            teacher_forcing_ratio: Ignored here as we compute all logits in parallel
            
        Returns:
            logits: (batch, tgt_len - 1, vocab_size)
            attentions: None (not tracked in this framework)
        """
        # Encode source sequence
        # BERT expects mask to be 1 for real tokens, 0 for padding
        encoder_output = self.encoder(
            src_ids, 
            attention_mask=src_mask,
            return_embeddings=True
        )
        
        # Format mask for cross-attention
        # Cross attention expects (batch, 1, 1, src_len) mask
        # 1.0 for keep, 0.0 for mask (in our GQA implementation)
        encoder_attn_mask = src_mask.unsqueeze(1).unsqueeze(2)
        
        # Decode target sequence
        # tgt_ids includes BOS token at index 0. We predict tgt_ids[1:]
        decoder_input = tgt_ids[:, :-1]
        
        # We can run the entire decoder sequence in parallel during training
        # because GPT-2 uses causal masking.
        logits, _ = self.decoder(
            decoder_input,
            encoder_output=encoder_output,
            encoder_mask=encoder_attn_mask,
            kv_caches=None, # Not used during training
        )
        
        return logits, None

    @torch.no_grad()
    def translate(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int = 200,
        beam_size: int = 1,
    ) -> Tuple[torch.Tensor, None]:
        """
        Autoregressive generation using KV caching for speed.
        """
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        self.eval()
        
        # Encode
        encoder_output = self.encoder(
            src_ids, 
            attention_mask=src_mask,
            return_embeddings=True
        )
        encoder_attn_mask = src_mask.unsqueeze(1).unsqueeze(2)
        
        if beam_size > 1:
            # We would implement beam search here.
            # For simplicity, returning greedy decoding placeholder if beam_size > 1.
            print("Warning: Beam search not implemented for BERT-GPT2 yet, falling back to greedy.")
            
        # Greedy decoding with KV Cache
        predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        # Start token
        input_token = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        kv_caches = None
        
        for t in range(max_len):
            # Forward pass (only for the last token, using KV cache for past tokens)
            logits, kv_caches = self.decoder(
                input_token,
                encoder_output=encoder_output,
                encoder_mask=encoder_attn_mask,
                kv_caches=kv_caches,
            )
            
            # Get prediction
            next_token_logits = logits[:, -1, :] # (batch, vocab_size)
            next_token = next_token_logits.argmax(dim=-1) # (batch,)
            
            predictions[:, t] = next_token
            
            # Check if all finished
            finished = finished | (next_token == self.eos_id)
            if finished.all():
                break
                
            input_token = next_token.unsqueeze(1)
            
        return predictions, None

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
