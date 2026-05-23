"""
Seq2Seq LSTM Encoder-Decoder with Attention for Hindi→Marathi NMT.

Architecture:
    - Encoder: Bidirectional multi-layer LSTM
    - Decoder: Unidirectional multi-layer LSTM with Bahdanau attention
    - Supports both random and pretrained (BERT) embeddings
    - Teacher forcing with configurable ratio for scheduled sampling

Design Decisions:
    - Bidirectional encoder doubles the hidden dimension, so we project
      it down before passing to the decoder
    - We concatenate attention context with decoder input at each step
    - Embedding dimension is 768 to match BERT for fair comparison
    - Dropout is applied at embedding, encoder, decoder, and attention levels

Reference:
    Bahdanau et al., ICLR 2015
    Sutskever et al., "Sequence to Sequence Learning with Neural Networks", NeurIPS 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional, Dict

from .attention import BahdanauAttention


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.

    Takes source token IDs, embeds them, and produces:
    - encoder_outputs: contextualized representations for attention
    - hidden/cell states: initial states for the decoder
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[nn.Embedding] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            vocab_size: Source vocabulary size
            embed_dim: Embedding dimension (768 to match BERT)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embedding layer
            freeze_embeddings: Whether to freeze pretrained embeddings
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = pretrained_embeddings
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            # Xavier initialization for random embeddings
            nn.init.xavier_uniform_(self.embedding.weight)
            self.embedding.weight.data[0].fill_(0)  # Zero padding embedding

        self.embed_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Project bidirectional hidden states to decoder dimension
        # Bidirectional doubles the hidden dim, so we need to reduce it
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src_ids: (batch_size, src_len) - source token IDs
            src_lengths: (batch_size,) - actual lengths for packing

        Returns:
            encoder_outputs: (batch_size, src_len, hidden_dim * 2)
            (hidden, cell): initial decoder states, each (num_layers, batch_size, hidden_dim)
        """
        # Embed: (batch_size, src_len, embed_dim)
        embedded = self.embed_dropout(self.embedding(src_ids))

        # Pack padded sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Encode: outputs are (batch_size, src_len, hidden_dim * 2) for bidirectional
        packed_outputs, (hidden, cell) = self.lstm(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )

        # hidden: (num_layers * 2, batch_size, hidden_dim) -> (num_layers, batch_size, hidden_dim)
        # Concatenate forward and backward hidden states, then project
        # Reshape: separate forward and backward
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)

        # Concatenate forward and backward: (num_layers, batch_size, hidden_dim * 2)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=-1)

        # Project to decoder dimension: (num_layers, batch_size, hidden_dim)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

        return encoder_outputs, (hidden, cell)


class Decoder(nn.Module):
    """
    LSTM Decoder with Bahdanau Attention.

    At each decoding step:
    1. Embed the previous target token
    2. Compute attention over encoder outputs
    3. Concatenate embedded token with attention context
    4. Pass through LSTM
    5. Project to vocabulary for next token prediction
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        encoder_dim: int = 1024,  # hidden_dim * 2 for bidirectional encoder
        attention_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[nn.Embedding] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            vocab_size: Target vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            encoder_dim: Encoder output dimension (bidirectional = hidden_dim * 2)
            attention_dim: Attention alignment dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embedding layer
            freeze_embeddings: Whether to freeze pretrained embeddings
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = pretrained_embeddings
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            nn.init.xavier_uniform_(self.embedding.weight)
            self.embedding.weight.data[0].fill_(0)

        self.embed_dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)

        # LSTM input is concatenation of embedded token and attention context
        self.lstm = nn.LSTM(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection: combine LSTM output, context, and embedding
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.

        Args:
            input_token: (batch_size,) - previous token IDs
            hidden: (num_layers, batch_size, hidden_dim)
            cell: (num_layers, batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, encoder_dim)
            encoder_mask: (batch_size, src_len)

        Returns:
            output: (batch_size, vocab_size) - logits over vocabulary
            hidden: updated hidden state
            cell: updated cell state
            attention_weights: (batch_size, src_len)
        """
        # Embed: (batch_size, embed_dim)
        embedded = self.embed_dropout(self.embedding(input_token))

        # Compute attention using top-layer hidden state
        # hidden[-1] is (batch_size, hidden_dim)
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, encoder_mask
        )

        # Concatenate embedding and context: (batch_size, 1, embed_dim + encoder_dim)
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)

        # LSTM step
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)  # (batch_size, hidden_dim)

        # Combine LSTM output, context, and embedding for prediction
        combined = torch.cat([lstm_output, context, embedded], dim=-1)
        combined = self.dropout(combined)

        # Project to vocabulary: (batch_size, vocab_size)
        output = self.fc_out(combined)

        return output, hidden, cell, attention_weights


class Seq2SeqModel(nn.Module):
    """
    Complete Seq2Seq NMT Model with Attention.

    Combines encoder and decoder with teacher forcing support.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        src_pretrained_embeddings: Optional[nn.Embedding] = None,
        tgt_pretrained_embeddings: Optional[nn.Embedding] = None,
        freeze_embeddings: bool = False,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ):
        """
        Args:
            src_vocab_size: Source (Hindi) vocabulary size
            tgt_vocab_size: Target (Marathi) vocabulary size
            embed_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            attention_dim: Attention alignment dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            src_pretrained_embeddings: Optional pretrained source embeddings
            tgt_pretrained_embeddings: Optional pretrained target embeddings
            freeze_embeddings: Whether to freeze pretrained embeddings
            pad_id: Padding token ID
            bos_id: Beginning of sentence token ID
            eos_id: End of sentence token ID
        """
        super().__init__()

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.tgt_vocab_size = tgt_vocab_size

        encoder_dim = hidden_dim * 2  # Bidirectional

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embeddings=src_pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout=dropout,
            pretrained_embeddings=tgt_pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            src_ids: (batch_size, src_len) - source token IDs
            src_lengths: (batch_size,) - source lengths
            src_mask: (batch_size, src_len) - source padding mask
            tgt_ids: (batch_size, tgt_len) - target token IDs (for teacher forcing)
            teacher_forcing_ratio: probability of using ground truth as next input

        Returns:
            outputs: (batch_size, tgt_len - 1, vocab_size) - logits for each step
            attentions: (batch_size, tgt_len - 1, src_len) - attention weights
        """
        batch_size = src_ids.size(0)
        tgt_len = tgt_ids.size(1)
        device = src_ids.device

        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src_ids, src_lengths)

        # Prepare decoder
        # We predict tgt_len - 1 tokens (skip BOS, predict up to including EOS)
        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=device)
        attentions = torch.zeros(
            batch_size, tgt_len - 1, encoder_outputs.size(1), device=device
        )

        # First input to decoder is BOS token
        input_token = tgt_ids[:, 0]  # BOS

        for t in range(tgt_len - 1):
            output, hidden, cell, attn_weights = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )

            outputs[:, t, :] = output
            attentions[:, t, :] = attn_weights

            # Teacher forcing: use ground truth or model prediction
            if random.random() < teacher_forcing_ratio:
                input_token = tgt_ids[:, t + 1]
            else:
                input_token = output.argmax(dim=-1)

        return outputs, attentions

    @torch.no_grad()
    def translate(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int = 200,
        beam_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translate source sentences (inference mode).

        Args:
            src_ids: (batch_size, src_len)
            src_lengths: (batch_size,)
            src_mask: (batch_size, src_len)
            max_len: Maximum generation length
            beam_size: Beam size (1 = greedy decoding)

        Returns:
            predictions: (batch_size, max_len) - predicted token IDs
            attentions: (batch_size, max_len, src_len) - attention weights
        """
        self.eval()

        if beam_size > 1:
            return self._beam_search(
                src_ids, src_lengths, src_mask, max_len, beam_size
            )
        else:
            return self._greedy_decode(src_ids, src_lengths, src_mask, max_len)

    def _greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding (beam_size=1)."""
        batch_size = src_ids.size(0)
        device = src_ids.device

        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src_ids, src_lengths)

        # Initialize
        predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attentions = torch.zeros(
            batch_size, max_len, encoder_outputs.size(1), device=device
        )

        input_token = torch.full(
            (batch_size,), self.bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_len):
            output, hidden, cell, attn_weights = self.decoder.forward_step(
                input_token, hidden, cell, encoder_outputs, src_mask
            )

            # Greedy selection
            top1 = output.argmax(dim=-1)

            predictions[:, t] = top1
            attentions[:, t, :] = attn_weights

            # Check for EOS
            finished = finished | (top1 == self.eos_id)
            if finished.all():
                break

            input_token = top1

        return predictions, attentions

    def _beam_search(
        self,
        src_ids: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        beam_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding.

        For simplicity, processes one sentence at a time.
        """
        batch_size = src_ids.size(0)
        device = src_ids.device

        all_predictions = []
        all_attentions = []

        for b in range(batch_size):
            # Get single example
            single_src = src_ids[b : b + 1]
            single_len = src_lengths[b : b + 1]
            single_mask = src_mask[b : b + 1]

            # Encode
            encoder_outputs, (hidden, cell) = self.encoder(single_src, single_len)

            # Expand for beam
            encoder_outputs = encoder_outputs.repeat(beam_size, 1, 1)
            single_mask = single_mask.repeat(beam_size, 1)
            hidden = hidden.repeat(1, beam_size, 1)
            cell = cell.repeat(1, beam_size, 1)

            # Initialize beams
            beam_scores = torch.zeros(beam_size, device=device)
            beam_scores[1:] = float("-inf")

            beam_tokens = torch.full(
                (beam_size, max_len), self.pad_id, dtype=torch.long, device=device
            )
            input_token = torch.full(
                (beam_size,), self.bos_id, dtype=torch.long, device=device
            )

            finished_beams = []

            for t in range(max_len):
                output, hidden, cell, attn_weights = self.decoder.forward_step(
                    input_token, hidden, cell, encoder_outputs, single_mask
                )

                log_probs = F.log_softmax(output, dim=-1)

                # Add to accumulated scores
                next_scores = beam_scores.unsqueeze(-1) + log_probs
                next_scores = next_scores.view(-1)

                # Get top-k candidates
                topk_scores, topk_indices = next_scores.topk(beam_size, dim=-1)

                beam_indices = topk_indices // self.tgt_vocab_size
                token_indices = topk_indices % self.tgt_vocab_size

                # Update beams
                beam_tokens = beam_tokens[beam_indices]
                beam_tokens[:, t] = token_indices
                beam_scores = topk_scores

                hidden = hidden[:, beam_indices, :]
                cell = cell[:, beam_indices, :]

                # Check for completed beams
                for i in range(beam_size):
                    if token_indices[i] == self.eos_id:
                        finished_beams.append(
                            (beam_scores[i].item() / (t + 1), beam_tokens[i].clone())
                        )

                if len(finished_beams) >= beam_size:
                    break

                input_token = token_indices

            # Select best beam
            if finished_beams:
                finished_beams.sort(key=lambda x: x[0], reverse=True)
                best_tokens = finished_beams[0][1]
            else:
                best_tokens = beam_tokens[0]

            all_predictions.append(best_tokens)

        predictions = torch.stack(all_predictions, dim=0)
        # Return empty attentions for beam search (not tracked per beam)
        attentions = torch.zeros(batch_size, max_len, src_ids.size(1), device=device)

        return predictions, attentions

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
