"""
SentencePiece BPE Tokenizer for Hindi-Marathi NMT.

Design Decisions:
- We use SentencePiece BPE (Byte Pair Encoding) for subword tokenization
- Shared vocabulary across Hindi and Marathi since both use Devanagari script
  (this enables better cross-lingual transfer and reduces vocabulary size)
- Vocabulary size: 32,000 tokens (standard for mid-size NMT systems)
- Special tokens: <pad>=0, <unk>=1, <s>=2 (BOS), </s>=3 (EOS)
- Character coverage: 0.9999 (capture nearly all Devanagari characters)
"""

import os
import sentencepiece as spm
from typing import List, Optional
from pathlib import Path


class TokenizerWrapper:
    """Wrapper around SentencePiece for training and using BPE tokenizer."""

    # Special token IDs (SentencePiece default mapping)
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(
        self,
        model_prefix: str = "tokenizer",
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9999,
        tokenizer_dir: str = "data/tokenizer",
    ):
        """
        Args:
            model_prefix: Prefix for the tokenizer model files
            vocab_size: Target vocabulary size
            model_type: 'bpe' or 'unigram'
            character_coverage: Coverage of characters to include
            tokenizer_dir: Directory to save/load tokenizer model
        """
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.tokenizer_dir / f"{model_prefix}.model"
        self.sp = None

    def train(self, input_file: str, **kwargs) -> None:
        """
        Train SentencePiece tokenizer on the given text file.

        Args:
            input_file: Path to text file (one sentence per line)
            **kwargs: Additional SentencePiece training arguments
        """
        model_prefix_path = str(self.tokenizer_dir / self.model_prefix)

        train_args = {
            "input": input_file,
            "model_prefix": model_prefix_path,
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "pad_id": self.PAD_ID,
            "unk_id": self.UNK_ID,
            "bos_id": self.BOS_ID,
            "eos_id": self.EOS_ID,
            "pad_piece": self.PAD_TOKEN,
            "unk_piece": self.UNK_TOKEN,
            "bos_piece": self.BOS_TOKEN,
            "eos_piece": self.EOS_TOKEN,
            # Treat whitespace-separated words as individual units for BPE
            "split_by_unicode_script": True,
            "split_by_whitespace": True,
            "split_by_number": True,
            # Regularization for robustness during training
            "shuffle_input_sentence": True,
            "input_sentence_size": 5000000,  # Use up to 5M sentences for training
            # Normalization
            "normalization_rule_name": "nfkc",
            "add_dummy_prefix": True,
            "remove_extra_whitespaces": True,
        }
        train_args.update(kwargs)

        print(f"Training SentencePiece {self.model_type.upper()} tokenizer...")
        print(f"  Input: {input_file}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Model type: {self.model_type}")
        print(f"  Character coverage: {self.character_coverage}")

        spm.SentencePieceTrainer.train(**train_args)

        print(f"  Model saved: {model_prefix_path}.model")
        print(f"  Vocab saved: {model_prefix_path}.vocab")

        # Load the trained model
        self.load()

    def load(self) -> None:
        """Load a trained SentencePiece model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {self.model_path}. "
                f"Train the tokenizer first using .train()"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))
        print(f"Tokenizer loaded: {self.model_path} (vocab_size={self.sp.get_piece_size()})")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of token IDs
        """
        assert self.sp is not None, "Tokenizer not loaded. Call .load() first."

        ids = self.sp.encode(text, out_type=int)

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special: Whether to remove special tokens before decoding

        Returns:
            Decoded text string
        """
        assert self.sp is not None, "Tokenizer not loaded. Call .load() first."

        if skip_special:
            special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
            ids = [i for i in ids if i not in special_ids]

        return self.sp.decode(ids)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces (strings)."""
        assert self.sp is not None, "Tokenizer not loaded. Call .load() first."
        return self.sp.encode(text, out_type=str)

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size of the loaded model."""
        assert self.sp is not None, "Tokenizer not loaded."
        return self.sp.get_piece_size()

    def id_to_piece(self, id: int) -> str:
        """Convert token ID to its string representation."""
        assert self.sp is not None, "Tokenizer not loaded."
        return self.sp.id_to_piece(id)

    def piece_to_id(self, piece: str) -> int:
        """Convert string piece to its token ID."""
        assert self.sp is not None, "Tokenizer not loaded."
        return self.sp.piece_to_id(piece)


if __name__ == "__main__":
    # Example: Train tokenizer on processed data
    tokenizer = TokenizerWrapper(
        model_prefix="hi_mr_bpe",
        vocab_size=32000,
        model_type="bpe",
        tokenizer_dir="data/tokenizer",
    )

    # Train on combined Hindi + Marathi text
    tokenizer.train("data/processed/all_text.txt")

    # Test encoding/decoding
    test_hi = "यह एक परीक्षण वाक्य है ।"
    test_mr = "हा एक चाचणी वाक्य आहे."

    for text in [test_hi, test_mr]:
        ids = tokenizer.encode(text)
        pieces = tokenizer.encode_as_pieces(text)
        decoded = tokenizer.decode(ids)
        print(f"\nOriginal:  {text}")
        print(f"Pieces:    {pieces}")
        print(f"IDs:       {ids}")
        print(f"Decoded:   {decoded}")
