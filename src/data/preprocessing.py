"""
Data Preprocessing Pipeline for Hindi-Marathi NMT.

This module handles:
- Loading raw parallel corpus data
- Text cleaning and normalization
- Filtering based on sentence length and quality
- Train/validation splitting
- Saving processed data

Design Decisions:
- We normalize Unicode (NFC) for consistent Devanagari representation
- We remove pairs where either side is empty or too short/long
- We use a 95/5 train/val split (test set is already provided)
- We filter sentences > 150 words to keep training manageable on limited GPU
"""

import os
import re
import random
import unicodedata
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class DataPreprocessor:
    """Preprocesses Hindi-Marathi parallel corpus for NMT training."""

    def __init__(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        max_len: int = 150,
        min_len: int = 1,
        val_ratio: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            raw_data_dir: Path to directory containing train.hi, train.mr, test.hi, test.mr
            processed_data_dir: Path to save processed files
            max_len: Maximum sentence length in words (filter out longer)
            min_len: Minimum sentence length in words (filter out shorter)
            val_ratio: Fraction of training data to use for validation
            seed: Random seed for reproducibility
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.max_len = max_len
        self.min_len = min_len
        self.val_ratio = val_ratio
        self.seed = seed

        # Create output directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def load_file(self, filepath: str) -> List[str]:
        """Load a text file and return list of stripped lines."""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    def normalize_text(self, text: str) -> str:
        """
        Normalize Unicode text for consistency.

        - NFC normalization for Devanagari
        - Collapse multiple spaces into single space
        - Strip leading/trailing whitespace
        - Normalize common punctuation variations
        """
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # Collapse multiple whitespace characters into single space
        text = re.sub(r"\s+", " ", text)

        # Strip
        text = text.strip()

        return text

    def filter_pair(self, src: str, tgt: str) -> bool:
        """
        Check if a sentence pair should be kept.

        Returns True if the pair passes all quality filters.
        """
        # Both sides must be non-empty
        if not src or not tgt:
            return False

        src_words = len(src.split())
        tgt_words = len(tgt.split())

        # Length constraints
        if src_words < self.min_len or tgt_words < self.min_len:
            return False
        if src_words > self.max_len or tgt_words > self.max_len:
            return False

        # Length ratio check: filter extreme length mismatches
        # (source and target should be roughly similar in length)
        ratio = max(src_words, tgt_words) / max(min(src_words, tgt_words), 1)
        if ratio > 3.0:
            return False

        return True

    def process_split(
        self, src_lines: List[str], tgt_lines: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Process a parallel corpus split: normalize and filter.

        Returns filtered (src, tgt) line lists.
        """
        assert len(src_lines) == len(tgt_lines), (
            f"Mismatched line counts: {len(src_lines)} vs {len(tgt_lines)}"
        )

        filtered_src, filtered_tgt = [], []

        for src, tgt in zip(src_lines, tgt_lines):
            src = self.normalize_text(src)
            tgt = self.normalize_text(tgt)

            if self.filter_pair(src, tgt):
                filtered_src.append(src)
                filtered_tgt.append(tgt)

        return filtered_src, filtered_tgt

    def create_val_split(
        self, src_lines: List[str], tgt_lines: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Split training data into train and validation sets.

        Returns (train_src, train_tgt, val_src, val_tgt).
        """
        random.seed(self.seed)

        indices = list(range(len(src_lines)))
        random.shuffle(indices)

        val_size = int(len(indices) * self.val_ratio)
        val_indices = set(indices[:val_size])

        train_src, train_tgt = [], []
        val_src, val_tgt = [], []

        for i in range(len(src_lines)):
            if i in val_indices:
                val_src.append(src_lines[i])
                val_tgt.append(tgt_lines[i])
            else:
                train_src.append(src_lines[i])
                train_tgt.append(tgt_lines[i])

        return train_src, train_tgt, val_src, val_tgt

    def save_lines(self, lines: List[str], filepath: str) -> None:
        """Save list of strings to a text file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def get_statistics(
        self, src_lines: List[str], tgt_lines: List[str], split_name: str
    ) -> Dict:
        """Compute and return dataset statistics."""
        src_lens = [len(s.split()) for s in src_lines]
        tgt_lens = [len(t.split()) for t in tgt_lines]

        stats = {
            "split": split_name,
            "num_pairs": len(src_lines),
            "src_avg_words": sum(src_lens) / max(len(src_lens), 1),
            "src_median_words": sorted(src_lens)[len(src_lens) // 2] if src_lens else 0,
            "src_max_words": max(src_lens) if src_lens else 0,
            "tgt_avg_words": sum(tgt_lens) / max(len(tgt_lens), 1),
            "tgt_median_words": sorted(tgt_lens)[len(tgt_lens) // 2] if tgt_lens else 0,
            "tgt_max_words": max(tgt_lens) if tgt_lens else 0,
        }
        return stats

    def run(self) -> Dict:
        """
        Execute the full preprocessing pipeline.

        Returns a dictionary with statistics for each split.
        """
        print("=" * 60)
        print("AdiVaani NMT - Data Preprocessing Pipeline")
        print("=" * 60)

        # 1. Load raw data
        print("\n[1/5] Loading raw data...")
        train_hi = self.load_file(self.raw_data_dir / "train.hi")
        train_mr = self.load_file(self.raw_data_dir / "train.mr")
        test_hi = self.load_file(self.raw_data_dir / "test.hi")
        test_mr = self.load_file(self.raw_data_dir / "test.mr")

        print(f"  Raw train: {len(train_hi):,} pairs")
        print(f"  Raw test:  {len(test_hi):,} pairs")

        # 2. Normalize and filter training data
        print("\n[2/5] Normalizing and filtering training data...")
        train_hi, train_mr = self.process_split(train_hi, train_mr)
        print(f"  After filtering: {len(train_hi):,} pairs")

        # 3. Normalize and filter test data
        print("\n[3/5] Normalizing and filtering test data...")
        test_hi, test_mr = self.process_split(test_hi, test_mr)
        print(f"  After filtering: {len(test_hi):,} pairs")

        # 4. Create validation split
        print("\n[4/5] Creating train/validation split...")
        train_hi, train_mr, val_hi, val_mr = self.create_val_split(
            train_hi, train_mr
        )
        print(f"  Train: {len(train_hi):,} pairs")
        print(f"  Val:   {len(val_hi):,} pairs")

        # 5. Save processed data
        print("\n[5/5] Saving processed data...")
        self.save_lines(train_hi, self.processed_data_dir / "train.hi")
        self.save_lines(train_mr, self.processed_data_dir / "train.mr")
        self.save_lines(val_hi, self.processed_data_dir / "val.hi")
        self.save_lines(val_mr, self.processed_data_dir / "val.mr")
        self.save_lines(test_hi, self.processed_data_dir / "test.hi")
        self.save_lines(test_mr, self.processed_data_dir / "test.mr")

        # Also save combined monolingual text for tokenizer training
        print("  Saving combined monolingual text for tokenizer training...")
        all_text = train_hi + train_mr + val_hi + val_mr
        self.save_lines(all_text, self.processed_data_dir / "all_text.txt")

        print(f"\n  All files saved to: {self.processed_data_dir}")

        # Compute statistics
        all_stats = {}
        for name, src, tgt in [
            ("train", train_hi, train_mr),
            ("val", val_hi, val_mr),
            ("test", test_hi, test_mr),
        ]:
            stats = self.get_statistics(src, tgt, name)
            all_stats[name] = stats
            print(f"\n  [{name.upper()}] {stats['num_pairs']:,} pairs")
            print(
                f"    Hindi:   avg={stats['src_avg_words']:.1f} words, "
                f"median={stats['src_median_words']}, max={stats['src_max_words']}"
            )
            print(
                f"    Marathi: avg={stats['tgt_avg_words']:.1f} words, "
                f"median={stats['tgt_median_words']}, max={stats['tgt_max_words']}"
            )

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)

        return all_stats


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        raw_data_dir="data/raw",
        processed_data_dir="data/processed",
        max_len=150,
        min_len=2,
        val_ratio=0.05,
        seed=42,
    )
    preprocessor.run()
