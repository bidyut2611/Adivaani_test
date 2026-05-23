"""
PyTorch Dataset and DataLoader for Hindi-Marathi NMT.

Design Decisions:
- Dynamic batching with padding to the longest sequence in each batch
- Bucketing by source length for more efficient batching (less padding waste)
- Collate function handles padding and creates attention masks
- Support for both training (src + tgt) and inference (src only) modes
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Tuple, Optional, Dict
import random
import numpy as np

from .tokenizer import TokenizerWrapper


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for parallel translation data.

    Each item returns:
        - src_ids: tokenized source sentence (Hindi) with BOS/EOS
        - tgt_ids: tokenized target sentence (Marathi) with BOS/EOS
        - src_text: original source text (for evaluation)
        - tgt_text: original target text (for evaluation)
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        tokenizer: TokenizerWrapper,
        max_len: int = 256,
    ):
        """
        Args:
            src_file: Path to source language file (one sentence per line)
            tgt_file: Path to target language file (one sentence per line)
            tokenizer: Trained TokenizerWrapper instance
            max_len: Maximum sequence length in tokens (truncate longer)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load raw text
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_texts = [line.strip() for line in f.readlines()]
        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_texts = [line.strip() for line in f.readlines()]

        assert len(self.src_texts) == len(self.tgt_texts), (
            f"Source and target files have different number of lines: "
            f"{len(self.src_texts)} vs {len(self.tgt_texts)}"
        )

        # Pre-tokenize all sentences for efficiency
        print(f"Pre-tokenizing {len(self.src_texts):,} sentence pairs...")
        self.src_ids_list = []
        self.tgt_ids_list = []

        for src, tgt in zip(self.src_texts, self.tgt_texts):
            src_ids = tokenizer.encode(src, add_bos=True, add_eos=True)
            tgt_ids = tokenizer.encode(tgt, add_bos=True, add_eos=True)

            # Truncate if necessary
            if len(src_ids) > max_len:
                src_ids = src_ids[: max_len - 1] + [tokenizer.EOS_ID]
            if len(tgt_ids) > max_len:
                tgt_ids = tgt_ids[: max_len - 1] + [tokenizer.EOS_ID]

            self.src_ids_list.append(src_ids)
            self.tgt_ids_list.append(tgt_ids)

        print(f"  Done. {len(self.src_ids_list):,} pairs tokenized.")

    def __len__(self) -> int:
        return len(self.src_ids_list)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "src_ids": self.src_ids_list[idx],
            "tgt_ids": self.tgt_ids_list[idx],
            "src_text": self.src_texts[idx],
            "tgt_text": self.tgt_texts[idx],
            "src_len": len(self.src_ids_list[idx]),
            "tgt_len": len(self.tgt_ids_list[idx]),
        }


class BucketBatchSampler(Sampler):
    """
    Sampler that groups sequences of similar length into batches.

    This reduces wasted computation on padding tokens. Sequences are sorted
    by source length within buckets, then shuffled across buckets each epoch.
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        batch_size: int,
        shuffle: bool = True,
        num_buckets: int = 100,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_buckets = num_buckets

    def __iter__(self):
        # Get all indices sorted by source length
        indices = list(range(len(self.dataset)))
        lengths = [self.dataset.src_ids_list[i] for i in indices]
        sorted_indices = sorted(indices, key=lambda i: len(self.dataset.src_ids_list[i]))

        # Create buckets
        bucket_size = max(1, len(sorted_indices) // self.num_buckets)
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i : i + bucket_size]
            if self.shuffle:
                random.shuffle(bucket)
            buckets.append(bucket)

        # Shuffle bucket order
        if self.shuffle:
            random.shuffle(buckets)

        # Flatten and create batches
        flat_indices = [idx for bucket in buckets for idx in bucket]
        batches = []
        for i in range(0, len(flat_indices), self.batch_size):
            batch = flat_indices[i : i + self.batch_size]
            if len(batch) > 0:
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict:
    """
    Collate function for DataLoader.

    Pads sequences to the maximum length in the batch and creates
    attention masks.

    Args:
        batch: List of dictionaries from TranslationDataset.__getitem__
        pad_id: Padding token ID

    Returns:
        Dictionary with padded tensors:
            - src_ids: (batch_size, max_src_len)
            - tgt_ids: (batch_size, max_tgt_len)
            - src_mask: (batch_size, max_src_len) - 1 for real tokens, 0 for padding
            - tgt_mask: (batch_size, max_tgt_len) - 1 for real tokens, 0 for padding
            - src_lengths: (batch_size,) - actual lengths
            - tgt_lengths: (batch_size,) - actual lengths
    """
    src_ids = [item["src_ids"] for item in batch]
    tgt_ids = [item["tgt_ids"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    src_lengths = [len(s) for s in src_ids]
    tgt_lengths = [len(t) for t in tgt_ids]

    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    # Pad sequences
    padded_src = [s + [pad_id] * (max_src_len - len(s)) for s in src_ids]
    padded_tgt = [t + [pad_id] * (max_tgt_len - len(t)) for t in tgt_ids]

    # Create masks (1 for real tokens, 0 for padding)
    src_mask = [[1] * l + [0] * (max_src_len - l) for l in src_lengths]
    tgt_mask = [[1] * l + [0] * (max_tgt_len - l) for l in tgt_lengths]

    return {
        "src_ids": torch.tensor(padded_src, dtype=torch.long),
        "tgt_ids": torch.tensor(padded_tgt, dtype=torch.long),
        "src_mask": torch.tensor(src_mask, dtype=torch.bool),
        "tgt_mask": torch.tensor(tgt_mask, dtype=torch.bool),
        "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
        "tgt_lengths": torch.tensor(tgt_lengths, dtype=torch.long),
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
    }


def create_dataloaders(
    processed_data_dir: str,
    tokenizer: TokenizerWrapper,
    batch_size: int = 32,
    max_len: int = 256,
    num_workers: int = 0,
    use_bucket_batching: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        processed_data_dir: Path to processed data directory
        tokenizer: Trained TokenizerWrapper
        batch_size: Batch size
        max_len: Maximum sequence length in tokens
        num_workers: Number of DataLoader workers
        use_bucket_batching: Whether to use length-based bucketing

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from pathlib import Path

    data_dir = Path(processed_data_dir)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(
        data_dir / "train.hi", data_dir / "train.mr", tokenizer, max_len
    )
    val_dataset = TranslationDataset(
        data_dir / "val.hi", data_dir / "val.mr", tokenizer, max_len
    )
    test_dataset = TranslationDataset(
        data_dir / "test.hi", data_dir / "test.mr", tokenizer, max_len
    )

    # Create DataLoaders
    if use_bucket_batching:
        train_sampler = BucketBatchSampler(train_dataset, batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset):,} samples, ~{len(train_loader):,} batches")
    print(f"  Val:   {len(val_dataset):,} samples, ~{len(val_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} samples, ~{len(test_loader):,} batches")

    return train_loader, val_loader, test_loader
