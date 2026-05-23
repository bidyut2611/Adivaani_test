"""
Data Preparation Script — Step 1 of the pipeline.

Usage: python scripts/prepare_data.py --config configs/part1_random_emb.yaml

Steps: 1) Copy raw data  2) Preprocess  3) Train tokenizer
"""

import argparse, shutil, sys, os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import DataPreprocessor
from src.data.tokenizer import TokenizerWrapper


def find_dataset(search_dir: Path) -> Path:
    """Search for dataset directory containing train.hi/train.mr."""
    candidates = [
        search_dir / "data" / "raw",
        search_dir.parent / "IIT-Delhi-MISN-Lab-Adivaani-Hindi-Marathi-Dataset" /
            "IIT-Delhi-MISN-Lab-Adivaani-Hindi-Marathi-Dataset-Hiring-2026",
    ]
    for c in candidates:
        if c.exists() and (c / "train.hi").exists():
            return c
    return None


def copy_raw_data(raw_dir: Path, dataset_dir: Path):
    """Copy dataset files to data/raw/ if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = ["train.hi", "train.mr", "test.hi", "test.mr"]
    if all((raw_dir / f).exists() for f in files):
        print("Raw data already in data/raw/. Skipping copy.")
        return
    if dataset_dir is None:
        print("ERROR: Cannot find Adivaani dataset. Use --dataset-dir.")
        sys.exit(1)
    print(f"Copying dataset from: {dataset_dir}")
    for f in files:
        if not (raw_dir / f).exists():
            shutil.copy2(dataset_dir / f, raw_dir / f)
            print(f"  Copied {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part1_random_emb.yaml")
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    dc = config["data"]
    raw_dir = PROJECT_ROOT / dc["raw_dir"]
    processed_dir = PROJECT_ROOT / dc["processed_dir"]
    tokenizer_dir = PROJECT_ROOT / dc["tokenizer_dir"]

    # Step 1: Copy raw data
    print("=" * 60)
    print("Step 1: Copy raw dataset")
    print("=" * 60)
    ds_dir = Path(args.dataset_dir) if args.dataset_dir else find_dataset(PROJECT_ROOT)
    copy_raw_data(raw_dir, ds_dir)

    # Step 2: Preprocess
    print("\n" + "=" * 60)
    print("Step 2: Preprocess (normalize, filter, split)")
    print("=" * 60)
    if (processed_dir / "train.hi").exists():
        print("Already processed. Delete data/processed/ to re-run.\n")
    else:
        preprocessor = DataPreprocessor(
            raw_data_dir=str(raw_dir),
            processed_data_dir=str(processed_dir),
            max_len=dc.get("max_word_len", 150),
            min_len=dc.get("min_word_len", 2),
            val_ratio=dc.get("val_ratio", 0.05),
            seed=config.get("seed", 42),
        )
        preprocessor.run()

    # Step 3: Train tokenizer
    print("\n" + "=" * 60)
    print("Step 3: Train SentencePiece BPE tokenizer")
    print("=" * 60)
    prefix = dc.get("tokenizer_prefix", "hi_mr_bpe")
    model_path = tokenizer_dir / f"{prefix}.model"
    if model_path.exists():
        print(f"Tokenizer already exists: {model_path}\n")
    else:
        tok = TokenizerWrapper(
            model_prefix=prefix,
            vocab_size=dc.get("vocab_size", 32000),
            tokenizer_dir=str(tokenizer_dir),
        )
        tok.train(str(processed_dir / "all_text.txt"))

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    tok = TokenizerWrapper(model_prefix=prefix, tokenizer_dir=str(tokenizer_dir))
    tok.load()
    with open(processed_dir / "train.hi", encoding="utf-8") as f:
        sample = f.readline().strip()
    ids = tok.encode(sample)
    print(f"Sample: {sample[:80]}")
    print(f"Tokens: {len(ids)}, IDs[:10]: {ids[:10]}")
    print(f"Vocab size: {tok.vocab_size_actual:,}")
    print("\nDone! Next: python scripts/train_seq2seq.py --config configs/part1_random_emb.yaml")


if __name__ == "__main__":
    main()
