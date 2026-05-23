"""
Execution script for GPT-2 Pretraining.
"""

import argparse
import sys
import torch
from pathlib import Path
import yaml
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import TokenizerWrapper
from src.models.gpt2 import CustomGPT2
from src.training.pretrain_gpt2 import GPT2Trainer

class MonolingualDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=256):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len-1] + [self.tokenizer.EOS_ID]
        return ids

def collate_monolingual(batch, pad_id):
    max_len = max(len(x) for x in batch)
    padded = [x + [pad_id] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part2_gpt2_pretrain.yaml")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = TokenizerWrapper(
        model_prefix=config["data"]["tokenizer_prefix"], 
        tokenizer_dir=str(PROJECT_ROOT / config["data"]["tokenizer_dir"])
    )
    tokenizer.load()

    train_ds = MonolingualDataset(
        PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["train_file"], 
        tokenizer, config["data"]["max_len"]
    )
    val_ds = MonolingualDataset(
        PROJECT_ROOT / config["data"]["processed_dir"] / config["data"]["val_file"], 
        tokenizer, config["data"]["max_len"]
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        collate_fn=lambda b: collate_monolingual(b, tokenizer.PAD_ID), num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        collate_fn=lambda b: collate_monolingual(b, tokenizer.PAD_ID), num_workers=config["training"]["num_workers"]
    )

    model = CustomGPT2(
        vocab_size=tokenizer.vocab_size_actual,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        num_kv_heads=config["model"]["num_kv_heads"],
        max_seq_len=config["model"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        pad_id=tokenizer.PAD_ID,
        add_cross_attention=config["model"].get("add_cross_attention", False)
    )

    config["training"]["checkpoint_dir"] = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    config["training"]["log_dir"] = PROJECT_ROOT / config["paths"]["log_dir"]

    trainer = GPT2Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=tokenizer.vocab_size_actual,
        pad_id=tokenizer.PAD_ID,
        config=config["training"],
        device=device
    )

    trainer.train()

if __name__ == "__main__":
    main()
