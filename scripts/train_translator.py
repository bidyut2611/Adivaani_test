"""
Training Script for Part II: BERT-GPT2 Translator Model.

Fine-tunes the pretrained Custom BERT and Custom GPT-2 models on the parallel corpus.
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import TokenizerWrapper
from src.data.dataset import create_dataloaders
from src.models.bert import CustomBERT
from src.models.gpt2 import CustomGPT2
from src.models.translator import BERTGPT2Translator
from src.training.trainer import Seq2SeqTrainer


def load_pretrained_weights(model: nn.Module, checkpoint_path: str):
    """Safely load weights, ignoring mismatched layers (like cross-attention which is new)."""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # Load with strict=False to allow for newly initialized cross-attention layers
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys (expected if layers were added, e.g., cross-attention):")
        for k in missing_keys[:5]: print(f"    - {k}")
        if len(missing_keys) > 5: print(f"    ... and {len(missing_keys) - 5} more")
        
    if unexpected_keys:
        print(f"  Unexpected keys:")
        for k in unexpected_keys[:5]: print(f"    - {k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/part2_translation.yaml")
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = TokenizerWrapper(
        model_prefix=data_cfg.get("tokenizer_prefix", "hi_mr_bpe"), 
        tokenizer_dir=str(PROJECT_ROOT / data_cfg["tokenizer_dir"])
    )
    tokenizer.load()
        
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data_dir=str(PROJECT_ROOT / data_cfg["processed_dir"]),
        tokenizer=tokenizer,
        batch_size=train_cfg.get("batch_size", 32),
        max_len=data_cfg.get("max_len", 256),
        num_workers=train_cfg.get("num_workers", 0),
        use_bucket_batching=train_cfg.get("use_bucket_batching", True),
    )
    
    vocab_size = tokenizer.vocab_size_actual
    
    # Instantiate BERT Encoder
    bert_encoder = CustomBERT(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 12),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        dropout=model_cfg.get("dropout", 0.1),
        pad_id=tokenizer.PAD_ID,
    )
    
    # Instantiate GPT-2 Decoder with cross-attention enabled
    gpt2_decoder = CustomGPT2(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 12),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        max_seq_len=model_cfg.get("gpt_max_seq_len", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        pad_id=tokenizer.PAD_ID,
        add_cross_attention=True,  # Crucial for translation
    )
    
    # Load pre-trained weights if available
    bert_ckpt = PROJECT_ROOT / paths_cfg.get("bert_checkpoint", "")
    gpt2_ckpt = PROJECT_ROOT / paths_cfg.get("gpt2_checkpoint", "")
    
    load_pretrained_weights(bert_encoder, str(bert_ckpt))
    load_pretrained_weights(gpt2_decoder, str(gpt2_ckpt))
    
    # Create the Translator Wrapper
    model = BERTGPT2Translator(
        encoder=bert_encoder,
        decoder=gpt2_decoder,
        pad_id=tokenizer.PAD_ID,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
    )
    
    # Setup Trainer
    config["checkpoint_dir"] = str(PROJECT_ROOT / paths_cfg["checkpoint_dir"])
    config["log_dir"] = str(PROJECT_ROOT / paths_cfg["log_dir"])
    
    for k, v in train_cfg.items():
        config[k] = v
        
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
