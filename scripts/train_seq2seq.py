"""
Training Script for Part I: Classical Seq2Seq NMT Model.

Usage:
    python scripts/train_seq2seq.py --config configs/part1_random_emb.yaml
    python scripts/train_seq2seq.py --config configs/part1_bert_emb.yaml
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import TokenizerWrapper
from src.data.dataset import create_dataloaders
from src.models.seq2seq import Seq2SeqModel
from src.training.trainer import Seq2SeqTrainer


def get_pretrained_embeddings(vocab_size: int, embed_dim: int, tokenizer: TokenizerWrapper, bert_model_name: str) -> nn.Embedding:
    """
    Extract embeddings from a pretrained BERT model and align with our vocabulary.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers library not installed. Please pip install transformers")
        sys.exit(1)
        
    print(f"Loading pretrained embeddings from {bert_model_name}...")
    
    # Load BERT model and its tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    bert_model = AutoModel.from_pretrained(bert_model_name)
    
    bert_embeddings = bert_model.embeddings.word_embeddings.weight.data
    assert bert_embeddings.shape[1] == embed_dim, \
        f"BERT embedding dim ({bert_embeddings.shape[1]}) doesn't match config ({embed_dim})"
    
    # Create new embedding layer for our Seq2Seq model
    new_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    # Initialize with normal distribution (same as BERT)
    nn.init.normal_(new_embeddings.weight, mean=0.0, std=0.02)
    
    print(f"Aligning vocabularies (Our Vocab: {vocab_size}, BERT Vocab: {bert_embeddings.shape[0]})...")
    
    # Match tokens between our SentencePiece tokenizer and BERT's tokenizer
    matched_count = 0
    
    # Zero is padding for both usually, but let's be explicit
    new_embeddings.weight.data[0].fill_(0)
    
    for i in range(1, vocab_size):
        # Skip our special tokens if they don't map cleanly
        piece = tokenizer.id_to_piece(i)
        
        # Clean up SentencePiece prefix (usually ' ')
        if piece.startswith(' '):
            piece = piece[1:]
            
        if not piece:
            continue
            
        # Get BERT token ID
        bert_ids = bert_tokenizer.encode(piece, add_special_tokens=False)
        
        # If it's a single token in BERT, we can copy the embedding directly
        if len(bert_ids) == 1:
            bert_id = bert_ids[0]
            if bert_id < bert_embeddings.shape[0]: # ensure within bounds
                new_embeddings.weight.data[i] = bert_embeddings[bert_id].clone()
                matched_count += 1
        # If it's multiple tokens in BERT, average their embeddings
        elif len(bert_ids) > 1:
            valid_ids = [bid for bid in bert_ids if bid < bert_embeddings.shape[0]]
            if valid_ids:
                avg_emb = bert_embeddings[valid_ids].mean(dim=0)
                new_embeddings.weight.data[i] = avg_emb.clone()
                matched_count += 1
                
    print(f"  Successfully initialized {matched_count}/{vocab_size} ({matched_count/vocab_size*100:.1f}%) embeddings from BERT.")
    
    # Free memory
    del bert_model
    del bert_tokenizer
    
    return new_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq NMT Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]
    
    # Set seed
    torch.manual_seed(config.get("seed", 42))
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_dir = PROJECT_ROOT / data_cfg["tokenizer_dir"]
    tokenizer_prefix = data_cfg.get("tokenizer_prefix", "hi_mr_bpe")
    
    tokenizer = TokenizerWrapper(model_prefix=tokenizer_prefix, tokenizer_dir=str(tokenizer_dir))
    try:
        tokenizer.load()
    except FileNotFoundError:
        print("ERROR: Tokenizer not found. Run scripts/prepare_data.py first.")
        sys.exit(1)
        
    # Create DataLoaders
    processed_dir = str(PROJECT_ROOT / data_cfg["processed_dir"])
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data_dir=processed_dir,
        tokenizer=tokenizer,
        batch_size=train_cfg.get("batch_size", 64),
        max_len=data_cfg.get("max_len", 256),
        num_workers=train_cfg.get("num_workers", 0),
        use_bucket_batching=train_cfg.get("use_bucket_batching", True),
    )
    
    # Setup Embeddings
    src_pretrained_embeddings = None
    tgt_pretrained_embeddings = None
    
    vocab_size = tokenizer.vocab_size_actual
    
    if model_cfg.get("embedding_type") == "bert":
        print("\nSetting up BERT embeddings...")
        src_bert = model_cfg.get("hindi_bert", "l3cube-pune/hindi-bert-v2")
        tgt_bert = model_cfg.get("marathi_bert", "l3cube-pune/marathi-bert-v2")
        
        src_pretrained_embeddings = get_pretrained_embeddings(
            vocab_size=vocab_size,
            embed_dim=model_cfg["embed_dim"],
            tokenizer=tokenizer,
            bert_model_name=src_bert
        )
        
        tgt_pretrained_embeddings = get_pretrained_embeddings(
            vocab_size=vocab_size,
            embed_dim=model_cfg["embed_dim"],
            tokenizer=tokenizer,
            bert_model_name=tgt_bert
        )
    else:
        print("\nUsing random embedding initialization.")
        
    # Create Model
    model = Seq2SeqModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=model_cfg.get("embed_dim", 768),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        attention_dim=model_cfg.get("attention_dim", 256),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        src_pretrained_embeddings=src_pretrained_embeddings,
        tgt_pretrained_embeddings=tgt_pretrained_embeddings,
        freeze_embeddings=model_cfg.get("freeze_embeddings", False),
        pad_id=tokenizer.PAD_ID,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
    )
    
    # Setup Trainer
    # Add absolute paths to config for trainer
    config["checkpoint_dir"] = str(PROJECT_ROOT / paths_cfg["checkpoint_dir"])
    config["log_dir"] = str(PROJECT_ROOT / paths_cfg["log_dir"])
    
    # Add flat training params from yaml for the Trainer
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
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        trainer.save_checkpoint(epoch=-1, val_bleu=trainer.best_val_bleu, is_best=False)
        print(f"Checkpoint saved in {config['checkpoint_dir']}")
        sys.exit(0)


if __name__ == "__main__":
    main()
