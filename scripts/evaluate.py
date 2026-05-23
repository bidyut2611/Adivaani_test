"""
Evaluation Script — Run evaluation on a saved checkpoint.

Usage:
    python scripts/evaluate.py --config configs/part1_random_emb.yaml --checkpoint checkpoints/part1_random/best_model.pt
    python scripts/evaluate.py --config configs/part1_random_emb.yaml --checkpoint checkpoints/part1_random/best_model.pt --beam-size 5
    python scripts/evaluate.py --config configs/part2_translation.yaml --checkpoint checkpoints/part2_translator/best_model.pt --model-type translator
"""

import argparse
import sys
import json
import torch
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import TokenizerWrapper
from src.data.dataset import create_dataloaders
from src.models.seq2seq import Seq2SeqModel
from src.models.bert import CustomBERT
from src.models.gpt2 import CustomGPT2
from src.models.translator import BERTGPT2Translator
from src.evaluation.inference import evaluate_model, print_examples
from src.evaluation.metrics import print_metrics


def load_seq2seq(config, tokenizer, device):
    """Load a Seq2Seq model from checkpoint."""
    model_cfg = config["model"]
    vocab_size = tokenizer.vocab_size_actual
    model = Seq2SeqModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=model_cfg.get("embed_dim", 768),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        attention_dim=model_cfg.get("attention_dim", 256),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pad_id=tokenizer.PAD_ID,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
    )
    return model


def load_translator(config, tokenizer, device):
    """Load a BERT-GPT2 Translator model from checkpoint."""
    model_cfg = config["model"]
    vocab_size = tokenizer.vocab_size_actual

    bert_encoder = CustomBERT(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 12),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        dropout=0.0,  # No dropout during eval
        pad_id=tokenizer.PAD_ID,
    )
    gpt2_decoder = CustomGPT2(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        num_layers=model_cfg.get("num_layers", 12),
        num_heads=model_cfg.get("num_heads", 12),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        max_seq_len=model_cfg.get("gpt_max_seq_len", 1024),
        dropout=0.0,
        pad_id=tokenizer.PAD_ID,
        add_cross_attention=True,
    )
    model = BERTGPT2Translator(
        encoder=bert_encoder,
        decoder=gpt2_decoder,
        pad_id=tokenizer.PAD_ID,
        bos_id=tokenizer.BOS_ID,
        eos_id=tokenizer.EOS_ID,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT Model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", choices=["seq2seq", "translator"], default="seq2seq")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    data_cfg = config["data"]
    tokenizer = TokenizerWrapper(
        model_prefix=data_cfg.get("tokenizer_prefix", "hi_mr_bpe"),
        tokenizer_dir=str(PROJECT_ROOT / data_cfg["tokenizer_dir"]),
    )
    tokenizer.load()

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data_dir=str(PROJECT_ROOT / data_cfg["processed_dir"]),
        tokenizer=tokenizer,
        batch_size=config.get("training", {}).get("batch_size", 32),
        max_len=data_cfg.get("max_len", 256),
        num_workers=0,
        use_bucket_batching=False,  # Deterministic eval
    )

    eval_loader = test_loader if args.split == "test" else val_loader

    # Build model architecture
    if args.model_type == "translator":
        model = load_translator(config, tokenizer, device)
    else:
        model = load_seq2seq(config, tokenizer, device)

    # Load checkpoint
    ckpt_path = PROJECT_ROOT / args.checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, val BLEU: {checkpoint.get('val_bleu', '?')}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Evaluate
    print(f"\nEvaluating on {args.split} set (beam_size={args.beam_size})...")
    results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        beam_size=args.beam_size,
        max_len=args.max_len,
        device=device,
        max_batches=None,  # Evaluate on full set
        desc=f"Eval [{args.split}]",
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Results on {args.split} set ({results['metrics']['num_sentences']} sentences)")
    print(f"{'='*60}")
    print_metrics(results["metrics"])
    print_examples(results["examples"], n=10)

    # Save results
    if args.output:
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            "metrics": results["metrics"],
            "config": args.config,
            "checkpoint": args.checkpoint,
            "beam_size": args.beam_size,
            "split": args.split,
            "examples": results["examples"][:20],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
