"""
CPU Smoke Test — Verifies every component works end-to-end.

Runs a tiny forward pass + backward pass through each model to catch
import errors, shape mismatches, and logic bugs BEFORE burning GPU hours.

Usage: python scripts/smoke_test.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_tokenizer():
    print("=" * 60)
    print("[1/6] Testing Tokenizer")
    print("=" * 60)
    from src.data.tokenizer import TokenizerWrapper

    tok = TokenizerWrapper(
        model_prefix="hi_mr_bpe",
        tokenizer_dir=str(PROJECT_ROOT / "data" / "tokenizer"),
    )
    tok.load()
    
    sample = "यह एक परीक्षण वाक्य है।"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    decoded = tok.decode(ids)
    
    assert len(ids) > 2, "Encoding produced too few tokens"
    assert ids[0] == tok.BOS_ID, "First token should be BOS"
    assert ids[-1] == tok.EOS_ID, "Last token should be EOS"
    
    print(f"  Input:   {sample}")
    print(f"  IDs:     {ids}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab:   {tok.vocab_size_actual:,}")
    print("  ✓ Tokenizer OK\n")
    return tok


def test_seq2seq(tok):
    print("=" * 60)
    print("[2/6] Testing Seq2Seq Model (forward + backward)")
    print("=" * 60)
    from src.models.seq2seq import Seq2SeqModel
    
    vocab_size = tok.vocab_size_actual
    model = Seq2SeqModel(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        embed_dim=64,       # Tiny for speed
        hidden_dim=32,
        attention_dim=16,
        num_layers=1,
        dropout=0.1,
        pad_id=tok.PAD_ID,
        bos_id=tok.BOS_ID,
        eos_id=tok.EOS_ID,
    )
    
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Fake batch
    batch_size, src_len, tgt_len = 4, 12, 10
    src_ids = torch.randint(4, vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(4, vocab_size, (batch_size, tgt_len))
    tgt_ids[:, 0] = tok.BOS_ID
    src_lengths = torch.full((batch_size,), src_len)
    src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    
    # Forward
    model.train()
    outputs, attentions = model(src_ids, src_lengths, src_mask, tgt_ids, teacher_forcing_ratio=0.5)
    
    assert outputs.shape == (batch_size, tgt_len - 1, vocab_size), f"Bad output shape: {outputs.shape}"
    assert attentions.shape[0] == batch_size, f"Bad attention shape: {attentions.shape}"
    
    # Backward
    loss = nn.CrossEntropyLoss(ignore_index=tok.PAD_ID)(
        outputs.reshape(-1, vocab_size), tgt_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    print(f"  Output shape: {outputs.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Inference (greedy)
    model.eval()
    with torch.no_grad():
        preds, _ = model.translate(src_ids, src_lengths, src_mask, max_len=15, beam_size=1)
    assert preds.shape[0] == batch_size
    print(f"  Greedy decode shape: {preds.shape}")
    
    # Inference (beam search)
    with torch.no_grad():
        preds_beam, _ = model.translate(src_ids[:1], src_lengths[:1], src_mask[:1], max_len=15, beam_size=3)
    print(f"  Beam search shape:  {preds_beam.shape}")
    print("  ✓ Seq2Seq OK\n")


def test_components():
    print("=" * 60)
    print("[3/6] Testing Components (RoPE, GQA, RMSNorm)")
    print("=" * 60)
    from src.models.components import RoPE, GroupedQueryAttention, RMSNorm
    
    batch, seq_len, hidden, heads, kv_heads = 2, 16, 64, 4, 2
    head_dim = hidden // heads
    
    # RoPE
    rope = RoPE(dim=head_dim, max_seq_len=128)
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, kv_heads, seq_len, head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print(f"  RoPE: q {q.shape} -> {q_rot.shape} ✓")
    
    # RMSNorm
    norm = RMSNorm(hidden)
    x = torch.randn(batch, seq_len, hidden)
    y = norm(x)
    assert y.shape == x.shape
    print(f"  RMSNorm: {x.shape} -> {y.shape} ✓")
    
    # GQA
    gqa = GroupedQueryAttention(
        hidden_dim=hidden, num_q_heads=heads, num_kv_heads=kv_heads, dropout=0.0
    )
    out, kv_cache = gqa(x, rope=rope)
    assert out.shape == x.shape
    print(f"  GQA: {x.shape} -> {out.shape} ✓")
    print("  ✓ Components OK\n")


def test_bert(tok):
    print("=" * 60)
    print("[4/6] Testing Custom BERT (forward + backward)")
    print("=" * 60)
    from src.models.bert import CustomBERT
    
    vocab_size = tok.vocab_size_actual
    model = CustomBERT(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
        dropout=0.1,
        pad_id=tok.PAD_ID,
    )
    print(f"  Parameters: {model.count_parameters():,}")
    
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # MLM forward
    model.train()
    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Bad shape: {logits.shape}"
    
    # Backward
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    print(f"  MLM logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Embedding mode
    model.eval()
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask=mask, return_embeddings=True)
    assert embeddings.shape == (batch_size, seq_len, 64)
    print(f"  Embeddings shape: {embeddings.shape}")
    print("  ✓ BERT OK\n")


def test_gpt2(tok):
    print("=" * 60)
    print("[5/6] Testing Custom GPT-2 (forward + backward)")
    print("=" * 60)
    from src.models.gpt2 import CustomGPT2
    
    vocab_size = tok.vocab_size_actual
    model = CustomGPT2(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
        dropout=0.1,
        pad_id=tok.PAD_ID,
        add_cross_attention=True,
    )
    print(f"  Parameters: {model.count_parameters():,}")
    
    batch_size, seq_len = 2, 15
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len))
    
    # Autoregressive forward
    model.train()
    logits, kv_caches = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Bad shape: {logits.shape}"
    
    # Backward
    targets = input_ids[:, 1:]
    inputs_logits = logits[:, :-1, :]
    loss = nn.CrossEntropyLoss()(inputs_logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()
    print(f"  LM logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # With cross-attention (encoder output)
    model.zero_grad()
    encoder_output = torch.randn(batch_size, 20, 64)
    encoder_mask = torch.ones(batch_size, 1, 1, 20)
    logits2, _ = model(input_ids, encoder_output=encoder_output, encoder_mask=encoder_mask)
    assert logits2.shape == (batch_size, seq_len, vocab_size)
    loss2 = nn.CrossEntropyLoss()(logits2[:, :-1].reshape(-1, vocab_size), targets.reshape(-1))
    loss2.backward()
    print(f"  Cross-attn logits shape: {logits2.shape}")
    print(f"  Cross-attn Loss: {loss2.item():.4f}")
    print("  ✓ GPT-2 OK\n")


def test_translator(tok):
    print("=" * 60)
    print("[6/6] Testing BERT-GPT2 Translator (forward + backward)")
    print("=" * 60)
    from src.models.bert import CustomBERT
    from src.models.gpt2 import CustomGPT2
    from src.models.translator import BERTGPT2Translator
    
    vocab_size = tok.vocab_size_actual
    
    encoder = CustomBERT(
        vocab_size=vocab_size, hidden_dim=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_seq_len=128, dropout=0.1, pad_id=tok.PAD_ID,
    )
    decoder = CustomGPT2(
        vocab_size=vocab_size, hidden_dim=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_seq_len=128, dropout=0.1,
        pad_id=tok.PAD_ID, add_cross_attention=True,
    )
    
    model = BERTGPT2Translator(
        encoder=encoder, decoder=decoder,
        pad_id=tok.PAD_ID, bos_id=tok.BOS_ID, eos_id=tok.EOS_ID,
    )
    print(f"  Parameters: {model.count_parameters():,}")
    
    batch_size, src_len, tgt_len = 2, 12, 10
    src_ids = torch.randint(4, vocab_size, (batch_size, src_len))
    tgt_ids = torch.randint(4, vocab_size, (batch_size, tgt_len))
    tgt_ids[:, 0] = tok.BOS_ID
    src_mask = torch.ones(batch_size, src_len, dtype=torch.long)
    src_lengths = torch.full((batch_size,), src_len)
    
    # Forward (training)
    model.train()
    logits, _ = model(src_ids, src_mask, tgt_ids)
    assert logits.shape == (batch_size, tgt_len - 1, vocab_size), f"Bad shape: {logits.shape}"
    
    # Backward
    targets = tgt_ids[:, 1:]
    loss = nn.CrossEntropyLoss(ignore_index=tok.PAD_ID)(
        logits.reshape(-1, vocab_size), targets.reshape(-1)
    )
    loss.backward()
    print(f"  Translator logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Inference (greedy)
    model.eval()
    with torch.no_grad():
        preds, _ = model.translate(src_ids, src_lengths, src_mask, max_len=15)
    assert preds.shape[0] == batch_size
    print(f"  Greedy decode shape: {preds.shape}")
    print("  ✓ Translator OK\n")


def main():
    print("\n" + "=" * 60)
    print("  AdiVaani NMT — CPU Smoke Test")
    print("=" * 60 + "\n")
    
    torch.manual_seed(42)
    
    tok = test_tokenizer()
    test_seq2seq(tok)
    test_components()
    test_bert(tok)
    test_gpt2(tok)
    test_translator(tok)
    
    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nThe entire codebase is verified and ready for GPU training.")
    print("Commands to run:")
    print("  Part I:  python scripts/train_seq2seq.py --config configs/part1_random_emb.yaml")
    print("  Part II: python scripts/train_bert.py --config configs/part2_bert_pretrain.yaml")


if __name__ == "__main__":
    main()
