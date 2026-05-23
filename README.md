# AdiVaani Hindi↔Marathi Neural Machine Translation

A complete, from-scratch Hindi↔Marathi Neural Machine Translation system developed for the **MISN Lab, IIT Delhi (AdiVaani Initiative) Hiring Assessment (2026)**. The project spans two phases — a classical Seq2Seq baseline and a modern Transformer-based encoder-decoder system — demonstrating mastery over both paradigms of sequence-to-sequence modelling.

> **All models, training loops, evaluation pipelines, and architectural components are implemented entirely from scratch** using PyTorch, without high-level NMT abstractions (e.g., `MarianMT`, `mT5`, or HuggingFace `Seq2SeqTrainer`).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
  - [Part I — Classical Seq2Seq](#part-i--classical-seq2seq)
  - [Part II — Modern Pretrained Framework](#part-ii--modern-pretrained-framework)
- [Results](#results)
- [Setup & Reproducibility](#setup--reproducibility)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Training Part I (Classical Seq2Seq)](#3-training-part-i-classical-seq2seq)
  - [4. Training Part II (Modern Framework)](#4-training-part-ii-modern-framework)
  - [5. Evaluation](#5-evaluation)
  - [6. Plotting](#6-plotting)
- [Configuration Reference](#configuration-reference)
- [CPU Smoke Test](#cpu-smoke-test)
- [Technical Report](#technical-report)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

```
adivaani-nmt/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── configs/                         # YAML configuration files
│   ├── part1_random_emb.yaml        #   Seq2Seq with random embeddings
│   ├── part1_bert_emb.yaml          #   Seq2Seq with pretrained BERT embeddings
│   ├── part2_bert_pretrain.yaml     #   Custom BERT pretraining (MLM)
│   ├── part2_gpt2_pretrain.yaml     #   Custom GPT-2 pretraining (CLM)
│   └── part2_translation.yaml       #   BERT-GPT2 Translator fine-tuning
│
├── data/
│   ├── raw/                         # Original parallel corpus
│   ├── processed/                   # Cleaned, filtered, train/val/test splits
│   └── tokenizer/                   # Trained SentencePiece BPE model (32K vocab)
│
├── src/                             # Core library
│   ├── data/
│   │   ├── preprocessing.py         #   Unicode normalisation, filtering, splitting
│   │   ├── tokenizer.py             #   SentencePiece BPE wrapper
│   │   └── dataset.py               #   PyTorch Dataset & BucketBatchSampler
│   │
│   ├── models/
│   │   ├── attention.py             #   Bahdanau & Luong attention mechanisms
│   │   ├── seq2seq.py               #   BiLSTM Encoder-Decoder with beam search
│   │   ├── components.py            #   RoPE, RMSNorm, GQA, CrossAttention
│   │   ├── bert.py                  #   Custom BERT (~110M params)
│   │   ├── gpt2.py                  #   Custom GPT-2 (~124M params)
│   │   └── translator.py            #   BERT-GPT2 Encoder-Decoder translator
│   │
│   ├── training/
│   │   ├── trainer.py               #   Seq2Seq & Translator training loop
│   │   ├── pretrain_bert.py         #   BERT MLM pretraining loop
│   │   └── pretrain_gpt2.py         #   GPT-2 CLM pretraining loop
│   │
│   └── evaluation/
│       ├── metrics.py               #   BLEU-100 & chrF++-100 via sacrebleu
│       └── inference.py             #   Greedy / beam search decoding
│
├── scripts/                         # Execution entry points
│   ├── prepare_data.py              #   Full data pipeline
│   ├── train_seq2seq.py             #   Train Part I models
│   ├── train_bert.py                #   Pretrain Custom BERT
│   ├── train_gpt2.py                #   Pretrain Custom GPT-2
│   ├── train_translator.py          #   Fine-tune BERT-GPT2 Translator
│   ├── evaluate.py                  #   Evaluate any model on test/val set
│   ├── plot_results.py              #   Generate training curves & comparisons
│   └── smoke_test.py               #   CPU end-to-end verification
│
├── checkpoints/                     # Saved model weights & training histories
├── results/                         # Evaluation JSON outputs
├── plots/                           # Training curve visualisations
├── logs/                            # TensorBoard logs
└── report/
    └── technical_report.md          # Detailed technical report
```

---

## Key Features

| Category | Details |
|:---------|:--------|
| **Data Pipeline** | Unicode NFC normalisation · Length & ratio filtering · 95/5 train/val split · Shared BPE tokeniser (32K vocab) · `BucketBatchSampler` for minimal padding |
| **Classical NMT (Part I)** | 2-layer BiLSTM Encoder · Bahdanau Attention · Unidirectional LSTM Decoder · Beam search (configurable width) |
| **Modern NMT (Part II)** | Custom BERT (~110M) · Custom GPT-2 (~124M) · RoPE · GQA (12Q / 4KV) · RMSNorm · Cross-Attention fusion |
| **Training** | Mixed-precision (AMP) · Label smoothing (ε=0.1) · Cosine LR with linear warmup · Scheduled teacher forcing (1.0→0.3) · Early stopping · Gradient clipping |
| **Evaluation** | `sacrebleu` BLEU-100 & chrF++-100 · Greedy & beam search decoding · Detokenised scoring |
| **Reproducibility** | Seed-controlled · YAML configs · TensorBoard logging · Automated smoke tests |

---

## Architecture Overview

### Part I — Classical Seq2Seq

A classical encoder-decoder NMT system based on LSTMs and additive attention.

```
┌─────────────────────────────────────────────────────────┐
│                    Seq2Seq Model                        │
│                                                         │
│  ┌──────────────────────┐   ┌────────────────────────┐  │
│  │   Bidirectional LSTM │   │  Unidirectional LSTM   │  │
│  │      Encoder         │──▶│      Decoder           │  │
│  │  (2 layers, 512d)    │   │  (2 layers, 512d)      │  │
│  └──────────┬───────────┘   └──────────┬─────────────┘  │
│             │                          │                │
│             ▼                          ▼                │
│       Encoder Outputs ───▶ Bahdanau Attention (256d)    │
│                                        │                │
│                                        ▼                │
│                              Vocabulary Projection      │
│                               (32K shared BPE)          │
└─────────────────────────────────────────────────────────┘
```

**Embedding strategies:**
- **Experiment 1 — Random:** Xavier-initialised embeddings (768d), trained from scratch.
- **Experiment 2 — BERT:** Initialised from `l3cube-pune/hindi-bert-v2` and `marathi-bert-v2`, aligned to the shared BPE vocabulary.

### Part II — Modern Pretrained Framework

A Transformer-based encoder-decoder system constructed via language model pretraining and cross-attention fusion.

```
┌───────────────────────────────────────────────────────────────────┐
│                  BERT-GPT2 Translator                             │
│                                                                   │
│  ┌───────────────────────┐         ┌───────────────────────────┐  │
│  │  Custom BERT Encoder  │         │  Custom GPT-2 Decoder     │  │
│  │  (~110M params)       │         │  (~124M params)           │  │
│  │                       │         │                           │  │
│  │  12 Transformer Layers│         │  12 Transformer Layers    │  │
│  │  • RMSNorm (Pre-Norm) │         │  • RMSNorm (Pre-Norm)    │  │
│  │  • GQA (12Q / 4KV)   │         │  • GQA (12Q / 4KV)       │  │
│  │  • RoPE               │         │  • RoPE                  │  │
│  │  • GELU MLP           │  ┌─────▶│  • Cross-Attention       │  │
│  │                       │  │      │  • GELU MLP              │  │
│  └───────────┬───────────┘  │      └──────────┬────────────────┘  │
│              │              │                 │                   │
│              └──────────────┘                 ▼                   │
│         Encoder Hidden States         LM Head (weight-tied)       │
│                                       (32K shared BPE)            │
└───────────────────────────────────────────────────────────────────┘
```

**Modern components (implemented from scratch in `src/models/components.py`):**

| Component | Description |
|:----------|:------------|
| **RoPE** (Rotary Positional Embeddings) | Injects positional information by rotating Q/K vectors at every layer. Superior length extrapolation over absolute embeddings. |
| **GQA** (Grouped Query Attention) | 12 query heads share 4 KV heads, reducing KV-cache memory by 3× while maintaining near-MHA quality. |
| **RMSNorm** | Replaces LayerNorm by omitting mean-centering — 10–15% faster with identical convergence. |
| **CrossAttention** | Decoder queries attend to encoder keys/values, bridging BERT and GPT-2 for translation. |

**Pretraining pipeline:**
1. **BERT** is pretrained on the Hindi corpus using Masked Language Modelling (MLM, 15% dynamic masking).
2. **GPT-2** is pretrained on the Marathi corpus using Causal Language Modelling (next-token prediction).
3. Cross-attention layers are injected into GPT-2, and the full **BERT-GPT2 Translator** is fine-tuned on the parallel Hindi→Marathi corpus.

---

## Results

All models were evaluated on the held-out test set (10,332 sentence pairs) using greedy decoding.

| Model | Strategy | Test BLEU-100 | Test chrF++ | Best Val BLEU | Epochs |
|:------|:---------|:-------------:|:-----------:|:-------------:|:------:|
| **Seq2Seq (Baseline)** | Random Embeddings | **9.26** | **31.02** | 18.19 | 30 |
| **Seq2Seq** | Pretrained BERT Embeddings | **7.18** | **27.47** | 14.12 | 15 |
| **BERT-GPT2 Translator** | Pretrained BERT + GPT-2 | **13.07** | **33.47** | 21.73 | 15 |

**Key observations:**
- The **BERT-GPT2 Translator** achieves the highest test BLEU (13.07) and chrF++ (33.47), demonstrating that pretrained Transformer representations significantly outperform the classical LSTM baseline.
- The Seq2Seq with **random embeddings** (9.26 BLEU) outperforms the **BERT embedding** variant (7.18 BLEU), likely because the frozen BERT tokeniser vocabulary does not align perfectly with the shared BPE vocabulary, introducing noise during embedding projection.
- **chrF++** scores are consistently higher than BLEU across all models, reflecting the metric's suitability for morphologically rich languages (Hindi/Marathi).

### Sample Translations (BERT-GPT2 Translator)

| Source (Hindi) | Reference (Marathi) | Hypothesis |
|:---------------|:--------------------|:-----------|
| यदि श्वास प्रणालिका में सूजन आ जाये तब भी रक्त मुँह के रास्ते बाहर आने लगता है | जर श्वासनलिकेला सूज आली तरीही रक्त तोंडावाटे बाहेर येऊ लागते | जर श्वासनलिकेला सूज आली असेल तरीही रक्त तोंडाच्या मार्गांना बाहेर येऊ लागते |
| नाश्ता नहीं करने पर आपका उपापचय दोपहर के भोजन तक शुरू नहीं होता | नाश्ता केल्यानंतर तुमचा उपापचय दुपारच्या जेवणापर्यंत सुरू होत नाही | न्याहारी नाही केली तर तुमचा उपापचय दुपारपर्यंत सुरू होत नाही |
| अनेक अवसरों पर आरोही गलत निर्णय ले लेते हैं | अनेक वेळेला गिर्यारोहक चुकीचे निर्णय घेतात | अनेक वेळा गिर्यारोहक चुकीच्या निर्णय घेऊ शकतात |

---

## Setup & Reproducibility

### 1. Environment Setup

**Requirements:** Python ≥ 3.8, CUDA-capable GPU (recommended: A100 80GB for Part II)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** `torch>=2.0`, `transformers>=4.30`, `sentencepiece>=0.1.99`, `sacrebleu>=2.3`, `tensorboard>=2.13`, `matplotlib>=3.7`, `seaborn>=0.12`, `pyyaml>=6.0`, `tqdm>=4.65`, `numpy>=1.24`, `pandas>=2.0`

### 2. Data Preparation

Place the raw dataset (`train.hi`, `train.mr`, `test.hi`, `test.mr`) in `data/raw/`, then run:

```bash
python scripts/prepare_data.py --config configs/part1_random_emb.yaml
```

This will:
1. Copy raw data into the project structure
2. Apply Unicode NFC normalisation
3. Filter by sentence length (2–150 words) and length ratio (≤3×)
4. Create a 95/5 train/validation split
5. Train a shared SentencePiece BPE tokeniser (32,000 vocab)

### 3. Training Part I (Classical Seq2Seq)

#### Experiment 1 — Random Embeddings
```bash
python scripts/train_seq2seq.py --config configs/part1_random_emb.yaml
```

#### Experiment 2 — BERT Pretrained Embeddings
```bash
python scripts/train_seq2seq.py --config configs/part1_bert_emb.yaml
```

### 4. Training Part II (Modern Framework)

#### Step 1: Pretrain Custom BERT (Hindi, MLM)
```bash
python scripts/train_bert.py --config configs/part2_bert_pretrain.yaml
```

#### Step 2: Pretrain Custom GPT-2 (Marathi, CLM)
```bash
python scripts/train_gpt2.py --config configs/part2_gpt2_pretrain.yaml
```

> **Note:** Steps 1 and 2 can be run in **parallel** on separate GPUs if available.

#### Step 3: Fine-tune BERT-GPT2 Translator
```bash
python scripts/train_translator.py --config configs/part2_translation.yaml
```

This automatically loads pretrained weights from `checkpoints/bert_pretrain/best_bert.pt` and `checkpoints/gpt2_pretrain/best_gpt2.pt`, injects cross-attention layers (trained from scratch), and fine-tunes the full model on the parallel corpus.

### 5. Evaluation

Evaluate any trained model on the test set:

```bash
# Part I — Seq2Seq
python scripts/evaluate.py \
  --config configs/part1_random_emb.yaml \
  --checkpoint checkpoints/part1_random/best_model.pt \
  --output results/part1_random_emb.json

# Part II — BERT-GPT2 Translator
python scripts/evaluate.py \
  --config configs/part2_translation.yaml \
  --checkpoint checkpoints/part2_translator/best_model.pt \
  --model-type translator \
  --output results/part2_translator.json
```

Optional flags: `--beam-size 5` for beam search, `--split val` for validation set evaluation.

### 6. Plotting

Generate training curves and comparison charts:

```bash
# Single experiment
python scripts/plot_results.py \
  --history checkpoints/part1_random/training_history.json \
  --names "Random Emb"

# Cross-experiment comparison
python scripts/plot_results.py \
  --history checkpoints/part1_random/training_history.json \
           checkpoints/part1_bert/training_history.json \
           checkpoints/part2_translator/training_history.json \
  --names "Random Emb" "BERT Emb" "BERT-GPT2" \
  --compare
```

Plots are saved to `plots/` and include per-experiment loss/BLEU/chrF++ curves and a cross-experiment comparison bar chart.

### TensorBoard

Monitor training in real-time:

```bash
tensorboard --logdir logs/
```

---

## Configuration Reference

All training runs are controlled via YAML configs in `configs/`. Key parameters:

| Parameter | Part I | Part II (Pretrain) | Part II (Translator) |
|:----------|:------:|:------------------:|:--------------------:|
| `hidden_dim` | 512 (LSTM) | 768 | 768 |
| `embed_dim` | 768 | — | — |
| `num_layers` | 2 | 12 | 12 |
| `num_heads` | — | 12 | 12 |
| `num_kv_heads` (GQA) | — | 4 | 4 |
| `batch_size` | 64 | 32 | 32 |
| `learning_rate` | 3e-4 | 1e-4 / 3e-4 | 5e-5 |
| `num_epochs` | 30 | 10 | 15 |
| `label_smoothing` | 0.1 | — | 0.1 |
| `use_amp` | ✓ | ✓ | ✓ |
| `warmup_steps` | 1,000 | 10,000 | 2,000 |

---

## CPU Smoke Test

Before committing to GPU training, validate the entire pipeline on CPU:

```bash
python scripts/smoke_test.py
```

This runs a complete forward + backward pass through every component:
1. **Tokeniser** — encode/decode roundtrip
2. **Seq2Seq** — forward, backward, greedy decode, beam search
3. **Components** — RoPE, GQA, RMSNorm shape verification
4. **Custom BERT** — MLM forward/backward, embedding extraction
5. **Custom GPT-2** — CLM forward/backward, cross-attention mode
6. **BERT-GPT2 Translator** — full encoder-decoder forward/backward, greedy decode

---

## Technical Report

A detailed technical report covering design decisions, architectural choices, training strategies, and quantitative analysis is available at:

```
report/technical_report.md
```

---

## Acknowledgements

- **Dataset:** IIT Delhi MISN Lab — AdiVaani Hindi-Marathi Parallel Corpus
- **BERT Embeddings (Part I):** [L3Cube Pune](https://huggingface.co/l3cube-pune) (`hindi-bert-v2`, `marathi-bert-v2`)
- **Tokenisation:** [SentencePiece](https://github.com/google/sentencepiece) BPE
- **Evaluation:** [sacrebleu](https://github.com/mjpost/sacrebleu)

---

**Author:** Bidyut · MISN Lab Assessment, IIT Delhi (2026)
