"""
Inference utilities for NMT models.

Handles:
- Batch translation with greedy/beam search decoding
- Post-processing of translations
- Qualitative analysis output
"""

import torch
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from ..data.tokenizer import TokenizerWrapper
from ..evaluation.metrics import compute_all_metrics, compute_sentence_bleu


def translate_batch(
    model,
    batch: Dict,
    tokenizer: TokenizerWrapper,
    beam_size: int = 1,
    max_len: int = 200,
    device: str = "cuda",
) -> List[str]:
    """
    Translate a batch of source sentences.

    Args:
        model: Seq2Seq model
        batch: Batch dictionary from DataLoader
        tokenizer: Tokenizer for decoding
        beam_size: Beam search width (1 = greedy)
        max_len: Maximum output length
        device: Device to run on

    Returns:
        List of translated strings
    """
    model.eval()

    src_ids = batch["src_ids"].to(device)
    src_lengths = batch["src_lengths"].to(device)
    src_mask = batch["src_mask"].to(device)

    predictions, _ = model.translate(
        src_ids, src_lengths, src_mask, max_len=max_len, beam_size=beam_size
    )

    # Decode predictions
    translations = []
    for pred in predictions:
        pred_list = pred.cpu().tolist()
        # Truncate at EOS if present
        if tokenizer.EOS_ID in pred_list:
            eos_idx = pred_list.index(tokenizer.EOS_ID)
            pred_list = pred_list[:eos_idx]
        translation = tokenizer.decode(pred_list, skip_special=True)
        translations.append(translation)

    return translations


def evaluate_model(
    model,
    dataloader,
    tokenizer: TokenizerWrapper,
    beam_size: int = 1,
    max_len: int = 200,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    desc: str = "Evaluating",
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Seq2Seq model
        dataloader: DataLoader to evaluate on
        tokenizer: Tokenizer for decoding
        beam_size: Beam search width
        max_len: Maximum output length
        device: Device
        max_batches: Limit evaluation to first N batches (for speed during training)
        desc: Progress bar description

    Returns:
        Dictionary with metrics and example translations
    """
    model.eval()
    all_hypotheses = []
    all_references = []
    examples = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches and i >= max_batches:
                break

            translations = translate_batch(
                model, batch, tokenizer, beam_size=beam_size,
                max_len=max_len, device=device
            )

            all_hypotheses.extend(translations)
            all_references.extend(batch["tgt_texts"])

            # Save a few examples for qualitative analysis
            if len(examples) < 10:
                for src, ref, hyp in zip(
                    batch["src_texts"], batch["tgt_texts"], translations
                ):
                    if len(examples) < 10:
                        examples.append({
                            "source": src,
                            "reference": ref,
                            "hypothesis": hyp,
                        })

    # Compute metrics
    metrics = compute_all_metrics(all_hypotheses, all_references)
    metrics["num_sentences"] = len(all_hypotheses)

    return {
        "metrics": metrics,
        "examples": examples,
        "hypotheses": all_hypotheses,
        "references": all_references,
    }


def print_examples(examples: List[Dict], n: int = 5) -> None:
    """Pretty-print translation examples."""
    print(f"\n{'='*70}")
    print(f"Sample Translations (showing {min(n, len(examples))} examples)")
    print(f"{'='*70}")

    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Example {i+1} ---")
        print(f"  Source (HI):    {ex['source'][:100]}")
        print(f"  Reference (MR): {ex['reference'][:100]}")
        print(f"  Hypothesis:     {ex['hypothesis'][:100]}")
    print()
