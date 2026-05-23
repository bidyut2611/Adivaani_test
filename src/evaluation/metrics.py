"""
Evaluation Metrics for Hindi-Marathi NMT.

Implements BLEU-100 and chrF++-100 scoring using sacrebleu.
Scores are reported on a 0-100 scale as required by the assessment.

BLEU (Bilingual Evaluation Understudy):
    - Measures n-gram precision between hypothesis and reference
    - Uses brevity penalty to penalize too-short translations
    - BLEU-100 = BLEU * 100 (scale 0 to 100)

chrF++ (Character F-score++):
    - Character-level n-gram F-score
    - More robust for morphologically rich languages like Hindi/Marathi
    - chrF++-100 = chrF++ * 100 (scale 0 to 100)
"""

import sacrebleu
from typing import List, Dict, Optional


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "intl",
) -> float:
    """
    Compute BLEU-100 score.

    Args:
        hypotheses: List of system translations
        references: List of reference translations
        tokenize: Tokenization method for sacrebleu
                   'intl' works well for non-Latin scripts

    Returns:
        BLEU score on 0-100 scale
    """
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],  # sacrebleu expects list of reference lists
        tokenize=tokenize,
    )
    return bleu.score  # Already on 0-100 scale


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    word_order: int = 2,
) -> float:
    """
    Compute chrF++-100 score.

    Args:
        hypotheses: List of system translations
        references: List of reference translations
        word_order: Word n-gram order (2 for chrF++)

    Returns:
        chrF++ score on 0-100 scale
    """
    chrf = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=word_order,  # chrF++ uses word_order=2
    )
    return chrf.score  # Already on 0-100 scale


def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        hypotheses: List of system translations
        references: List of reference translations

    Returns:
        Dictionary with BLEU-100 and chrF++-100 scores
    """
    bleu = compute_bleu(hypotheses, references)
    chrf = compute_chrf(hypotheses, references)

    return {
        "bleu_100": round(bleu, 2),
        "chrf_100": round(chrf, 2),
    }


def compute_sentence_bleu(
    hypothesis: str,
    reference: str,
) -> float:
    """Compute sentence-level BLEU score."""
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu.score


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix_str = f"[{prefix}] " if prefix else ""
    print(f"  {prefix_str}BLEU-100:  {metrics['bleu_100']:.2f}")
    print(f"  {prefix_str}chrF++-100: {metrics['chrf_100']:.2f}")


if __name__ == "__main__":
    # Quick sanity check
    hyps = ["यह एक परीक्षण है ।", "मैं घर जा रहा हूँ ।"]
    refs = ["यह एक परीक्षण है ।", "मैं घर जा रहा हूँ ।"]

    metrics = compute_all_metrics(hyps, refs)
    print("Perfect match test:")
    print_metrics(metrics)
    print()

    # Imperfect match
    hyps2 = ["यह एक परीक्षा है ।", "मैं स्कूल जा रहा हूँ ।"]
    metrics2 = compute_all_metrics(hyps2, refs)
    print("Partial match test:")
    print_metrics(metrics2)
