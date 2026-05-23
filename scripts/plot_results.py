"""
Plotting Utility for Training Curves and Comparisons.

Generates the 6 required plots per experiment:
1. Train loss vs. epochs
2. Val loss vs. epochs
3. Train BLEU-100 vs. epochs
4. Val BLEU-100 vs. epochs
5. Train chrF++-100 vs. epochs
6. Val chrF++-100 vs. epochs

Also generates comparative charts across experiments.

Usage:
    python scripts/plot_results.py --history checkpoints/part1_random/training_history.json --name "Random Emb"
    python scripts/plot_results.py --compare checkpoints/part1_random/training_history.json checkpoints/part1_bert/training_history.json --names "Random" "BERT"
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.2)
except ImportError:
    print("ERROR: matplotlib/seaborn not installed.")
    sys.exit(1)


def load_history(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def plot_single_experiment(history: dict, name: str, output_dir: Path):
    """Generate the 6 required plots for one experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plots = [
        ("train_loss", "Train Loss", "Loss", "tab:blue"),
        ("val_loss", "Val Loss", "Loss", "tab:orange"),
        ("train_bleu", "Train BLEU-100", "BLEU-100", "tab:green"),
        ("val_bleu", "Val BLEU-100", "BLEU-100", "tab:red"),
        ("train_chrf", "Train chrF++-100", "chrF++-100", "tab:purple"),
        ("val_chrf", "Val chrF++-100", "chrF++-100", "tab:brown"),
    ]

    for key, title, ylabel, color in plots:
        if key not in history or not history[key]:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs[:len(history[key])], history[key], color=color, linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name} — {title}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {key}.png")

    # Combined loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    if history.get("val_loss"):
        ax.plot(epochs[:len(history["val_loss"])], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} — Train vs. Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_combined.png", dpi=150)
    plt.close(fig)

    # Combined BLEU plot
    if history.get("train_bleu") and history.get("val_bleu"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs[:len(history["train_bleu"])], history["train_bleu"], label="Train BLEU-100", linewidth=2)
        ax.plot(epochs[:len(history["val_bleu"])], history["val_bleu"], label="Val BLEU-100", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BLEU-100")
        ax.set_title(f"{name} — BLEU-100 Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "bleu_combined.png", dpi=150)
        plt.close(fig)

    print(f"  All plots saved to {output_dir}")


def plot_comparison(histories: list, names: list, output_dir: Path):
    """Generate comparison plots across experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_compare = [
        ("val_loss", "Val Loss Comparison", "Loss"),
        ("val_bleu", "Val BLEU-100 Comparison", "BLEU-100"),
        ("val_chrf", "Val chrF++-100 Comparison", "chrF++-100"),
    ]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for key, title, ylabel in metrics_to_compare:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (hist, name) in enumerate(zip(histories, names)):
            if key in hist and hist[key]:
                epochs = list(range(1, len(hist[key]) + 1))
                color = colors[i % len(colors)]
                ax.plot(epochs, hist[key], label=name, linewidth=2, color=color, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"compare_{key}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved compare_{key}.png")

    # Summary bar chart of final metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bleu_scores = [h.get("val_bleu", [0])[-1] for h in histories]
    chrf_scores = [h.get("val_chrf", [0])[-1] for h in histories]

    axes[0].bar(names, bleu_scores, color=colors[:len(names)], alpha=0.8)
    axes[0].set_ylabel("BLEU-100")
    axes[0].set_title("Final Val BLEU-100")
    for i, v in enumerate(bleu_scores):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

    axes[1].bar(names, chrf_scores, color=colors[:len(names)], alpha=0.8)
    axes[1].set_ylabel("chrF++-100")
    axes[1].set_title("Final Val chrF++-100")
    for i, v in enumerate(chrf_scores):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "compare_final_metrics.png", dpi=150)
    plt.close(fig)
    print(f"  All comparison plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Training Results")
    parser.add_argument("--history", type=str, nargs="+", help="Path(s) to training_history.json")
    parser.add_argument("--names", type=str, nargs="+", help="Names for each experiment")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Generate comparison charts")
    args = parser.parse_args()

    if not args.history:
        print("No history files provided. Nothing to plot.")
        return

    output_dir = PROJECT_ROOT / args.output_dir
    names = args.names or [f"Experiment {i+1}" for i in range(len(args.history))]

    histories = [load_history(PROJECT_ROOT / h) for h in args.history]

    if len(histories) == 1:
        plot_single_experiment(histories[0], names[0], output_dir / names[0].lower().replace(" ", "_"))
    else:
        # Plot each individually
        for hist, name in zip(histories, names):
            plot_single_experiment(hist, name, output_dir / name.lower().replace(" ", "_"))
        # Plot comparison
        if args.compare or len(histories) > 1:
            plot_comparison(histories, names, output_dir / "comparison")


if __name__ == "__main__":
    main()
