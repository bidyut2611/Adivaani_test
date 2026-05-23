"""
Training Loop for Seq2Seq NMT Models.

Features:
- Teacher forcing with scheduled sampling (linear decay)
- Gradient clipping for training stability
- Learning rate scheduling (cosine with warmup)
- BLEU/chrF++ evaluation at configurable intervals
- TensorBoard logging
- Best model checkpointing (by validation BLEU)
- Training/validation loss tracking
- Mixed precision training support (AMP)

Design Decisions:
- We use label smoothing cross-entropy to prevent overconfident predictions
- Gradient clipping at 1.0 to prevent exploding gradients in LSTMs
- Teacher forcing ratio decays linearly from 1.0 to 0.3 over training
- We evaluate on a subset of validation data every N steps for speed
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm

from ..data.tokenizer import TokenizerWrapper
from ..evaluation.inference import evaluate_model, print_examples
from ..evaluation.metrics import print_metrics


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident
    and improves generalization, especially for translation tasks.
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (batch_size * seq_len, vocab_size) - log probabilities
            target: (batch_size * seq_len,) - target token IDs
        """
        output = output.log_softmax(dim=-1)

        # Create smooth distribution
        smooth_dist = torch.full_like(output, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.padding_idx] = 0

        # Mask padding positions
        mask = target != self.padding_idx
        smooth_dist = smooth_dist * mask.unsqueeze(1)

        loss = self.criterion(output, smooth_dist)
        # Normalize by number of non-padding tokens
        num_tokens = mask.sum().item()
        if num_tokens > 0:
            loss = loss / num_tokens

        return loss


class Seq2SeqTrainer:
    """
    Trainer for Seq2Seq NMT models.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        train_loader,
        val_loader,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Args:
            model: Seq2Seq model
            tokenizer: Trained tokenizer
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training hyperparameters
        self.num_epochs = config.get("num_epochs", 30)
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.weight_decay = float(config.get("weight_decay", 1e-5))
        self.grad_clip = config.get("grad_clip", 1.0)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.tf_start = config.get("teacher_forcing_start", 1.0)
        self.tf_end = config.get("teacher_forcing_end", 0.3)
        self.eval_every = config.get("eval_every_epochs", 1)
        self.eval_max_batches = config.get("eval_max_batches", 50)
        self.use_amp = config.get("use_amp", True)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.early_stopping_patience = config.get("early_stopping_patience", 3)

        # Directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        vocab_size = model.tgt_vocab_size
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            padding_idx=0,
            smoothing=self.label_smoothing,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp and device != "cpu" else None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.global_step = 0
        self.best_val_bleu = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_bleu": [],
            "val_bleu": [],
            "train_chrf": [],
            "val_chrf": [],
        }

    def _create_scheduler(self):
        """Create cosine annealing LR scheduler with linear warmup."""

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.num_epochs * len(self.train_loader) - self.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Linear decay of teacher forcing ratio."""
        progress = epoch / max(1, self.num_epochs - 1)
        return self.tf_start - (self.tf_start - self.tf_end) * progress

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tf_ratio = self._get_teacher_forcing_ratio(epoch)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs} [TF={tf_ratio:.2f}]",
        )

        for batch in pbar:
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler and self.device != "cpu":
                with autocast():
                    outputs, _ = self.model(
                        src_ids, src_lengths, src_mask, tgt_ids, tf_ratio
                    )
                    # outputs: (batch, tgt_len-1, vocab)
                    # targets: shift right (skip BOS)
                    targets = tgt_ids[:, 1:]  # (batch, tgt_len-1)

                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1),
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, _ = self.model(
                    src_ids, src_lengths, src_mask, tgt_ids, tf_ratio
                )
                targets = tgt_ids[:, 1:]

                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    targets.reshape(-1),
                )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{total_loss/num_batches:.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

            # TensorBoard logging
            if self.global_step % 50 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.scheduler.get_last_lr()[0], self.global_step
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """
        Validate the model.

        Returns:
            avg_loss: Average validation loss
            eval_results: Dictionary with metrics and examples
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Compute validation loss
        for batch in tqdm(self.val_loader, desc="Computing val loss"):
            src_ids = batch["src_ids"].to(self.device)
            tgt_ids = batch["tgt_ids"].to(self.device)
            src_lengths = batch["src_lengths"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)

            outputs, _ = self.model(
                src_ids, src_lengths, src_mask, tgt_ids,
                teacher_forcing_ratio=0.0,  # No teacher forcing during validation
            )
            targets = tgt_ids[:, 1:]

            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute BLEU and chrF++ on validation set
        eval_results = evaluate_model(
            self.model,
            self.val_loader,
            self.tokenizer,
            beam_size=1,  # Greedy for speed during training
            max_len=200,
            device=self.device,
            max_batches=self.eval_max_batches,
            desc="Val BLEU/chrF++",
        )

        return avg_loss, eval_results

    def save_checkpoint(self, epoch: int, val_bleu: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_bleu": val_bleu,
            "global_step": self.global_step,
            "config": self.config,
            "history": self.history,
        }

        # Save periodic checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ★ New best model saved! BLEU-100: {val_bleu:.2f}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Label smoothing: {self.label_smoothing}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, eval_results = self.validate(epoch)
            val_metrics = eval_results["metrics"]
            val_bleu = val_metrics["bleu_100"]
            val_chrf = val_metrics["chrf_100"]

            self.history["val_loss"].append(val_loss)
            self.history["val_bleu"].append(val_bleu)
            self.history["val_chrf"].append(val_chrf)

            # Quick train set evaluation (subset)
            train_eval = evaluate_model(
                self.model,
                self.train_loader,
                self.tokenizer,
                beam_size=1,
                max_len=200,
                device=self.device,
                max_batches=20,
                desc="Train BLEU/chrF++",
            )
            train_bleu = train_eval["metrics"]["bleu_100"]
            train_chrf = train_eval["metrics"]["chrf_100"]
            self.history["train_bleu"].append(train_bleu)
            self.history["train_chrf"].append(train_chrf)

            # TensorBoard logging
            self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/bleu_100", val_bleu, epoch)
            self.writer.add_scalar("val/chrf_100", val_chrf, epoch)
            self.writer.add_scalar("train/bleu_100", train_bleu, epoch)
            self.writer.add_scalar("train/chrf_100", train_chrf, epoch)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ({epoch_time:.0f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train BLEU-100: {train_bleu:.2f} | Val BLEU-100: {val_bleu:.2f}")
            print(f"  Train chrF++-100: {train_chrf:.2f} | Val chrF++-100: {val_chrf:.2f}")

            # Save checkpoint
            is_best = val_bleu > self.best_val_bleu
            if is_best:
                self.best_val_bleu = val_bleu
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            self.save_checkpoint(epoch, val_bleu, is_best)
            self.save_history()

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered! Validation BLEU hasn't improved for {self.early_stopping_patience} epochs.")
                break

            # Print examples
            if (epoch + 1) % self.eval_every == 0:
                print_examples(eval_results["examples"], n=3)

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete! Total time: {total_time/3600:.1f} hours")
        print(f"Best validation BLEU-100: {self.best_val_bleu:.2f}")
        print(f"{'='*70}")

        self.writer.close()
        return self.history
