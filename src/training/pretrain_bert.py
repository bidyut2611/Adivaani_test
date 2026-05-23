"""
Masked Language Modeling (MLM) Pretraining Loop for Custom BERT.

Features:
- Dynamic masking strategy (15% tokens masked)
- Standard BERT masking breakdown (80% [MASK], 10% random, 10% keep)
- Distributed Data Parallel (DDP) / Gradient Accumulation support
- Cosine LR scheduling with warmup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from ..models.bert import CustomBERT


class BERTMlmTrainer:
    def __init__(
        self,
        model: CustomBERT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab_size: int,
        pad_id: int,
        mask_id: int,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.num_epochs = config.get("num_epochs", 10)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 10000)
        self.mask_prob = config.get("mask_prob", 0.15)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.accum_steps = config.get("grad_accum_steps", 1)
        self.early_stopping_patience = config.get("early_stopping_patience", 3)
        
        # Directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/bert"))
        self.log_dir = Path(config.get("log_dir", "logs/bert"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if config.get("use_amp", True) else None
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.num_epochs * len(self.train_loader) // self.accum_steps - self.warmup_steps
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def apply_masking(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masking to inputs."""
        labels = input_ids.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mask_prob, device=self.device)
        # Don't mask padding
        padding_mask = labels == self.pad_id
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Sample tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set targets for unmasked tokens to -100 (ignored in CrossEntropyLoss)
        labels[~masked_indices] = -100
        
        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_id
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long, device=labels.device)
        input_ids[indices_random] = random_words[indices_random]
        
        # The rest 10% of the time, keep the original word
        return input_ids, labels

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for i, batch in enumerate(pbar):
            # Batch shape varies depending on dataloader, assuming tensor of (batch, seq_len)
            input_ids = batch.to(self.device)
            
            # Create attention mask before corrupting inputs
            attention_mask = (input_ids != self.pad_id).long()
            
            # Apply dynamic masking
            masked_inputs, labels = self.apply_masking(input_ids.clone())
            
            # Forward pass
            if self.scaler:
                with autocast():
                    logits = self.model(masked_inputs, attention_mask)
                    loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
                    loss = loss / self.accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
            else:
                logits = self.model(masked_inputs, attention_mask)
                loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
                loss = loss / self.accum_steps
                loss.backward()
                
                if (i + 1) % self.accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
            
            total_loss += loss.item() * self.accum_steps
            
            # Logging
            if self.global_step % 100 == 0 and (i + 1) % self.accum_steps == 0:
                current_loss = loss.item() * self.accum_steps
                pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.2e}")
                self.writer.add_scalar("train/mlm_loss", current_loss, self.global_step)
                
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch.to(self.device)
            attention_mask = (input_ids != self.pad_id).long()
            masked_inputs, labels = self.apply_masking(input_ids.clone())
            
            logits = self.model(masked_inputs, attention_mask)
            loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
            total_loss += loss.item()
            
        return total_loss / max(1, len(self.val_loader))

    def train(self):
        print(f"Starting BERT Pretraining ({self.model.count_parameters():,} params)")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.writer.add_scalar("val/mlm_loss", val_loss, epoch)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_bert.pt")
                print("  Saved best model!")
            else:
                self.patience_counter += 1
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered! Validation loss hasn't improved for {self.early_stopping_patience} epochs.")
                break
