"""
Autoregressive Pretraining Loop for Custom GPT-2.

Features:
- Causal Language Modeling (Next-token prediction)
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
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from ..models.gpt2 import CustomGPT2


class GPT2Trainer:
    def __init__(
        self,
        model: CustomGPT2,
        train_loader: DataLoader,
        val_loader: DataLoader,
        vocab_size: int,
        pad_id: int,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.num_epochs = config.get("num_epochs", 10)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_steps = config.get("warmup_steps", 10000)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.accum_steps = config.get("grad_accum_steps", 1)
        self.early_stopping_patience = config.get("early_stopping_patience", 3)
        
        # Directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/gpt2"))
        self.log_dir = Path(config.get("log_dir", "logs/gpt2"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CrossEntropyLoss automatically ignores pad_id if set
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
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

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for i, batch in enumerate(pbar):
            input_ids = batch.to(self.device)
            
            # For autoregressive modeling:
            # Inputs: x_0, x_1, ..., x_{t-1}
            # Targets: x_1, x_2, ..., x_t
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            if self.scaler:
                with autocast():
                    logits, _ = self.model(inputs)
                    loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
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
                logits, _ = self.model(inputs)
                loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
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
                self.writer.add_scalar("train/lm_loss", current_loss, self.global_step)
                
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch.to(self.device)
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            
            logits, _ = self.model(inputs)
            loss = self.criterion(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            total_loss += loss.item()
            
        return total_loss / max(1, len(self.val_loader))

    def train(self):
        print(f"Starting GPT-2 Pretraining ({self.model.count_parameters():,} params)")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.writer.add_scalar("val/lm_loss", val_loss, epoch)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best_gpt2.pt")
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
