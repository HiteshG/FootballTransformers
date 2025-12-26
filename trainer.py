"""
SoccerTransformer Trainer
- Training loop with gradient accumulation
- Validation with metrics
- Learning rate scheduling
- Checkpointing
- Comprehensive logging
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from typing import Dict, Optional
import time
import os
import json
from collections import defaultdict

from config import Config
from model import SoccerTransformer
from losses import CombinedLoss, check_embedding_collapse


class Trainer:
    """
    Trainer for SoccerTransformer model.
    """
    
    def __init__(
        self,
        model: SoccerTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = CombinedLoss(
            lambda_mr=config.training.lambda_mr,
            lambda_cl=config.training.lambda_cl,
            temperature=config.training.temperature
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler: warmup + cosine annealing
        warmup_steps = config.training.warmup_epochs * len(train_loader)
        total_steps = config.training.epochs * len(train_loader)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(train_loader),  # Restart every epoch
            T_mult=2,
            eta_min=config.training.learning_rate * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics history
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Create checkpoint directory
        os.makedirs(config.data.checkpoint_path, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            mask_anchor = batch['mask_anchor'].to(self.device)
            mask_positive = batch['mask_positive'].to(self.device)
            features_negative = batch['features_negative'].to(self.device)
            mask_negative = batch['mask_negative'].to(self.device)
            
            # Ground truth for reconstruction (x_norm, y_norm are first 2 features)
            targets = features[:, :, :, :2]  # [B, A, T, 2]
            
            # Forward pass - anchor view
            out_anchor = self.model(features, mask_anchor)
            
            # Forward pass - positive view (same data, different mask)
            out_positive = self.model(features, mask_positive)
            
            # Compute loss
            loss, metrics = self.criterion(
                pred_anchor=out_anchor['reconstructed'],
                pred_positive=out_positive['reconstructed'],
                targets=targets,
                mask_anchor=mask_anchor,
                mask_positive=mask_positive,
                embed_anchor=out_anchor['embedding'],
                embed_positive=out_positive['embedding']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate metrics
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % self.config.training.log_every_n_steps == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.config.training.batch_size / elapsed
                
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {metrics['loss_total']:.4f} | "
                      f"MR: {metrics['loss_mr']:.4f} | "
                      f"CL: {metrics['loss_cl']:.4f} | "
                      f"Acc: {metrics['contrastive_accuracy']:.3f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                      f"Speed: {samples_per_sec:.1f} samples/s")
                
                # Check for embedding collapse
                if check_embedding_collapse(metrics['embedding_std']):
                    print("  ⚠️ WARNING: Embedding collapse detected! Std={:.4f}".format(
                        metrics['embedding_std']))
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        epoch_metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        epoch_metrics['epoch_time'] = time.time() - start_time
        
        return dict(epoch_metrics)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        all_embeddings = []
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = features[:, :, :, :2]
            
            # Forward pass
            out = self.model(features, mask)
            
            # Reconstruction loss only (no contrastive for validation)
            from losses import ReconstructionLoss
            mr_loss = ReconstructionLoss()(out['reconstructed'], targets, mask)
            
            epoch_metrics['loss_mr'] += mr_loss.item()
            
            # Collect embeddings for analysis
            all_embeddings.append(out['embedding'].cpu())
            num_batches += 1
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        # Embedding statistics
        all_embeddings = torch.cat(all_embeddings, dim=0)
        epoch_metrics['embedding_std'] = all_embeddings.std(dim=0).mean().item()
        epoch_metrics['embedding_mean_norm'] = all_embeddings.mean(dim=0).norm().item()
        
        # Intra-batch similarity distribution
        sample_idx = torch.randperm(len(all_embeddings))[:min(1000, len(all_embeddings))]
        sample_embeds = all_embeddings[sample_idx]
        sim_matrix = torch.mm(sample_embeds, sample_embeds.t())
        mask = ~torch.eye(len(sample_embeds), dtype=torch.bool)
        epoch_metrics['mean_similarity'] = sim_matrix[mask].mean().item()
        epoch_metrics['std_similarity'] = sim_matrix[mask].std().item()
        
        return dict(epoch_metrics)
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'model': self.config.model.__dict__,
                'training': self.config.training.__dict__,
            },
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
        }
        
        path = os.path.join(self.config.data.checkpoint_path, filename)
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
        
        if is_best:
            best_path = os.path.join(self.config.data.checkpoint_path, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.data.checkpoint_path, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = defaultdict(list, checkpoint.get('train_history', {}))
        self.val_history = defaultdict(list, checkpoint.get('val_history', {}))
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Full training loop."""
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Learning rate: {self.config.training.learning_rate}")
        print(f"Lambda MR: {self.config.training.lambda_mr}")
        print(f"Lambda CL: {self.config.training.lambda_cl}")
        print("=" * 70)
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Store training metrics
            for k, v in train_metrics.items():
                self.train_history[k].append(v)
            
            print(f"\n  Train Summary:")
            print(f"    Loss Total: {train_metrics['loss_total']:.4f}")
            print(f"    Loss MR: {train_metrics['loss_mr']:.4f} (weighted: {train_metrics['loss_mr_weighted']:.4f})")
            print(f"    Loss CL: {train_metrics['loss_cl']:.4f} (weighted: {train_metrics['loss_cl_weighted']:.4f})")
            print(f"    Contrastive Accuracy: {train_metrics['contrastive_accuracy']:.3f}")
            print(f"    Positive Similarity: {train_metrics['positive_similarity']:.3f}")
            print(f"    Negative Similarity: {train_metrics['negative_similarity']:.3f}")
            print(f"    Embedding Std: {train_metrics['embedding_std']:.4f}")
            print(f"    Epoch Time: {train_metrics['epoch_time']:.1f}s")
            
            # Validation
            if (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                val_metrics = self.validate()
                
                for k, v in val_metrics.items():
                    self.val_history[k].append(v)
                
                print(f"\n  Validation Summary:")
                print(f"    Loss MR: {val_metrics['loss_mr']:.4f}")
                print(f"    Embedding Std: {val_metrics['embedding_std']:.4f}")
                print(f"    Mean Similarity: {val_metrics['mean_similarity']:.3f}")
                print(f"    Std Similarity: {val_metrics['std_similarity']:.3f}")
                
                # Check for best model
                is_best = val_metrics['loss_mr'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss_mr']
                    print(f"    ✓ New best validation loss!")
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', is_best=is_best)
        
        # Final save
        self.save_checkpoint('final_model.pt', is_best=False)
        
        # Save training history
        history_path = os.path.join(self.config.data.checkpoint_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train': dict(self.train_history),
                'val': dict(self.val_history)
            }, f, indent=2)
        print(f"\nSaved training history: {history_path}")
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 70)
