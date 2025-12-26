"""
Loss Functions for SoccerTransformer
- MSE Loss for Motion Reconstruction
- NT-Xent Loss for Contrastive Learning
- Metrics for monitoring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ReconstructionLoss(nn.Module):
    """
    MSE Loss for motion reconstruction.
    Only computes loss on masked positions.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch, agents, timesteps, 2] - predicted (x, y)
            targets: [batch, agents, timesteps, 2] - ground truth (x, y)
            mask: [batch, agents, timesteps] - True means this position was masked
        Returns:
            loss: scalar tensor
        """
        # Expand mask for (x, y) dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # [B, A, T, 2]
        
        # Compute MSE
        mse = self.mse(predictions, targets)  # [B, A, T, 2]
        
        # Only compute loss on masked positions
        masked_mse = mse * mask_expanded.float()
        
        # Average over masked positions only
        num_masked = mask_expanded.float().sum()
        if num_masked > 0:
            loss = masked_mse.sum() / num_masked
        else:
            loss = masked_mse.sum() * 0.0  # No masked positions
        
        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent)
    for contrastive learning.
    
    Positive pairs: anchor and its augmented version (different mask)
    Negative pairs: all other samples in batch
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_anchor: [batch, embed_dim] - embeddings of anchor samples
            z_positive: [batch, embed_dim] - embeddings of positive samples (same data, different mask)
        Returns:
            loss: scalar tensor
            metrics: dict with contrastive accuracy and other metrics
        """
        batch_size = z_anchor.shape[0]
        device = z_anchor.device
        
        # Concatenate anchor and positive: [2*batch, embed_dim]
        z = torch.cat([z_anchor, z_positive], dim=0)
        
        # Compute cosine similarity matrix: [2*batch, 2*batch]
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels: positive pairs are (i, i+batch) and (i+batch, i)
        # For sample i in anchor, its positive is at index i+batch
        # For sample i in positive, its positive is at index i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Compute metrics
        with torch.no_grad():
            # Contrastive accuracy: how often is the positive the most similar?
            predictions = sim_matrix.argmax(dim=1)
            correct = (predictions == labels).float()
            accuracy = correct.mean().item()
            
            # Average positive similarity
            pos_sim = F.cosine_similarity(z_anchor, z_positive, dim=1).mean().item()
            
            # Average negative similarity (approximate - sample some negatives)
            neg_mask = ~mask
            neg_sims = sim_matrix[neg_mask].view(2 * batch_size, -1)
            neg_sim = (neg_sims * self.temperature).mean().item()  # Undo temperature scaling
            
            # Embedding statistics for collapse detection
            z_std = z.std(dim=0).mean().item()  # Should not collapse to 0
            z_mean_norm = z.mean(dim=0).norm().item()  # Should be small if centered
        
        metrics = {
            'contrastive_accuracy': accuracy,
            'positive_similarity': pos_sim,
            'negative_similarity': neg_sim,
            'embedding_std': z_std,
            'embedding_mean_norm': z_mean_norm
        }
        
        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss: L = L_MR + λ_CL * L_CL
    """
    
    def __init__(self, lambda_mr: float = 1.0, lambda_cl: float = 100.0, temperature: float = 0.5):
        super().__init__()
        self.lambda_mr = lambda_mr
        self.lambda_cl = lambda_cl
        
        self.reconstruction_loss = ReconstructionLoss()
        self.contrastive_loss = NTXentLoss(temperature=temperature)
    
    def forward(
        self,
        pred_anchor: torch.Tensor,
        pred_positive: torch.Tensor,
        targets: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_positive: torch.Tensor,
        embed_anchor: torch.Tensor,
        embed_positive: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_anchor: [B, A, T, 2] - reconstruction from anchor mask
            pred_positive: [B, A, T, 2] - reconstruction from positive mask
            targets: [B, A, T, 2] - ground truth (x, y)
            mask_anchor: [B, A, T] - anchor mask
            mask_positive: [B, A, T] - positive mask
            embed_anchor: [B, embed_dim] - anchor embedding
            embed_positive: [B, embed_dim] - positive embedding
        Returns:
            total_loss: scalar tensor
            metrics: dict with all loss components and metrics
        """
        # Reconstruction loss (average over both views)
        loss_mr_anchor = self.reconstruction_loss(pred_anchor, targets, mask_anchor)
        loss_mr_positive = self.reconstruction_loss(pred_positive, targets, mask_positive)
        loss_mr = (loss_mr_anchor + loss_mr_positive) / 2
        
        # Contrastive loss
        loss_cl, cl_metrics = self.contrastive_loss(embed_anchor, embed_positive)
        
        # Combined loss
        total_loss = self.lambda_mr * loss_mr + self.lambda_cl * loss_cl
        
        metrics = {
            'loss_total': total_loss.item(),
            'loss_mr': loss_mr.item(),
            'loss_cl': loss_cl.item(),
            'loss_mr_weighted': (self.lambda_mr * loss_mr).item(),
            'loss_cl_weighted': (self.lambda_cl * loss_cl).item(),
            **cl_metrics
        }
        
        return total_loss, metrics


def check_embedding_collapse(embedding_std: float, threshold: float = 0.01) -> bool:
    """
    Check if embeddings are collapsing (all becoming similar).
    Returns True if collapse is detected.
    """
    return embedding_std < threshold
