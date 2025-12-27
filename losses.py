"""
Loss Functions for SoccerTransformer
- MSE Loss for Motion Reconstruction
- NT-Xent Loss with In-Batch Negatives
- Triplet Loss with In-Batch Semi-Hard Mining
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
        mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
        mse = self.mse(predictions, targets)
        masked_mse = mse * mask_expanded.float()
        
        num_masked = mask_expanded.float().sum()
        if num_masked > 0:
            loss = masked_mse.sum() / num_masked
        else:
            loss = masked_mse.sum() * 0.0
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss with In-Batch Negatives.
    
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
            z_positive: [batch, embed_dim] - embeddings of positive samples
        Returns:
            loss: scalar tensor
            metrics: dict with contrastive accuracy and other metrics
        """
        batch_size = z_anchor.shape[0]
        device = z_anchor.device
        
        # Normalize embeddings
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)
        
        # Concatenate: [2*batch, embed_dim]
        z = torch.cat([z_anchor, z_positive], dim=0)
        
        # Compute similarity matrix: [2*batch, 2*batch]
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels: positive for anchor i is at i+batch, and vice versa
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Compute metrics
        with torch.no_grad():
            predictions = sim_matrix.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
            pos_sim = F.cosine_similarity(z_anchor, z_positive, dim=1).mean().item()
            
            # Average negative similarity
            neg_mask = ~mask
            # Also exclude positive pairs
            for i in range(batch_size):
                neg_mask[i, i + batch_size] = False
                neg_mask[i + batch_size, i] = False
            neg_sims = sim_matrix[neg_mask] * self.temperature  # Undo temperature
            neg_sim = neg_sims.mean().item() if neg_sims.numel() > 0 else 0.0
            
            z_std = z.std(dim=0).mean().item()
        
        metrics = {
            'contrastive_accuracy': accuracy,
            'positive_similarity': pos_sim,
            'negative_similarity': neg_sim,
            'embedding_std': z_std,
        }
        
        return loss, metrics


class TripletLossInBatch(nn.Module):
    """
    Triplet Loss with In-Batch Semi-Hard Negative Mining.
    
    For each anchor-positive pair, finds semi-hard negatives from
    other samples in the batch.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        z_anchor: torch.Tensor,
        z_positive: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_anchor: [batch, embed_dim] - anchor embeddings
            z_positive: [batch, embed_dim] - positive embeddings
        Returns:
            loss: scalar tensor
            metrics: dict with triplet-specific metrics
        """
        batch_size = z_anchor.shape[0]
        device = z_anchor.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device), {
                'triplet_loss': 0.0,
                'num_valid_triplets': 0,
                'avg_positive_dist': 0.0,
                'avg_hard_negative_dist': 0.0,
            }
        
        # Normalize embeddings
        z_anchor = F.normalize(z_anchor, dim=1)
        z_positive = F.normalize(z_positive, dim=1)
        
        # Positive distances (1 - cosine_similarity)
        pos_sim = (z_anchor * z_positive).sum(dim=1)
        pos_dist = 1.0 - pos_sim  # [batch]
        
        # All pairwise distances to other positives (use as negatives)
        # Each anchor uses OTHER samples' positives as negatives
        all_neg_sim = torch.mm(z_anchor, z_positive.t())  # [batch, batch]
        all_neg_dist = 1.0 - all_neg_sim  # [batch, batch]
        
        # Mask out the positive pair (diagonal)
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        all_neg_dist = all_neg_dist.masked_fill(mask, float('inf'))
        
        # Semi-hard mining: find negatives where pos_dist < neg_dist < pos_dist + margin
        pos_dist_expanded = pos_dist.unsqueeze(1)  # [batch, 1]
        
        semi_hard_mask = (all_neg_dist > pos_dist_expanded) & \
                         (all_neg_dist < pos_dist_expanded + self.margin)
        
        # For samples with no semi-hard negatives, use hardest negative
        triplet_losses = []
        num_valid = 0
        hard_neg_dists = []
        
        for i in range(batch_size):
            semi_hard_indices = semi_hard_mask[i].nonzero(as_tuple=True)[0]
            
            if len(semi_hard_indices) > 0:
                # Use mean of semi-hard negatives
                neg_dist = all_neg_dist[i, semi_hard_indices].mean()
                num_valid += 1
            else:
                # Fall back to hardest negative (smallest distance, excluding inf)
                valid_neg_dist = all_neg_dist[i][all_neg_dist[i] < float('inf')]
                if len(valid_neg_dist) > 0:
                    neg_dist = valid_neg_dist.min()
                else:
                    continue
            
            hard_neg_dists.append(neg_dist.item())
            loss = F.relu(pos_dist[i] - neg_dist + self.margin)
            triplet_losses.append(loss)
        
        if len(triplet_losses) > 0:
            loss = torch.stack(triplet_losses).mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        metrics = {
            'triplet_loss': loss.item(),
            'num_valid_triplets': num_valid,
            'avg_positive_dist': pos_dist.mean().item(),
            'avg_hard_negative_dist': sum(hard_neg_dists) / len(hard_neg_dists) if hard_neg_dists else 0.0,
        }
        
        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss with in-batch negatives:
    L = λ_MR * L_MR + λ_CL * L_NT-Xent + λ_triplet * L_triplet
    """
    
    def __init__(
        self,
        lambda_mr: float = 1.0,
        lambda_cl: float = 100.0,
        lambda_triplet: float = 30.0,
        temperature: float = 0.5,
        triplet_margin: float = 0.3,
    ):
        super().__init__()
        self.lambda_mr = lambda_mr
        self.lambda_cl = lambda_cl
        self.lambda_triplet = lambda_triplet
        
        self.reconstruction_loss = ReconstructionLoss()
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.triplet_loss = TripletLossInBatch(margin=triplet_margin)
    
    def forward(
        self,
        pred_anchor: torch.Tensor,
        pred_positive: torch.Tensor,
        targets: torch.Tensor,
        mask_anchor: torch.Tensor,
        mask_positive: torch.Tensor,
        embed_anchor: torch.Tensor,
        embed_positive: torch.Tensor,
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
            metrics: dict with all loss components
        """
        # Reconstruction loss (average over both views)
        loss_mr_anchor = self.reconstruction_loss(pred_anchor, targets, mask_anchor)
        loss_mr_positive = self.reconstruction_loss(pred_positive, targets, mask_positive)
        loss_mr = (loss_mr_anchor + loss_mr_positive) / 2
        
        # Contrastive loss (in-batch negatives)
        loss_cl, cl_metrics = self.contrastive_loss(embed_anchor, embed_positive)
        
        # Triplet loss (in-batch semi-hard mining)
        loss_triplet, triplet_metrics = self.triplet_loss(embed_anchor, embed_positive)
        
        # Combined loss
        total_loss = (
            self.lambda_mr * loss_mr + 
            self.lambda_cl * loss_cl + 
            self.lambda_triplet * loss_triplet
        )
        
        metrics = {
            'loss_total': total_loss.item(),
            'loss_mr': loss_mr.item(),
            'loss_cl': loss_cl.item(),
            'loss_triplet': triplet_metrics['triplet_loss'],
            'loss_mr_weighted': (self.lambda_mr * loss_mr).item(),
            'loss_cl_weighted': (self.lambda_cl * loss_cl).item(),
            'loss_triplet_weighted': (self.lambda_triplet * loss_triplet).item(),
            **cl_metrics,
            'num_valid_triplets': triplet_metrics['num_valid_triplets'],
            'avg_positive_dist': triplet_metrics['avg_positive_dist'],
            'avg_hard_negative_dist': triplet_metrics['avg_hard_negative_dist'],
        }
        
        return total_loss, metrics


def check_embedding_collapse(embedding_std: float, threshold: float = 0.01) -> bool:
    """
    Check if embeddings are collapsing.
    Returns True if collapse is detected.
    """
    return embedding_std < threshold