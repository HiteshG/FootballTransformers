"""
SoccerTransformer Model Architecture
- Axial attention encoder (temporal + spatial)
- Decoder for motion reconstruction
- Pooling head for contrastive embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
from config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""
    
    def __init__(self, d_model: int, dropout: float, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, agents, timesteps, features]
        Returns:
            x with positional encoding added to temporal dimension
        """
        B, A, T, D = x.shape
        # Add positional encoding to each agent's temporal sequence
        x = x + self.pe[:, :T, :].unsqueeze(1)  # broadcast over agents
        return self.dropout(x)


class AxialAttentionLayer(nn.Module):
    """
    Single axial attention layer.
    Can operate across time (for each agent) or across agents (for each timestep).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        across_time: bool
    ):
        super().__init__()
        self.across_time = across_time
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, agents, timesteps, features]
            mask: [batch, agents, timesteps] - True means masked/padded
        Returns:
            x: [batch, agents, timesteps, features]
        """
        B, A, T, D = x.shape
        
        if self.across_time:
            # Temporal attention: each agent attends across its own timesteps
            # Reshape: [B*A, T, D]
            x_reshape = x.view(B * A, T, D)
            
            # Prepare key_padding_mask if provided [B*A, T]
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask.view(B * A, T)
            
        else:
            # Spatial attention: all agents attend to each other at each timestep
            # Reshape: [B*T, A, D]
            x_reshape = x.permute(0, 2, 1, 3).contiguous().view(B * T, A, D)
            
            # Prepare key_padding_mask if provided [B*T, A]
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = mask.permute(0, 2, 1).contiguous().view(B * T, A)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(
            x_reshape, x_reshape, x_reshape,
            key_padding_mask=key_padding_mask
        )
        x_reshape = self.norm1(x_reshape + self.dropout(attn_out))
        
        # FFN with residual
        x_reshape = self.norm2(x_reshape + self.ffn(x_reshape))
        
        # Reshape back to [B, A, T, D]
        if self.across_time:
            x = x_reshape.view(B, A, T, D)
        else:
            x = x_reshape.view(B, T, A, D).permute(0, 2, 1, 3).contiguous()
        
        return x


class AxialEncoder(nn.Module):
    """
    Encoder with alternating temporal and spatial attention layers.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection: [6] -> [256]
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_features, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Positional encoding for temporal dimension
        self.pos_encoding = PositionalEncoding(
            config.hidden_dim,
            config.dropout,
            max_len=config.num_timesteps
        )
        
        # Alternating axial attention layers (5 pairs = 10 layers)
        self.layers = nn.ModuleList()
        for i in range(config.num_encoder_layers):
            across_time = (i % 2 == 0)  # Alternate: temporal, spatial, temporal, ...
            self.layers.append(
                AxialAttentionLayer(
                    d_model=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    across_time=across_time
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, agents, timesteps, features] - raw input
            mask: [batch, agents, timesteps] - True means masked
        Returns:
            encoded: [batch, agents, timesteps, hidden_dim]
        """
        # Project input features
        x = self.input_projection(x)  # [B, A, T, D]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply axial attention layers
        for layer in self.layers:
            x = layer(x, mask=None)  # Don't mask attention, we want full context
        
        return x


class ReconstructionDecoder(nn.Module):
    """
    Decoder for motion reconstruction task.
    Predicts (x, y) coordinates for masked positions.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Axial attention decoder layers (2 pairs = 4 layers)
        self.layers = nn.ModuleList()
        for i in range(config.num_decoder_layers):
            across_time = (i % 2 == 0)
            self.layers.append(
                AxialAttentionLayer(
                    d_model=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    across_time=across_time
                )
            )
        
        # Output projection: predict (x, y) normalized coordinates
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 2)  # Predict x_norm, y_norm
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: [batch, agents, timesteps, hidden_dim]
        Returns:
            predictions: [batch, agents, timesteps, 2] - predicted (x, y)
        """
        x = encoded
        for layer in self.layers:
            x = layer(x)
        
        return self.output_projection(x)


class ContrastiveHead(nn.Module):
    """
    Pooling and projection head for contrastive learning.
    Generates possession-level embeddings.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Global average pooling is done in forward
        
        # Projection MLP (as per SimCLR)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: [batch, agents, timesteps, hidden_dim]
        Returns:
            embeddings: [batch, embedding_dim] - L2 normalized
        """
        # Global average pooling across agents and timesteps
        pooled = encoded.mean(dim=[1, 2])  # [batch, hidden_dim]
        
        # Project to embedding space
        embeddings = self.projection(pooled)  # [batch, embedding_dim]
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class SoccerTransformer(nn.Module):
    """
    Complete SoccerTransformer model.
    - Encoder: Axial attention for spatiotemporal features
    - Decoder: Motion reconstruction
    - Contrastive Head: Embedding generation
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.encoder = AxialEncoder(config)
        self.decoder = ReconstructionDecoder(config)
        self.contrastive_head = ContrastiveHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, agents, timesteps, features]
            mask: [batch, agents, timesteps] - True means masked (for reconstruction target)
        Returns:
            dict with:
                - encoded: [batch, agents, timesteps, hidden_dim]
                - reconstructed: [batch, agents, timesteps, 2]
                - embedding: [batch, embedding_dim]
        """
        # Apply mask to input (zero out masked positions)
        if mask is not None:
            x_masked = x.clone()
            x_masked[mask] = 0.0
        else:
            x_masked = x
        
        # Encode
        encoded = self.encoder(x_masked, mask)
        
        # Decode for reconstruction
        reconstructed = self.decoder(encoded)
        
        # Generate embedding for contrastive learning
        embedding = self.contrastive_head(encoded)
        
        return {
            'encoded': encoded,
            'reconstructed': reconstructed,
            'embedding': embedding
        }
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding without reconstruction (for inference)."""
        encoded = self.encoder(x)
        return self.contrastive_head(encoded)