"""
SoccerTransformer Configuration
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    # Input dimensions
    num_agents: int = 23              # 1 ball + 11 home + 11 away
    num_timesteps: int = 100          # frames per window
    input_features: int = 6           # x_norm, y_norm, vx_norm, vy_norm, home_away_player, role_id
    
    # Model architecture
    hidden_dim: int = 256             # internal feature dimension
    num_heads: int = 8                # attention heads
    num_encoder_layers: int = 10      # 5 pairs of temporal/spatial attention
    embedding_dim: int = 128          # final embedding dimension for similarity search
    dropout: float = 0.1
    
    # Decoder for reconstruction
    num_decoder_layers: int = 4       # 2 pairs of temporal/spatial attention


@dataclass
class TrainingConfig:
    # Data split
    train_ratio: float = 0.85
    val_ratio: float = 0.15
    
    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 5e-5       # Low LR as per paper guidance
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    
    # Loss weights
    lambda_mr: float = 1.0            # Motion reconstruction weight
    lambda_cl: float = 100.0          # Contrastive learning weight (from Hoop-MSSL)
    
    # Masking
    mask_ratio: float = 0.8           # 80% masking as per paper
    
    # Contrastive learning
    temperature: float = 0.5          # NT-Xent temperature
    
    # Logging
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5


@dataclass
class DataConfig:
    # CSV columns to use as features (in order)
    feature_columns: Tuple[str, ...] = (
        'x_norm', 'y_norm', 'vx_norm', 'vy_norm', 'home_away_player', 'role_id'
    )
    
    # Columns for indexing
    window_column: str = 'global_window_id'
    match_column: str = 'match_id'
    agent_column: str = 'agent_idx'
    frame_column: str = 'frame'
    
    # Data paths (Modal volume)
    data_volume_path: str = '/data'
    csv_filename: str = 'tracking_data.csv'
    checkpoint_path: str = '/data/checkpoints'


@dataclass
class Config:
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()


def get_config() -> Config:
    return Config()
