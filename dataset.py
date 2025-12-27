"""
SoccerTransformer Dataset
- Loads windows from CSV
- Generates random masks for reconstruction
- Creates contrastive pairs (same match, different window)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import Config


class SoccerTrackingDataset(Dataset):
    """
    Dataset for soccer tracking data.
    Each item returns a window with shape [23, 100, 6] and its contrastive pair.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        is_training: bool = True
    ):
        self.config = config
        self.is_training = is_training
        self.mask_ratio = config.training.mask_ratio
        
        # Feature columns
        self.feature_cols = list(config.data.feature_columns)
        
        # Build window index (string IDs)
        self.global_window_ids = df[config.data.window_column].unique().tolist()
        self.global_window_to_match = df.groupby(config.data.window_column)[config.data.match_column].first().to_dict()
        
        # Build match to windows mapping for contrastive sampling
        self.match_to_global_windows = {
            k: v.tolist() for k, v in 
            df.groupby(config.data.match_column)[config.data.window_column].unique().items()
        }
        
        # Create string to index mappings for returning integer IDs
        self.global_window_id_to_idx = {wid: idx for idx, wid in enumerate(self.global_window_ids)}
        self.match_ids = list(self.match_to_global_windows.keys())
        self.match_id_to_idx = {mid: idx for idx, mid in enumerate(self.match_ids)}
        
        # Preprocess data into tensors per window
        self.windows_data = self._preprocess_windows(df)
        
    def _preprocess_windows(self, df: pd.DataFrame) -> Dict[int, torch.Tensor]:
        """Preprocess all windows into tensors."""
        windows_data = {}
        
        grouped = df.groupby(self.config.data.window_column)
        
        for window_id, window_df in grouped:
            # Sort by frame and agent_idx to ensure consistent ordering
            window_df = window_df.sort_values(
                [self.config.data.frame_column, self.config.data.agent_column]
            )
            
            # Extract features [frames * agents, features]
            features = window_df[self.feature_cols].values
            
            # Reshape to [agents, timesteps, features]
            # Data is sorted by frame then agent, so reshape accordingly
            num_agents = self.config.model.num_agents
            num_timesteps = self.config.model.num_timesteps
            num_features = self.config.model.input_features
            
            try:
                features = features.reshape(num_timesteps, num_agents, num_features)
                features = features.transpose(1, 0, 2)  # [agents, timesteps, features]
                windows_data[window_id] = torch.tensor(features, dtype=torch.float32)
            except ValueError:
                # Skip malformed windows
                continue
                
        return windows_data
    
    def __len__(self) -> int:
        return len(self.windows_data)
    
    def _create_random_mask(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Create random mask with specified ratio. True = masked (to predict)."""
        return torch.rand(shape) < self.mask_ratio
    
    def _get_contrastive_pair_global_window_id(self, global_window_id: str) -> Optional[str]:
        """Get a different window from the same match for hard negative mining."""
        match_id = self.global_window_to_match[global_window_id]
        match_windows = self.match_to_global_windows[match_id]
        
        # Filter out current window
        other_windows = [w for w in match_windows if w != global_window_id and w in self.windows_data]
        
        if len(other_windows) == 0:
            return None
        
        return np.random.choice(other_windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_window_id = self.global_window_ids[idx]  # String ID
        features = self.windows_data[global_window_id].clone()  # [23, 100, 6]
        
        A, T, D = features.shape
        
        # Get integer indices for window and match
        global_window_idx = self.global_window_id_to_idx[global_window_id]
        match_id = self.global_window_to_match[global_window_id]
        match_idx = self.match_id_to_idx[match_id]
        
        if self.is_training:
            # Create two different random masks for contrastive learning
            mask_anchor = self._create_random_mask((A, T))
            mask_positive = self._create_random_mask((A, T))
            
            # Get contrastive negative from same match, different window
            neg_global_window_id = self._get_contrastive_pair_global_window_id(global_window_id)
            if neg_global_window_id is not None:
                features_negative = self.windows_data[neg_global_window_id].clone()
                mask_negative = self._create_random_mask((A, T))
            else:
                # Fallback: use same window with very different mask
                features_negative = features.clone()
                mask_negative = self._create_random_mask((A, T))
            
            return {
                'features': features,                    # [23, 100, 6]
                'mask_anchor': mask_anchor,              # [23, 100]
                'mask_positive': mask_positive,          # [23, 100]
                'features_negative': features_negative,  # [23, 100, 6]
                'mask_negative': mask_negative,          # [23, 100]
                'global_window_idx': torch.tensor(global_window_idx, dtype=torch.long),
                'match_idx': torch.tensor(match_idx, dtype=torch.long),
            }
        else:
            # Validation: single mask
            mask = self._create_random_mask((A, T))
            return {
                'features': features,
                'mask': mask,
                'global_window_idx': torch.tensor(global_window_idx, dtype=torch.long),
                'match_idx': torch.tensor(match_idx, dtype=torch.long),
            }


def load_and_split_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV and split by match_id."""
    import os
    
    csv_path = os.path.join(config.data.data_volume_path, config.data.csv_filename)
    print(f"Loading data from {csv_path}...")
    
    # Load CSV in chunks to handle large file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    # Get unique match IDs
    match_ids = df[config.data.match_column].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)
    
    # Split by match
    split_idx = int(len(match_ids) * config.training.train_ratio)
    train_matches = set(match_ids[:split_idx])
    val_matches = set(match_ids[split_idx:])
    
    print(f"Train matches: {len(train_matches)}, Val matches: {len(val_matches)}")
    
    train_df = df[df[config.data.match_column].isin(train_matches)]
    val_df = df[df[config.data.match_column].isin(val_matches)]
    
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    
    return train_df, val_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = SoccerTrackingDataset(train_df, config, is_training=True)
    val_dataset = SoccerTrackingDataset(val_df, config, is_training=False)
    
    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Important for contrastive learning batch consistency
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader