"""
SoccerTransformer Inference
- Load trained model
- Generate embeddings for all windows
- Find similar windows with temporal constraints
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    checkpoint_path: str = '/data/checkpoints/best_model.pt'
    data_path: str = '/data/tracking_data.csv'
    embeddings_output_path: str = '/data/embeddings'
    
    # Model config (must match training)
    num_agents: int = 23
    num_timesteps: int = 100
    input_features: int = 6
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 10
    embedding_dim: int = 128
    dropout: float = 0.1
    num_decoder_layers: int = 4
    
    # Feature columns
    feature_columns: Tuple[str, ...] = (
        'x_norm', 'y_norm', 'vx_norm', 'vy_norm', 'home_away_player', 'role_id'
    )
    window_column: str = 'global_window_id'
    match_column: str = 'match_id'
    agent_column: str = 'agent_idx'
    frame_column: str = 'frame'


class SoccerTransformerInference:
    """Inference wrapper for SoccerTransformer."""
    
    def __init__(self, config: InferenceConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Storage for embeddings
        self.embeddings = None
        self.global_window_ids = None
        self.match_ids = None
        self.window_to_idx = None
        self.match_to_windows = None
        
    def _load_model(self):
        """Load trained model from checkpoint."""
        from model import SoccerTransformer
        from config import ModelConfig
        
        # Create model config
        model_config = ModelConfig(
            num_agents=self.config.num_agents,
            num_timesteps=self.config.num_timesteps,
            input_features=self.config.input_features,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_encoder_layers=self.config.num_encoder_layers,
            embedding_dim=self.config.embedding_dim,
            dropout=self.config.dropout,
            num_decoder_layers=self.config.num_decoder_layers,
        )
        
        # Create model
        model = SoccerTransformer(model_config)
        
        # Load checkpoint
        print(f"Loading checkpoint from {self.config.checkpoint_path}")
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        return model
    
    def _preprocess_window(self, window_df: pd.DataFrame) -> torch.Tensor:
        """Preprocess a single window into tensor."""
        feature_cols = list(self.config.feature_columns)
        
        # Sort by frame and agent
        window_df = window_df.sort_values([self.config.frame_column, self.config.agent_column])
        
        # Extract features
        features = window_df[feature_cols].values
        
        # Reshape to [agents, timesteps, features]
        features = features.reshape(
            self.config.num_timesteps, 
            self.config.num_agents, 
            self.config.input_features
        )
        features = features.transpose(1, 0, 2)  # [A, T, F]
        
        return torch.tensor(features, dtype=torch.float32)
    
    @torch.no_grad()
    def generate_embeddings(self, df: pd.DataFrame, batch_size: int = 64) -> Dict:
        """Generate embeddings for all windows in dataframe."""
        print("Generating embeddings...")
        
        # Get unique windows
        self.global_window_ids = df[self.config.window_column].unique().tolist()
        self.window_to_idx = {wid: idx for idx, wid in enumerate(self.global_window_ids)}
        
        # Build match to windows mapping
        self.match_to_windows = {}
        window_to_match = df.groupby(self.config.window_column)[self.config.match_column].first().to_dict()
        for wid, mid in window_to_match.items():
            if mid not in self.match_to_windows:
                self.match_to_windows[mid] = []
            self.match_to_windows[mid].append(wid)
        
        self.match_ids = list(self.match_to_windows.keys())
        
        # Preprocess all windows
        print(f"Preprocessing {len(self.global_window_ids)} windows...")
        all_tensors = []
        valid_window_ids = []
        
        grouped = df.groupby(self.config.window_column)
        for wid in self.global_window_ids:
            try:
                window_df = grouped.get_group(wid)
                tensor = self._preprocess_window(window_df)
                all_tensors.append(tensor)
                valid_window_ids.append(wid)
            except Exception as e:
                print(f"Skipping window {wid}: {e}")
                continue
        
        # Update window IDs to only valid ones
        self.global_window_ids = valid_window_ids
        self.window_to_idx = {wid: idx for idx, wid in enumerate(self.global_window_ids)}
        
        # Stack into batches and generate embeddings
        all_tensors = torch.stack(all_tensors)  # [N, A, T, F]
        num_windows = len(all_tensors)
        
        embeddings_list = []
        
        for i in range(0, num_windows, batch_size):
            batch = all_tensors[i:i+batch_size].to(self.device)
            
            with torch.cuda.amp.autocast():
                emb = self.model.get_embedding(batch)
            
            embeddings_list.append(emb.cpu())
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"  Processed {min(i + batch_size, num_windows)}/{num_windows} windows")
        
        self.embeddings = torch.cat(embeddings_list, dim=0)  # [N, embedding_dim]
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)  # L2 normalize
        
        print(f"Generated {self.embeddings.shape[0]} embeddings of dimension {self.embeddings.shape[1]}")
        
        return {
            'embeddings': self.embeddings.numpy(),
            'global_window_ids': self.global_window_ids,
            'match_ids': self.match_ids,
        }
    
    def save_embeddings(self, output_dir: str = None):
        """Save embeddings to disk."""
        output_dir = output_dir or self.config.embeddings_output_path
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings as numpy
        np.save(os.path.join(output_dir, 'embeddings.npy'), self.embeddings.numpy())
        
        # Save metadata
        metadata = {
            'global_window_ids': self.global_window_ids,
            'match_ids': self.match_ids,
            'window_to_idx': self.window_to_idx,
            'match_to_windows': self.match_to_windows,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"Saved embeddings to {output_dir}")
    
    def load_embeddings(self, input_dir: str = None):
        """Load embeddings from disk."""
        input_dir = input_dir or self.config.embeddings_output_path
        
        self.embeddings = torch.tensor(np.load(os.path.join(input_dir, 'embeddings.npy')))
        
        with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.global_window_ids = metadata['global_window_ids']
        self.match_ids = metadata['match_ids']
        self.window_to_idx = metadata['window_to_idx']
        self.match_to_windows = metadata['match_to_windows']
        
        print(f"Loaded {self.embeddings.shape[0]} embeddings from {input_dir}")
    
    def _parse_window_id(self, global_window_id: str) -> Tuple[str, float, int]:
        """
        Parse global_window_id into components.
        Format: {match_id}_{period}_{window_num}
        Example: 1021404_1.0_0 -> (1021404, 1.0, 0)
        """
        parts = global_window_id.rsplit('_', 2)
        match_id = parts[0]
        period = float(parts[1])
        window_num = int(parts[2])
        return match_id, period, window_num
    
    def _get_temporal_distance(self, wid1: str, wid2: str) -> Optional[int]:
        """
        Get temporal distance between two windows in frames.
        Returns None if windows are from different matches or periods.
        
        Each window is 100 frames with 50 stride, so:
        - window_num difference of 1 = 50 frames apart
        - 1 minute = 600 frames = 12 window difference
        """
        try:
            match1, period1, num1 = self._parse_window_id(wid1)
            match2, period2, num2 = self._parse_window_id(wid2)
            
            if match1 != match2 or period1 != period2:
                return None
            
            # Window stride is 50 frames
            frame_distance = abs(num2 - num1) * 50
            return frame_distance
        except:
            return None
    
    def find_similar_windows(
        self,
        query_global_window_id: str,
        top_k: int = 5,
        min_temporal_distance_seconds: float = 60.0,
        same_match_only: bool = False,
        exclude_same_match: bool = False,
    ) -> List[Dict]:
        """
        Find top-k most similar windows to query.
        
        Args:
            query_global_window_id: The window to find similarities for
            top_k: Number of similar windows to return
            min_temporal_distance_seconds: Minimum time apart (in seconds) for same-match results
            same_match_only: If True, only return windows from same match
            exclude_same_match: If True, exclude windows from same match
            
        Returns:
            List of dicts with window_id, similarity, match_id, temporal_distance_seconds
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Run generate_embeddings() or load_embeddings() first.")
        
        if query_global_window_id not in self.window_to_idx:
            raise ValueError(f"Window {query_global_window_id} not found in embeddings.")
        
        # Get query embedding
        query_idx = self.window_to_idx[query_global_window_id]
        query_emb = self.embeddings[query_idx:query_idx+1]  # [1, D]
        
        # Compute cosine similarities
        similarities = torch.mm(query_emb, self.embeddings.t()).squeeze()  # [N]
        
        # Parse query window info
        query_match, query_period, query_num = self._parse_window_id(query_global_window_id)
        
        # Convert min_temporal_distance to frames (10 FPS)
        min_frame_distance = min_temporal_distance_seconds * 10
        
        # Filter and sort
        results = []
        
        # Get indices sorted by similarity (descending)
        sorted_indices = torch.argsort(similarities, descending=True)
        
        for idx in sorted_indices.tolist():
            wid = self.global_window_ids[idx]
            
            # Skip query itself
            if wid == query_global_window_id:
                continue
            
            sim = similarities[idx].item()
            
            # Parse candidate window
            try:
                cand_match, cand_period, cand_num = self._parse_window_id(wid)
            except:
                continue
            
            # Check match constraints
            is_same_match = (cand_match == query_match)
            
            if same_match_only and not is_same_match:
                continue
            if exclude_same_match and is_same_match:
                continue
            
            # Check temporal distance for same match
            temporal_distance_frames = None
            temporal_distance_seconds = None
            
            if is_same_match and cand_period == query_period:
                temporal_distance_frames = abs(cand_num - query_num) * 50
                temporal_distance_seconds = temporal_distance_frames / 10.0
                
                # Skip if too close temporally
                if temporal_distance_frames < min_frame_distance:
                    continue
            
            results.append({
                'global_window_id': wid,
                'similarity': sim,
                'match_id': cand_match,
                'period': cand_period,
                'is_same_match': is_same_match,
                'temporal_distance_seconds': temporal_distance_seconds,
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def list_matches(self) -> List[str]:
        """List all available match IDs."""
        return self.match_ids
    
    def list_windows_for_match(self, match_id: str) -> List[str]:
        """List all window IDs for a given match."""
        if match_id not in self.match_to_windows:
            raise ValueError(f"Match {match_id} not found.")
        return sorted(self.match_to_windows[match_id])


def print_similar_windows(results: List[Dict], query_id: str):
    """Pretty print similar windows results."""
    print(f"\n{'='*70}")
    print(f"Query: {query_id}")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Window ID':<30}{'Similarity':<12}{'Same Match':<12}{'Time Apart':<12}")
    print(f"{'-'*70}")
    
    for i, r in enumerate(results, 1):
        time_str = f"{r['temporal_distance_seconds']:.1f}s" if r['temporal_distance_seconds'] else "N/A"
        same_match_str = "Yes" if r['is_same_match'] else "No"
        print(f"{i:<6}{r['global_window_id']:<30}{r['similarity']:<12.4f}{same_match_str:<12}{time_str:<12}")
    
    print(f"{'='*70}\n")