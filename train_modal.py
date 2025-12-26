"""
Modal App for SoccerTransformer Training
- Uses Modal 1.2.5 (App, not Stub)
- A100 GPU with 80GB memory
- Volume for data and checkpoints

Usage:
    1. Upload your CSV to the Modal volume:
       modal volume put soccer-transformer-data tracking_data.csv /tracking_data.csv
    
    2. Check data is accessible:
       modal run train_modal.py --action check
    
    3. Start training:
       modal run train_modal.py --action train
    
    4. List checkpoints:
       modal run train_modal.py --action list
"""
import modal
from pathlib import Path

# Get local directory for mounting modules
LOCAL_DIR = Path(__file__).parent

# Create Modal app
app = modal.App("soccer-transformer-training")

# Create persistent volume for data and checkpoints
volume = modal.Volume.from_name("soccer-transformer-data", create_if_missing=True)

# Define container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
    )
    .add_local_file(LOCAL_DIR / "config.py", "/root/config.py")
    .add_local_file(LOCAL_DIR / "dataset.py", "/root/dataset.py")
    .add_local_file(LOCAL_DIR / "model.py", "/root/model.py")
    .add_local_file(LOCAL_DIR / "losses.py", "/root/losses.py")
    .add_local_file(LOCAL_DIR / "trainer.py", "/root/trainer.py")
)

# Training configuration
VOLUME_PATH = "/data"
GPU_CONFIG = "A100-40GB"
TIMEOUT = 60 * 60 * 12  # 12 hours


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={VOLUME_PATH: volume},
    timeout=TIMEOUT,
    memory=32768,  # 32GB RAM
)
def train():
    """Main training function."""
    import torch
    import sys
    import os
    
    # Add current directory to path
    sys.path.insert(0, "/root")
    
    # Import local modules (will be mounted)
    from config import Config, ModelConfig, TrainingConfig, DataConfig
    from dataset import load_and_split_data, create_dataloaders
    from model import SoccerTransformer
    from trainer import Trainer
    
    print("=" * 70)
    print("SoccerTransformer Training")
    print("=" * 70)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = Config(
        model=ModelConfig(
            num_agents=23,
            num_timesteps=100,
            input_features=6,
            hidden_dim=256,
            num_heads=8,
            num_encoder_layers=10,
            embedding_dim=128,
            dropout=0.1,
            num_decoder_layers=4,
        ),
        training=TrainingConfig(
            train_ratio=0.85,
            val_ratio=0.15,
            batch_size=64,
            epochs=50,
            learning_rate=5e-5,
            weight_decay=1e-4,
            warmup_epochs=3,
            lambda_mr=1.0,
            lambda_cl=100.0,
            mask_ratio=0.8,
            temperature=0.5,
            log_every_n_steps=50,
            val_every_n_epochs=1,
            save_every_n_epochs=5,
        ),
        data=DataConfig(
            feature_columns=('x_norm', 'y_norm', 'vx_norm', 'vy_norm', 'home_away_player', 'role_id'),
            window_column='global_window_id',
            match_column='match_id',
            agent_column='agent_idx',
            frame_column='frame',
            data_volume_path=VOLUME_PATH,
            csv_filename='tracking_data.csv',
            checkpoint_path=f'{VOLUME_PATH}/checkpoints',
        )
    )
    
    # Load data
    print("\nLoading data...")
    train_df, val_df = load_and_split_data(config)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_df, val_df, config)
    
    # Free memory
    del train_df, val_df
    
    # Create model
    print("\nInitializing model...")
    model = SoccerTransformer(config.model)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()
    
    # Commit volume changes
    volume.commit()
    
    print("\nTraining complete! Checkpoints saved to volume.")
    
    return {"status": "success", "best_val_loss": trainer.best_val_loss}


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def list_checkpoints():
    """List available checkpoints."""
    import os
    
    checkpoint_dir = f"{VOLUME_PATH}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return {"checkpoints": [], "message": "No checkpoint directory found"}
    
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if f.endswith('.pt')]
    
    return {
        "checkpoints": sorted(checkpoints),
        "checkpoint_dir": checkpoint_dir
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def check_data():
    """Check if data file exists and preview it."""
    import os
    import pandas as pd
    
    csv_path = f"{VOLUME_PATH}/tracking_data.csv"
    
    if not os.path.exists(csv_path):
        return {
            "exists": False,
            "message": f"Data file not found at {csv_path}. Please upload tracking_data.csv to the volume."
        }
    
    # Get file size
    file_size = os.path.getsize(csv_path) / (1024 ** 3)  # GB
    
    # Load small sample
    df_sample = pd.read_csv(csv_path, nrows=1000)
    
    return {
        "exists": True,
        "file_size_gb": round(file_size, 2),
        "columns": list(df_sample.columns),
        "sample_rows": len(df_sample),
        "dtypes": {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
        "unique_windows_in_sample": df_sample['global_window_id'].nunique() if 'global_window_id' in df_sample.columns else None,
    }


@app.local_entrypoint()
def main(action: str = "train"):
    """
    Local entrypoint for running the app.
    
    Usage:
        modal run train_modal.py --action train
        modal run train_modal.py --action check
        modal run train_modal.py --action list
    """
    if action == "train":
        print("Starting training...")
        result = train.remote()
        print(f"Result: {result}")
    
    elif action == "check":
        print("Checking data...")
        result = check_data.remote()
        print(f"Result: {result}")
    
    elif action == "list":
        print("Listing checkpoints...")
        result = list_checkpoints.remote()
        print(f"Result: {result}")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, check, list")
