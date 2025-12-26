# SoccerTransformer

Self-supervised learning model for soccer tracking data, adapted from HoopTransformer/Hoop-MSSL.

## Architecture

- **Encoder**: 10 axial attention layers (5 temporal + 5 spatial, alternating)
- **Decoder**: 4 axial attention layers for motion reconstruction
- **Contrastive Head**: 2D pooling + MLP projection to 128-dim embedding

## Training Tasks

1. **Motion Reconstruction (MR)**: Predict masked (x, y) coordinates
2. **Contrastive Learning (CL)**: NT-Xent loss with masking augmentation

Combined loss: `L = L_MR + 100 × L_CL`

## Input Features

| Feature | Description |
|---------|-------------|
| x_norm | Normalized x coordinate |
| y_norm | Normalized y coordinate |
| vx_norm | Normalized x velocity |
| vy_norm | Normalized y velocity |
| home_away_player | -1 (ball), 0 (home), 1 (away) |
| role_id | Encoded player role |

Input tensor shape: `[batch, 23, 100, 6]`

## Usage

### 1. Setup Modal Volume

```bash
# Create volume and upload data
modal volume create soccer-transformer-data
modal volume put soccer-transformer-data /path/to/tracking_data.csv /tracking_data.csv
```

### 2. Verify Data

```bash
modal run train_modal.py --action check
```

### 3. Start Training

```bash
modal run train_modal.py --action train
```

### 4. Check Checkpoints

```bash
modal run train_modal.py --action list
```

### 5. Download Checkpoints

```bash
modal volume get soccer-transformer-data /checkpoints/best_model.pt ./best_model.pt
```

## Configuration

Edit `config.py` to modify:

- Model architecture (hidden_dim, num_heads, etc.)
- Training hyperparameters (lr, batch_size, epochs)
- Loss weights (lambda_mr, lambda_cl)
- Masking ratio

## Monitoring

Training logs include:

- **loss_total**: Combined loss
- **loss_mr**: Motion reconstruction loss
- **loss_cl**: Contrastive loss (NT-Xent)
- **contrastive_accuracy**: How often positive is most similar
- **embedding_std**: Standard deviation (collapse detection)
- **positive_similarity**: Average similarity of positive pairs
- **negative_similarity**: Average similarity of negative pairs

## Files

```
soccer_transformer/
├── config.py          # Hyperparameters and configuration
├── dataset.py         # Data loading and masking
├── model.py           # SoccerTransformer architecture
├── losses.py          # MSE + NT-Xent losses
├── trainer.py         # Training loop and validation
├── train_modal.py     # Modal app for cloud training
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Expected Output

After training:
- `checkpoints/best_model.pt`: Best model by validation loss
- `checkpoints/final_model.pt`: Final epoch model
- `checkpoints/training_history.json`: All metrics per epoch
