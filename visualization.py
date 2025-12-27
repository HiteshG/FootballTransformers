"""
SoccerTransformer Visualization
- Visualize windows using mplsoccer
- Generate static frames and animated GIFs
- Compare query window with similar windows
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os


def create_pitch_frame(
    frame_data: pd.DataFrame,
    pitch,
    ax,
    title: str = "",
    show_trails: bool = False,
    trail_data: Optional[pd.DataFrame] = None,
    trail_alpha: float = 0.3,
):
    """
    Draw a single frame on a pitch.
    
    Args:
        frame_data: DataFrame with columns x, y, home_away_player for single frame
        pitch: mplsoccer Pitch object
        ax: matplotlib axis
        title: Title for the frame
        show_trails: Whether to show trajectory trails
        trail_data: Historical positions for trails
        trail_alpha: Alpha for trail lines
    """
    ax.clear()
    pitch.draw(ax=ax)
    
    # Separate ball, home, away
    ball = frame_data[frame_data['home_away_player'] == -1]
    home = frame_data[frame_data['home_away_player'] == 0]
    away = frame_data[frame_data['home_away_player'] == 1]
    
    # Draw trails if requested
    if show_trails and trail_data is not None:
        for player_id in frame_data['player_id'].unique():
            player_trail = trail_data[trail_data['player_id'] == player_id]
            if len(player_trail) > 1:
                home_away = player_trail['home_away_player'].iloc[0]
                if home_away == -1:
                    color = 'black'
                elif home_away == 0:
                    color = 'blue'
                else:
                    color = 'red'
                ax.plot(
                    player_trail['x'].values,
                    player_trail['y'].values,
                    color=color,
                    alpha=trail_alpha,
                    linewidth=1,
                    zorder=1
                )
    
    # Draw players
    # Ball - black, larger
    if len(ball) > 0:
        pitch.scatter(
            ball['x'].values,
            ball['y'].values,
            ax=ax,
            s=150,
            c='black',
            edgecolors='white',
            linewidth=2,
            zorder=4,
            label='Ball'
        )
    
    # Home team - blue
    if len(home) > 0:
        pitch.scatter(
            home['x'].values,
            home['y'].values,
            ax=ax,
            s=200,
            c='#2E86AB',  # Blue
            edgecolors='white',
            linewidth=1.5,
            zorder=3,
            label='Home'
        )
    
    # Away team - red
    if len(away) > 0:
        pitch.scatter(
            away['x'].values,
            away['y'].values,
            ax=ax,
            s=200,
            c='#E94F37',  # Red
            edgecolors='white',
            linewidth=1.5,
            zorder=3,
            label='Away'
        )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return ax


def visualize_window(
    window_df: pd.DataFrame,
    output_path: str,
    title: str = "",
    fps: int = 10,
    show_trails: bool = True,
    trail_length: int = 20,
    pitch_type: str = 'statsbomb',
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Create an animated GIF for a window.
    
    Args:
        window_df: DataFrame with columns: frame, x, y, player_id, home_away_player
        output_path: Path to save the GIF
        title: Title for the animation
        fps: Frames per second
        show_trails: Whether to show trajectory trails
        trail_length: Number of frames to show in trail
        pitch_type: Pitch type for mplsoccer
        figsize: Figure size
    """
    from mplsoccer import Pitch
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Create pitch
    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_color='#22312b',
        line_color='white',
        goal_type='box',
    )
    
    fig, ax = pitch.draw(figsize=figsize)
    
    # Get unique frames sorted
    frames = sorted(window_df['frame'].unique())
    
    def animate(frame_idx):
        frame_num = frames[frame_idx]
        frame_data = window_df[window_df['frame'] == frame_num]
        
        # Get trail data
        trail_data = None
        if show_trails:
            trail_start_idx = max(0, frame_idx - trail_length)
            trail_frames = frames[trail_start_idx:frame_idx + 1]
            trail_data = window_df[window_df['frame'].isin(trail_frames)]
        
        frame_title = f"{title}\nFrame {frame_idx + 1}/{len(frames)}"
        create_pitch_frame(
            frame_data, pitch, ax, frame_title,
            show_trails=show_trails, trail_data=trail_data
        )
        
        # Add legend on first frame
        if frame_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        return ax,
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(frames),
        interval=1000 // fps, blit=False
    )
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Saved animation: {output_path}")


def visualize_window_static(
    window_df: pd.DataFrame,
    output_path: str,
    title: str = "",
    num_frames: int = 6,
    pitch_type: str = 'statsbomb',
    figsize: Tuple[int, int] = (18, 12),
):
    """
    Create a static image showing multiple frames from a window.
    
    Args:
        window_df: DataFrame with window data
        output_path: Path to save image
        title: Overall title
        num_frames: Number of frames to show
        pitch_type: Pitch type
        figsize: Figure size
    """
    from mplsoccer import Pitch
    import matplotlib.pyplot as plt
    
    pitch = Pitch(
        pitch_type=pitch_type,
        pitch_color='#22312b',
        line_color='white',
        goal_type='box',
    )
    
    frames = sorted(window_df['frame'].unique())
    
    # Select evenly spaced frames
    if len(frames) <= num_frames:
        selected_frames = frames
    else:
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]
    
    # Create subplot grid
    ncols = min(3, len(selected_frames))
    nrows = (len(selected_frames) + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    for idx, frame_num in enumerate(selected_frames):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        pitch.draw(ax=ax)
        frame_data = window_df[window_df['frame'] == frame_num]
        
        # Get trail up to this point
        frame_idx = frames.index(frame_num)
        trail_start = max(0, frame_idx - 20)
        trail_frames = frames[trail_start:frame_idx + 1]
        trail_data = window_df[window_df['frame'].isin(trail_frames)]
        
        create_pitch_frame(
            frame_data, pitch, ax,
            title=f"Frame {frame_idx + 1}",
            show_trails=True,
            trail_data=trail_data
        )
    
    # Hide empty subplots
    for idx in range(len(selected_frames), nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved static visualization: {output_path}")


def visualize_similar_windows(
    query_window_df: pd.DataFrame,
    similar_windows_dfs: List[Tuple[str, float, pd.DataFrame]],
    output_dir: str,
    query_id: str,
    create_gifs: bool = True,
    create_static: bool = True,
    fps: int = 10,
):
    """
    Visualize query window and its similar windows.
    
    Args:
        query_window_df: DataFrame for query window
        similar_windows_dfs: List of (window_id, similarity, DataFrame) tuples
        output_dir: Directory to save visualizations
        query_id: Query window ID
        create_gifs: Whether to create animated GIFs
        create_static: Whether to create static images
        fps: FPS for animations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize query window
    if create_gifs:
        visualize_window(
            query_window_df,
            os.path.join(output_dir, f"query_{query_id.replace('/', '_')}.gif"),
            title=f"Query: {query_id}",
            fps=fps
        )
    
    if create_static:
        visualize_window_static(
            query_window_df,
            os.path.join(output_dir, f"query_{query_id.replace('/', '_')}.png"),
            title=f"Query: {query_id}"
        )
    
    # Visualize similar windows
    for rank, (window_id, similarity, window_df) in enumerate(similar_windows_dfs, 1):
        safe_id = window_id.replace('/', '_')
        
        if create_gifs:
            visualize_window(
                window_df,
                os.path.join(output_dir, f"similar_{rank}_{safe_id}.gif"),
                title=f"Rank {rank}: {window_id}\nSimilarity: {similarity:.4f}",
                fps=fps
            )
        
        if create_static:
            visualize_window_static(
                window_df,
                os.path.join(output_dir, f"similar_{rank}_{safe_id}.png"),
                title=f"Rank {rank}: {window_id} (Similarity: {similarity:.4f})"
            )
    
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"  - Query: query_{query_id.replace('/', '_')}.[gif/png]")
    for rank, (window_id, _, _) in enumerate(similar_windows_dfs, 1):
        safe_id = window_id.replace('/', '_')
        print(f"  - Rank {rank}: similar_{rank}_{safe_id}.[gif/png]")


def create_comparison_grid(
    query_window_df: pd.DataFrame,
    similar_windows_dfs: List[Tuple[str, float, pd.DataFrame]],
    output_path: str,
    query_id: str,
    frame_idx: int = 50,  # Middle frame
):
    """
    Create a single image comparing query with all similar windows at same frame.
    
    Args:
        query_window_df: Query window data
        similar_windows_dfs: List of (window_id, similarity, DataFrame)
        output_path: Path to save comparison image
        query_id: Query window ID
        frame_idx: Which frame index to show
    """
    from mplsoccer import Pitch
    import matplotlib.pyplot as plt
    
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color='#22312b',
        line_color='white',
        goal_type='box',
    )
    
    num_windows = 1 + len(similar_windows_dfs)
    ncols = min(3, num_windows)
    nrows = (num_windows + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if num_windows == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f"Query: {query_id} - Similar Windows Comparison", fontsize=14, fontweight='bold')
    
    # All windows to visualize
    all_windows = [(query_id, 1.0, query_window_df, "QUERY")] + [
        (wid, sim, df, f"Rank {i+1}") for i, (wid, sim, df) in enumerate(similar_windows_dfs)
    ]
    
    for idx, (window_id, similarity, window_df, label) in enumerate(all_windows):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        pitch.draw(ax=ax)
        
        frames = sorted(window_df['frame'].unique())
        actual_frame_idx = min(frame_idx, len(frames) - 1)
        frame_num = frames[actual_frame_idx]
        
        frame_data = window_df[window_df['frame'] == frame_num]
        
        # Trail data
        trail_start = max(0, actual_frame_idx - 20)
        trail_frames = frames[trail_start:actual_frame_idx + 1]
        trail_data = window_df[window_df['frame'].isin(trail_frames)]
        
        if label == "QUERY":
            title = f"QUERY\n{window_id}"
        else:
            title = f"{label} (sim: {similarity:.3f})\n{window_id}"
        
        create_pitch_frame(
            frame_data, pitch, ax, title,
            show_trails=True, trail_data=trail_data
        )
    
    # Hide empty subplots
    for idx in range(num_windows, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved comparison grid: {output_path}")