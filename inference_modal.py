"""
Modal App for SoccerTransformer Inference
- Generate embeddings for all windows
- Find similar windows interactively

Usage:
    # Generate embeddings for all data
    modal run inference_modal.py --action generate
    
    # List available matches
    modal run inference_modal.py --action list-matches
    
    # List windows for a match
    modal run inference_modal.py --action list-windows --match-id 1021404
    
    # Find similar windows
    modal run inference_modal.py --action find-similar --window-id "1021404_1.0_100" --top-k 5
"""
import modal
from pathlib import Path

LOCAL_DIR = Path(__file__).parent

app = modal.App("soccer-transformer-inference")

volume = modal.Volume.from_name("soccer-transformer-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "mplsoccer==1.2.4",
        "matplotlib==3.7.2",
        "pillow==10.0.0",
    )
    .add_local_file(LOCAL_DIR / "config.py", "/root/config.py")
    .add_local_file(LOCAL_DIR / "model.py", "/root/model.py")
    .add_local_file(LOCAL_DIR / "losses.py", "/root/losses.py")
    .add_local_file(LOCAL_DIR / "inference.py", "/root/inference.py")
    .add_local_file(LOCAL_DIR / "visualization.py", "/root/visualization.py")
)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 2,  # 2 hours
    memory=32768,
)
def generate_embeddings(
    csv_filename: str = "tracking_data.csv",
    checkpoint_filename: str = "best_model.pt",
    batch_size: int = 64,
):
    """Generate embeddings for all windows in the dataset."""
    import sys
    sys.path.insert(0, "/root")
    
    import pandas as pd
    import torch
    from inference import SoccerTransformerInference, InferenceConfig
    
    print("=" * 70)
    print("Generating Embeddings")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    config = InferenceConfig(
        checkpoint_path=f"{VOLUME_PATH}/checkpoints/{checkpoint_filename}",
        data_path=f"{VOLUME_PATH}/{csv_filename}",
        embeddings_output_path=f"{VOLUME_PATH}/embeddings",
    )
    
    # Load data
    print(f"\nLoading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Unique windows: {df['global_window_id'].nunique()}")
    print(f"Unique matches: {df['match_id'].nunique()}")
    
    # Create inference engine
    engine = SoccerTransformerInference(config, device)
    
    # Generate embeddings
    result = engine.generate_embeddings(df, batch_size=batch_size)
    
    # Save embeddings
    engine.save_embeddings()
    
    # Commit volume
    volume.commit()
    
    return {
        "status": "success",
        "num_embeddings": len(result['global_window_ids']),
        "num_matches": len(result['match_ids']),
        "embedding_dim": result['embeddings'].shape[1],
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def list_matches():
    """List all available matches."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    
    metadata_path = f"{VOLUME_PATH}/embeddings/metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        matches = metadata['match_ids']
        return {
            "status": "success",
            "num_matches": len(matches),
            "matches": sorted(matches)[:50],  # Return first 50
            "message": f"Showing first 50 of {len(matches)} matches" if len(matches) > 50 else None
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Embeddings not found. Run 'generate' action first."
        }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def list_windows(match_id: str):
    """List all windows for a given match."""
    import sys
    sys.path.insert(0, "/root")
    
    import json
    
    metadata_path = f"{VOLUME_PATH}/embeddings/metadata.json"
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if match_id not in metadata['match_to_windows']:
            return {
                "status": "error",
                "message": f"Match {match_id} not found."
            }
        
        windows = sorted(metadata['match_to_windows'][match_id])
        return {
            "status": "success",
            "match_id": match_id,
            "num_windows": len(windows),
            "windows": windows[:100],  # Return first 100
            "message": f"Showing first 100 of {len(windows)} windows" if len(windows) > 100 else None
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Embeddings not found. Run 'generate' action first."
        }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=300,
)
def find_similar(
    window_id: str,
    top_k: int = 5,
    min_temporal_distance: float = 60.0,
    same_match_only: bool = False,
    exclude_same_match: bool = False,
):
    """Find similar windows to a query window."""
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    import numpy as np
    import json
    
    # Load embeddings directly (no model needed)
    embeddings_dir = f"{VOLUME_PATH}/embeddings"
    
    try:
        embeddings = torch.tensor(np.load(f"{embeddings_dir}/embeddings.npy"))
        with open(f"{embeddings_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Embeddings not found. Run 'generate' action first."
        }
    
    global_window_ids = metadata['global_window_ids']
    window_to_idx = metadata['window_to_idx']
    
    if window_id not in window_to_idx:
        return {
            "status": "error",
            "message": f"Window {window_id} not found in embeddings."
        }
    
    # Get query embedding
    query_idx = window_to_idx[window_id]
    query_emb = embeddings[query_idx:query_idx+1]
    
    # Compute similarities
    similarities = torch.mm(query_emb, embeddings.t()).squeeze()
    
    # Parse query window
    def parse_window_id(wid):
        parts = wid.rsplit('_', 2)
        return parts[0], float(parts[1]), int(parts[2])
    
    query_match, query_period, query_num = parse_window_id(window_id)
    min_frame_distance = min_temporal_distance * 10
    
    # Sort and filter
    sorted_indices = torch.argsort(similarities, descending=True)
    
    results = []
    for idx in sorted_indices.tolist():
        wid = global_window_ids[idx]
        
        if wid == window_id:
            continue
        
        sim = similarities[idx].item()
        
        try:
            cand_match, cand_period, cand_num = parse_window_id(wid)
        except:
            continue
        
        is_same_match = (cand_match == query_match)
        
        if same_match_only and not is_same_match:
            continue
        if exclude_same_match and is_same_match:
            continue
        
        temporal_distance_seconds = None
        if is_same_match and cand_period == query_period:
            temporal_distance_frames = abs(cand_num - query_num) * 50
            temporal_distance_seconds = temporal_distance_frames / 10.0
            
            if temporal_distance_frames < min_frame_distance:
                continue
        
        results.append({
            'global_window_id': wid,
            'similarity': round(sim, 4),
            'match_id': cand_match,
            'period': cand_period,
            'is_same_match': is_same_match,
            'temporal_distance_seconds': temporal_distance_seconds,
        })
        
        if len(results) >= top_k:
            break
    
    return {
        "status": "success",
        "query_window_id": window_id,
        "results": results,
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,  # 30 minutes for generating GIFs
    memory=16384,
)
def visualize_similar(
    window_id: str,
    top_k: int = 5,
    min_temporal_distance: float = 60.0,
    same_match_only: bool = False,
    exclude_same_match: bool = False,
    create_gifs: bool = True,
    create_static: bool = True,
    fps: int = 10,
):
    """
    Find similar windows and generate visualizations.
    
    Creates:
    - Animated GIFs showing trajectories
    - Static images with key frames
    - Comparison grid
    """
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    import numpy as np
    import json
    import pandas as pd
    from visualization import (
        visualize_similar_windows,
        create_comparison_grid,
    )
    
    print(f"Finding similar windows for: {window_id}")
    
    # Load embeddings
    embeddings_dir = f"{VOLUME_PATH}/embeddings"
    
    try:
        embeddings = torch.tensor(np.load(f"{embeddings_dir}/embeddings.npy"))
        with open(f"{embeddings_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "Embeddings not found. Run 'generate' action first."
        }
    
    global_window_ids = metadata['global_window_ids']
    window_to_idx = metadata['window_to_idx']
    
    if window_id not in window_to_idx:
        return {
            "status": "error",
            "message": f"Window {window_id} not found in embeddings."
        }
    
    # Find similar windows (same logic as find_similar)
    query_idx = window_to_idx[window_id]
    query_emb = embeddings[query_idx:query_idx+1]
    similarities = torch.mm(query_emb, embeddings.t()).squeeze()
    
    def parse_window_id(wid):
        parts = wid.rsplit('_', 2)
        return parts[0], float(parts[1]), int(parts[2])
    
    query_match, query_period, query_num = parse_window_id(window_id)
    min_frame_distance = min_temporal_distance * 10
    
    sorted_indices = torch.argsort(similarities, descending=True)
    
    similar_results = []
    for idx in sorted_indices.tolist():
        wid = global_window_ids[idx]
        
        if wid == window_id:
            continue
        
        sim = similarities[idx].item()
        
        try:
            cand_match, cand_period, cand_num = parse_window_id(wid)
        except:
            continue
        
        is_same_match = (cand_match == query_match)
        
        if same_match_only and not is_same_match:
            continue
        if exclude_same_match and is_same_match:
            continue
        
        if is_same_match and cand_period == query_period:
            temporal_distance_frames = abs(cand_num - query_num) * 50
            if temporal_distance_frames < min_frame_distance:
                continue
        
        similar_results.append({
            'global_window_id': wid,
            'similarity': sim,
        })
        
        if len(similar_results) >= top_k:
            break
    
    print(f"Found {len(similar_results)} similar windows")
    
    # Load tracking data
    print("Loading tracking data...")
    df = pd.read_csv(f"{VOLUME_PATH}/tracking_data.csv")
    
    # Get query window data
    query_df = df[df['global_window_id'] == window_id].copy()
    
    # Convert normalized coords back to pitch coords for visualization
    # Assuming StatsBomb pitch: 120 x 80
    query_df['x'] = query_df['x_norm'] * 120
    query_df['y'] = query_df['y_norm'] * 80
    
    # Get similar windows data
    similar_windows_dfs = []
    for result in similar_results:
        wid = result['global_window_id']
        sim = result['similarity']
        
        window_df = df[df['global_window_id'] == wid].copy()
        window_df['x'] = window_df['x_norm'] * 120
        window_df['y'] = window_df['y_norm'] * 80
        
        similar_windows_dfs.append((wid, sim, window_df))
    
    # Create output directory
    safe_query_id = window_id.replace('/', '_').replace('.', '_')
    output_dir = f"{VOLUME_PATH}/visualizations/{safe_query_id}"
    
    print(f"Generating visualizations in: {output_dir}")
    
    # Generate visualizations
    visualize_similar_windows(
        query_window_df=query_df,
        similar_windows_dfs=similar_windows_dfs,
        output_dir=output_dir,
        query_id=window_id,
        create_gifs=create_gifs,
        create_static=create_static,
        fps=fps,
    )
    
    # Create comparison grid
    create_comparison_grid(
        query_window_df=query_df,
        similar_windows_dfs=similar_windows_dfs,
        output_path=f"{output_dir}/comparison_grid.png",
        query_id=window_id,
    )
    
    # Commit volume
    volume.commit()
    
    return {
        "status": "success",
        "query_window_id": window_id,
        "similar_windows": [r['global_window_id'] for r in similar_results],
        "output_dir": output_dir,
        "files_created": {
            "gifs": create_gifs,
            "static": create_static,
            "comparison_grid": True,
        }
    }


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def interactive_demo(match_id: str = None, num_queries: int = 3):
    """
    Run an interactive demo:
    - Pick random windows from a match
    - Find similar windows for each
    """
    import sys
    sys.path.insert(0, "/root")
    
    import torch
    import numpy as np
    import json
    import random
    
    # Load embeddings (no model needed)
    embeddings_dir = f"{VOLUME_PATH}/embeddings"
    
    embeddings = torch.tensor(np.load(f"{embeddings_dir}/embeddings.npy"))
    with open(f"{embeddings_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    global_window_ids = metadata['global_window_ids']
    window_to_idx = metadata['window_to_idx']
    match_to_windows = metadata['match_to_windows']
    match_ids = metadata['match_ids']
    
    # Pick a match if not provided
    if match_id is None:
        match_id = random.choice(match_ids)
    
    print(f"\n{'='*70}")
    print(f"Interactive Demo - Match: {match_id}")
    print(f"{'='*70}")
    
    # Get windows for this match
    windows = sorted(match_to_windows.get(match_id, []))
    print(f"Total windows in match: {len(windows)}")
    
    if len(windows) == 0:
        return {"status": "error", "message": f"No windows found for match {match_id}"}
    
    # Pick random query windows
    query_windows = random.sample(windows, min(num_queries, len(windows)))
    
    def parse_window_id(wid):
        parts = wid.rsplit('_', 2)
        return parts[0], float(parts[1]), int(parts[2])
    
    def find_similar_for_query(query_id, top_k=5, min_time=60.0, same_match_only=False, exclude_same_match=False):
        query_idx = window_to_idx[query_id]
        query_emb = embeddings[query_idx:query_idx+1]
        similarities = torch.mm(query_emb, embeddings.t()).squeeze()
        
        query_match, query_period, query_num = parse_window_id(query_id)
        min_frame_distance = min_time * 10
        
        sorted_indices = torch.argsort(similarities, descending=True)
        
        results = []
        for idx in sorted_indices.tolist():
            wid = global_window_ids[idx]
            if wid == query_id:
                continue
            
            sim = similarities[idx].item()
            
            try:
                cand_match, cand_period, cand_num = parse_window_id(wid)
            except:
                continue
            
            is_same_match = (cand_match == query_match)
            
            if same_match_only and not is_same_match:
                continue
            if exclude_same_match and is_same_match:
                continue
            
            temporal_distance_seconds = None
            if is_same_match and cand_period == query_period:
                temporal_distance_frames = abs(cand_num - query_num) * 50
                temporal_distance_seconds = temporal_distance_frames / 10.0
                if temporal_distance_frames < min_frame_distance:
                    continue
            
            results.append({
                'global_window_id': wid,
                'similarity': round(sim, 4),
                'is_same_match': is_same_match,
                'temporal_distance_seconds': temporal_distance_seconds,
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    all_results = []
    
    for query_id in query_windows:
        print(f"\n--- Query: {query_id} ---")
        
        # Same match, 1+ min apart
        print("\n[Same match, 1+ min apart]")
        results_same = find_similar_for_query(query_id, same_match_only=True)
        for i, r in enumerate(results_same, 1):
            time_str = f"{r['temporal_distance_seconds']:.1f}s" if r['temporal_distance_seconds'] else "N/A"
            print(f"  {i}. {r['global_window_id']} (sim: {r['similarity']:.4f}, time: {time_str})")
        
        # Different matches
        print("\n[Different matches]")
        results_other = find_similar_for_query(query_id, exclude_same_match=True)
        for i, r in enumerate(results_other, 1):
            print(f"  {i}. {r['global_window_id']} (sim: {r['similarity']:.4f})")
        
        all_results.append({
            "query": query_id,
            "same_match": results_same,
            "other_matches": results_other,
        })
    
    return {
        "status": "success",
        "match_id": match_id,
        "results": all_results,
    }


@app.local_entrypoint()
def main(
    action: str = "generate",
    window_id: str = None,
    match_id: str = None,
    top_k: int = 5,
    min_time: float = 60.0,
    same_match: bool = False,
    exclude_same: bool = False,
    create_gifs: bool = True,
    create_static: bool = True,
    fps: int = 10,
):
    """
    Local entrypoint for inference.
    
    Actions:
        generate       - Generate embeddings for all windows
        list-matches   - List available matches
        list-windows   - List windows for a match (requires --match-id)
        find-similar   - Find similar windows (requires --window-id)
        visualize      - Find similar + generate visualizations (requires --window-id)
        demo           - Run interactive demo (optional --match-id)
    
    Examples:
        modal run inference_modal.py --action generate
        modal run inference_modal.py --action list-matches
        modal run inference_modal.py --action list-windows --match-id 1021404
        modal run inference_modal.py --action find-similar --window-id "1021404_1.0_100"
        modal run inference_modal.py --action find-similar --window-id "1021404_1.0_100" --same-match
        modal run inference_modal.py --action visualize --window-id "1021404_1.0_100" --top-k 5
        modal run inference_modal.py --action visualize --window-id "1021404_1.0_100" --same-match --no-create-gifs
        modal run inference_modal.py --action demo --match-id 1021404
    """
    if action == "generate":
        print("Generating embeddings for all windows...")
        result = generate_embeddings.remote()
        print(f"Result: {result}")
    
    elif action == "list-matches":
        result = list_matches.remote()
        if result['status'] == 'success':
            print(f"\nAvailable matches ({result['num_matches']} total):")
            for m in result['matches']:
                print(f"  {m}")
            if result.get('message'):
                print(f"\n{result['message']}")
        else:
            print(f"Error: {result['message']}")
    
    elif action == "list-windows":
        if not match_id:
            print("Error: --match-id required for list-windows action")
            return
        result = list_windows.remote(match_id)
        if result['status'] == 'success':
            print(f"\nWindows for match {match_id} ({result['num_windows']} total):")
            for w in result['windows']:
                print(f"  {w}")
            if result.get('message'):
                print(f"\n{result['message']}")
        else:
            print(f"Error: {result['message']}")
    
    elif action == "find-similar":
        if not window_id:
            print("Error: --window-id required for find-similar action")
            return
        result = find_similar.remote(
            window_id=window_id,
            top_k=top_k,
            min_temporal_distance=min_time,
            same_match_only=same_match,
            exclude_same_match=exclude_same,
        )
        if result['status'] == 'success':
            print(f"\nTop {top_k} similar windows for {window_id}:")
            for i, r in enumerate(result['results'], 1):
                time_str = f"{r['temporal_distance_seconds']:.1f}s" if r['temporal_distance_seconds'] else "N/A"
                print(f"  {i}. {r['global_window_id']} (sim: {r['similarity']:.4f}, same_match: {r['is_same_match']}, time_apart: {time_str})")
        else:
            print(f"Error: {result['message']}")
    
    elif action == "visualize":
        if not window_id:
            print("Error: --window-id required for visualize action")
            return
        print(f"Finding similar windows and generating visualizations for: {window_id}")
        result = visualize_similar.remote(
            window_id=window_id,
            top_k=top_k,
            min_temporal_distance=min_time,
            same_match_only=same_match,
            exclude_same_match=exclude_same,
            create_gifs=create_gifs,
            create_static=create_static,
            fps=fps,
        )
        if result['status'] == 'success':
            print(f"\n✓ Visualizations generated!")
            print(f"  Query: {result['query_window_id']}")
            print(f"  Similar windows: {result['similar_windows']}")
            print(f"  Output directory: {result['output_dir']}")
            print(f"\nDownload with:")
            print(f"  modal volume get soccer-transformer-data {result['output_dir']} ./visualizations/")
        else:
            print(f"Error: {result['message']}")
    
    elif action == "demo":
        print("Running interactive demo...")
        result = interactive_demo.remote(match_id=match_id)
        print(f"\nDemo complete for match: {result['match_id']}")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: generate, list-matches, list-windows, find-similar, visualize, demo")