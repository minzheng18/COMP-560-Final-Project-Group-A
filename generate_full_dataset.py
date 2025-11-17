import numpy as np
from pathlib import Path
import random

import maze_generator
from bfs_groundtruth import get_bfs_distances
from state_encoding import (
    build_dataset_from_distance_grid,
    build_residual_dataset_from_distance_grid,
)



# Set to True to use residual labels (distance - Manhattan), False for true distance labels
USE_RESIDUAL_LABELS = True

# Dataset parameters
NUM_MAPS = 200
WIDTH = 15
HEIGHT = 15
MIN_DENSITY = 0.1
MAX_DENSITY = 0.4

# Dataset split
TRAIN_RATIO = 0.8

# Random seed
SEED = 42

# ---------------------------------------------------------------------

def save_npz(filename: str, X: np.ndarray, y: np.ndarray) -> None:
    """Save X, y to a compressed .npz file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    print(f"Saved {path} with X.shape={X.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    print("=== Generating random mazes ===")
    maps = maze_generator.generate_map_batch(
        num_maps=NUM_MAPS,
        width=WIDTH,
        height=HEIGHT,
        min_density=MIN_DENSITY,
        max_density=MAX_DENSITY,
    )
    print(f"Generated {len(maps)} maps.")

    # Choose which builder to use
    if USE_RESIDUAL_LABELS:
        print("Using RESIDUAL labels: y = distance - Manhattan")
        builder = build_residual_dataset_from_distance_grid
    else:
        print("Using TRUE DISTANCE labels: y = shortest path length")
        builder = build_dataset_from_distance_grid

    X_list = []
    y_list = []

    for idx_map, maze in enumerate(maps):
        print(f"\nProcessing map {idx_map + 1}/{len(maps)}")

        goal = (WIDTH - 1, HEIGHT - 1)

        dist_grid = get_bfs_distances(maze, goal)

        try:
            X_map, y_map = builder(
                maze=maze,
                goal_pos=goal,
                distance_grid=dist_grid,
                unreachable_value=-1,
                max_samples=None,
            )
        except RuntimeError as e:
            print("  ->", e, "Skipping map.")
            continue

        print(f"  -> collected {X_map.shape[0]} samples")
        X_list.append(X_map)
        y_list.append(y_map)

    if not X_list:
        raise RuntimeError("No samples collected from any map; check maze/BFS settings.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print("\n=== Final dataset stats ===")
    print("Total samples:", X.shape[0])
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    split = int(TRAIN_RATIO * X.shape[0])
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print("\nTrain samples:", X_train.shape[0])
    print("Val samples:  ", X_val.shape[0])

    save_npz("dataset_train.npz", X_train, y_train)
    save_npz("dataset_val.npz", X_val, y_val)
