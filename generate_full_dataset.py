"""
Generate full train/val dataset using random mazes.

Uses:
- maze_generator.generate_map_batch
- bfs_groundtruth.get_bfs_distances
- state_encoding.build_dataset_from_distance_grid

Outputs:
- dataset_train.npz
- dataset_val.npz
"""

import numpy as np
from pathlib import Path
import random

import maze_generator
from bfs_groundtruth import get_bfs_distances
from state_encoding import build_dataset_from_distance_grid


def choose_random_free_goal(maze):
    """Pick a random free cell (0) as goal."""
    free_cells = np.argwhere(maze == 0)
    if len(free_cells) == 0:
        return None
    gy, gx = random.choice(free_cells)
    return (int(gx), int(gy))


def save_npz(path, X, y):
    path = Path(path)
    np.savez(path, X=X.astype(np.float32), y=y.astype(np.float32))
    print(f"[OK] Saved {path}  Samples = {X.shape[0]}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    num_maps = 50        
    width, height = 15, 15
    min_density, max_density = 0.1, 0.4
    train_ratio = 0.8

    print("=== Generating random map batch ===")
    maps = maze_generator.generate_map_batch(
        num_maps=num_maps,
        width=width,
        height=height,
        min_density=min_density,
        max_density=max_density,
    )

    X_all = []
    y_all = []

    for i, maze in enumerate(maps):
        print(f"[Map {i+1}/{num_maps}]")

        goal = choose_random_free_goal(maze)
        if goal is None:
            print("  -> no free cells, skipping map.")
            continue

        dist_grid = get_bfs_distances(maze, goal)

        try:
            X_map, y_map = build_dataset_from_distance_grid(
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

        X_all.append(X_map)
        y_all.append(y_map)

    if not X_all:
        raise RuntimeError("No data collected. Check maze/BFS settings.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    print("Total samples:", X.shape[0])
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    split = int(train_ratio * X.shape[0])
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print("Train samples:", X_train.shape[0])
    print("Val samples:", X_val.shape[0])

    save_npz("dataset_train.npz", X_train, y_train)
    save_npz("dataset_val.npz", X_val, y_val)
