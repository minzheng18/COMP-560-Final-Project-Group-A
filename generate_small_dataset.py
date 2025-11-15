"""
generate_dataset_small.py
Person B (Data Engineer)

Goal (Day 1):
- Use A's BFS + training-data generator to produce a small demo dataset
- Save dataset_small.npz for ML Engineer (C) to test model code
"""

import numpy as np
import maze_generator
from bfs_groundtruth import generate_training_data_for_map
from pathlib import Path


def save_npz(path, X, y):
    path = Path(path)
    np.savez(path, X=X.astype(np.float32), y=y.astype(np.float32))
    print(f"[OK] Saved {path}   Samples = {X.shape[0]}")


if __name__ == "__main__":
    print("=== Person B: Generating small demo dataset ===")

    maze = maze_generator.create_test_maze()
    H, W = maze.shape
    goal = (W-1, H-1)

    X, y = generate_training_data_for_map(maze, goal)

    max_samples = 80
    if X.shape[0] > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    save_npz("dataset_small.npz", X, y)
    print("Dataset shape:", X.shape, y.shape)
