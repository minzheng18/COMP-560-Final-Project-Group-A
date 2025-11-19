# A* Search with Learned Residual Heuristic

This project implements the A* search algorithm enhanced with a Convolutional Neural Network (CNN) based on residual learning. The primary objective is to significantly improve pathfinding efficiency by providing an an obstacle-aware heuristic function $h(s)$ that reduces expanded nodes.

## File Structure and Purpose

* **`maze_generator.py`** # Utility for generating randomized mazes and fixed test maps.
* **`astar.py`** # Core application. Implements the A* algorithm, the Learned Heuristic adapter, and contains the comparison test framework.
* **`bfs_groundtruth.py`** # Computes the true shortest-path distance (Ground Truth) using BFS.
* **`generate_full_dataset.py`, `generate_small_dataset.py`** # Automated script for dataset generation for the training of the NN model.
* **`dataset_train.npz`, `dataset_val.npz`, `dataset_small.npz`** # The aggregated training and validation datasets used by the NN model.
* **`state_encoding.py`** # Utility for encoding the (Map, Agent, Goal) state into the 3 * H * W tensor format.
* **`train_residual_heuristic.py`** # Neural Network model definition and training logic (Residual Learning mode).
* **`heuristic.pt`** # The final trained Neural Network model file.
* **`visualize_compare.py`** # Generates dynamic and static visualizations (search animations, comparison plots) for the final report.
* **`demo.gif`, `compare_static.png`, `viz_seed7.gif`** # Visualization outputs (search animations and static performance plots) for the final demo.

## Usage

1. Ensure the model file (`heuristic.pt`) and dependencies are present.

2. Run the main script:

   ```bash
   python astar.py
