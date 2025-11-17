"""
State encoding utilities for learned heuristic project.

Matches the format used in bfs_groundtruth.generate_training_data_for_map:
- X: (N, 3, H, W), float32
    channel 0: obstacle map (1 = obstacle, 0 = free)
    channel 1: agent position one-hot
    channel 2: goal position one-hot
- y: (N,), float32
    shortest path length (steps) from agent to goal (from BFS)
"""

from typing import Tuple, List, Optional
import numpy as np


def encode_single_state(
    maze: np.ndarray,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
) -> np.ndarray:
    """
    Encode a single (maze, agent, goal) into a (3, H, W) tensor.

    maze: (H, W), 0 = free, 1 = obstacle
    agent_pos: (ax, ay)
    goal_pos:  (gx, gy)

    Returns: state_tensor (3, H, W), float32
    """
    if maze.ndim != 2:
        raise ValueError(f"maze must be 2D, got shape {maze.shape}")

    H, W = maze.shape
    ax, ay = agent_pos
    gx, gy = goal_pos

    if not (0 <= ax < W and 0 <= ay < H):
        raise ValueError("Agent position out of bounds")
    if not (0 <= gx < W and 0 <= gy < H):
        raise ValueError("Goal position out of bounds")
    if maze[ay, ax] == 1:
        raise ValueError("Agent must be on a free cell")
    if maze[gy, gx] == 1:
        raise ValueError("Goal must be on a free cell")


    obstacle_channel = maze.astype(np.float32)  # 1 = obstacle, 0 = free
    agent_channel = np.zeros((H, W), dtype=np.float32)
    agent_channel[ay, ax] = 1.0
    goal_channel = np.zeros((H, W), dtype=np.float32)
    goal_channel[gy, gx] = 1.0
    
    state_tensor = np.stack(
        [obstacle_channel, agent_channel, goal_channel],
        axis=0
    ).astype(np.float32)

    return state_tensor


def build_dataset_from_distance_grid(
    maze: np.ndarray,
    goal_pos: Tuple[int, int],
    distance_grid: np.ndarray,
    unreachable_value: int = -1,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) dataset from a BFS distance grid.

    maze: (H, W), 0 = free, 1 = obstacle
    goal_pos: (gx, gy)
    distance_grid: (H, W)
        distance_grid[y, x] = shortest path steps from (x, y) to goal
        unreachable cells = unreachable_value (e.g. -1)
    unreachable_value: value used for unreachable cells in distance_grid
    max_samples: optional cap on number of samples returned

    Returns:
        X: (N, 3, H, W), float32
        y: (N,), float32
    """
    H, W = maze.shape
    if distance_grid.shape != maze.shape:
        raise ValueError(
            f"distance_grid shape {distance_grid.shape} must match maze shape {maze.shape}"
        )

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    gx, gy = goal_pos

    for y in range(H):
        for x in range(W):
            if maze[y, x] == 1:
                continue

            d = distance_grid[y, x]
            if d == unreachable_value:
                continue

            state = encode_single_state(
                maze=maze,
                agent_pos=(x, y),
                goal_pos=(gx, gy),
            )
            X_list.append(state)
            y_list.append(float(d))

            if max_samples is not None and len(X_list) >= max_samples:
                break
        if max_samples is not None and len(X_list) >= max_samples:
            break

    if not X_list:
        raise RuntimeError("No reachable states found; check BFS or maze.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """
    Manhattan distance on grid between points a and b.

    a, b: (x, y)
    """
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


def build_residual_dataset_from_distance_grid(
    maze: np.ndarray,
    goal_pos: Tuple[int, int],
    distance_grid: np.ndarray,
    unreachable_value: int = -1,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    H, W = maze.shape
    if distance_grid.shape != maze.shape:
        raise ValueError(
            f"distance_grid shape {distance_grid.shape} must match maze shape {maze.shape}"
        )

    gx, gy = goal_pos
    if not (0 <= gx < W and 0 <= gy < H):
        raise ValueError("Goal position out of bounds in build_residual_dataset_from_distance_grid")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for y in range(H):
        for x in range(W):
            if maze[y, x] == 1:
                continue

            d = distance_grid[y, x]
            if d == unreachable_value:
                continue

            m = manhattan_distance((x, y), (gx, gy))
            residual = float(d - m)
            if residual < 0:
                residual = 0.0

            state = encode_single_state(
                maze=maze,
                agent_pos=(x, y),
                goal_pos=(gx, gy),
            )

            X_list.append(state)
            y_list.append(residual)

            if max_samples is not None and len(X_list) >= max_samples:
                break
        if max_samples is not None and len(X_list) >= max_samples:
            break

    if not X_list:
        raise RuntimeError(
            "No reachable states found when building residual dataset; "
            "check BFS result or maze."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y