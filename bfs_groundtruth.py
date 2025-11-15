import numpy as np
from collections import deque
import maze_generator

def get_bfs_distances(maze, goal):
    """
    Run BFS to compute shortest distances from all reachable cells to a single goal cell.
    """
    height, width = maze.shape
    distance_grid = np.full((height, width), -1) 
    
    if maze[goal[1], goal[0]] == 1:
        return distance_grid

    queue = deque([goal])
    distance_grid[goal[1], goal[0]] = 0
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        current_x, current_y = queue.popleft()
        current_dist = distance_grid[current_y, current_x]
        
        for dx, dy in neighbors:
            next_x, next_y = current_x + dx, current_y + dy
            
            if 0 <= next_x < width and 0 <= next_y < height:
                if maze[next_y, next_x] == 0 and distance_grid[next_y, next_x] == -1:
                    distance_grid[next_y, next_x] = current_dist + 1
                    queue.append((next_x, next_y))
                    
    return distance_grid

def generate_training_data_for_map(maze, goal):
    """
    Create training data (X, y) for a single map and goal.
    """
    height, width = maze.shape
    
    distance_grid = get_bfs_distances(maze, goal)
    
    obstacle_channel = maze.astype(np.float32)
    
    goal_channel = np.zeros((height, width), dtype=np.float32)
    goal_channel[goal[1], goal[0]] = 1.0
    
    X_list = []
    y_list = []
    
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 0:
                distance = distance_grid[y, x]
                if distance != -1:
                    agent_channel = np.zeros((height, width), dtype=np.float32)
                    agent_channel[y, x] = 1.0
                    
                    X_sample = np.stack([obstacle_channel, agent_channel, goal_channel], axis=0)
                    
                    X_list.append(X_sample)
                    y_list.append(float(distance))

    if not X_list:
        return np.array([]), np.array([])
        
    X_batch = np.stack(X_list, axis=0)
    y_batch = np.array(y_list, dtype=np.float32)
    
    return X_batch, y_batch

if __name__ == "__main__":
    print("BFS Ground Truth Test:")
    test_maze = maze_generator.create_test_maze()
    goal_pos = (4, 4)
    distances = get_bfs_distances(test_maze, goal_pos)
    print(f"Distance grid (goal at {goal_pos}):")
    print(distances)

    print("Training Data Generation Test:")
    X, y = generate_training_data_for_map(test_maze, goal_pos)
    
    if X.shape[0] > 0:
        print(f"Generated {X.shape[0]} training samples.")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
    else:
        print("No valid paths found for this map.")