import numpy as np
import random

def create_maze(width, height, obstacle_density=0.3):
    """
    Generate a random maze (0 = free cell, 1 = obstacle).
    """
    maze = np.random.rand(height, width) < obstacle_density
    maze = maze.astype(int)
    
    maze[0, 0] = 0
    maze[height-1, width-1] = 0
    return maze

def create_test_maze():
    maze = np.array([
        [0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    return maze

def generate_map_batch(num_maps, width, height, min_density=0.1, max_density=0.4):
    """
    Generate a batch of random mazes.
    """
    maps = []
    for _ in range(num_maps):
        density = random.uniform(min_density, max_density)
        maps.append(create_maze(width, height, density))
    return maps

if __name__ == "__main__":
    print("Create Maze Test:")
    test_maze = create_maze(5, 5, )
    print(test_maze)
    
    print("Batch Generation Test:")
    batch = generate_map_batch(num_maps=2, width=5, height=5, min_density=0.1, max_density=0.3)
    print("Map 1:\n", batch[0])
    print("Map 2:\n", batch[1])