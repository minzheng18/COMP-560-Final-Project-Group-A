import heapq
import numpy as np
import maze_generator
import torch
from train_heuristic import load_trained_model, heuristic_fn 

def manhattan_distance(a, b):
    """
    Calculate the Manhattan distance between points a and b.
    """
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(prev, current):
    """
    Reconstruct the path from start to goal.
    """
    path = [current]
    while current in prev:
        current = prev[current]
        path.append(current)
    path.reverse()
    return path

def a_star_search(maze, start, goal, heuristic_func):
    """
    A star Search Algorithm.
    """
    height, width = maze.shape
    
    if maze[start[1], start[0]] == 1 or maze[goal[1], goal[0]] == 1:
        return [], 0
    
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    prev = {}
    
    g_score = np.full((height, width), float('inf'))
    g_score[start[1], start[0]] = 0
    
    f_score = np.full((height, width), float('inf'))
    try:
        h_start = heuristic_func(start, goal)
    except TypeError:
        h_start = heuristic_func(start)
    f_score[start[1], start[0]] = h_start

    heapq.heappush(open_set, (f_score[start[1], start[0]], start))

    open_set_hash = {start}
    
    expanded_nodes_count = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current not in open_set_hash:
            continue
            
        open_set_hash.remove(current)
        expanded_nodes_count += 1
        
        if current == goal:
            path = reconstruct_path(prev, current)
            return path, expanded_nodes_count
            
        current_g = g_score[current[1], current[0]]
        
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            nx, ny = neighbor
            
            if not (0 <= nx < width and 0 <= ny < height):
                continue
                
            if maze[ny, nx] == 1:
                continue
                
            tentative_g_score = current_g + 1
            
            if tentative_g_score < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g_score

                try:
                    h = heuristic_func(neighbor, goal)
                except TypeError:
                    h = heuristic_func(neighbor)
                new_f_score = tentative_g_score + h
                f_score[ny, nx] = new_f_score
                
                prev[neighbor] = current
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (new_f_score, neighbor))
                    open_set_hash.add(neighbor)

    return [], expanded_nodes_count

class LearnedHeuristic:
    """
    Enhanced Heuristic for A* using learned AI model.
    """
    def __init__(self, model_path, maze, goal):  
        self.maze = maze
        self.goal = goal
        self.height, self.width = maze.shape
        
        self.model, self.device = load_trained_model(
            model_path, 
            h=self.height, 
            w=self.width
        )
        
        self.obstacle_channel = maze.astype(np.float32)
        self.goal_channel = np.zeros((self.height, self.width), dtype=np.float32)
        self.goal_channel[goal[1], goal[0]] = 1.0
        
        self.agent_channel = np.zeros((self.height, self.width), dtype=np.float32)

    def _state_to_numpy(self, state):
        """
        Convert (x, y) state to (3, H, W) NumPy array
        """
        ax, ay = state
        
        self.agent_channel.fill(0)
        self.agent_channel[ay, ax] = 1.0
        
        state_np = np.stack([self.obstacle_channel, self.agent_channel, self.goal_channel], axis=0)
        
        return state_np

    def __call__(self, state):
        """
        Callable method: return heuristic value for given state.
        """
        if state == self.goal:
            return 0.0
            
        state_np = self._state_to_numpy(state)
        
        h_value = heuristic_fn(self.model, self.device, state_np)
        
        return h_value
    
if __name__ == "__main__":
    # # Basic Test of A* Baseline with 5x5 Maze
    # print("A* Baseline (Manhattan) Test:")
    # maze = maze_generator.create_test_maze()
    # start = (0, 0)
    # goal = (4, 4)
    
    # path_manhattan, expanded_manhattan = a_star_search(maze, start, goal, manhattan_distance)
    
    # print(f"Maze Shape: {maze.shape}")
    # print(f"Path (Manhattan): {len(path_manhattan)-1} steps")
    # print(path_manhattan)
    # print(f"Expanded Nodes (Manhattan): {expanded_manhattan}")

    print("A* Baseline vs. Learned (15x15 Map)")
    
    model_file = "heuristic_cnn.pt"
    maze_15 = maze_generator.create_maze(15, 15, )
    start_15 = (0, 0)
    goal_15 = (14, 14)
    
    path_m_15, expanded_m_15 = a_star_search(maze_15, start_15, goal_15, manhattan_distance)
    while not path_m_15:
        maze_15 = maze_generator.create_maze(15, 15, )
        path_m_15, expanded_m_15 = a_star_search(maze_15, start_15, goal_15, manhattan_distance)
    print(f"\nPath (Manhattan): {len(path_m_15) - 1} steps")
    print(f"Expanded Nodes (Manhattan): {expanded_m_15}")

    learned_h = LearnedHeuristic(model_path=model_file, maze=maze_15, goal=goal_15)
    
    path_l_15, expanded_l_15 = a_star_search(maze_15, start_15, goal_15, learned_h)
    print(f"\nPath (Learned): {len(path_l_15) - 1} steps")
    print(f"Expanded Nodes (Learned): {expanded_l_15}")

    print("\nComparison of Expanded Nodes:")
    print(f"Manhattan: {expanded_m_15} expanded nodes")
    print(f"Learned:   {expanded_l_15} expanded nodes")