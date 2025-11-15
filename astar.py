import heapq
import numpy as np
import maze_generator

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
    f_score[start[1], start[0]] = heuristic_func(start, goal)
    
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

                new_f_score = tentative_g_score + heuristic_func(neighbor, goal)
                f_score[ny, nx] = new_f_score
                
                prev[neighbor] = current
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (new_f_score, neighbor))
                    open_set_hash.add(neighbor)

    return [], expanded_nodes_count

if __name__ == "__main__":
    print("A* Baseline (Manhattan) Test:")
    maze = maze_generator.create_test_maze()
    start = (0, 0)
    goal = (4, 4)
    
    path_manhattan, expanded_manhattan = a_star_search(maze, start, goal, manhattan_distance)
    
    print(f"Maze Shape: {maze.shape}")
    print(f"Path (Manhattan): {len(path_manhattan)-1} steps")
    print(path_manhattan)
    print(f"Expanded Nodes (Manhattan): {expanded_manhattan}")