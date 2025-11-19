import numpy as np
import matplotlib.pyplot as plt
from astar import run_random_comparison

def generate_comparison_chart(manhattan_nodes, learned_nodes):
    """
    Generate a bar chart comparing the average expanded nodes
    """
    avg_m = np.mean(manhattan_nodes)
    std_m = np.std(manhattan_nodes)
    
    avg_l = np.mean(learned_nodes)
    std_l = np.std(learned_nodes)
    
    labels = ['Manhattan (Baseline)', 'Learned Heuristic (CNN)']
    averages = [avg_m, avg_l]
    std_devs = [std_m, std_l]
    colors = ['#4CAF50', '#2196F3']

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(labels, averages, yerr=std_devs, capsize=7, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('Average Expanded Nodes', fontsize=12)
    ax.set_title('Quantitative Comparison of Heuristic Efficiency (20 Random Maps)', fontsize=14)
    ax.set_ylim(0, (max(averages) + max(std_devs)) * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    def add_labels(bars):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            text_offset = height + std_devs[i] + 2.0 
            ax.text(bar.get_x() + bar.get_width() / 2., text_offset, f'{height:.2f}', ha='center', va='bottom', fontsize=11)
            
    add_labels(bars)

    plt.tight_layout()
    plt.savefig('heuristic_comparison_chart.png')
    plt.close()
    print("Generated 'heuristic_comparison_chart.png' in the current directory.")


if __name__ == "__main__":
    MAP_WIDTH = 15
    MAP_HEIGHT = 15
    NUM_RANDOM_MAPS = 20
    OBSTACLE_DENSITY = 0.3

    MODEL_FILE = "heuristic.pt"

    manhattan_data, learned_data = run_random_comparison(NUM_RANDOM_MAPS, MAP_HEIGHT, MAP_WIDTH, MODEL_FILE, OBSTACLE_DENSITY)

    generate_comparison_chart(manhattan_data, learned_data)