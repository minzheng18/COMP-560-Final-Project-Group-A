#!/usr/bin/env python
"""
Side-by-side visualization of A* using Manhattan vs. learned residual heuristic.

Runs both searches on the same random maze and produces:
- An animated GIF/MP4 showing each expansion step.
- A static PNG snapshot of the final paths.

Usage (dl_normal conda env):
    python visualize_compare.py --width 15 --height 15 --density 0.30 \
        --model heuristic.pt --output compare.gif --seed 0

Flags:
    --width/--height   Grid size.
    --density          Obstacle density in [0,1].
    --model            Trained CNN checkpoint (default: heuristic.pt).
    --tries            Max attempts to sample a solvable maze (default: 30).
    --seed             RNG seed for reproducibility.
    --fps              Frames per second for animation.
    --no-show          Skip interactive display.
    --mp4              Save MP4 instead of GIF (needs ffmpeg installed).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, animation
import maze_generator
from astar import a_star_search, manhattan_distance, LearnedHeuristic


# ------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------
COLORS = {
    "free": "#ffffff",
    "wall": "#1f1f1f",
    "start": "#2ca02c",
    "goal": "#d62728",
    "manhattan": plt.get_cmap("Blues"),
    "learned": plt.get_cmap("Oranges"),
    "path": "#ffd700",
}


def draw_state(ax, maze, exp_seq, path, step, title, algo_cmap, start, goal):
    """Render one subplot for a given algorithm at a specific expansion step."""
    ax.clear()
    h, w = maze.shape
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

    # Base maze (0 free -> white, 1 wall -> black)
    base_cmap = colors.ListedColormap([COLORS["free"], COLORS["wall"]])
    ax.imshow(maze, cmap=base_cmap, vmin=0, vmax=1)

    # Expanded nodes heatmap
    if exp_seq:
        upto = min(step, len(exp_seq) - 1)
        visited = exp_seq[: upto + 1]
        if visited:
            ys, xs = zip(*[(y, x) for x, y in visited])
            grad = np.linspace(0.3, 1.0, len(visited))
            ax.scatter(xs, ys, c=grad, cmap=algo_cmap, marker='s', s=140, edgecolors='none', alpha=0.85)

    # Path (only after goal expanded)
    if path and step >= len(exp_seq) - 1:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=COLORS["path"], linewidth=2.5, alpha=0.95)
        ax.scatter(xs, ys, color=COLORS["path"], s=30, edgecolors='k', linewidths=0.3)

    # Start / Goal markers
    ax.scatter([start[0]], [start[1]], color=COLORS["start"], s=120, edgecolors='k', linewidths=0.5, zorder=5)
    ax.scatter([goal[0]], [goal[1]], color=COLORS["goal"], s=120, edgecolors='k', linewidths=0.5, zorder=5)


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------

def find_solvable_maze(width, height, density, model_path, tries, seed=None):
    rng = np.random.default_rng(seed)
    for attempt in range(1, tries + 1):
        maze = maze_generator.create_maze(width, height, density)
        start = (0, 0)
        goal = (width - 1, height - 1)

        # Manhattan heuristic run first
        path_m, nodes_m, stats_m = a_star_search(maze, start, goal, manhattan_distance, return_stats=True)

        # Learned heuristic
        learned = LearnedHeuristic(model_path=model_path, maze=maze, goal=goal)
        path_l, nodes_l, stats_l = a_star_search(maze, start, goal, learned, return_stats=True)

        if path_m and path_l:
            return {
                "maze": maze,
                "start": start,
                "goal": goal,
                "manhattan": {
                    "path": path_m,
                    "expanded": stats_m["expanded_sequence"],
                    "nodes": nodes_m,
                },
                "learned": {
                    "path": path_l,
                    "expanded": stats_l["expanded_sequence"],
                    "nodes": nodes_l,
                }
            }
    raise RuntimeError(f"Failed to sample a solvable maze after {tries} tries; try lowering density.")


def main():
    parser = argparse.ArgumentParser(description="Visualize Manhattan vs learned heuristic on a random maze")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--density", type=float, default=0.30)
    parser.add_argument("--model", type=str, default="heuristic.pt")
    parser.add_argument("--tries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output", type=str, default="compare.gif")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    parser.add_argument("--mp4", action="store_true", help="Save MP4 instead of GIF (needs ffmpeg)")
    args = parser.parse_args()

    if (args.width, args.height) != (15, 15):
        print("[warn] Model heuristic.pt was trained on 15x15; other sizes may mismatch unless you retrain.")

    np.random.seed(args.seed)

    info = find_solvable_maze(args.width, args.height, args.density, args.model, args.tries, seed=args.seed)
    maze = info["maze"]
    start, goal = info["start"], info["goal"]

    manh = info["manhattan"]
    learn = info["learned"]

    print(f"Maze {args.width}x{args.height}, density={args.density:.2f}")
    print(f"Manhattan: nodes expanded={manh['nodes']}, path length={len(manh['path'])-1}")
    print(f"Learned:   nodes expanded={learn['nodes']}, path length={len(learn['path'])-1}")

    max_steps = max(len(manh['expanded']), len(learn['expanded']))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    def animate(step):
        draw_state(
            axes[0], maze, manh['expanded'], manh['path'], step,
            f"Manhattan (expanded {min(step+1, len(manh['expanded']))}/{len(manh['expanded'])})",
            COLORS["manhattan"], start, goal
        )
        draw_state(
            axes[1], maze, learn['expanded'], learn['path'], step,
            f"Learned (expanded {min(step+1, len(learn['expanded']))}/{len(learn['expanded'])})",
            COLORS["learned"], start, goal
        )
        fig.suptitle("A* Expansion Comparison", fontsize=13)

    anim = animation.FuncAnimation(fig, animate, frames=max_steps, interval=1000/args.fps, repeat=False)

    if args.mp4:
        writer = animation.FFMpegWriter(fps=args.fps)
        anim.save(args.output.replace('.gif', '.mp4'), writer=writer, dpi=200)
    else:
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(args.output, writer=writer, dpi=200)

    # Final static snapshot
    animate(max_steps - 1)
    fig.savefig("compare_static.png", dpi=200)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
