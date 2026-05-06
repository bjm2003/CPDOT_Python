"""Matplotlib plotting and animation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

from .env import CircleObstacle, Map2D, RectangleObstacle


def plot_map(map2d: Map2D, ax=None):
    """Plot map bounds, obstacles, start, and goal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.add_patch(Rectangle((0, 0), map2d.width, map2d.height, fill=False, lw=1.5, color="black"))
    for obs in map2d.obstacles:
        if isinstance(obs, CircleObstacle):
            patch = Circle(obs.center, obs.radius, color="#555555", alpha=0.55)
        elif isinstance(obs, RectangleObstacle):
            patch = Polygon(obs.polygon(), closed=True, color="#555555", alpha=0.55)
        else:
            patch = Polygon(obs.polygon(), closed=True, color="#555555", alpha=0.55)
        ax.add_patch(patch)
    ax.scatter([map2d.start[0]], [map2d.start[1]], c="#15803d", s=55, marker="o", label="start")
    ax.scatter([map2d.goal[0]], [map2d.goal[1]], c="#b91c1c", s=55, marker="*", label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, map2d.width + 0.5)
    ax.set_ylim(-0.5, map2d.height + 0.5)
    ax.grid(True, alpha=0.25)
    return ax


def plot_result(
    map2d: Map2D,
    topo_paths: list[np.ndarray],
    trajectory: np.ndarray,
    output: str | Path,
    *,
    selected_guide: np.ndarray | None = None,
    seed_trajectory: np.ndarray | None = None,
    robot_topo_paths: list[list[np.ndarray]] | None = None,
):
    """Save a static planning result figure."""
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_map(map2d, ax)
    for i, path in enumerate(topo_paths):
        ax.plot(
            path[:, 0],
            path[:, 1],
            "--",
            color="#94a3b8",
            lw=0.9,
            alpha=0.45,
            label="center topo candidates" if i == 0 else None,
        )
    if selected_guide is not None:
        ax.plot(
            selected_guide[:, 0],
            selected_guide[:, 1],
            color="#111827",
            lw=2.0,
            alpha=0.8,
            label="selected center guide",
        )
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    if seed_trajectory is not None:
        for r in range(seed_trajectory.shape[1]):
            ax.plot(
                seed_trajectory[:, r, 0],
                seed_trajectory[:, r, 1],
                color=colors[r % len(colors)],
                lw=1.0,
                ls=":",
                alpha=0.55,
                label="robot coarse seeds" if r == 0 else None,
            )
    if robot_topo_paths is not None:
        for r, paths in enumerate(robot_topo_paths):
            for j, path in enumerate(paths):
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=colors[r % len(colors)],
                    lw=0.85,
                    ls="--",
                    alpha=0.35,
                    label="robot topo candidates" if r == 0 and j == 0 else None,
                )
    for r in range(trajectory.shape[1]):
        ax.plot(
            trajectory[:, r, 0],
            trajectory[:, r, 1],
            color=colors[r % len(colors)],
            lw=2.0,
            label=f"robot {r}",
        )
    for idx in np.linspace(0, len(trajectory) - 1, 6, dtype=int):
        ax.plot(*np.vstack([trajectory[idx], trajectory[idx, 0]]).T, color="#111827", alpha=0.25, lw=1.0)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title("CPDOT Python reproduction")
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def animate_result(map2d: Map2D, trajectory: np.ndarray, output: str | Path | None = None):
    """Create a simple animation of the optimized formation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_map(map2d, ax)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    scatters = [ax.plot([], [], "o", color=colors[i % len(colors)], ms=7)[0] for i in range(trajectory.shape[1])]
    sheet_line, = ax.plot([], [], "-", color="#111827", alpha=0.45)

    def update(frame):
        points = trajectory[frame]
        for i, artist in enumerate(scatters):
            artist.set_data([points[i, 0]], [points[i, 1]])
        closed = np.vstack([points, points[0]])
        sheet_line.set_data(closed[:, 0], closed[:, 1])
        return [*scatters, sheet_line]

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=80, blit=True)
    if output is not None:
        anim.save(output)
        plt.close(fig)
    return anim
