"""Topological PRM inspired by ``formation_planner/topo_prm.cpp``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .env import Map2D
from .geometry import as_point, resample_polyline


@dataclass(eq=False)
class GraphNode:
    """Guard or connector node in the sparse topological roadmap."""

    pos: np.ndarray
    kind: str
    node_id: int
    neighbors: list["GraphNode"] = field(default_factory=list)

    def connect(self, other: "GraphNode") -> None:
        if all(n.node_id != other.node_id for n in self.neighbors):
            self.neighbors.append(other)
        if all(n.node_id != self.node_id for n in other.neighbors):
            other.neighbors.append(self)


class TopologyPRM:
    """Sparse visibility PRM that keeps topologically distinct paths.

    The structure follows the C++ implementation: samples that see no guard
    become guards; samples that see exactly two guards become connectors if the
    connection is topologically new; DFS then enumerates raw paths, shortcuts
    them, removes equivalent homotopy classes, and keeps short representatives.
    """

    def __init__(
        self,
        map2d: Map2D,
        *,
        max_samples: int = 3000,
        sample_inflate: tuple[float, float] = (25.0, 3.0),
        clearance: float = 0.15,
        resolution: float = 0.35,
        max_raw_paths: int = 40,
        reserve_num: int = 8,
        ratio_to_short: float = 5.5,
        seed: int = 7,
    ):
        self.map = map2d
        self.max_samples = max_samples
        self.sample_inflate = np.asarray(sample_inflate, dtype=float)
        self.clearance = clearance
        self.resolution = resolution
        self.max_raw_paths = max_raw_paths
        self.reserve_num = reserve_num
        self.ratio_to_short = ratio_to_short
        self.rng = np.random.default_rng(seed)
        self.graph: list[GraphNode] = []
        self.raw_paths: list[np.ndarray] = []
        self.short_paths: list[np.ndarray] = []

    def find_topo_paths(
        self, start: Iterable[float], goal: Iterable[float], rectangle_ratio: float = 1.0
    ) -> list[np.ndarray]:
        """Build the roadmap and return selected topologically distinct paths."""
        self.create_graph(start, goal, rectangle_ratio)
        self.raw_paths = self.search_paths()
        self.short_paths = [self.shortcut_path(path, iterations=2) for path in self.raw_paths]
        pruned = self.prune_equivalent(self.short_paths)
        return self.select_short_paths(pruned)

    def create_graph(
        self, start: Iterable[float], goal: Iterable[float], rectangle_ratio: float = 1.0
    ) -> list[GraphNode]:
        """Create a sparse guard/connector roadmap."""
        start = as_point(start)
        goal = as_point(goal)
        self.graph = [GraphNode(start, "guard", 0), GraphNode(goal, "guard", 1)]

        center = 0.5 * (start + goal)
        direction = goal - center
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        rotation = np.column_stack([direction, normal])
        radii = np.array([self.sample_inflate[0], self.sample_inflate[1] * rectangle_ratio])

        next_id = 2
        for _ in range(self.max_samples):
            local = self.rng.uniform(-1.0, 1.0, size=2) * radii
            point = rotation @ local + center
            if self.map.is_collision(point, self.clearance):
                continue

            guards = self.find_visible_guards(point)
            if len(guards) == 0:
                self.graph.append(GraphNode(point, "guard", next_id))
                next_id += 1
            elif len(guards) == 2 and self.need_connection(guards[0], guards[1], point):
                connector = GraphNode(point, "connector", next_id)
                next_id += 1
                self.graph.append(connector)
                guards[0].connect(connector)
                guards[1].connect(connector)

        self.prune_graph()
        return self.graph

    def find_visible_guards(self, point: np.ndarray) -> list[GraphNode]:
        """Return up to three visible guards from a point."""
        visible: list[GraphNode] = []
        for node in self.graph:
            if node.kind != "guard":
                continue
            if self.line_visible(point, node.pos):
                visible.append(node)
                if len(visible) > 2:
                    break
        return visible

    def need_connection(self, g1: GraphNode, g2: GraphNode, point: np.ndarray) -> bool:
        """Check whether a new connector between two guards is non-redundant."""
        candidate = np.array([g1.pos, point, g2.pos])
        for n1 in g1.neighbors:
            for n2 in g2.neighbors:
                if n1.node_id == n2.node_id:
                    existing = np.array([g1.pos, n1.pos, g2.pos])
                    if self.same_topo_path(candidate, existing):
                        if self.path_length(candidate) < self.path_length(existing):
                            n1.pos = point
                        return False
        return True

    def line_visible(self, p1: Iterable[float], p2: Iterable[float]) -> bool:
        """Visibility equals segment collision freedom."""
        return self.map.segment_is_collision_free(p1, p2, self.clearance)

    def prune_graph(self) -> None:
        """Remove non-terminal nodes with degree 0 or 1 until stable."""
        changed = True
        while changed and len(self.graph) > 2:
            changed = False
            for node in list(self.graph):
                if node.node_id <= 1:
                    continue
                if len(node.neighbors) <= 1:
                    for other in self.graph:
                        other.neighbors = [n for n in other.neighbors if n.node_id != node.node_id]
                    self.graph = [n for n in self.graph if n.node_id != node.node_id]
                    changed = True
                    break

    def search_paths(self) -> list[np.ndarray]:
        """Enumerate raw start-to-goal paths with DFS."""
        if not self.graph:
            return []
        raw: list[np.ndarray] = []

        def dfs(path: list[GraphNode]) -> None:
            if len(raw) >= self.max_raw_paths:
                return
            current = path[-1]
            for nb in current.neighbors:
                if nb.node_id == 1:
                    raw.append(np.array([n.pos for n in path + [nb]], dtype=float))
                    return
            seen = {n.node_id for n in path}
            for nb in current.neighbors:
                if nb.node_id in seen or nb.node_id == 1:
                    continue
                dfs(path + [nb])
                if len(raw) >= self.max_raw_paths:
                    return

        dfs([self.graph[0]])
        return sorted(raw, key=lambda p: (len(p), self.path_length(p)))[: self.max_raw_paths]

    def shortcut_path(self, path: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Greedy visibility shortcutting for a path."""
        short = np.asarray(path, dtype=float)
        for _ in range(iterations):
            dense = self.discretize_path(short)
            if len(dense) <= 2:
                return dense
            out = [dense[0]]
            i = 0
            while i < len(dense) - 1:
                j = len(dense) - 1
                while j > i + 1 and not self.line_visible(dense[i], dense[j]):
                    j -= 1
                out.append(dense[j])
                i = j
            new_short = np.asarray(out)
            if self.path_length(new_short) <= self.path_length(short) + 1e-6:
                short = new_short
        return short

    def prune_equivalent(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Remove paths in the same topological class."""
        kept: list[np.ndarray] = []
        for path in sorted(paths, key=self.path_length):
            if not any(self.same_topo_path(path, other) for other in kept):
                kept.append(path)
        return kept

    def select_short_paths(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        """Keep short representatives after topological pruning."""
        if not paths:
            return []
        ordered = sorted(paths, key=self.path_length)
        best = self.path_length(ordered[0])
        selected = [
            p
            for p in ordered[: self.reserve_num]
            if self.path_length(p) / max(best, 1e-9) <= self.ratio_to_short
        ]
        return self.prune_equivalent([self.shortcut_path(p, iterations=4) for p in selected])

    def same_topo_path(self, path1: np.ndarray, path2: np.ndarray) -> bool:
        """Two paths are equivalent if corresponding samples see each other."""
        max_len = max(self.path_length(path1), self.path_length(path2))
        count = max(2, int(np.ceil(max_len / self.resolution)))
        pts1 = resample_polyline(path1, count)
        pts2 = resample_polyline(path2, count)
        return all(self.line_visible(a, b) for a, b in zip(pts1, pts2))

    def discretize_path(self, path: np.ndarray) -> np.ndarray:
        count = max(2, int(np.ceil(self.path_length(path) / self.resolution)) + 1)
        return resample_polyline(path, count)

    @staticmethod
    def path_length(path: np.ndarray) -> float:
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
