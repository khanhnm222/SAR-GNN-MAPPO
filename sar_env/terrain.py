"""Terrain generation for the SAR simulation environment.

Three modes as specified in the thesis proposal (Bang 2 / muc 1.5.1):
  - flat:      no obstacles.
  - obstacle:  static obstacles covering 5-15% of cells (buildings/trees/hills).
  - dynamic:   obstacles spread over time (flood simulation).
"""
from __future__ import annotations

import numpy as np


class Terrain:
    def __init__(self, width: int, height: int, mode: str, rng: np.random.Generator,
                 obstacle_ratio_range=(0.05, 0.15), flood_growth_rate: float = 0.01):
        self.width = width
        self.height = height
        self.mode = mode
        self.rng = rng
        self.obstacle_ratio_range = obstacle_ratio_range
        self.flood_growth_rate = flood_growth_rate
        self.occupancy = np.zeros((width, height), dtype=np.float32)  # 1 = blocked
        self._flood_frontier: list[tuple[int, int]] = []
        self.reset()

    def reset(self):
        self.occupancy[:] = 0.0
        if self.mode == "flat":
            return
        ratio = self.rng.uniform(*self.obstacle_ratio_range)
        n_cells = self.width * self.height
        n_obstacles = int(ratio * n_cells)
        if self.mode == "obstacle":
            self._place_clustered_obstacles(n_obstacles)
        elif self.mode == "dynamic":
            # start with a small seed region (e.g. a riverbank) that grows over time
            n_seed = max(1, n_obstacles // 6)
            seeds = self._place_clustered_obstacles(n_seed, return_cells=True)
            self._flood_frontier = list(seeds)
        else:
            raise ValueError(f"Unknown terrain mode: {self.mode}")

    def _place_clustered_obstacles(self, n_obstacles: int, return_cells: bool = False):
        placed = []
        n_clusters = max(1, n_obstacles // 20)
        remaining = n_obstacles
        for _ in range(n_clusters):
            cx = self.rng.integers(0, self.width)
            cy = self.rng.integers(0, self.height)
            cluster_size = min(remaining, self.rng.integers(5, 25))
            for _ in range(cluster_size):
                dx = int(self.rng.normal(0, 3))
                dy = int(self.rng.normal(0, 3))
                x, y = np.clip(cx + dx, 0, self.width - 1), np.clip(cy + dy, 0, self.height - 1)
                if self.occupancy[x, y] == 0:
                    self.occupancy[x, y] = 1.0
                    placed.append((int(x), int(y)))
            remaining -= cluster_size
            if remaining <= 0:
                break
        return placed if return_cells else None

    def step(self):
        """Advance dynamic terrain (flood growth). No-op for flat/obstacle."""
        if self.mode != "dynamic" or not self._flood_frontier:
            return
        n_new = max(1, int(self.flood_growth_rate * self.width * self.height))
        new_frontier = []
        for _ in range(n_new):
            if not self._flood_frontier:
                break
            idx = self.rng.integers(0, len(self._flood_frontier))
            x, y = self._flood_frontier[idx]
            dx, dy = self.rng.integers(-1, 2), self.rng.integers(-1, 2)
            nx, ny = int(np.clip(x + dx, 0, self.width - 1)), int(np.clip(y + dy, 0, self.height - 1))
            if self.occupancy[nx, ny] == 0:
                self.occupancy[nx, ny] = 1.0
                new_frontier.append((nx, ny))
        self._flood_frontier.extend(new_frontier)

    def is_blocked(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return bool(self.occupancy[x, y] > 0.5)

    def local_patch(self, x: int, y: int, radius: int = 2) -> np.ndarray:
        """Return a (2r+1, 2r+1) occupancy patch centered on (x, y), padded with 1 (blocked) outside bounds."""
        size = 2 * radius + 1
        patch = np.ones((size, size), dtype=np.float32)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                xi, yj = x + i, y + j
                if 0 <= xi < self.width and 0 <= yj < self.height:
                    patch[i + radius, j + radius] = self.occupancy[xi, yj]
        return patch
