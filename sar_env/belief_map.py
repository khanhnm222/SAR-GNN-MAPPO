"""Bayesian belief map over victim presence, following the discrete Bayes filter
update rule in Thrun, Burgard & Fox (2005), "Probabilistic Robotics", Ch. 4 —
cited in the thesis proposal (muc 1.5.1, Bang 2).

Update rule when a cell c is scanned without detection:
    P(c) <- P(c) * (1 - p_d) / (1 - P(c) * p_d)
where p_d = 0.9 is the sensor's detection probability.
When a victim is found at c, P(c) is set to 0 (resolved).

With `ScenarioConfig.local_belief` each UAV owns one of these and they are
merged ONE HOP PER STEP along live communication links, so a UAV only learns
what a neighbour saw by being (transitively) in radio range of it. That is what
gives the communication graph a real information role — see `sync`.
"""
from __future__ import annotations

import numpy as np


class BeliefMap:
    def __init__(self, width: int, height: int, p_detect: float = 0.9):
        self.width = width
        self.height = height
        self.p_detect = p_detect
        self.prob = np.zeros((width, height), dtype=np.float32)
        self.visited = np.zeros((width, height), dtype=np.float32)

    def reset(self, prior: np.ndarray | None = None):
        if prior is not None:
            self.prob = prior.astype(np.float32).copy()
        else:
            self.prob[:] = 0.05  # small uniform prior
        self.visited[:] = 0.0

    def copy(self) -> "BeliefMap":
        other = BeliefMap(self.width, self.height, self.p_detect)
        other.prob = self.prob.copy()
        other.visited = self.visited.copy()
        return other

    def scan_no_detection(self, x: int, y: int) -> bool:
        """Apply the Bayes update at (x, y). Returns True if this cell had not
        been visited by THIS belief map before (i.e. it is newly revealed)."""
        was_new = self.visited[x, y] < 0.5
        p = self.prob[x, y]
        denom = 1.0 - p * self.p_detect
        denom = max(denom, 1e-6)
        self.prob[x, y] = p * (1.0 - self.p_detect) / denom
        self.visited[x, y] = 1.0
        return bool(was_new)

    def mark_resolved(self, x: int, y: int):
        self.prob[x, y] = 0.0
        self.visited[x, y] = 1.0

    def sync(self, other: "BeliefMap"):
        """Merge in another UAV's belief map (communication-range synchronization).
        Takes the pointwise minimum probability (more information = lower residual
        uncertainty where either UAV has scanned) and the union of visited cells.
        """
        np.minimum(self.prob, other.prob, out=self.prob)
        np.maximum(self.visited, other.visited, out=self.visited)

    def local_patch(self, x: int, y: int, radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(2r+1, 2r+1) probability and visited patches centred on (x, y).

        Out-of-bounds cells are padded prob=0 / visited=1, i.e. "nothing there
        and nothing left to explore". They used to pad visited=0, which told
        every UAV that the area beyond the map edge was unexplored and pulled
        the swarm into the borders -- the edge-hugging seen in every replay.
        """
        size = 2 * radius + 1
        prob_patch = np.zeros((size, size), dtype=np.float32)
        visited_patch = np.ones((size, size), dtype=np.float32)

        x0, x1 = max(0, x - radius), min(self.width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.height, y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return prob_patch, visited_patch
        px0, py0 = x0 - (x - radius), y0 - (y - radius)
        prob_patch[px0:px0 + (x1 - x0), py0:py0 + (y1 - y0)] = self.prob[x0:x1, y0:y1]
        visited_patch[px0:px0 + (x1 - x0), py0:py0 + (y1 - y0)] = self.visited[x0:x1, y0:y1]
        return prob_patch, visited_patch

    def coarse_patch(self, x: int, y: int, radius: int = 2,
                      block: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Same (2r+1, 2r+1) shape, but each cell averages a `block` x `block`
        square, so the patch spans (2r+1)*block cells instead of (2r+1).

        This exists because the fine patch is provably uninformative. With
        r_sense = 3 the UAV scans a radius-3 disc every step, while
        LOCAL_MAP_RADIUS = 2 means its belief window is a radius-2 square
        strictly INSIDE that disc. Measured over 480 agent-steps: the visited
        channel was all-ones 100% of the time and the probability channel had
        a standard deviation of 0.0003. Fifty of the eighty-six observation
        dimensions were constant, so no policy -- graph or otherwise -- could
        see where unexplored ground lay, and turning could only ever be noise.

        block=5 makes the window span 25x25 cells (radius 12), well outside the
        sensor disc, so the outer ring finally carries a real exploration
        gradient. Padding follows local_patch: prob=0, visited=1 outside.
        """
        if block <= 1:
            return self.local_patch(x, y, radius)
        size = 2 * radius + 1
        span = size * block
        half = span // 2
        x0, y0 = x - half, y - half

        prob = np.zeros((span, span), dtype=np.float32)
        visited = np.ones((span, span), dtype=np.float32)
        sx0, sx1 = max(0, x0), min(self.width, x0 + span)
        sy0, sy1 = max(0, y0), min(self.height, y0 + span)
        if sx0 < sx1 and sy0 < sy1:
            prob[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = self.prob[sx0:sx1, sy0:sy1]
            visited[sx0 - x0:sx1 - x0, sy0 - y0:sy1 - y0] = self.visited[sx0:sx1, sy0:sy1]
        prob = prob.reshape(size, block, size, block).mean(axis=(1, 3))
        visited = visited.reshape(size, block, size, block).mean(axis=(1, 3))
        return prob.astype(np.float32), visited.astype(np.float32)

    def coverage_rate(self) -> float:
        return float(self.visited.mean())
