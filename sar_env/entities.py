"""UAV and Victim entities for the SAR simulation environment."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Discrete(13) action space (muc 1.5.4, Bang 3)
ACTIONS = [
    "N", "S", "E", "W", "NE", "NW", "SE", "SW",   # 0-7: horizontal moves
    "ALT_UP", "ALT_DOWN",                          # 8-9: altitude change
    "HOVER",                                        # 10: hold position
    "FAST_FORWARD",                                  # 11: double speed along current heading, 2x energy
    "RETURN_BASE",                                   # 12: head towards base station
]
ACTION_DELTAS = {
    0: (0, 1, 0), 1: (0, -1, 0), 2: (1, 0, 0), 3: (-1, 0, 0),
    4: (1, 1, 0), 5: (-1, 1, 0), 6: (1, -1, 0), 7: (-1, -1, 0),
    8: (0, 0, 1), 9: (0, 0, -1),
    10: (0, 0, 0),
}
N_ACTIONS = len(ACTIONS)
MAX_ALTITUDE = 5


@dataclass
class Victim:
    x: int
    y: int
    priority: int = 1  # 1, 2, 3
    detected: bool = False
    detected_at_step: int | None = None
    detected_by: int | None = None   # uid of the UAV that found this victim


@dataclass
class UAV:
    uid: int
    x: float
    y: float
    z: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    energy: float = 1000.0
    max_energy: float = 1000.0
    alive: bool = True
    base_x: float = 0.0
    base_y: float = 0.0
    heading: tuple[int, int] = field(default_factory=lambda: (0, 1))
    # Diagnostics for the replay view (web/components/ReplayCanvas.tsx): the
    # action actually taken this step and whether it was rejected by the
    # terrain. Without these the replay cannot distinguish "UAV chose to
    # hover" from "UAV tried to move into an obstacle and was blocked", which
    # is exactly the obstacle-avoidance behaviour the thesis needs to show.
    last_action: int = 10
    last_blocked: bool = False

    def energy_ratio(self) -> float:
        return float(np.clip(self.energy / self.max_energy, 0.0, 1.0))

    def apply_action(self, action: int, width: int, height: int, terrain,
                      c0: float = 1.0, c1: float = 2.0) -> float:
        """Move the UAV according to `action`, consume energy, return energy spent.
        Illegal moves (into obstacles / out of bounds) are treated as HOVER.
        """
        if not self.alive:
            return 0.0

        self.last_action = int(action)
        speed_mult = 1.0
        if action == 12:  # RETURN_BASE: move one step towards base
            dx = np.sign(self.base_x - self.x)
            dy = np.sign(self.base_y - self.y)
            dz = 0
        elif action == 11:  # FAST_FORWARD along last heading, double distance & energy
            dx, dy = self.heading
            dz = 0
            speed_mult = 2.0
        elif action in ACTION_DELTAS:
            dx, dy, dz = ACTION_DELTAS[action]
        else:
            dx, dy, dz = 0, 0, 0

        nx = self.x + dx * speed_mult
        ny = self.y + dy * speed_mult
        nz = float(np.clip(self.z + dz, 0, MAX_ALTITUDE))

        blocked = terrain.is_blocked(int(round(nx)), int(round(ny)))
        self.last_blocked = bool(blocked and (dx, dy) != (0, 0))
        if not blocked:
            self.vx, self.vy, self.vz = nx - self.x, ny - self.y, nz - self.z
            self.x, self.y, self.z = float(np.clip(nx, 0, width - 1)), float(np.clip(ny, 0, height - 1)), nz
            if (dx, dy) != (0, 0):
                self.heading = (int(np.sign(dx)), int(np.sign(dy)))
        else:
            self.vx = self.vy = self.vz = 0.0

        speed = float(np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2))
        energy_cost = c0 + c1 * speed
        if action == 11:
            energy_cost *= 2.0  # fast_forward tieu hao gap doi (muc 1.5.4)
        self.energy = max(0.0, self.energy - energy_cost)
        if self.energy <= 0.0:
            self.alive = False
        return energy_cost

    def self_state(self) -> np.ndarray:
        return np.array([
            self.x, self.y, self.z, self.vx, self.vy, self.vz, self.energy_ratio(),
        ], dtype=np.float32)
