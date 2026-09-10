"""Non-learning heuristic baselines (Bang 5, nhom "Heuristic"):
  - Random Walk: each UAV moves randomly, no coordination.
  - Greedy: each UAV moves towards the nearest highest victim-probability cell
    within its own local observation window (no global information / no
    coordination between UAVs — a fair decentralized heuristic).

Both operate purely on the 116-dim observation vector produced by
`SARSwarmEnv`, exactly like the learned policies, so evaluation code is shared.
"""
from __future__ import annotations

import numpy as np

SELF_DIM = 7
MAP_RADIUS = 2
MAP_SIZE = 2 * MAP_RADIUS + 1
MAP_CHANNELS = 3
MAP_DIM = MAP_SIZE * MAP_SIZE * MAP_CHANNELS

# direction (dx_sign, dy_sign) -> action id, matches sar_env.entities.ACTION_DELTAS
_DIR_TO_ACTION = {
    (0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3,
    (1, 1): 4, (-1, 1): 5, (1, -1): 6, (-1, -1): 7,
    (0, 0): 10,
}


def _decode_local_map(obs: np.ndarray) -> np.ndarray:
    flat = obs[SELF_DIM: SELF_DIM + MAP_DIM]
    return flat.reshape(MAP_SIZE, MAP_SIZE, MAP_CHANNELS)


class RandomWalkPolicy:
    def __init__(self, n_actions: int = 13, seed: int | None = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        # bias slightly towards movement actions (0-10) over fast_forward/return_base
        return int(self.rng.choice(list(range(11)) + [11, 12], p=_rw_probs()))

    def act_batch(self, obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        return {a: self.act(o) for a, o in obs_dict.items()}


def _rw_probs():
    p = np.ones(13)
    p[11] = 0.3  # fast_forward less likely (burns energy fast)
    p[12] = 0.1  # return_base rarely chosen randomly
    return p / p.sum()


class GreedyPolicy:
    """Moves toward the highest victim-probability cell in the local window;
    falls back to exploring the least-visited cell when the window carries no
    useful signal (all probabilities near the uninformative prior)."""

    def __init__(self, prior_threshold: float = 0.06, seed: int | None = None):
        self.prior_threshold = prior_threshold
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        local_map = _decode_local_map(obs)
        occ, prob, visited = local_map[..., 0], local_map[..., 1], local_map[..., 2]

        masked_prob = np.where(occ > 0.5, -1.0, prob)
        best_val = masked_prob.max()

        if best_val > self.prior_threshold:
            # random tie-break among equally-good cells (argmax alone is deterministic
            # and would otherwise bias every UAV towards the same corner)
            noisy = masked_prob + self.rng.uniform(0, 1e-3, size=masked_prob.shape)
            idx = np.unravel_index(np.argmax(noisy), noisy.shape)
        else:
            masked_unvisited = np.where(occ > 0.5, 2.0, visited) + self.rng.uniform(0, 1e-3, size=visited.shape)
            idx = np.unravel_index(np.argmin(masked_unvisited), masked_unvisited.shape)

        di, dj = idx[0] - MAP_RADIUS, idx[1] - MAP_RADIUS
        if di == 0 and dj == 0:
            return 10  # already at the target cell -> hover / scan
        dx, dy = int(np.sign(di)), int(np.sign(dj))
        return _DIR_TO_ACTION.get((dx, dy), 10)

    def act_batch(self, obs_dict: dict[str, np.ndarray]) -> dict[str, int]:
        return {a: self.act(o) for a, o in obs_dict.items()}
 