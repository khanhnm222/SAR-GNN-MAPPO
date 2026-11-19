"""Unit checks of the belief merge of equation (1) (review issue 1, 15/11/2026).

The merge sets every member of a connected component to the cell-wise minimum of
the members' probabilities and the cell-wise maximum of their visited flags. This
file checks on random maps that the result

  * equals that formula,
  * does not depend on the order of the UAVs (commutative and associative),
  * is unchanged by a second merge (idempotent: merging the same evidence twice
    does not count it twice),
  * does not depend on how many paths connect two members (a chain and a clique
    with the same members give the same maps),
  * leaves UAVs in other components untouched.

    python -m comm_robustness.test_merge
"""
from __future__ import annotations

import numpy as np

from sar_env import get_scenario
from sar_env.sar_parallel_env import SARSwarmEnv


def _env_with_maps(positions, rng):
    env = SARSwarmEnv(get_scenario("easy"), seed=0)
    env.reset(seed=0)
    for u, (x, y) in zip(env.uavs, positions):
        u.x, u.y, u.alive = float(x), float(y), True
    for bm in env._beliefs:
        bm.prob[:] = rng.uniform(0.0, 0.05, bm.prob.shape).astype(np.float32)
        bm.visited[:] = (rng.random(bm.visited.shape) < 0.2).astype(np.float32)
    return env


def _snapshot(env):
    return [(b.prob.copy(), b.visited.copy()) for b in env._beliefs]


def main():
    rng = np.random.default_rng(7)
    r = get_scenario("easy").r_comm
    chain = [(5, 5), (5 + 0.9 * r, 5), (5 + 1.8 * r, 5), (45, 45)]   # 0-1-2 linked as a chain, 3 alone
    clique = [(5, 5), (6, 5), (7, 5), (45, 45)]                       # 0-1-2 all linked, 3 alone
    for trial in range(20):
        seed = int(rng.integers(1 << 30))
        env = _env_with_maps(chain, np.random.default_rng(seed))
        before = _snapshot(env)
        env._merge_beliefs_over_comm_graph()
        after = _snapshot(env)
        want_p = np.minimum.reduce([before[i][0] for i in range(3)])
        want_v = np.maximum.reduce([before[i][1] for i in range(3)])
        for i in range(3):
            assert np.array_equal(after[i][0], want_p) and np.array_equal(after[i][1], want_v), "formula"
        assert np.array_equal(after[3][0], before[3][0]) and np.array_equal(after[3][1], before[3][1]), \
            "other component touched"

        env._merge_beliefs_over_comm_graph()
        again = _snapshot(env)
        assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                   for a, b in zip(after, again)), "not idempotent"

        perm = [2, 0, 1, 3]
        env2 = _env_with_maps([chain[k] for k in perm], np.random.default_rng(seed))
        for k, bm in zip(perm, env2._beliefs):
            bm.prob[:], bm.visited[:] = before[k]
        env2._merge_beliefs_over_comm_graph()
        assert np.array_equal(env2._beliefs[0].prob, want_p), "order dependent"

        env3 = _env_with_maps(clique, np.random.default_rng(seed))
        for k, bm in enumerate(env3._beliefs):
            bm.prob[:], bm.visited[:] = before[k]
        env3._merge_beliefs_over_comm_graph()
        assert np.array_equal(env3._beliefs[1].prob, want_p) and \
            np.array_equal(env3._beliefs[1].visited, want_v), "depends on the paths"
    print("merge checks passed (20 random trials)")


if __name__ == "__main__":
    main()
