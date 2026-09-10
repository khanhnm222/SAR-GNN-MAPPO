"""SARSwarmEnv: a PettingZoo ParallelEnv for cooperative UAV swarm search &
rescue, implementing the specification in muc 1.5.1 of the thesis proposal.

Four design flags on `ScenarioConfig` control behaviour that the 2026-09-03
review found had made the graph architecture structurally unable to beat the
MLP baseline: `local_belief`, `include_neighbor_obs`, `per_agent_reward` and
`normalize_obs`. See `sar_env/scenarios.py` for what each one fixes and
`sar_env.scenarios.LEGACY_OVERRIDES` to reproduce the Giai doan 6/7 runs.
"""
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from .belief_map import BeliefMap
from .entities import MAX_ALTITUDE, UAV, N_ACTIONS, Victim
from .graph_builder import build_graph, k_nearest_neighbors
from .rewards import RewardWeights, compute_agent_rewards, compute_step_reward
from .scenarios import ScenarioConfig, get_scenario, obs_dim_for

K_NEIGHBORS = 5
SELF_STATE_DIM = 7
LOCAL_MAP_RADIUS = 2  # 5x5 patch
LOCAL_MAP_CHANNELS = 3
TASK_STATE_DIM = 4
# Legacy observation length (self 7 + local map 75 + neighbours 30 + task 4).
# The live length depends on `ScenarioConfig.include_neighbor_obs`; use
# `env.obs_dim` or `sar_env.scenarios.obs_dim_for(cfg)` instead of this
# constant in new code.
OBS_DIM = (SELF_STATE_DIM + (2 * LOCAL_MAP_RADIUS + 1) ** 2 * LOCAL_MAP_CHANNELS +
           K_NEIGHBORS * 6 + TASK_STATE_DIM)  # = 116
SAFETY_DIST = 1.0


class SARSwarmEnv(ParallelEnv):
    metadata = {"name": "sar_swarm_v0", "render_modes": ["state"]}

    def __init__(self, scenario: ScenarioConfig, seed: int | None = None,
                 reward_weights: RewardWeights | None = None):
        self.cfg = scenario
        self.possible_agents = [f"uav_{i}" for i in range(scenario.n_uavs)]
        self.agents = list(self.possible_agents)
        self.weights = reward_weights or RewardWeights()
        self._rng = np.random.default_rng(seed)
        self.total_cells = scenario.width * scenario.height
        self.obs_dim = obs_dim_for(scenario)

        self.terrain = None
        self.belief = None            # team union map, used for metrics
        self._beliefs: list[BeliefMap] = []   # per-UAV maps when local_belief
        self._offsets = None          # cached circular sensor footprint
        self._prev_occ = None
        self.uavs: list[UAV] = []
        self.victims: list[Victim] = []
        self.t = 0
        self._n_victims_detected = 0
        self._last_step_log = {}

        from .terrain import Terrain  # local import to avoid cycle in type hints
        self._Terrain = Terrain

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent):
        return Box(low=-1e4, high=1e4, shape=(self.obs_dim,), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(N_ACTIONS)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.agents = list(self.possible_agents)
        self.terrain = self._Terrain(cfg.width, cfg.height, cfg.terrain_mode, self._rng)
        self.belief = BeliefMap(cfg.width, cfg.height)
        self.belief.reset()

        self.victims = self._spawn_victims()
        self.uavs = self._spawn_uavs()
        # Per-UAV belief maps. Each starts from the same uninformative prior and
        # then diverges: a UAV only learns what a neighbour scanned by being in
        # radio range of it (see _merge_beliefs_over_comm_graph).
        self._beliefs = ([BeliefMap(cfg.width, cfg.height) for _ in self.uavs]
                         if cfg.local_belief else [])
        for bm in self._beliefs:
            bm.reset()
        self.t = 0
        self._n_victims_detected = 0
        self._prev_occ = None

        observations = {a: self._build_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _spawn_victims(self) -> list[Victim]:
        cfg = self.cfg
        n = int(self._rng.integers(cfg.n_victims_range[0], cfg.n_victims_range[1] + 1))
        victims = []
        free = lambda x, y: not self.terrain.is_blocked(x, y)

        def rand_free_cell():
            for _ in range(100):
                x, y = int(self._rng.integers(0, cfg.width)), int(self._rng.integers(0, cfg.height))
                if free(x, y):
                    return x, y
            return int(cfg.width // 2), int(cfg.height // 2)

        if cfg.victim_mode == "uniform":
            for _ in range(n):
                x, y = rand_free_cell()
                victims.append(Victim(x, y, priority=self._sample_priority()))
        elif cfg.victim_mode == "cluster":
            n_clusters = int(self._rng.integers(2, 4))
            centers = [rand_free_cell() for _ in range(n_clusters)]
            for i in range(n):
                cx, cy = centers[i % n_clusters]
                x = int(np.clip(cx + self._rng.normal(0, 4), 0, cfg.width - 1))
                y = int(np.clip(cy + self._rng.normal(0, 4), 0, cfg.height - 1))
                if self.terrain.is_blocked(x, y):
                    x, y = rand_free_cell()
                victims.append(Victim(x, y, priority=self._sample_priority()))
        elif cfg.victim_mode == "corridor":
            # a corridor = a straight band across the map (road / riverbank)
            horizontal = bool(self._rng.integers(0, 2))
            if horizontal:
                y0 = int(self._rng.integers(cfg.height // 4, 3 * cfg.height // 4))
                for _ in range(n):
                    x = int(self._rng.integers(0, cfg.width))
                    y = int(np.clip(y0 + self._rng.integers(-3, 4), 0, cfg.height - 1))
                    if self.terrain.is_blocked(x, y):
                        x, y = rand_free_cell()
                    victims.append(Victim(x, y, priority=self._sample_priority()))
            else:
                x0 = int(self._rng.integers(cfg.width // 4, 3 * cfg.width // 4))
                for _ in range(n):
                    y = int(self._rng.integers(0, cfg.height))
                    x = int(np.clip(x0 + self._rng.integers(-3, 4), 0, cfg.width - 1))
                    if self.terrain.is_blocked(x, y):
                        x, y = rand_free_cell()
                    victims.append(Victim(x, y, priority=self._sample_priority()))
        else:
            raise ValueError(f"Unknown victim_mode {cfg.victim_mode}")
        return victims

    def _sample_priority(self) -> int:
        if not self.cfg.priority_enabled:
            return 1
        return int(self._rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))

    def _spawn_uavs(self) -> list[UAV]:
        cfg = self.cfg
        base_x, base_y = cfg.width / 2.0, cfg.height / 2.0
        uavs = []
        # Grid layout centered on base, spacing >= SAFETY_DIST with margin, plus
        # small per-UAV jitter for episode-to-episode variety. Independent
        # uniform(-2,2) jitter for every UAV around the SAME center used to
        # cause guaranteed spawn-time collisions at N>=8 (verified empirically:
        # Medium/Hard started 100% of episodes with multiple colliding pairs,
        # 4.4/18.5 on average) — the collision penalty from that then dominated
        # the reward signal for undertrained policies, matching the same class
        # of bug found and fixed in the tim-kiem-bay-dan reference project.
        spacing = max(1.5, SAFETY_DIST * 2.0)
        per_row = 5
        n = cfg.n_uavs
        grid_w = (min(n, per_row) - 1) * spacing
        grid_h = ((n - 1) // per_row) * spacing
        for i in range(n):
            gx = (i % per_row) * spacing - grid_w / 2.0
            gy = (i // per_row) * spacing - grid_h / 2.0
            jitter_x = self._rng.uniform(-0.3, 0.3)
            jitter_y = self._rng.uniform(-0.3, 0.3)
            x = float(np.clip(base_x + gx + jitter_x, 0, cfg.width - 1))
            y = float(np.clip(base_y + gy + jitter_y, 0, cfg.height - 1))
            uavs.append(UAV(uid=i, x=x, y=y, z=1.0, energy=1000.0, max_energy=1000.0,
                             base_x=base_x, base_y=base_y))
        return uavs

    # ------------------------------------------------------------------- step
    def step(self, actions: dict[str, int]):
        cfg = self.cfg
        n = len(self.uavs)
        per_agent_energy = [0.0] * n
        for i, agent in enumerate(self.agents):
            u = self.uavs[i]
            if not u.alive:
                continue
            a = actions.get(agent, 10)  # default HOVER if missing
            per_agent_energy[i] = u.apply_action(int(a), cfg.width, cfg.height, self.terrain)

        collisions, per_agent_collision = self._count_collisions()
        (per_agent_new_cells, per_agent_victim_priority,
         new_cells, victim_priority_sum, newly_detected) = self._sense_and_update_belief()
        self._merge_beliefs_over_comm_graph()
        self.terrain.step()
        self.t += 1

        if cfg.per_agent_reward:
            agent_rewards, comp = compute_agent_rewards(
                per_agent_new_cells=per_agent_new_cells, total_cells=self.total_cells,
                per_agent_victim_priority=per_agent_victim_priority,
                per_agent_collision=per_agent_collision,
                per_agent_energy=per_agent_energy,
                weights=self.weights, team_share=cfg.team_reward_share,
            )
            reward = float(np.mean(agent_rewards)) if agent_rewards else 0.0
        else:
            reward, comp = compute_step_reward(
                n_new_cells=new_cells, total_cells=self.total_cells,
                victims_found_priority_sum=victim_priority_sum,
                n_collisions=collisions, energy_spent=sum(per_agent_energy),
                weights=self.weights,
            )
            agent_rewards = [reward] * n

        all_detected = self._n_victims_detected >= len(self.victims)
        all_dead = all(not u.alive for u in self.uavs)
        truncated = self.t >= cfg.max_steps
        terminated = all_detected or all_dead

        coverage = self.team_coverage_rate()
        observations = {a: self._build_obs(i) for i, a in enumerate(self.agents)}
        rewards = {a: float(agent_rewards[i]) for i, a in enumerate(self.agents)}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        infos = {a: {
            "reward_components": comp.__dict__,
            "collisions": collisions,
            "newly_detected": newly_detected,
            "coverage_rate": coverage,
            "victim_detection_rate": self._n_victims_detected / max(len(self.victims), 1),
        } for a in self.agents}

        self._last_step_log = {
            "t": self.t, "reward": reward, "collisions": collisions,
            "coverage_rate": coverage,
            "victims_detected": self._n_victims_detected, "n_victims": len(self.victims),
        }

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _count_collisions(self) -> tuple[int, list[bool]]:
        """Returns (number of violating PAIRS, per-UAV violated flag).

        The per-UAV flag is what `compute_agent_rewards` charges: a UAV is only
        penalised for a collision it is actually part of, instead of the whole
        swarm sharing one team-wide collision flag.
        """
        flags = [False] * len(self.uavs)
        alive = [(i, u) for i, u in enumerate(self.uavs) if u.alive]
        n = 0
        for a in range(len(alive)):
            ia, ua = alive[a]
            for b in range(a + 1, len(alive)):
                ib, ub = alive[b]
                if abs(ua.z - ub.z) < 0.5 and np.hypot(ua.x - ub.x, ua.y - ub.y) < SAFETY_DIST:
                    n += 1
                    flags[ia] = True
                    flags[ib] = True
        return n, flags

    def _sense_and_update_belief(self):
        """Sense with every alive UAV, updating that UAV's belief map.

        Returns per-agent and team aggregates. `per_agent_new_cells[i]` counts
        cells that were new to UAV i's OWN belief, which is what makes sharing
        a map over the comm graph worth something: a UAV that flies where a
        radio neighbour has already been earns nothing for it.
        """
        cfg = self.cfg
        n = len(self.uavs)
        per_agent_new_cells = [0] * n
        per_agent_victim_priority = [0.0] * n
        newly_detected = []
        detected_this_step = set()
        offsets = self._sense_offsets()

        for i, u in enumerate(self.uavs):
            if not u.alive:
                continue
            bm = self._belief_of(i)
            ux, uy = int(round(u.x)), int(round(u.y))
            for dx, dy in offsets:
                cx, cy = ux + dx, uy + dy
                if not (0 <= cx < cfg.width and 0 <= cy < cfg.height):
                    continue
                if bm.scan_no_detection(cx, cy):
                    per_agent_new_cells[i] += 1
                if bm is not self.belief:
                    self.belief.scan_no_detection(cx, cy)  # team union map, metrics only

            for v in self.victims:
                if v.detected or id(v) in detected_this_step:
                    continue
                if np.hypot(v.x - u.x, v.y - u.y) <= cfg.r_sense:
                    v.detected = True
                    v.detected_at_step = self.t
                    v.detected_by = u.uid
                    detected_this_step.add(id(v))
                    bm.mark_resolved(v.x, v.y)
                    self.belief.mark_resolved(v.x, v.y)
                    per_agent_victim_priority[i] += v.priority
                    newly_detected.append({"x": v.x, "y": v.y, "priority": v.priority,
                                            "detected_by": u.uid})

        self._n_victims_detected += len(newly_detected)
        return (per_agent_new_cells, per_agent_victim_priority,
                sum(per_agent_new_cells), sum(per_agent_victim_priority), newly_detected)

    def _sense_offsets(self):
        if self._offsets is None:
            r = int(round(self.cfg.r_sense))
            self._offsets = [(dx, dy)
                             for dx in range(-r, r + 1)
                             for dy in range(-r, r + 1)
                             if dx * dx + dy * dy <= r * r]
        return self._offsets

    def _belief_of(self, idx: int):
        return self._beliefs[idx] if self.cfg.local_belief else self.belief

    def _merge_beliefs_over_comm_graph(self):
        """Share belief maps across each connected component of the r_comm graph.

        This is the only channel through which a UAV learns what another UAV has
        seen. With `local_belief=False` (legacy) every UAV read the same global
        map, so communication was decorative and the graph carried no
        information the policy did not already have -- the structural reason
        swapping an MLP encoder for a GNN could not help.

        Merging per connected component (rather than pairwise) models radio
        relaying, which is fast relative to flight, and costs one array pass per
        UAV instead of one per edge plus a full copy per UAV -- measured at 2.6x
        faster than the pairwise version on the Medium scenario.
        """
        if not self.cfg.local_belief:
            return
        alive = [i for i, u in enumerate(self.uavs) if u.alive]
        if len(alive) < 2:
            return
        r2 = self.cfg.r_comm ** 2

        parent = {i: i for i in alive}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for pos, i in enumerate(alive):
            ui = self.uavs[i]
            for j in alive[pos + 1:]:
                uj = self.uavs[j]
                if (ui.x - uj.x) ** 2 + (ui.y - uj.y) ** 2 <= r2:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        components: dict[int, list[int]] = {}
        for i in alive:
            components.setdefault(find(i), []).append(i)

        for members in components.values():
            if len(members) < 2:
                continue
            maps = [self._beliefs[i] for i in members]
            prob = maps[0].prob.copy()
            visited = maps[0].visited.copy()
            for bm in maps[1:]:
                np.minimum(prob, bm.prob, out=prob)
                np.maximum(visited, bm.visited, out=visited)
            for bm in maps:
                bm.prob[:] = prob
                bm.visited[:] = visited

    def team_coverage_rate(self) -> float:
        """Union of every UAV's scanned cells -- the Coverage Rate of muc 1.5.5.
        Unaffected by `local_belief`, so the metric stays comparable across runs.
        """
        return self.belief.coverage_rate()

    # -------------------------------------------------------------- observations
    def _build_obs(self, idx: int) -> np.ndarray:
        cfg = self.cfg
        u = self.uavs[idx]
        norm = cfg.normalize_obs

        if norm:
            # Every feature into roughly [-1, 1]. Raw grid coordinates (up to
            # 199 on Hard) previously sat in the same vector as probabilities
            # in [0, 1]; the first Linear layer was dominated by absolute
            # position and the informative channels were effectively noise.
            self_state = np.array([
                2.0 * u.x / max(cfg.width - 1, 1) - 1.0,
                2.0 * u.y / max(cfg.height - 1, 1) - 1.0,
                u.z / MAX_ALTITUDE,
                np.clip(u.vx / 2.0, -1, 1), np.clip(u.vy / 2.0, -1, 1), np.clip(u.vz, -1, 1),
                u.energy_ratio(),
            ], dtype=np.float32)
        else:
            self_state = u.self_state()

        ux, uy = int(round(u.x)), int(round(u.y))
        occ = self.terrain.local_patch(ux, uy, LOCAL_MAP_RADIUS)
        prob_patch, visited_patch = self._belief_of(idx).coarse_patch(
            ux, uy, LOCAL_MAP_RADIUS, cfg.belief_patch_block)
        local_map = np.stack([occ, prob_patch, visited_patch], axis=-1).astype(np.float32).flatten()

        parts = [self_state, local_map]

        if cfg.include_neighbor_obs:
            neighbors = k_nearest_neighbors(self.uavs, idx, K_NEIGHBORS, cfg.r_comm)
            neigh_feat = np.zeros((K_NEIGHBORS, 6), dtype=np.float32)
            for k, (_uid, feat) in enumerate(neighbors):
                neigh_feat[k] = feat
            if norm:
                scale = np.array([cfg.r_comm, cfg.r_comm, MAX_ALTITUDE, 2.0, 2.0, cfg.r_comm],
                                  dtype=np.float32)
                neigh_feat = np.clip(neigh_feat / scale, -1.0, 1.0)
            parts.append(neigh_feat.flatten())

        task = [
            self._n_victims_detected / max(len(self.victims), 1),
            self.t / max(cfg.max_steps, 1),
            self._belief_of(idx).coverage_rate(),
            u.energy_ratio(),
        ]
        if cfg.include_heading_obs:
            # appended at the END so the local_map offset that
            # algorithms/heuristics.py slices by a hard-coded index stays at 7
            task += [float(u.heading[0]), float(u.heading[1])]
        parts.append(np.array(task, dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.obs_dim, f"obs dim mismatch: {obs.shape[0]} != {self.obs_dim}"
        return obs

    # ------------------------------------------------------------------- graph
    def build_communication_graph(self):
        return build_graph(self.uavs, self.cfg.r_comm,
                            normalize=self.cfg.normalize_edge_attr)

    # ----------------------------------------------------------------- replay
    def get_render_state(self) -> dict:
        """Full state snapshot used for JSON logging / web replay.

        Beyond positions this exports the three behaviours the thesis needs to
        *show*, not just tabulate:
          - communication: `comm_edges`, the live r_comm graph with per-link
            signal strength — the same edges/edge_attr the DGAT attends over,
            so the replay draws the actual message-passing topology;
          - obstacle avoidance: per-UAV `action` and `blocked`, so a UAV that
            was refused a move into terrain is visually distinguishable from
            one that chose to hover;
          - victim search: `detected_at_step` per victim plus `detected_by`,
            so the replay can flash the moment and the agent of each find.
        `terrain_new` carries cells that became blocked since the previous
        step (flood growth in the Hard scenario), which a single static
        occupancy snapshot at t=0 could not represent.
        """
        graph = self.build_communication_graph()
        alive_idx = graph["alive_idx"]
        ei, ea = graph["edge_index"], graph["edge_attr"]
        comm_edges = []
        if ei.numel() > 0:
            src = ei[0].tolist()
            dst = ei[1].tolist()
            sig = ea[:, -1].tolist()
            for a, b, q in zip(src, dst, sig):
                if a < b:  # graph is symmetric; emit each link once
                    comm_edges.append([self.uavs[alive_idx[a]].uid,
                                       self.uavs[alive_idx[b]].uid,
                                       round(float(q), 3)])

        return {
            "t": self.t,
            "uavs": [{"uid": u.uid, "x": u.x, "y": u.y, "z": u.z, "alive": u.alive,
                      "energy_ratio": round(u.energy_ratio(), 4),
                      "action": u.last_action, "blocked": u.last_blocked} for u in self.uavs],
            "victims": [{"x": v.x, "y": v.y, "priority": v.priority, "detected": v.detected,
                         "detected_at_step": v.detected_at_step,
                         "detected_by": getattr(v, "detected_by", None)}
                        for v in self.victims],
            "comm_edges": comm_edges,
            "terrain_new": self._terrain_new_cells(),
            "coverage_rate": self.belief.coverage_rate(),
            "victims_detected": self._n_victims_detected,
            "n_victims": len(self.victims),
        }

    def _terrain_new_cells(self) -> list[list[int]]:
        """Cells that became blocked since the last call (dynamic/flood terrain)."""
        occ = self.terrain.occupancy > 0.5
        prev = getattr(self, "_prev_occ", None)
        if prev is None:
            self._prev_occ = occ.copy()
            return []
        new = occ & ~prev
        self._prev_occ = occ.copy()
        return np.argwhere(new).tolist()

    def close(self):
        pass


def make_env(scenario_name: str, seed: int | None = None, **overrides) -> SARSwarmEnv:
    scenario = get_scenario(scenario_name, **overrides)
    return SARSwarmEnv(scenario, seed=seed)
