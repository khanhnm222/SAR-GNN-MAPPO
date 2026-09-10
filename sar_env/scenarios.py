"""Scenario configurations: Easy / Medium / Hard (muc 1.4.3 of the thesis proposal)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioConfig:
    name: str
    n_uavs: int
    width: int
    height: int
    n_victims_range: tuple[int, int]
    victim_mode: str          # "uniform" | "cluster" | "corridor"
    terrain_mode: str         # "flat" | "obstacle" | "dynamic"
    r_comm: float = 15.0
    r_sense: float = 3.0
    max_steps: int = 500
    priority_enabled: bool = False
    energy_full_model: bool = True

    # ---------------------------------------------------------------- v2 flags
    # Four design flags added 2026-09-03 after the review of Giai doan 1-7.
    # All four default to the *corrected* behaviour; set them to the legacy
    # value in `get_scenario(...)` to reproduce the Giai doan 6/7 numbers
    # (results_stage7_argmax_eval/).

    normalize_obs: bool = True
    """Scale every observation feature into roughly [-1, 1].

    Legacy (False) fed raw grid coordinates — x, y up to 199 on Hard — into the
    same vector as belief probabilities in [0, 1] and an energy ratio in
    [0, 1]. The first Linear layer is then dominated by absolute position and
    the informative channels are effectively noise. Standard practice in every
    MAPPO reference implementation; costs nothing and does not change obs_dim.
    """

    include_neighbor_obs: bool = False
    """Whether the K=5 nearest-neighbour block (30 dims) stays in the local
    observation vector.

    Legacy (True) put each UAV's 5 nearest neighbours' relative position,
    velocity and distance directly into every agent's observation. That means
    MAPPO-MLP -- the "no graph structure at all" baseline of Bang 1/Bang 5 --
    already received complete 1-hop neighbour state, so the GNN's message
    passing was largely redundant and RQ1 ("does graph structure help?") could
    not be answered by the experiment. With False, neighbour information
    reaches a policy ONLY through the communication graph, which is the
    comparison the thesis actually claims to make. obs_dim: 116 -> 86.
    """

    local_belief: bool = True
    """Whether each UAV keeps its own belief map, merged only along live
    communication links.

    Legacy (False) kept ONE global BeliefMap shared by every UAV regardless of
    distance, and `BeliefMap.sync` was dead code. Communication therefore
    carried no information a UAV did not already have, which removes the entire
    functional motivation for the graph. With True, a UAV's belief spreads one
    hop per step, so multi-hop message passing has something real to exploit
    and `coverage_rate` (scored on the union map) rewards the swarm for
    actually sharing what it has seen.
    """

    per_agent_reward: bool = True
    """Whether reward is decomposed per UAV instead of one global scalar.

    Legacy (False) gave every UAV the identical team reward at every step, and
    algorithms/mappo.py broadcast one scalar advantage to all N agents. With no
    per-agent signal there is nothing for a policy to differentiate on, so
    division of labour -- the behaviour the graph is supposed to enable --
    could not be learned at any budget. With True each UAV is credited with the
    cells it personally revealed, the victims it personally found, its own
    collisions and its own energy, plus a small shared team term.
    """
    include_heading_obs: bool = False
    """Expose the UAV's own heading (2 dims) in its observation. obs_dim 86 -> 88.

    Motivation: action 11 (FAST_FORWARD) advances along the stored heading and
    cannot change it, and the heading was not observable, so a policy had no
    state on which to decide when to turn.

    TESTED, AND IT DOES NOT FIX THE PROBLEM -- hence the False default.
    Training MAPPO-MLP on Easy for 150k steps with the flag on gives
    VDR 0.777 (vs 0.764 without, i.e. within seed noise), but an argmax rollout
    of the resulting policy still picks action 11 on 800/800 agent-steps and
    still ends up frozen against the map edge (coverage 0.121, 0/5 victims).

    FAST_FORWARD is simply the highest-value action almost everywhere -- it
    covers two cells per step -- and turning pays off only occasionally, which
    a deterministic policy cannot express. The real fix is to the ACTION SPACE:
    give the 8 directional actions a fast variant (turn and accelerate in one
    action), or drop action 11. Until then, stochastic evaluation is required,
    not optional -- see evaluation/controllers.py.

    Left in place as a flag so the experiment can be repeated; results in
    results_v3_heading/.
    """

    belief_patch_block: int = 5
    """Block size for the belief/visited channels of the local map (1 = legacy).

    The fine 5x5 window (LOCAL_MAP_RADIUS=2) sits strictly inside the radius-3
    sensor disc, so those two channels were CONSTANT: measured over 480
    agent-steps the visited channel was all-ones 100% of the time and the
    probability channel had std 0.0003. Fifty of eighty-six observation
    dimensions carried no information, leaving no policy any basis for choosing
    a direction -- which is why a deterministic policy (argmax, or MADDPG,
    whose actor is deterministic by construction) degenerates to one action,
    and why sharing coverage over the graph could not help.

    block=5 keeps the 5x5x3 shape of Bang 2 but makes each belief cell average
    a 5x5 square, so the window spans 25x25 cells (radius 12) and its outer
    ring finally shows unexplored ground. The occupancy channel stays fine
    (radius 2) for obstacle avoidance.
    """

    normalize_edge_attr: bool = True
    """Scale communication edge features by r_comm before they reach GATv2Conv.

    Legacy (False) passed raw metres: dx, dy and dist all range over
    [-r_comm, r_comm] while node embeddings are LayerNorm'd, so the attention
    logits were driven by edge magnitude rather than content.
    """

    team_reward_share: float = 0.3
    """Fraction of the per-agent reward replaced by the team mean, when
    `per_agent_reward` is on. 0 = fully selfish, 1 = the legacy global reward.
    0.3 keeps a cooperative signal while leaving most of the gradient
    attributable to the individual UAV."""


SCENARIOS: dict[str, ScenarioConfig] = {
    # r_comm duoc dat lai 04/09/2026 (Giai doan 10): 10/10/10 -> 26/34/46.
    #
    # Giai doan 6 ha r_comm tu 15/20/25 xuong 10/10/10 dua tren phep do cho
    # thay do thi "gan dac (54%/83%/96%)". Phep do do chay 100 buoc voi chinh
    # sach NGAU NHIEN — luc UAV van con nam trong luoi xuat phat quanh tam ban
    # do. Duoi mot chinh sach thuc su phan tan, cung r_comm=10 cho mat do
    # 0.14/0.012/0.020, va tren chinh sach GNN-MAPPO da huan luyen thi
    # **56% so buoc cua Easy va 35% cua Medium khong co MOT canh nao**.
    # GATv2Conv voi add_self_loops=True khi do suy bien thanh mot phep bien doi
    # tuyen tinh theo tung node: DGAT da la mot MLP trong phan lon thoi gian,
    # nen khong the khac MLP.
    #
    # Gia tri moi lay tu bac trung binh ky vong cho dan phan tan deu,
    #     deg = (N-1) * pi * r^2 / (W*H),
    # dat deg ~ 2.5 (du de message passing co noi dung, chua den muc moi UAV
    # noi voi moi UAV khien attention khong con gi de phan biet):
    #     Easy   N=4,  50x50  -> r = 26
    #     Medium N=8,  100x100 -> r = 34
    #     Hard   N=16, 200x200 -> r = 46
    # Khop voi phep quet thuc nghiem (do duoc 25 / 40 / 40-60).
    # Dung **LEGACY_OVERRIDES de tai lap r_comm=10 cua Giai doan 6-9.
    "easy": ScenarioConfig(
        name="easy", n_uavs=4, width=50, height=50,
        n_victims_range=(3, 5), victim_mode="uniform", terrain_mode="flat",
        r_comm=26.0, r_sense=3.0, max_steps=500, priority_enabled=False,
    ),
    "medium": ScenarioConfig(
        name="medium", n_uavs=8, width=100, height=100,
        n_victims_range=(5, 8), victim_mode="cluster", terrain_mode="obstacle",
        r_comm=34.0, r_sense=3.0, max_steps=500, priority_enabled=False,
    ),
    "hard": ScenarioConfig(
        name="hard", n_uavs=16, width=200, height=200,
        n_victims_range=(8, 12), victim_mode="corridor", terrain_mode="dynamic",
        r_comm=46.0, r_sense=3.0, max_steps=500, priority_enabled=True,
    ),
}

LEGACY_OVERRIDES = dict(normalize_obs=False, include_neighbor_obs=True,
                        local_belief=False, per_agent_reward=False,
                        normalize_edge_attr=False, include_heading_obs=False,
                        belief_patch_block=1, r_comm=10.0)
"""Pass `**LEGACY_OVERRIDES` to `get_scenario` to reproduce Giai doan 6/7."""


def get_scenario(name: str, **overrides) -> ScenarioConfig:
    base = SCENARIOS[name]
    cfg = ScenarioConfig(**{**base.__dict__, **overrides})
    return cfg


def obs_dim_for(cfg: ScenarioConfig) -> int:
    """Observation length implied by a scenario's flags (see SARSwarmEnv)."""
    from .sar_parallel_env import (K_NEIGHBORS, LOCAL_MAP_CHANNELS, LOCAL_MAP_RADIUS,
                                    SELF_STATE_DIM, TASK_STATE_DIM)
    dim = (SELF_STATE_DIM + (2 * LOCAL_MAP_RADIUS + 1) ** 2 * LOCAL_MAP_CHANNELS
           + TASK_STATE_DIM)
    if cfg.include_neighbor_obs:
        dim += K_NEIGHBORS * 6
    if cfg.include_heading_obs:
        dim += 2
    return dim
