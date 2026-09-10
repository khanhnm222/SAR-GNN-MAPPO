"""Multi-component reward function (muc 1.5.4, cong thuc 1.4):

    r = alpha * r_cover + beta * r_victim + gamma * r_collision + delta * r_energy

Weights alpha/beta are positive (reward), gamma/delta are negative (penalty),
matching Bang 4 of the thesis proposal. `RewardComponents` stores each raw,
*unsigned magnitude* term; `compute_step_reward` applies the signed weights.

Two shapes are provided:
  - `compute_step_reward`  : one scalar for the whole team (legacy, muc 1.5.4
    as literally written).
  - `compute_agent_rewards`: one scalar PER UAV, crediting each UAV with the
    cells it personally revealed, the victims it personally found, its own
    collisions and its own energy, blended with a `team_share` of the team
    mean. See ScenarioConfig.per_agent_reward for why this matters.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    # alpha. Raised 1.0 -> 20.0 together with per-agent credit assignment.
    # r_cover is a FRACTION of the whole map: on Easy a UAV reveals ~29 cells
    # of 2500 on its first step and far fewer later, so alpha=1.0 caps the
    # total coverage reward of a perfect episode at 1.0 -- against 15-25 for
    # victim detection. Exploration, the behaviour that actually leads to
    # victims, was therefore worth almost nothing per step and the search
    # signal was effectively sparse. alpha=20 makes a full sweep worth ~20,
    # comparable to finding every victim, without dominating it.
    cover: float = 20.0
    victim: float = 5.0      # beta
    # gamma (negative: penalty). Was -2.0, matching the thesis's stated *starting*
    # value, but that accumulates unbounded over an episode (collision is a
    # per-STEP flag, not a one-off event): an easy-scenario episode with a
    # reasonable heuristic hit ~10 collision-steps/300, i.e. -20.0 total,
    # versus at most 4.0 total possible victim reward (4 victims x priority 1)
    # -- collision penalty alone made total reward NEGATIVE for a policy that
    # outperformed random on every task metric, so PPO's gradient pointed away
    # from actual task success. -0.05 keeps collisions clearly worse than
    # coverage per step without an episode's worth of them outweighing victim
    # detection.
    collision: float = -0.05
    energy: float = -0.01    # delta (negative: penalty)


@dataclass
class RewardComponents:
    cover: float = 0.0        # fraction of newly-explored cells this step
    victim: float = 0.0       # sum of priority over victims newly detected this step
    collision: float = 0.0    # 1.0 if >=1 collision event this step, else 0.0
    energy: float = 0.0       # energy spent this step, normalized by E_max


def _combine(comp: RewardComponents, weights: RewardWeights) -> float:
    return (weights.cover * comp.cover + weights.victim * comp.victim +
            weights.collision * comp.collision + weights.energy * comp.energy)


def compute_step_reward(n_new_cells: int, total_cells: int, victims_found_priority_sum: float,
                         n_collisions: int, energy_spent: float,
                         weights: RewardWeights) -> tuple[float, RewardComponents]:
    # r_collision la mot CO/KHONG (bool), khop dinh nghia cua de cuong "Collision
    # Rate: ti le buoc co va cham tren tong so buoc" (buoc-co-va-cham, khong phai
    # tong so cap va cham). Truoc day comp.collision = float(n_collisions) cong
    # DON TAT CA cap UAV vi pham cung luc, khong gioi han tren: voi N=16 (Hard),
    # hang chuc cap co the vi pham dong thoi -> phat -2.0*n_collisions co the la
    # -20 den -60 trong MOT buoc, at han toan tin hieu r_cover va ca r_victim.
    comp = RewardComponents(
        cover=n_new_cells / max(total_cells, 1),
        victim=victims_found_priority_sum,
        collision=1.0 if n_collisions > 0 else 0.0,
        energy=energy_spent / 1000.0,
    )
    return _combine(comp, weights), comp


def compute_agent_rewards(per_agent_new_cells: list[int], total_cells: int,
                           per_agent_victim_priority: list[float],
                           per_agent_collision: list[bool],
                           per_agent_energy: list[float],
                           weights: RewardWeights,
                           team_share: float = 0.3
                           ) -> tuple[list[float], RewardComponents]:
    """Per-UAV reward decomposition.

    Returns `(rewards_per_agent, team_components)` where `team_components`
    aggregates the same quantities for logging / the `info` dict, so the
    metrics reported in muc 1.5.5 are unchanged.
    """
    n = len(per_agent_new_cells)
    individual = []
    for i in range(n):
        comp = RewardComponents(
            cover=per_agent_new_cells[i] / max(total_cells, 1),
            victim=per_agent_victim_priority[i],
            collision=1.0 if per_agent_collision[i] else 0.0,
            energy=per_agent_energy[i] / 1000.0,
        )
        individual.append(_combine(comp, weights))

    team_mean = sum(individual) / max(n, 1)
    blended = [(1.0 - team_share) * r + team_share * team_mean for r in individual]

    team_comp = RewardComponents(
        cover=sum(per_agent_new_cells) / max(total_cells, 1),
        victim=sum(per_agent_victim_priority),
        collision=1.0 if any(per_agent_collision) else 0.0,
        energy=sum(per_agent_energy) / 1000.0,
    )
    return blended, team_comp
