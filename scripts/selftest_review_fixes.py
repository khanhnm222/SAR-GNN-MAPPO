"""Self-tests for the corrections made in REVIEW_2026-09-03.md.

Run:  python -m scripts.selftest_review_fixes

Covers the four environment flags, per-agent GAE, return normalisation, the
replay payload, and a short end-to-end MAPPO iteration for every encoder in
Bang 5. Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from models import build_encoder  # noqa: E402
from sar_env import make_env  # noqa: E402
from sar_env.scenarios import LEGACY_OVERRIDES, get_scenario, obs_dim_for  # noqa: E402
from training.rollout_buffer import RolloutBuffer, RunningNorm, Transition  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- observations
def test_obs_dims():
    print("observation dimensions")
    legacy = get_scenario("easy", **LEGACY_OVERRIDES)
    new = get_scenario("easy")
    check("legacy obs_dim == 116", obs_dim_for(legacy) == 116, str(obs_dim_for(legacy)))
    check("reviewed obs_dim == 86 (neighbour block removed)",
          obs_dim_for(new) == 86, str(obs_dim_for(new)))
    head = get_scenario("easy", include_heading_obs=True)
    check("include_heading_obs adds exactly 2 dims", obs_dim_for(head) == 88)

    env = make_env("hard")
    obs, _ = env.reset(seed=3)
    o = next(iter(obs.values()))
    check("obs length matches env.obs_dim", o.shape[0] == env.obs_dim)
    check("normalised obs stays within [-1.05, 1.05]",
          bool(np.all(np.abs(o) <= 1.05)), f"max |x| = {np.abs(o).max():.3f}")

    env_l = make_env("hard", **LEGACY_OVERRIDES)
    obs_l, _ = env_l.reset(seed=3)
    o_l = next(iter(obs_l.values()))
    check("legacy obs really was unnormalised (raw grid coords present)",
          bool(np.abs(o_l).max() > 50.0), f"max |x| = {np.abs(o_l).max():.1f}")

    # heading must actually track the UAV's last movement direction
    env2 = make_env("easy", seed=1, include_heading_obs=True)
    obs2, _ = env2.reset(seed=1)
    obs2, _, _, _, _ = env2.step({a: 2 for a in env2.agents})   # 2 = East -> (1, 0)
    h_east = next(iter(obs2.values()))[-2:]
    obs2, _, _, _, _ = env2.step({a: 1 for a in env2.agents})   # 1 = South -> (0, -1)
    h_south = next(iter(obs2.values()))[-2:]
    check("heading observable and updated by movement",
          bool(np.allclose(h_east, [1, 0]) and np.allclose(h_south, [0, -1])),
          f"east={h_east} south={h_south}")


# ------------------------------------------------------- local map is useful
def test_local_map_informative():
    """The single most damaging defect found: with r_sense=3 and
    LOCAL_MAP_RADIUS=2 the belief window sits strictly inside the sensor disc,
    so its visited/prob channels are constant and no policy can see where to
    explore. `belief_patch_block` spans the window past the sensor."""
    print("local map carries an exploration gradient")
    from sar_env.sar_parallel_env import SELF_STATE_DIM

    def measure(**ov):
        env = make_env("easy", seed=5, **ov)
        obs, _ = env.reset(seed=5)
        rng = np.random.default_rng(0)
        allones, tot, stds = 0, 0, []
        for _ in range(60):
            if not env.agents:
                break
            obs, _, _, _, _ = env.step({a: int(rng.integers(0, 13)) for a in env.agents})
            for o in obs.values():
                lm = o[SELF_STATE_DIM:SELF_STATE_DIM + 75].reshape(5, 5, 3)
                tot += 1
                if lm[..., 2].min() > 0.999:
                    allones += 1
                stds.append(float(lm[..., 2].std()))
        return allones / max(tot, 1), float(np.mean(stds))

    frac_new, std_new = measure()
    frac_old, std_old = measure(belief_patch_block=1)
    check("legacy visited channel was constant (the defect)",
          frac_old > 0.99 and std_old < 1e-6, f"all-ones {frac_old:.0%}, std {std_old:.6f}")
    check("coarse window carries a real exploration gradient",
          frac_new < 0.05 and std_new > 0.05, f"all-ones {frac_new:.0%}, std {std_new:.4f}")

    # out-of-bounds must read as "already covered", not "unexplored"
    from sar_env.belief_map import BeliefMap
    bm = BeliefMap(20, 20)
    bm.reset()
    bm.visited[:] = 1.0
    _, v = bm.local_patch(0, 0, 2)
    check("outside the map reads visited=1, not 0",
          bool(np.all(v > 0.999)), f"min {v.min():.2f}")


# ------------------------------------------------------------- local belief
def test_local_belief():
    print("per-UAV belief maps merged over the comm graph")
    env = make_env("medium", seed=7)
    env.reset(seed=7)
    check("one belief map per UAV", len(env._beliefs) == env.cfg.n_uavs)

    # Split the swarm into two groups placed further apart than r_comm, so the
    # comm graph really has two components. (Driving them apart for a fixed
    # number of steps is not enough now that r_comm is sized to the map.)
    half = env.cfg.n_uavs // 2
    for i, u in enumerate(env.uavs):
        u.x, u.y = (5.0, 5.0) if i < half else (env.cfg.width - 6.0, env.cfg.height - 6.0)
    for t in range(3):
        env.step({a: 10 for a in env.agents})   # hover; only sensing + merging
    # Compare the MAPS, not the coverage rates: two symmetric groups scan the
    # same NUMBER of cells, so equal rates would hide completely disjoint maps.
    a, b = env._beliefs[0].visited, env._beliefs[-1].visited
    overlap = float((a * b).sum()) / max(float(np.minimum(a.sum(), b.sum())), 1.0)
    check("UAV beliefs diverge once the swarm splits",
          overlap < 0.05, f"chong lan giua hai cum = {overlap:.1%}")
    check("UAVs inside one component share an identical map",
          bool(np.array_equal(env._beliefs[0].visited, env._beliefs[1].visited)))
    covs = [b_.coverage_rate() for b_ in env._beliefs]
    team = env.team_coverage_rate()
    check("team coverage >= every individual coverage",
          all(team >= c - 1e-6 for c in covs), f"team={team:.6f} max_ind={max(covs):.6f}")

    env_l = make_env("medium", seed=7, **LEGACY_OVERRIDES)
    env_l.reset(seed=7)
    check("legacy keeps a single shared map", env_l._beliefs == [])


# ----------------------------------------------------------- per-agent reward
def test_per_agent_reward():
    print("per-agent reward decomposition")
    env = make_env("medium", seed=11)
    env.reset(seed=11)
    seen_spread = False
    for t in range(25):
        _, rewards, _, _, _ = env.step({a: (0 if i < 4 else 1) for i, a in enumerate(env.agents)})
        vals = list(rewards.values())
        if len(set(round(v, 8) for v in vals)) > 1:
            seen_spread = True
    check("UAVs receive different rewards", seen_spread)

    env_l = make_env("medium", seed=11, **LEGACY_OVERRIDES)
    env_l.reset(seed=11)
    _, rewards_l, _, _, _ = env_l.step({a: 0 for a in env_l.agents})
    check("legacy gives one identical team reward",
          len(set(round(v, 8) for v in rewards_l.values())) == 1)


# ---------------------------------------------------- graph is not empty
def test_graph_connectivity():
    """r_comm must be large enough that a DISPERSED swarm still has edges.

    Measured on the trained results_v2 policies at the old r_comm=10, 56% of
    Easy steps and 35% of Medium steps had NO edge at all — GATv2Conv with
    self-loops then degenerates to a per-node linear map, i.e. the DGAT was an
    MLP most of the time. See EXPERIMENT_LOG Giai doan 10."""
    print("communication graph survives dispersal")
    from evaluation.controllers import HeuristicController

    def measure(sc, **ov):
        env = make_env(sc, seed=3, **ov)
        obs, _ = env.reset(seed=3)
        ctrl = HeuristicController("greedy", seed=0)
        ctrl.reset()
        degs, empty, tot = [], 0, 0
        for t in range(320):
            if not env.agents:
                break
            obs, _, _, _, _ = env.step(ctrl.act(env, obs))
            if t < 100:
                continue
            g = env.build_communication_graph()
            n = g["n_nodes"]
            if n < 2:
                continue
            tot += 1
            e = int(g["edge_index"].shape[1])
            degs.append(e / n)
            if e == 0:
                empty += 1
        return (sum(degs) / max(len(degs), 1)), empty / max(tot, 1)

    for sc, lo in [("easy", 1.0), ("medium", 1.0)]:
        deg, frac_empty = measure(sc)
        deg_old, empty_old = measure(sc, r_comm=10.0)
        check(f"{sc}: dispersed swarm keeps a usable graph",
              deg >= lo and frac_empty < 0.25,
              f"bac {deg:.2f}, {frac_empty:.0%} buoc rong (r_comm=10 cu: bac {deg_old:.2f}, {empty_old:.0%})")


# ------------------------------------------------------------------ edge attr
def test_edge_attr():
    print("communication edge features")
    env = make_env("medium", seed=5)
    env.reset(seed=5)
    for _ in range(5):
        env.step({a: 2 for a in env.agents})
    g = env.build_communication_graph()
    ea = g["edge_attr"]
    check("edge_attr has 6 columns", ea.shape[1] == 6)
    check("edge_attr normalised to [-1, 1]",
          bool(ea.numel() == 0 or ea.abs().max() <= 1.0 + 1e-6),
          f"max |x| = {float(ea.abs().max()) if ea.numel() else 0:.3f}")
    adj = g["adjacency"]
    check("adjacency symmetric", bool(np.allclose(adj, adj.T, atol=1e-6)))


# ------------------------------------------------------------------ replay
def test_replay_payload():
    print("replay payload")
    env = make_env("hard", seed=2)
    env.reset(seed=2)
    saw_new_terrain = False
    for _ in range(20):
        env.step({a: 4 for a in env.agents})
        s = env.get_render_state()
        if s["terrain_new"]:
            saw_new_terrain = True
    for key in ["comm_edges", "terrain_new", "uavs", "victims"]:
        check(f"render state has '{key}'", key in s)
    check("UAV entries carry action + blocked",
          all("action" in u and "blocked" in u for u in s["uavs"]))
    check("victims carry detected_by", all("detected_by" in v for v in s["victims"]))
    check("dynamic terrain reports newly blocked cells (Hard flood)", saw_new_terrain)
    if s["comm_edges"]:
        a, b, q = s["comm_edges"][0]
        check("comm edge is [uidA, uidB, signal in 0..1]", 0.0 <= q <= 1.0 and a != b)


# ------------------------------------------------------------------- GAE
def test_per_agent_gae():
    print("per-agent GAE")
    # two agents, three steps, agent 1 dies after step 1
    def tr(uids, rewards, values, done=False):
        k = len(uids)
        return Transition(
            x=torch.zeros(k, 4), edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 6)), h_prev=torch.zeros(k, 8),
            h_prev_critic=torch.zeros(k, 8), uids=list(uids),
            actions=torch.zeros(k, dtype=torch.long), log_probs_old=torch.zeros(k),
            values_old=torch.tensor(values, dtype=torch.float32),
            rewards=torch.tensor(rewards, dtype=torch.float32), done=done)

    buf = RolloutBuffer()
    buf.add(tr([0, 1], [1.0, 0.0], [0.0, 0.0]))
    buf.add(tr([0], [1.0], [0.0]))
    buf.add(tr([0], [1.0], [0.0], done=True))
    buf.compute_gae_per_agent(gamma=1.0, lam=1.0)

    a0 = [float(buf.transitions[t].advantages[0]) for t in range(3)]
    check("agent 0 advantage = remaining reward-to-go [3,2,1]",
          np.allclose(a0, [3.0, 2.0, 1.0]), str([round(v, 3) for v in a0]))
    a1 = float(buf.transitions[0].advantages[1])
    check("agent 1 (dead after step 0) bootstraps at 0", np.isclose(a1, 0.0), str(round(a1, 3)))
    check("advantages differ between agents at the same timestep", not np.isclose(a0[0], a1))

    buf.normalize_advantages()
    allv = torch.cat([t.advantages for t in buf.transitions])
    check("normalised advantages have ~zero mean", abs(float(allv.mean())) < 1e-5)

    rn = RunningNorm()
    rn.update(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    z = rn.normalize(torch.tensor([25.0]))
    check("return normaliser round-trips", np.isclose(float(rn.denormalize(z)[0]), 25.0, atol=1e-4))


# ------------------------------------------------------------------- BPTT
def test_bptt():
    """The GRU used to be trained with BPTT length 1 -- it carried state but
    never got a gradient for carrying it, and measurably HURT (Easy: 0.890 with
    GRU vs 0.910 without, p=0.0116). `bptt_len` rolls the encoder through
    contiguous within-episode chunks with the hidden state carried under
    autograd. Only recurrent encoders take that path."""
    print("truncated BPTT for the recurrent encoder")
    from algorithms.mappo import MAPPOConfig, MAPPOTrainer
    from training.rollout_buffer import collate_transitions

    def gru_grad(bptt):
        torch.manual_seed(0)
        np.random.seed(0)
        env0 = make_env("easy", seed=0)
        enc = build_encoder("dgat", in_dim=env0.obs_dim, hidden_dim=32)
        cfg = MAPPOConfig(hidden_dim=32, rollout_len=96, minibatch_size=48, ppo_epochs=1,
                          total_env_steps=100000, bptt_len=bptt)
        tr = MAPPOTrainer(lambda seed: make_env("easy", seed=seed), enc, env0.obs_dim,
                          config=cfg, seed=0)
        buf = tr.collect_rollout(96)
        if bptt > 1:
            chunks = tr._chunks(buf)
            assert max(len(c) for c in chunks) <= bptt
            loss, _ = tr._recurrent_minibatch_loss(buf, chunks[:6], 0.01)
        else:
            b = collate_transitions(buf.transitions[:6])
            h = tr.policy.encode(b["x"], b["edge_index"], b["edge_attr"], b["h_prev"])
            lp, _e = tr.policy.evaluate_actions(h, b["actions"])
            loss = -(lp * b["advantages"]).mean()
        tr.policy.zero_grad()
        loss.backward()
        g = tr.policy.encoder.gru.weight_hh.grad
        return float(g.norm()) if g is not None else 0.0

    g1, g8 = gru_grad(1), gru_grad(8)
    check("BPTT changes the GRU gradient (it is not a no-op)",
          g8 > 0 and abs(g8 - g1) > 1e-9, f"len1 {g1:.6f} vs len8 {g8:.6f}")

    # chunks must never cross an episode boundary
    env0 = make_env("easy", seed=0)
    enc = build_encoder("dgat", in_dim=env0.obs_dim, hidden_dim=32)
    tr = MAPPOTrainer(lambda seed: make_env("easy", seed=seed), enc, env0.obs_dim,
                      config=MAPPOConfig(hidden_dim=32, rollout_len=96, bptt_len=8), seed=1)
    buf = tr.collect_rollout(96)
    ok = True
    for c in tr._chunks(buf):
        if any(buf.transitions[t].done for t in c[:-1]):
            ok = False
    check("no chunk crosses an episode boundary", ok)

    # non-recurrent encoders keep the flat (faster) update
    enc_m = build_encoder("mlp", in_dim=env0.obs_dim, hidden_dim=32)
    tr_m = MAPPOTrainer(lambda seed: make_env("easy", seed=seed), enc_m, env0.obs_dim,
                        config=MAPPOConfig(hidden_dim=32, rollout_len=64, minibatch_size=32,
                                            ppo_epochs=1, bptt_len=8), seed=0)
    check("MLP still uses the flat update", tr_m.train_iteration()["bptt_len"] == 1)


# ------------------------------------------------------------ reproducibility
def test_seeding():
    """`torch.manual_seed` was never called, so weight init and every rollout
    sample came from an unseeded global RNG -- `--seed` only controlled the
    environment and the numpy shuffle. Two runs of byte-identical code differed
    by up to 0.017 VDR, comparable to the between-seed std."""
    print("seed protocol")
    from training.train import seed_everything

    def draw():
        seed_everything(7)
        env0 = make_env("easy", seed=0)
        enc = build_encoder("gcn", in_dim=env0.obs_dim, hidden_dim=32)
        w = next(enc.parameters()).detach().clone()
        return w, torch.rand(3), np.random.rand(3)

    w1, t1, n1 = draw()
    w2, t2, n2 = draw()
    check("weight init reproducible", bool(torch.equal(w1, w2)))
    check("torch RNG reproducible", bool(torch.equal(t1, t2)))
    check("numpy RNG reproducible", bool(np.array_equal(n1, n2)))

    seed_everything(8)
    env0 = make_env("easy", seed=0)
    w3 = next(build_encoder("gcn", in_dim=env0.obs_dim, hidden_dim=32).parameters()).detach()
    check("different seed gives a different init", not bool(torch.equal(w1, w3)))


# --------------------------------------------------------------- end-to-end
def test_training_iteration():
    print("one MAPPO iteration per encoder")
    from algorithms.mappo import MAPPOConfig, MAPPOTrainer

    for name in ["mlp", "gcn", "gat", "dgat"]:
        env0 = make_env("easy", seed=0)
        obs_dim = env0.obs_dim
        enc = build_encoder(name, in_dim=obs_dim, hidden_dim=32)
        cfg = MAPPOConfig(hidden_dim=32, rollout_len=40, minibatch_size=16, ppo_epochs=2,
                           total_env_steps=1000)
        tr = MAPPOTrainer(lambda seed: make_env("easy", seed=seed), enc, obs_dim,
                           config=cfg, seed=0)
        st = tr.train_iteration()
        finite = all(np.isfinite(st[k]) for k in ("actor_loss", "value_loss", "entropy"))
        check(f"{name}: finite losses, entropy={st['entropy']:.3f}, kl={st['approx_kl']:.5f}",
              finite and st["global_step"] == 40)

    # legacy path still runs
    env0 = make_env("easy", seed=0, **LEGACY_OVERRIDES)
    enc = build_encoder("dgat", in_dim=env0.obs_dim, hidden_dim=32)
    cfg = MAPPOConfig(hidden_dim=32, rollout_len=40, minibatch_size=16, ppo_epochs=2, legacy=True)
    tr = MAPPOTrainer(lambda seed: make_env("easy", seed=seed, **LEGACY_OVERRIDES),
                       enc, env0.obs_dim, config=cfg, seed=0)
    st = tr.train_iteration()
    check("legacy=True trainer still runs", np.isfinite(st["actor_loss"]))


def main():
    for fn in [test_obs_dims, test_local_map_informative, test_graph_connectivity, test_local_belief, test_per_agent_reward, test_edge_attr,
               test_replay_payload, test_per_agent_gae, test_bptt, test_seeding, test_training_iteration]:
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
