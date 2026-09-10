"""Writes learning-curve + evaluation JSON logs consumed by evaluation/plots.py
and by the Next.js visualization site (web/public/data/*.json).
"""
from __future__ import annotations

import json
import os


class RunLogger:
    def __init__(self, out_dir: str, method: str, scenario: str, seed: int):
        self.dir = os.path.join(out_dir, scenario, method, f"seed_{seed}")
        os.makedirs(self.dir, exist_ok=True)
        self.method, self.scenario, self.seed = method, scenario, seed
        self.curve: list[dict] = []
        self.meta = {"method": method, "scenario": scenario, "seed": seed}

    def log_iteration(self, stats: dict):
        self.curve.append(stats)
        self._flush_curve()

    def _flush_curve(self):
        with open(os.path.join(self.dir, "learning_curve.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "curve": self.curve}, f)

    def save_eval(self, eval_metrics: dict):
        with open(os.path.join(self.dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "metrics": eval_metrics}, f, indent=2)

    def save_trajectory(self, trajectory: list[dict], terrain_occupancy: list[list[int]] | None = None):
        payload = {"meta": self.meta, "frames": trajectory}
        if terrain_occupancy is not None:
            payload["terrain_occupancy"] = terrain_occupancy
        with open(os.path.join(self.dir, "trajectory_sample.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def save_checkpoint(self, state_dict, name: str = "policy.pt"):
        import torch
        torch.save(state_dict, os.path.join(self.dir, name))
