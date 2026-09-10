import type { Scenario } from "./types";

// Mirrors sar_env/scenarios.py SCENARIOS
export const SCENARIO_DIMS: Record<Scenario, { width: number; height: number; nUavs: number }> = {
  easy: { width: 50, height: 50, nUavs: 4 },
  medium: { width: 100, height: 100, nUavs: 8 },
  hard: { width: 200, height: 200, nUavs: 16 },
};

export const SCENARIO_LABELS_SHORT: Record<Scenario, string> = {
  easy: "Easy · 4 UAV",
  medium: "Medium · 8 UAV",
  hard: "Hard · 16 UAV",
};
