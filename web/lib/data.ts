import type {
  AblationReport,
  Comparison,
  LearningCurves,
  Manifest,
  Scenario,
  TrajectoryData,
  ZeroShotReport,
} from "./types";

async function getJSON<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export const fetchManifest = () => getJSON<Manifest>("/data/manifest.json");

export const fetchLearningCurves = (scenario: Scenario) =>
  getJSON<LearningCurves>(`/data/${scenario}/learning_curves.json`);

export const fetchComparison = (scenario: Scenario) =>
  getJSON<Comparison>(`/data/${scenario}/comparison.json`);

export const fetchAblation = (scenario: Scenario) =>
  getJSON<AblationReport>(`/data/${scenario}/ablation.json`);

export const fetchZeroShot = (scenario: Scenario) =>
  getJSON<ZeroShotReport>(`/data/${scenario}/zero_shot.json`);

export const fetchTrajectory = (scenario: Scenario, method: string) =>
  getJSON<TrajectoryData>(`/data/${scenario}/trajectories/${method}.json`);

export const fetchTrajectoryManifest = (scenario: Scenario) =>
  getJSON<Record<string, boolean>>(`/data/${scenario}/trajectories/manifest.json`);
