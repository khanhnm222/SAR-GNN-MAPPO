export type Scenario = "easy" | "medium" | "hard";

export type Method =
  | "random_walk"
  | "greedy"
  | "maddpg"
  | "mappo_mlp"
  | "mappo_gcn"
  | "mappo_gat"
  | "gnn_mappo";

export const METHOD_ORDER: Method[] = [
  "random_walk",
  "greedy",
  "maddpg",
  "mappo_mlp",
  "mappo_gcn",
  "mappo_gat",
  "gnn_mappo",
];

export const METHOD_LABELS: Record<Method, string> = {
  random_walk: "Random Walk",
  greedy: "Greedy",
  maddpg: "MADDPG",
  mappo_mlp: "MAPPO-MLP",
  mappo_gcn: "MAPPO-GCN",
  mappo_gat: "MAPPO-GAT (no GRU)",
  gnn_mappo: "GNN-MAPPO (đề xuất)",
};

export const METHOD_GROUP: Record<Method, "heuristic" | "marl" | "ablation" | "proposed"> = {
  random_walk: "heuristic",
  greedy: "heuristic",
  maddpg: "marl",
  mappo_mlp: "marl",
  mappo_gcn: "ablation",
  mappo_gat: "ablation",
  gnn_mappo: "proposed",
};

export const METHOD_COLORS: Record<Method, string> = {
  random_walk: "#94a3b8",
  greedy: "#64748b",
  maddpg: "#f59e0b",
  mappo_mlp: "#ef4444",
  mappo_gcn: "#8b5cf6",
  mappo_gat: "#3b82f6",
  gnn_mappo: "#10b981",
};

export type MetricKey =
  | "coverage_rate"
  | "victim_detection_rate"
  | "ttfd"
  | "collision_rate"
  | "energy_efficiency";

export const METRIC_LABELS_VI: Record<MetricKey, string> = {
  coverage_rate: "Tỷ lệ bao phủ (Coverage Rate)",
  victim_detection_rate: "Tỷ lệ phát hiện nạn nhân (VDR)",
  ttfd: "Thời gian phát hiện đầu tiên (TTFD, bước)",
  collision_rate: "Tỷ lệ va chạm (Collision Rate)",
  energy_efficiency: "Hiệu suất năng lượng",
};

export const METRIC_ORDER: MetricKey[] = [
  "victim_detection_rate",
  "coverage_rate",
  "ttfd",
  "collision_rate",
  "energy_efficiency",
];

export interface Manifest {
  scenarios: Scenario[];
  methods: Method[];
  method_labels: Record<string, string>;
}

export interface LearningCurvePoint {
  step: number;
  vdr: number | null;
  coverage: number | null;
}

export interface LearningCurves {
  scenario: Scenario;
  methods: Partial<Record<Method, { label: string; seeds: { seed: number; points: LearningCurvePoint[] }[] }>>;
}

export interface MetricStat {
  mean: number | null;
  std: number | null;
  values: number[];
}

export interface ComparisonEntry {
  label: string;
  metrics: Partial<Record<MetricKey, MetricStat>>;
}

export interface StatTestResult {
  test: string;
  statistic: number | null;
  p_value: number | null;
  significant: boolean;
  mean_a: number | null;
  mean_b: number | null;
  cohens_d: number | null;
  alpha_corrected: number;
  significant_bonferroni: boolean;
}

export interface Comparison {
  scenario: Scenario;
  methods: Partial<Record<Method, ComparisonEntry>>;
  gnn_mappo_vs_baselines?: Partial<Record<MetricKey, Record<string, StatTestResult>>>;
}

export interface AblationReport {
  scenario: Scenario;
  metrics: Partial<Record<MetricKey, { gnn_mappo_values: number[]; comparisons?: Record<string, StatTestResult> }>>;
}

export interface TrajectoryUAV {
  uid: number;
  x: number;
  y: number;
  z: number;
  alive: boolean;
  energy_ratio: number;
  /** action id 0-12 taken at this step (sar_env/entities.py ACTIONS) */
  action?: number;
  /** true when the chosen move was refused by terrain / map bounds */
  blocked?: boolean;
}

export interface TrajectoryVictim {
  x: number;
  y: number;
  priority: number;
  detected: boolean;
  detected_at_step?: number | null;
  /** uid of the UAV that found this victim */
  detected_by?: number | null;
}

/** one live communication link: [uidA, uidB, signalStrength in 0..1] */
export type CommEdge = [number, number, number];

export interface TrajectoryFrame {
  t: number;
  uavs: TrajectoryUAV[];
  victims: TrajectoryVictim[];
  /** r_comm graph at this step — the same edges the DGAT attends over */
  comm_edges?: CommEdge[];
  /** cells that became blocked since the previous step (flood growth) */
  terrain_new?: [number, number][];
  coverage_rate: number;
  victims_detected: number;
  n_victims: number;
}

export interface TrajectoryData {
  meta: { method: Method; scenario: Scenario; seed: number };
  frames: TrajectoryFrame[];
  /** static obstacle cells [x, y] — Hinh 2: "vat can tinh" (o vuong xam) */
  terrain_occupancy?: [number, number][];
}

/** action id -> short Vietnamese label, matches sar_env/entities.py ACTIONS */
export const ACTION_LABELS: string[] = [
  "Bắc", "Nam", "Đông", "Tây", "ĐB", "TB", "ĐN", "TN",
  "Lên cao", "Xuống thấp", "Đứng yên", "Tăng tốc", "Về trạm",
];

export interface ZeroShotSeries {
  mean: (number | null)[];
  std: (number | null)[];
}

export interface ZeroShotMethod {
  label: string;
  n_seeds: number;
  victim_detection_rate: ZeroShotSeries;
  coverage_rate: ZeroShotSeries;
  collision_rate: ZeroShotSeries;
}

/** Cross-architecture zero-shot comparison: every method measured at every N. */
export interface ZeroShotReport {
  scenario: Scenario;
  train_n: number;
  ns: number[];
  eval_protocol: string;
  methods: Partial<Record<Method, ZeroShotMethod>>;
}
