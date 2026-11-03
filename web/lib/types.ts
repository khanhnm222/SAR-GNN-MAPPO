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
  mappo_mlp: "MAPPO-MLP (không đồ thị)",
  mappo_gcn: "MAPPO-GCN (đề xuất)",
  mappo_gat: "MAPPO-GAT (GATv2, không GRU)",
  gnn_mappo: "MAPPO-DGAT (GATv2 + GRU)",
};

/** Tên ngắn dùng trên trục biểu đồ / chú giải. */
export const METHOD_SHORT: Record<Method, string> = {
  random_walk: "Random",
  greedy: "Greedy",
  maddpg: "MADDPG",
  mappo_mlp: "MAPPO-MLP",
  mappo_gcn: "MAPPO-GCN",
  mappo_gat: "MAPPO-GAT",
  gnn_mappo: "MAPPO-DGAT",
};

/** Phiên bản đề xuất của framework GNN-MAPPO (chọn bằng ablation có kiểm soát). */
export const PROPOSED_METHOD: Method = "mappo_gcn";

/** Họ encoder của framework GNN-MAPPO: cùng bộ huấn luyện MAPPO, chỉ khác bộ mã hoá. */
export const ENCODER_FAMILY: Method[] = ["mappo_mlp", "mappo_gcn", "mappo_gat", "gnn_mappo"];

export const METHOD_NOTES: Record<Method, string> = {
  random_walk: "Heuristic — di chuyển ngẫu nhiên, không phối hợp",
  greedy: "Heuristic — về ô có xác suất nạn nhân cao nhất gần nhất",
  maddpg: "MARL không đồ thị, actor tất định (Lowe et al. 2017)",
  mappo_mlp: "Encoder MLP — không đồ thị; đối chứng của họ encoder",
  mappo_gcn: "Encoder GCN — tổng hợp lân cận trọng số cố định; phiên bản đề xuất",
  mappo_gat: "Encoder GATv2 có đặc trưng cạnh — cô lập đóng góp của attention",
  gnn_mappo: "Encoder GATv2 + GRU (DGAT) — biến thể đầy đủ; cô lập đóng góp của bộ nhớ thời gian (RQ2)",
};

export type MethodGroup = "heuristic" | "marl" | "encoder";
export const METHOD_GROUP: Record<Method, MethodGroup> = {
  random_walk: "heuristic",
  greedy: "heuristic",
  maddpg: "marl",
  mappo_mlp: "encoder",
  mappo_gcn: "encoder",
  mappo_gat: "encoder",
  gnn_mappo: "encoder",
};
export const METHOD_GROUP_ORDER: MethodGroup[] = ["heuristic", "marl", "encoder"];
export const METHOD_GROUP_LABELS: Record<MethodGroup, string> = {
  heuristic: "Heuristic (không học)",
  marl: "MARL không đồ thị",
  encoder: "Họ encoder GNN-MAPPO — cùng bộ huấn luyện MAPPO, chỉ khác bộ mã hoá",
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
  proposed_method?: Method;
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
  /** phương pháp được lấy làm mốc trong kiểm định (phiên bản đề xuất, MAPPO-GCN) */
  proposed_method?: Method;
  /** đề xuất so với từng phương pháp còn lại, Bonferroni theo số so sánh */
  proposed_vs_baselines?: Partial<Record<MetricKey, Record<string, StatTestResult>>>;
  /** RQ2: MAPPO-DGAT (có GRU) so với MAPPO-GAT (không GRU) */
  gru_ablation?: Partial<Record<MetricKey, StatTestResult>>;
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

export type ZeroShotMode = "fixed_map" | "density";

export const ZERO_SHOT_MODE_LABELS: Record<ZeroShotMode, string> = {
  fixed_map: "Fixed-map — giữ nguyên bản đồ, chỉ đổi N",
  density: "Density-controlled — bản đồ và số nạn nhân scale theo N",
};

/** Cross-architecture zero-shot comparison: every method measured at every N. */
export interface ZeroShotReport {
  scenario: Scenario;
  train_n: number;
  ns: number[];
  mode?: ZeroShotMode;
  /** kích thước môi trường tại từng N (chỉ khác nhau ở chế độ density) */
  env_by_n?: Record<string, { width: number; height: number; n_victims_range: [number, number]; r_comm: number }> | null;
  eval_protocol: string;
  methods: Partial<Record<Method, ZeroShotMethod>>;
}
