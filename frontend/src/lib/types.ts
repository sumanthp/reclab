// Mirrors the dataclasses in src/reclab/{data_profiler,reasoning_engine,eval,api}/*.py.
// Kept as one file since the backend is the single source of truth for these
// shapes — if a field changes there, it should change here, not the other
// way around.

export interface DataProfile {
  n_users: number;
  n_items: number;
  n_interactions: number;
  sparsity: number;
  cold_start_ratio: number;
  median_sequence_length: number;
  has_item_text: boolean;
  has_item_image: boolean;
}

export interface Recommendation {
  architecture: string;
  rank: number;
  score: number;
  rationale: string;
}

export interface ProfileResponse {
  profile: DataProfile;
  recommendations: Recommendation[];
}

export interface ArchitectureInfo {
  name: string;
  description: string;
  strengths: string[];
  weaknesses: string[];
  relative_train_cost: "low" | "medium" | "high";
  relative_serving_latency: "low" | "medium" | "high";
}

export interface EvalResult {
  architecture: string;
  k: number;
  n_test_users: number;
  recall_at_k: number;
  ndcg_at_k: number;
  coverage_at_k: number;
  cold_start_recall_at_k: number | null;
  cold_start_surfaced_rate: number;
}

export interface SkippedEvalResult {
  skipped: string;
}

export function isSkipped(r: EvalResult | SkippedEvalResult): r is SkippedEvalResult {
  return "skipped" in r;
}

export type EvalResultsMap = Record<string, EvalResult | SkippedEvalResult>;

export interface ComparisonSummary {
  shortlist_pick: string;
  measured_best_recall: string | null;
  measured_best_cold_start_recall: string | null;
  matches_on_recall: boolean | null;
  note: string | null;
}

export interface CompareResult {
  profile: DataProfile;
  reasoning_engine_shortlist: Recommendation[];
  eval_results: EvalResultsMap;
  // null when the run was cancelled before enough architectures finished
  // for a fair verdict — see summarize_comparison's caller in main.py.
  comparison: ComparisonSummary | null;
}

export type JobStatus = "pending" | "running" | "done" | "error" | "cancelled";

export interface RunResponse {
  id: string;
  status: JobStatus;
  dataset_label: string | null;
  created_at: string;
  updated_at: string;
  result: CompareResult | null;
  error: string | null;
}

// GET /runs list items — no `result`/`error`, kept cheap to list. Fetch
// GET /runs/{id} (RunResponse) for the full detail of one run.
export interface RunSummary {
  id: string;
  status: JobStatus;
  dataset_label: string | null;
  created_at: string;
  updated_at: string;
}

export class ApiError extends Error {}
