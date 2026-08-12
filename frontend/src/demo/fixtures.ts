import movielensJson from "./data/movielens.json";
import amazonJson from "./data/amazon-reviews.json";
import syntheticDefaultJson from "./data/synthetic-default.json";
import syntheticColdStartJson from "./data/synthetic-cold-start-heavy.json";
import type { ArchitectureInfo, CompareResult, RunResponse, RunSummary } from "../lib/types";

// Real output of scripts/run_benchmark.py, checked into
// benchmarks/results/ and copied here unmodified — the same numbers
// benchmarks/README.md quotes and this session validated against the real
// downloaded datasets. Nothing in this file is synthesized for the demo.
interface DemoFixture {
  id: string;
  label: string;
  createdAt: string;
  result: CompareResult;
}

const FIXTURES: DemoFixture[] = [
  {
    id: "movielens-100k",
    label: "MovieLens 100K (real)",
    createdAt: "2026-08-12T19:39:42Z",
    result: movielensJson as CompareResult,
  },
  {
    id: "amazon-reviews-all-beauty",
    label: "Amazon Reviews · All_Beauty (real)",
    createdAt: "2026-08-12T19:40:59Z",
    result: amazonJson as CompareResult,
  },
  {
    id: "synthetic-default",
    label: "Synthetic — moderate sparsity",
    createdAt: "2026-08-12T18:55:43Z",
    result: syntheticDefaultJson as CompareResult,
  },
  {
    id: "synthetic-cold-start-heavy",
    label: "Synthetic — cold-start heavy",
    createdAt: "2026-08-12T18:55:55Z",
    result: syntheticColdStartJson as CompareResult,
  },
];

// From the real GET /architectures response — see src/reclab/architectures/
// */info() in the Python source. Hardcoded here only because there's no
// live backend on GitHub Pages to fetch it from; the content is identical.
export const DEMO_ARCHITECTURES: ArchitectureInfo[] = [
  {
    name: "two_tower",
    description:
      "Separate user and item embedding towers trained with a BPR pairwise ranking loss (plain NumPy SGD, a real matrix-factorization recommender). Fast to train and serve; handles dense collaborative signal well but has no native answer for cold-start items or sequence order.",
    strengths: [
      "Fast to train and serve",
      "Strong on dense interaction data",
      "Simple to reason about and debug",
    ],
    weaknesses: [
      "Weak on cold-start items with few interactions",
      "Ignores interaction order / recency",
      "No use of item text or image metadata",
    ],
    relative_train_cost: "low",
    relative_serving_latency: "low",
  },
  {
    name: "sasrec",
    description:
      "Single-head causal self-attention over a user's interaction history, trained via next-item prediction (plain NumPy, hand-derived backprop — see module docstring). Captures order and recency signal that two-tower ignores; needs reasonably long user sequences to earn its extra training cost.",
    strengths: [
      "Captures sequence order and recency",
      "Strong on session-based / repeat-engagement patterns",
    ],
    weaknesses: [
      "Needs sufficient per-user sequence length to outperform simpler baselines",
      "Still weak on cold-start items with no interaction history",
      "More expensive to train than two-tower",
    ],
    relative_train_cost: "medium",
    relative_serving_latency: "medium",
  },
  {
    name: "hybrid_llm",
    description:
      "SASRec-style encoder for candidate generation from collaborative signal, plus a pluggable re-ranker (default: TF-IDF lexical similarity, no API key required) that injects and re-scores candidates using item text — including items the encoder has never seen. A real LLM-based re-ranker is a drop-in swap (see rerankers.py) once you have API access; the interface doesn't change.",
    strengths: [
      "Strong cold-start performance when item text/metadata is rich",
      "Can recommend items with zero training interactions via text alone",
      "Handles sparse interaction data better than pure collaborative models",
    ],
    weaknesses: [
      "Highest serving latency and cost of the three candidates",
      "Requires usable item text metadata to earn its complexity",
      "Default re-ranker is lexical (TF-IDF), not a real LLM — semantic matches beyond shared vocabulary need a real LLM re-ranker swapped in",
    ],
    relative_train_cost: "high",
    relative_serving_latency: "high",
  },
];

export function demoRunSummaries(): RunSummary[] {
  return FIXTURES.map((f) => ({
    id: f.id,
    status: "done",
    dataset_label: f.label,
    created_at: f.createdAt,
    updated_at: f.createdAt,
  }));
}

export function demoRun(id: string): RunResponse | null {
  const fixture = FIXTURES.find((f) => f.id === id);
  if (!fixture) return null;
  return {
    id: fixture.id,
    status: "done",
    dataset_label: fixture.label,
    created_at: fixture.createdAt,
    updated_at: fixture.createdAt,
    result: fixture.result,
    error: null,
  };
}

export function demoFixtureList(): { id: string; label: string }[] {
  return FIXTURES.map((f) => ({ id: f.id, label: f.label }));
}
