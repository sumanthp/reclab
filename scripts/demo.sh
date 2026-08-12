#!/usr/bin/env bash
# The reclab demo: profile a dataset, get the reasoning engine's architecture
# shortlist, then actually train and evaluate all three architectures so you
# can see whether the shortlist holds up — no GPU, no API keys, no cloud
# account, runs in well under a minute on a laptop.
#
# This is the "real demo" called for in docs/architecture/mvp-plan.md
# section 6 (Phase 2) in place of a screen recording — every number below is
# live output from this run, not a canned transcript.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Setting up environment (uv sync)..."
  uv sync
fi

echo
echo "############################################################"
echo "# Scenario 1: a typical dataset (moderate cold-start ratio)"
echo "############################################################"
uv run python scripts/run_benchmark.py --dataset synthetic

echo
echo "############################################################"
echo "# Scenario 2: sparse, high cold-start ratio, rich item text"
echo "# — the case the reasoning engine says hybrid_llm should win"
echo "############################################################"
uv run python scripts/run_benchmark.py --dataset synthetic --cold-start-heavy

echo
echo "Both runs are saved under benchmarks/results/. See benchmarks/README.md"
echo "for what these numbers mean and where the reasoning engine's shortlist"
echo "currently does and doesn't hold up."
