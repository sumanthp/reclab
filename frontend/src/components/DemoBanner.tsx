export function DemoBanner() {
  return (
    <div className="demo-banner">
      <span className="demo-banner-tag">Static demo</span>
      <p>
        This is a static build with real, pre-computed results — the exact output of{" "}
        <code>scripts/run_benchmark.py</code> against MovieLens 100K, Amazon Reviews, and
        synthetic data (see{" "}
        <a href="https://github.com/sumanthp/reclab/blob/main/benchmarks/README.md" target="_blank" rel="noreferrer">
          benchmarks/README.md
        </a>
        ). GitHub Pages can't run the Python backend, so uploading your own data isn't available
        here — clone{" "}
        <a href="https://github.com/sumanthp/reclab" target="_blank" rel="noreferrer">
          the repo
        </a>{" "}
        and run <code>docker compose up</code> for that.
      </p>
    </div>
  );
}
