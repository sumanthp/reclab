import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, devices } from "@playwright/test";

// Scoped, not comprehensive: 2-3 golden-path tests exercising the real
// backend + real frontend together (no mocking), as a check that the two
// sides actually integrate — not a substitute for the Vitest component
// suite (fast, isolated, exhaustive on edge cases) or the pytest suite
// (the same for the API/architectures). See README.md's "what's not done"
// for the coverage this deliberately doesn't attempt.
const dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(dirname, "..");
const backendPort = 8000;
const frontendPort = 5173;

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"]] : [["html", { open: "never" }]],
  use: {
    baseURL: `http://localhost:${frontendPort}`,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run uvicorn reclab.api.main:app --port 8000",
      cwd: repoRoot,
      url: `http://localhost:${backendPort}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
      env: {
        // A dedicated, disposable job store — must not touch a real
        // reclab.db a developer might have from `docker compose up` /
        // manual runs elsewhere in this repo.
        RECLAB_STORAGE: `sqlite:///${path.join(dirname, "e2e-runs.db")}`,
      },
    },
    {
      command: `npm run dev -- --port ${frontendPort} --strictPort`,
      cwd: dirname,
      url: `http://localhost:${frontendPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
