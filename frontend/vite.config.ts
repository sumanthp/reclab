import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Proxies API calls to the FastAPI backend during `npm run dev` so the
// frontend can just call same-origin paths (/profile, /compare, /runs/...)
// with no CORS setup needed in dev. The backend also sets permissive CORS
// headers itself (see src/reclab/api/main.py) for the case the built
// frontend is served separately, e.g. docker-compose.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://localhost:8000',
      '/architectures': 'http://localhost:8000',
      '/profile': 'http://localhost:8000',
      '/compare': 'http://localhost:8000',
      '/runs': 'http://localhost:8000',
    },
  },
  test: {
    // happy-dom, not jsdom: jsdom 30 hits a hard incompatibility on Node 24
    // ("webidl.util.markAsUncloneable is not a function", inside undici's
    // CacheStorage, on jsdom's own environment init) — reproduces in CI's
    // Node 24 runner even though it works locally on Node 22. happy-dom is
    // the standard Vitest-recommended alternative and avoids the whole
    // dependency chain that bug lives in.
    environment: 'happy-dom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
