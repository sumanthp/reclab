# reclab frontend

React + TypeScript (Vite). Upload → Profile → Shortlist → Compare → Result
detail — see [`../docs/architecture/ui-ux-plan.md`](../docs/architecture/ui-ux-plan.md)
for the screen-by-screen spec this was built against, and the top-level
[`../README.md`](../README.md) for how to run the full stack.

## Local development

```bash
npm install
npm run dev
```

Requires the API running separately at `localhost:8000` (`uv run uvicorn
reclab.api.main:app --reload` from the repo root) — Vite's dev server
proxies API calls to it (see `vite.config.ts`), so there's no CORS setup
needed and the frontend just calls same-origin relative paths
(`/profile`, `/compare`, `/runs/...`).

```bash
npm run build   # type-checks (tsc -b) then builds to dist/
npm run lint    # oxlint
npm test        # vitest (unit + component tests)
```

## Demo mode

`VITE_DEMO_MODE=true npm run build` builds the static site deployed to
[GitHub Pages](https://sumanthp.github.io/reclab/) (`.github/workflows/ci.yml`,
`deploy-demo` job) — same components, same code paths, but `App.tsx` reads
from `src/demo/fixtures.ts` (real `scripts/run_benchmark.py` output, copied
from `benchmarks/results/`) instead of calling the network functions in
`lib/api.ts`, since Pages can't run the Python backend. `vite.config.ts`
also sets `base: '/reclab/'` for that build only — GitHub Pages project
sites are served from a subpath, not the domain root.

## Structure

```
src/
  lib/
    types.ts     # mirrors the backend's Pydantic/dataclass response shapes
    api.ts        # typed fetch wrappers for every endpoint
    format.ts     # number/percent formatting helpers
  components/      # one component per screen section
  demo/            # static fixtures + picker UI for the GitHub Pages build
  App.tsx          # the Upload -> Profile -> Shortlist -> Compare state machine
```

No component library or CSS framework — plain CSS (`index.css` for design
tokens, `App.css` for component styles) and native `<details>`/`<summary>`
for the Layer 1/Layer 2 disclosure pattern. See
`docs/architecture/ui-ux-plan.md` section 6 for why.
