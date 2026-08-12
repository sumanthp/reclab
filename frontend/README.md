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
```

## Structure

```
src/
  lib/
    types.ts     # mirrors the backend's Pydantic/dataclass response shapes
    api.ts        # typed fetch wrappers for every endpoint
    format.ts     # number/percent formatting helpers
  components/      # one component per screen section
  App.tsx          # the Upload -> Profile -> Shortlist -> Compare state machine
```

No component library or CSS framework — plain CSS (`index.css` for design
tokens, `App.css` for component styles) and native `<details>`/`<summary>`
for the Layer 1/Layer 2 disclosure pattern. See
`docs/architecture/ui-ux-plan.md` section 6 for why.
