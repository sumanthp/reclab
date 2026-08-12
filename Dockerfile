# Minimal image for the reclab API. Kept to the core (non-ml) dependency set
# so `docker compose up` stays fast — the `ml` extra (torch) is only needed
# for actually training architectures, not for profiling/reasoning-engine
# calls over the API.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY README.md LICENSE ./

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "reclab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
