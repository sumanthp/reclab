# Minimal image for the reclab API. `uv sync --no-dev` installs the base
# dependency set (numpy/pandas/scikit-learn, no PyTorch — see pyproject.toml)
# which is also what /compare's training+eval job runs on; there's no
# separate "ml" extra or training service.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY README.md LICENSE ./

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "reclab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
