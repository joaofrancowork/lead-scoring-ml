# syntax=docker/dockerfile:1.7
# Multi-stage: train in the build stage, ship only the runtime and the model.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

FROM base AS train
RUN lead-scoring train --split time --features client --out /app/artifacts/model

FROM base AS serve
COPY --from=train /app/artifacts/model /app/artifacts/model
RUN useradd --create-home app && chown -R app /app
USER app
ENV LEAD_SCORING_MODEL_DIR=/app/artifacts/model
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["lead-scoring", "serve", "--port", "8000"]
