# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────────────────────
# Flan-T5 Summariser  ·  Inference API
# Target runtime : AWS App Runner (linux/amd64)
# Build          : docker buildx build --platform linux/amd64 -f inference/Dockerfile .
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Pin uv for reproducible builds — bump this tag when upgrading uv
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

# ── Layer 1: dependencies ────────────────────────────────────────────────────
# Copied before source so this heavy layer (torch + transformers ≈ 1 GB)
# is only re-built when pyproject.toml or uv.lock changes, not on code edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Layer 2: application source ──────────────────────────────────────────────
# Invalidated only when inference/ changes; deps layer above stays cached.
COPY inference/ ./inference/

# Drop root privileges after all filesystem writes are complete
RUN useradd --create-home --no-log-init --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# App Runner performs its own health checks via GET /health, but this gives
# Docker-level visibility during local runs and ECS-style deployments.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Call the venv's uvicorn directly — avoids the `uv run` subprocess wrapper.
CMD ["/app/.venv/bin/uvicorn", "inference.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
