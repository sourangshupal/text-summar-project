# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (creates .venv automatically)
uv sync

# Run fast tests (no model download, uses mocks) — default for CI
uv run pytest tests/ -v -m "not slow"

# Run a single test
uv run pytest tests/test_inference.py::test_health_endpoint -v

# Run slow integration tests (downloads ~900MB flan-t5-base)
uv run pytest tests/ -v -m slow

# Start the inference API locally
HF_MODEL_ID=google/flan-t5-base WANDB_API_KEY=<key> \
  uv run uvicorn inference.app:app --reload --port 8080

# Run fine-tuning (requires GPU; intended for Colab)
uv run python -m training.train

# Push a trained checkpoint to HF Hub
HF_TOKEN=<token> uv run python -m training.push_to_hub results/flan-t5-samsum your-username/repo-name

# Build Docker image for App Runner (must be linux/amd64)
docker buildx build --platform linux/amd64 -t flan-t5-summarizer:local -f inference/Dockerfile .
```

## Architecture

This project has two independent halves that share nothing at runtime:

**Training** (`training/`) runs in Google Colab, produces a model checkpoint, and pushes it to HF Hub. It never runs in the deployed container.

**Inference** (`inference/`) is a FastAPI service that loads a checkpoint from HF Hub at startup and serves it over HTTP. This is what gets containerised and deployed to AWS App Runner.

### Data flow

```
SAMSum (HF dataset)
  → training/dataset.py   tokenize with "summarize: " prefix, -100 label masking
  → training/train.py     Seq2SeqTrainer → results/flan-t5-samsum/
  → training/push_to_hub.py → HuggingFace Hub

HuggingFace Hub (HF_MODEL_ID env var)
  → inference/model.py    loaded once at startup, cached in module-level globals
  → inference/app.py      POST /summarize calls predict(), then log_inference()
  → inference/logger.py   @weave.op() traces to W&B Weave
```

### Key implementation details

**`inference/model.py`** holds two module-level singletons (`_model`, `_tokenizer`). Tests reset these to `None` before each test to avoid cross-test contamination. The model is loaded in the FastAPI `lifespan` context manager, not at import time.

**`inference/app.py`** uses FastAPI's `lifespan=` parameter (not the deprecated `@app.on_event`). Pydantic v2 request models use `model_config = ConfigDict(strict=False)` (not an inner `Config` class).

**`training/train.py`** uses `processing_class=tokenizer` in `Seq2SeqTrainer` (transformers 5.x) and `eval_strategy="epoch"` — both of these replaced deprecated arguments.

**transformers 5.x breaking changes:** `pipeline("summarization")` and `pipeline("text2text-generation")` were removed. This project uses `AutoModelForSeq2SeqLM` + `AutoTokenizer` + `model.generate()` directly throughout.

### Testing strategy

Fast tests (default, no download) mock `AutoTokenizer`, `AutoModelForSeq2SeqLM`, `torch`, and `weave` at the module level. The `client` fixture also patches these during FastAPI `lifespan` startup.

Slow integration tests (`@pytest.mark.slow`) actually download the model and are excluded from CI (`pytest -m "not slow"`).

## Environment variables

| Variable | Used in | Default |
|---|---|---|
| `HF_MODEL_ID` | `inference/model.py` | `google/flan-t5-base` |
| `WANDB_API_KEY` | weave (implicit) | — |
| `WANDB_PROJECT` | `inference/logger.py` | `flan-t5-summarizer` |
| `HF_TOKEN` | `training/push_to_hub.py` | — |

## Deployment

CI (`deploy.yml`) triggers on pushes to `main` that touch `inference/**`, `pyproject.toml`, or `uv.lock`. It runs fast tests, then builds a `linux/amd64` image and pushes to ECR, then calls `aws apprunner start-deployment`. Required GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPOSITORY`, `APP_RUNNER_SERVICE_ARN`.

The Dockerfile installs only production deps (`uv sync --frozen --no-dev`) and does not include the `training/` package.
