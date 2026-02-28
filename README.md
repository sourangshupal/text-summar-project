# Flan-T5 Text Summarization — MLOps Project

Fine-tune `google/flan-t5-base` on the [SAMSum](https://huggingface.co/datasets/samsum) dialogue
dataset, deploy a FastAPI inference API to AWS App Runner, and log every inference to W&B Weave.

## Architecture

```
Training (Colab)          Inference (App Runner)
─────────────────         ──────────────────────
SAMSum dataset            FastAPI  POST /summarize
     │                         │
Flan-T5-base                model.py (pipeline)
     │                         │
Seq2SeqTrainer             logger.py (weave.op)
     │                         │
HF Hub ──────────────────► ECR → App Runner
```

## Quickstart

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) 0.10.6+
- Docker (with buildx for cross-platform builds)
- AWS CLI configured

### Local setup

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Verify imports
uv run python -c "from inference.app import app; print('OK')"
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Run the API locally

```bash
HF_MODEL_ID=google/flan-t5-base \
WANDB_API_KEY=<your-key> \
uv run uvicorn inference.app:app --reload --port 8080
```

Health check:
```bash
curl http://localhost:8080/health
```

Summarize:
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "A: Hello! B: Hi! How are you? A: Great, thanks!"}'
```

### Build Docker image locally

```bash
docker buildx build \
  --platform linux/amd64 \
  -t flan-t5-summarizer:local \
  -f inference/Dockerfile .
```

### Run Docker container

```bash
docker run -p 8080:8080 \
  -e HF_MODEL_ID=your-username/flan-t5-samsum-summarizer \
  -e WANDB_API_KEY=<your-key> \
  flan-t5-summarizer:local
```

## Training (Google Colab)

Open `notebooks/training_demo.ipynb` in Colab with a T4 GPU runtime and run all cells.

The notebook covers:
1. Install dependencies
2. Load & tokenize SAMSum
3. Baseline ROUGE evaluation
4. Fine-tune with `Seq2SeqTrainer`
5. Post-training ROUGE comparison
6. Push model to Hugging Face Hub

## CI/CD — GitHub Actions

Push to `main` triggers `.github/workflows/deploy.yml`:

1. **test** — `uv run pytest tests/ -v`
2. **build-and-deploy** — build `linux/amd64` Docker image → push to ECR → trigger App Runner deployment

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | e.g. `flan-t5-summarizer` |
| `APP_RUNNER_SERVICE_ARN` | ARN from AWS App Runner console |
| `HF_MODEL_ID` | `your-username/flan-t5-samsum-summarizer` |
| `WANDB_API_KEY` | W&B API key from wandb.ai |

## Project Structure

```
text-summar-project/
├── training/
│   ├── dataset.py          # SAMSum load + tokenization
│   ├── train.py            # Seq2SeqTrainer fine-tuning
│   ├── evaluate.py         # ROUGE baseline + post-training
│   └── push_to_hub.py      # Upload to HF Hub
├── inference/
│   ├── app.py              # FastAPI app (pydantic v2, lifespan)
│   ├── model.py            # pipeline() wrapper
│   ├── logger.py           # W&B Weave logging
│   └── Dockerfile          # python:3.12-slim, linux/amd64
├── notebooks/
│   └── training_demo.ipynb # Colab-ready notebook
├── tests/
│   └── test_inference.py   # pytest smoke tests
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD pipeline
├── pyproject.toml          # UV-managed dependencies
└── uv.lock                 # Pinned lockfile (committed)
```

## Key Technology Choices (2026)

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 | Best PyTorch 2.10 support |
| transformers | ≥5.2.0 | `processing_class=` replaces `tokenizer=` |
| datasets | ≥4.6.0 | Latest stable |
| fastapi | ≥0.133.1 | `lifespan=` replaces `@app.on_event` |
| pydantic | ≥2.12.5 | `ConfigDict` replaces inner `Config` class |
| weave | ≥0.52.28 | W&B Weave tracing |
| UV | 0.10.6+ | Fast resolver, replaces pip + requirements.txt |
| Docker base | python:3.12-slim | `--platform linux/amd64` for App Runner |
