# 🤖 Flan-T5 Text Summarization — MLOps Project

> Fine-tune `google/flan-t5-base` on the [SAMSum](https://huggingface.co/datasets/samsum) dialogue dataset, deploy a FastAPI inference API to AWS App Runner, and log every inference to W&B Weave.

---

## 🏗️ Architecture

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

---

## ⚡ Quickstart

### 🔧 Prerequisites

Make sure the following are installed before you begin:

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | with buildx | [docker.com](https://docker.com) |
| AWS CLI | v2 | [AWS docs](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) |

---

## 💻 Local Development

### Step 1 — Clone & install dependencies

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into .venv (reads from uv.lock for reproducibility)
uv sync

# Verify the inference app imports cleanly
uv run python -c "from inference.app import app; print('✅ OK')"
```

### Step 2 — Set up your `.env` file

Copy the template and fill in your credentials (`.env` is already gitignored — never commit it):

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx        # https://huggingface.co/settings/tokens (Read access)
HF_MODEL_ID=google/flan-t5-base         # or your fine-tuned model ID from HF Hub
WANDB_API_KEY=<your-wandb-key>          # https://wandb.ai/settings
WANDB_PROJECT=flan-t5-summarizer
# TORCH_DEVICE=mps                      # optional — auto-detected (mps on Apple Silicon, cpu elsewhere)
```

### Step 3 — Run the API locally

```bash
# Load all env vars from .env and start the server
set -a && source .env && set +a
uv run uvicorn inference.app:app --reload --port 8080
```

> 💡 The first run downloads the model from Hugging Face Hub (~1 GB). Subsequent starts are instant.

### Step 4 — Smoke test the running API

**Health check:**
```bash
curl http://localhost:8080/health
```

**Summarize a conversation:**
```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "A: Hello! B: Hi! How are you? A: Great, thanks!"}'
```

Expected response:
```json
{"summary": "A and B greet each other."}
```

> 💡 More ready-to-use dialogue examples with expected outputs are in [`examples.md`](examples.md).

### Step 5 — Run the test suite

```bash
# Fast unit tests only (no model download)
uv run pytest tests/ -v -m "not slow"

# Full test suite including model inference (slow, requires GPU or patience)
uv run pytest tests/ -v
```

---

## 🐳 Docker

### Build the image

```bash
docker buildx build \
  --platform linux/amd64 \
  -t flan-t5-summarizer:local \
  -f inference/Dockerfile .
```

```bash
docker buildx build --platform linux/amd64 -f inference/Dockerfile.cpu -t flan-t5-cpu:test .
```

> ⚠️ The `--platform linux/amd64` flag is required — AWS App Runner runs on x86 hardware. Building on Apple Silicon without this flag produces an image that will fail to start on App Runner.

### Run the container locally

```bash
docker run -p 8080:8080 \
  -e HF_TOKEN=<your-hf-token> \
  -e HF_MODEL_ID=google/flan-t5-base \
  -e WANDB_API_KEY=<your-wandb-key> \
  -e WANDB_PROJECT=flan-t5-summarizer \
  flan-t5-summarizer:local
```

> 💡 Tip — pass all vars from your `.env` file in one go:
> ```bash
> docker run -p 8080:8080 --env-file .env flan-t5-summarizer:local
> ```

### Verify the container is healthy

```bash
# Health check
curl http://localhost:8080/health

# Test summarization
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "A: Hello! B: Hi! How are you? A: Great, thanks!"}'
```

### Useful Docker commands

```bash
# List running containers
docker ps

# View container logs
docker logs <container-id>

# Stop the container
docker stop <container-id>

# Remove the image
docker rmi flan-t5-summarizer:local
```

---

## 🎓 Training (Google Colab)

Open `notebooks/training_demo.ipynb` in Colab with a **T4 GPU runtime** and run all cells.

The notebook walks through:

1. 📦 Install dependencies
2. 📚 Load & tokenize SAMSum dataset
3. 📊 Baseline ROUGE evaluation (before fine-tuning)
4. 🏋️ Fine-tune with `Seq2SeqTrainer`
5. 📈 Post-training ROUGE comparison
6. 🚀 Push fine-tuned model to Hugging Face Hub

---

## ☁️ AWS Deployment

For a full CLI walkthrough — IAM setup, ECR, App Runner service creation, manual redeploy, rollback, and teardown — see [`aws_deployment.md`](aws_deployment.md).

---

## 🚀 CI/CD — GitHub Actions

Push to `main` triggers `.github/workflows/deploy.yml` automatically:

```
git push origin main
       │
       ▼
  ┌─────────┐     pass      ┌──────────────────┐
  │  test   │ ──────────►  │ build-and-deploy  │
  │ pytest  │               │ ECR → App Runner  │
  └─────────┘               └──────────────────┘
```

| Job | What it does |
|-----|-------------|
| `test` | `uv run pytest tests/ -v -m "not slow"` |
| `build-and-deploy` | Builds `linux/amd64` image → pushes to ECR → triggers App Runner deployment |

### 🔑 Required GitHub Secrets

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | e.g. `flan-t5-summarizer` |
| `APP_RUNNER_SERVICE_ARN` | ARN from AWS App Runner console |
| `HF_TOKEN` | Hugging Face read token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `HF_MODEL_ID` | `your-username/flan-t5-samsum-summarizer` |
| `WANDB_API_KEY` | W&B API key from [wandb.ai](https://wandb.ai) |

---

## 📁 Project Structure

```
text-summar-project/
├── inference/
│   ├── app.py              # FastAPI app (pydantic v2, lifespan)
│   ├── model.py            # AutoModelForSeq2SeqLM wrapper + singleton cache
│   ├── logger.py           # W&B Weave inference logging
│   └── Dockerfile          # python:3.12-slim, linux/amd64, non-root
├── training/
│   ├── dataset.py          # SAMSum load + tokenization
│   ├── train.py            # Seq2SeqTrainer fine-tuning
│   ├── evaluate.py         # ROUGE baseline + post-training eval
│   └── push_to_hub.py      # Upload fine-tuned model to HF Hub
├── notebooks/
│   └── training_demo.ipynb # Colab-ready end-to-end training notebook
├── design_workflows/       # Mermaid architecture diagrams
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD: test → build → ECR → App Runner
├── .dockerignore           # Keeps build context < 5 MB (excludes .venv, .git, notebooks …)
├── .env.example            # Env var template — copy to .env and fill in secrets
├── aws_deployment.md       # Step-by-step CLI guide: IAM → ECR → App Runner
├── examples.md             # Ready-to-use curl examples for /summarize
├── pyproject.toml          # Single source of truth for dependencies
└── uv.lock                 # Pinned lockfile — always committed
```

---

## 🛠️ Key Technology Choices

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Best PyTorch 2.10 support |
| transformers | ≥ 5.2.0 | `processing_class=` replaces deprecated `tokenizer=` |
| datasets | ≥ 4.6.0 | Latest stable |
| fastapi | ≥ 0.133.1 | `lifespan=` replaces deprecated `@app.on_event` |
| pydantic | ≥ 2.12.5 | `ConfigDict` replaces inner `Config` class |
| weave | ≥ 0.52.28 | W&B Weave tracing |
| uv | latest | Fast resolver — replaces pip + requirements.txt |
| Docker base | python:3.12-slim | `--platform linux/amd64` for App Runner compatibility |
