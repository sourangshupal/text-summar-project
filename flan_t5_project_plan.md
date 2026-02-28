# Text Summarization with Flan-T5
### End-to-End Fine-Tuning & Deployment Project Plan
> **Stack:** Google Colab · Hugging Face Hub · AWS App Runner · GitHub Actions · W&B Weave

---

## Quick Reference Card

| Component | Decision |
|---|---|
| **Model** | `google/flan-t5-base` (250M params) |
| **Dataset** | `samsum` (16k dialogues, ~4 MB) |
| **Training Platform** | Google Colab Free Tier (T4 GPU, 12 GB RAM) |
| **Model Registry** | Hugging Face Hub (`push_to_hub`) |
| **AWS Deployment** | AWS App Runner (Docker + FastAPI) |
| **CI/CD** | GitHub Actions (build → test → deploy) |
| **Inference Logging** | Weights & Biases Weave (free tier) |
| **Evaluation Metric** | ROUGE-1, ROUGE-2, ROUGE-L |
| **Estimated Training Time** | ~45 min on Colab T4 (3 epochs, 1k samples) |

---

## 1. Project Overview

This project teaches students the complete MLOps lifecycle for NLP by walking through every stage — from fine-tuning a pre-trained language model to deploying it as a live inference API with full query/response logging. The architecture is deliberately minimal but follows professional modular design principles so each component can be swapped, extended, or replaced independently.

### 1.1 Learning Objectives

- Understand the text-to-text fine-tuning paradigm with encoder-decoder models
- Practice dataset loading, preprocessing, and tokenization with Hugging Face Datasets
- Execute a training loop on a GPU with `Seq2SeqTrainer` and evaluate with ROUGE
- Version and publish a fine-tuned model to the Hugging Face Hub
- Build a containerised inference API and deploy it to AWS using GitHub Actions CI/CD
- Log every inference query and response to a free observability dashboard (W&B Weave)

### 1.2 High-Level Architecture

```
GOOGLE COLAB  ──►  HF HUB  ──►  GITHUB  ──►  AWS APP RUNNER  ──►  W&B WEAVE
  (Training)      (Registry)    (CI/CD)       (Inference API)      (Logging)
```

Each stage is decoupled. Swapping the dataset, changing the model size, or replacing the cloud provider only requires changing one module.

---

## 2. Dataset Selection: SAMSum

### 2.1 Why SAMSum for Colab Free Tier

SAMSum is the most widely used dataset for fine-tuning Flan-T5 on summarization, appears in the official Hugging Face Flan-T5 fine-tuning tutorial, and is small enough to train on Colab's free T4 GPU in under an hour.

| Property | Value |
|---|---|
| **HF Dataset ID** | `samsum` |
| **Total samples** | ~16,369 (train 14,732 / val 818 / test 819) |
| **Avg dialogue length** | ~94 words |
| **Avg summary length** | ~23 words |
| **Download size** | ~4 MB |
| **Format** | `dialogue` (string) + `summary` (string) |
| **License** | CC BY-NC-ND 4.0 |
| **Colab RAM needed** | < 2 GB to load |

### 2.2 Load Command

```python
from datasets import load_dataset

dataset = load_dataset("samsum")

# For Colab free tier demo: use 1,000 training samples
train_small = dataset["train"].select(range(1000))
```

> **Why 1,000 samples?** Colab free tier provides ~12 GB GPU RAM and disconnects after ~90 minutes of inactivity. 1,000 samples trains in ~45 minutes with Flan-T5-base. Students can see a clear ROUGE improvement over the baseline, which is the pedagogical goal.

### 2.3 Why Not CNN/DailyMail?

CNN/DailyMail is the classic benchmark but each article is 600–800 words. Tokenization alone exceeds Colab free RAM for a full training run. SAMSum dialogues are short (4–15 turns), so tokenization fits easily in memory and students can read the raw examples to understand what the model is learning.

---

## 3. Modular Project Structure

### 3.1 Repository Layout

```
flan-t5-summarizer/
├── training/
│   ├── train.py           # Seq2SeqTrainer fine-tuning script
│   ├── dataset.py         # Dataset loading & preprocessing
│   ├── evaluate.py        # ROUGE evaluation
│   └── push_to_hub.py     # Upload model to HF Hub
├── inference/
│   ├── app.py             # FastAPI inference endpoint
│   ├── model.py           # Model loading & predict()
│   ├── logger.py          # W&B Weave query/response logging
│   └── Dockerfile         # Container for App Runner
├── .github/
│   └── workflows/
│       ├── train.yml      # (Optional) trigger training
│       └── deploy.yml     # Build Docker → ECR → App Runner
├── notebooks/
│   └── training_demo.ipynb  # Colab notebook for students
├── tests/
│   └── test_inference.py  # Pytest: model loads, prediction works
└── requirements.txt
```

### 3.2 Module Responsibilities

| Module | Responsibility | Key Libraries |
|---|---|---|
| `dataset.py` | Load SAMSum, prepend `"summarize:"` prefix, tokenize | `datasets`, `transformers` |
| `train.py` | Full fine-tuning loop with `Seq2SeqTrainer` | `transformers`, `evaluate` |
| `evaluate.py` | Compute ROUGE before/after fine-tuning | `evaluate`, `rouge_score` |
| `push_to_hub.py` | Authenticate & push model + tokenizer | `huggingface_hub` |
| `model.py` | Download model from HF Hub, wrap `pipeline()` | `transformers` |
| `app.py` | FastAPI `POST /summarize` endpoint | `fastapi`, `uvicorn` |
| `logger.py` | Log `{query, response, timestamp}` to W&B Weave | `weave` |
| `Dockerfile` | Python 3.11 slim + requirements install | Docker |
| `deploy.yml` | Build image, push to ECR, deploy to App Runner | GitHub Actions |

---

## 4. Module 1 — Fine-Tuning in Google Colab

### 4.1 Colab Setup

Open Google Colab and connect to a T4 GPU runtime: **Runtime → Change runtime type → T4 GPU**. Install dependencies in the first cell:

```bash
!pip install transformers datasets evaluate rouge_score accelerate -q
```

### 4.2 Dataset Preprocessing (`dataset.py`)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

CHECKPOINT = "google/flan-t5-base"
tokenizer  = AutoTokenizer.from_pretrained(CHECKPOINT)

def preprocess(batch):
    inputs = tokenizer(
        ["summarize: " + d for d in batch["dialogue"]],
        max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        batch["summary"],
        max_length=128, truncation=True, padding="max_length"
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset  = load_dataset("samsum")
train_ds = dataset["train"].select(range(1000)).map(preprocess, batched=True)
eval_ds  = dataset["validation"].map(preprocess, batched=True)
```

### 4.3 Training Loop (`train.py`)

```python
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate, numpy as np

model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels    = eval_pred
    decoded_preds    = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels   = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

args = Seq2SeqTrainingArguments(
    output_dir="./results",  num_train_epochs=3,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    eval_strategy="epoch",   save_strategy="epoch",
    learning_rate=5e-4,      predict_with_generate=True,
    fp16=True,               load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    tokenizer=tokenizer, compute_metrics=compute_metrics,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
```

### 4.4 Push to Hugging Face Hub (`push_to_hub.py`)

```python
from huggingface_hub import login

login(token="hf_YOUR_TOKEN")   # Store as a Colab secret

model.push_to_hub("your-username/flan-t5-samsum-summarizer")
tokenizer.push_to_hub("your-username/flan-t5-samsum-summarizer")
```

> **Expected ROUGE Improvement:** Baseline Flan-T5-base on SAMSum: ROUGE-1 ~0.38. After fine-tuning 3 epochs on 1k samples: ROUGE-1 ~0.42–0.45. Students will clearly see measurable improvement, which validates the fine-tuning step.

---

## 5. Module 2 — AWS Deployment: App Runner

### 5.1 Why AWS App Runner (Not Lambda or SageMaker)

| Service | Suitability | Reason |
|---|---|---|
| **App Runner** | ✅ Best for this project | Deploy a container in < 5 min. No VPC, no IAM complexity, no scaling policies. Just push a Dockerfile and get a URL. |
| Lambda | ❌ Not ideal for ML models | Loading a 250M-param model on cold start takes 60+ seconds. Tight 10 GB image limit. |
| SageMaker | ❌ Overkill for demo | Powerful but complex IAM setup, expensive instances ($0.75+/hr), too many concepts for one project. |
| EC2 | ❌ Too much ops overhead | Students must manage OS, security groups, and process management. |
| ECS / Fargate | ⚠️ Good but more complex | Requires cluster + task definition + service + load balancer. More to teach than App Runner. |

> **App Runner Free Tier:** 1 million requests/month free, with automatic scale-to-zero so you pay nothing when the service is idle.

### 5.2 Inference FastAPI App (`inference/app.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, predict
from logger import log_inference

app   = FastAPI(title="Summarizer API")
model = load_model()   # Downloads from HF Hub on startup

class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 128

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    summary = predict(model, req.text, req.max_length)
    log_inference(query=req.text, response=summary)
    return {"summary": summary}

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 5.3 Model Module (`inference/model.py`)

```python
from transformers import pipeline
import os

HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "your-username/flan-t5-samsum-summarizer"
)

def load_model():
    return pipeline("summarization", model=HF_MODEL_ID)

def predict(pipe, text, max_length=128):
    result = pipe(text, max_length=max_length, min_length=20, num_beams=4)
    return result[0]["summary_text"]
```

### 5.4 Dockerfile (`inference/Dockerfile`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY inference/ .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 6. Module 3 — Inference Logging: W&B Weave

### 6.1 Why W&B Weave (Free Tier)

| Service | Free Tier | Best For |
|---|---|---|
| **W&B Weave** | Unlimited traces, free for personal use | Best UX. Logs inputs, outputs, latency with a simple `@weave.op()` decorator. |
| MLflow | Fully free (self-hosted) | Good but requires hosting your own tracking server. |
| Langfuse | Free cloud tier (50k traces/month) | Great OSS alternative. More LLM-specific. Self-hostable. |
| AWS DynamoDB | 25 GB free forever | Simple key-value storage but no dashboard UI. |
| Comet Opik | Free cloud tier | Good LLM tracing. Apache 2.0 open-source. |

> **Recommendation:** Use W&B Weave. It integrates in 3 lines of code, provides a real-time dashboard at wandb.ai that students can view live, and has a genuinely free personal tier with no time limit.

### 6.2 Logger Module (`inference/logger.py`)

```python
import weave, os
from datetime import datetime

# Initialize once at module load
weave.init(os.getenv("WANDB_PROJECT", "flan-t5-summarizer"))

@weave.op()
def log_inference(query: str, response: str) -> dict:
    return {
        "query":     query,
        "response":  response,
        "timestamp": datetime.utcnow().isoformat(),
    }
```

> **What gets logged automatically:** W&B Weave captures the full input text, model output, call latency, timestamp, and any exceptions — all visible at `wandb.ai/your-username/flan-t5-summarizer`.

---

## 7. Module 4 — CI/CD with GitHub Actions

### 7.1 Required GitHub Secrets

| Secret Name | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM credentials for ECR + App Runner |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret (pair with above) |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | Name of your ECR repo (e.g. `flan-t5-summarizer`) |
| `APP_RUNNER_SERVICE_ARN` | ARN of the App Runner service to update |
| `HF_MODEL_ID` | `your-username/flan-t5-samsum-summarizer` |
| `WANDB_API_KEY` | W&B API key for inference logging |

### 7.2 Deployment Workflow (`.github/workflows/deploy.yml`)

```yaml
name: Build and Deploy Inference API

on:
  push:
    branches: [main]
    paths: ["inference/**", "Dockerfile", "requirements.txt"]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/test_inference.py -v

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region:            ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push Docker image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG:    ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push  $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Deploy to App Runner
        run: |
          aws apprunner start-deployment \
            --service-arn ${{ secrets.APP_RUNNER_SERVICE_ARN }}
```

### 7.3 Test Module (`tests/test_inference.py`)

```python
from inference.model import load_model, predict

def test_model_loads():
    model = load_model()
    assert model is not None

def test_predict_returns_string():
    model   = load_model()
    dialogue = "A: Do you want coffee? B: Yes please!"
    result  = predict(model, dialogue)
    assert isinstance(result, str)
    assert len(result) > 5
```

---

## 8. Evaluation Strategy

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the standard metric for summarization. Students compute ROUGE before and after fine-tuning to demonstrate improvement.

| Metric | What It Measures |
|---|---|
| **ROUGE-1** | Unigram (word-level) overlap between prediction and reference |
| **ROUGE-2** | Bigram overlap — measures phrase-level coherence |
| **ROUGE-L** | Longest common subsequence — measures fluency and order |

> **Expected Results:** Flan-T5-base zero-shot on SAMSum: ROUGE-1 ~0.38, ROUGE-L ~0.35. After fine-tuning 3 epochs on 1,000 samples: ROUGE-1 ~0.43–0.46, ROUGE-L ~0.40–0.43.

---

## 9. Project Execution Checklist

### Module 1: Environment & Data
- [ ] Create Hugging Face account and generate a write token
- [ ] Create W&B account and copy API key
- [ ] Create AWS account (free tier); note access key + secret
- [ ] Fork the project GitHub repo and add all secrets
- [ ] Open the training Colab notebook; connect to T4 GPU
- [ ] Run `dataset.py`: load SAMSum, verify 1,000 training samples tokenize correctly

### Module 2: Fine-Tuning
- [ ] Run baseline ROUGE evaluation on raw Flan-T5-base (no fine-tuning)
- [ ] Run `train.py` for 3 epochs; monitor loss in Trainer output
- [ ] Run post-training ROUGE evaluation; compare with baseline
- [ ] Push model to HF Hub; confirm model card is visible

### Module 3: Inference API
- [ ] Run `app.py` locally: `uvicorn app:app --reload`
- [ ] Test `POST /summarize` with a sample dialogue using curl or Postman
- [ ] Verify `log_inference` writes to W&B Weave dashboard
- [ ] Build Docker image locally; confirm it runs on port 8080

### Module 4: CI/CD & Cloud
- [ ] Create ECR repository in AWS Console
- [ ] Create App Runner service pointing to ECR repo
- [ ] Push a change to `inference/app.py`; watch GitHub Actions pipeline run
- [ ] Confirm deployment completes; test the live App Runner URL
- [ ] Send 5 test queries; verify all appear in W&B Weave dashboard

---

## 10. Suggested Teaching Timeline

| Session | Topics Covered |
|---|---|
| **Session 1 (2 hrs)** | Project intro, Flan-T5 architecture review, SAMSum dataset exploration, baseline inference demo |
| **Session 2 (2 hrs)** | Tokenization, `Seq2SeqTrainer` setup, start fine-tuning run (runs in background during lecture) |
| **Session 3 (2 hrs)** | Review ROUGE results, push model to HF Hub, build FastAPI app, test locally |
| **Session 4 (2 hrs)** | Dockerfile, GitHub Actions YAML walkthrough, deploy to App Runner, W&B Weave dashboard demo |
| **Take-home project** | Students fine-tune on a different dataset subset, push their own model version, and share their App Runner URL |

---

## 11. Key Links & Resources

| Resource | URL |
|---|---|
| SAMSum Dataset | [huggingface.co/datasets/samsum](https://huggingface.co/datasets/samsum) |
| Flan-T5-base | [huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base) |
| Official HF Fine-Tuning Tutorial | [philschmid.de/fine-tune-flan-t5](https://www.philschmid.de/fine-tune-flan-t5) |
| W&B Weave Docs | [weave-docs.wandb.ai](https://weave-docs.wandb.ai) |
| AWS App Runner Docs | [docs.aws.amazon.com/apprunner](https://docs.aws.amazon.com/apprunner) |
| GitHub Actions aws-actions | [github.com/aws-actions](https://github.com/aws-actions) |
| ROUGE Metric | [huggingface.co/spaces/evaluate-metric/rouge](https://huggingface.co/spaces/evaluate-metric/rouge) |

---

*Flan-T5 Text Summarization Project · KrishAI Technologies · University of San Diego AI Course*
