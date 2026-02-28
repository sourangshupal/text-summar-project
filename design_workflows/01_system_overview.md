# System Overview — High-Level Architecture

This diagram gives a bird's-eye view of the entire project. The system is split into two independent halves that only communicate through the **HuggingFace Hub**: a **Training** side that runs in Google Colab, and an **Inference** side that runs as a containerised FastAPI service on AWS App Runner. GitHub Actions automates the deployment pipeline, and W&B Weave provides end-to-end observability of every prediction request.

```mermaid
graph TD
    %% ── Actors ──────────────────────────────────────────────
    DEV(["👩‍💻 Developer"])
    STUDENT(["🧑‍🎓 Student / Client"])
    DATASET[("📦 SAMSum Dataset\nHuggingFace Datasets")]

    %% ── Training side ────────────────────────────────────────
    subgraph TRAIN["☁️  Training  (Google Colab)"]
        direction TB
        T1["dataset.py\nLoad & tokenise SAMSum"]
        T2["train.py\nSeq2SeqTrainer · 3 epochs · fp16"]
        T3["evaluate.py\nROUGE baseline vs fine-tuned"]
        T4["push_to_hub.py\nSave checkpoint → Hub"]
        T1 --> T2 --> T3 --> T4
    end

    %% ── HuggingFace Hub (bridge) ─────────────────────────────
    HUB[("🤗 HuggingFace Hub\nModel Registry")]

    %% ── CI/CD ────────────────────────────────────────────────
    subgraph CICD["⚙️  GitHub Actions  (deploy.yml)"]
        direction TB
        C1["pytest · unit tests"]
        C2["docker buildx\nlinux/amd64"]
        C3["Push image → ECR"]
        C4["apprunner start-deployment"]
        C1 --> C2 --> C3 --> C4
    end

    %% ── Inference side ───────────────────────────────────────
    subgraph INFER["🚀  Inference  (AWS App Runner)"]
        direction TB
        I1["app.py\nFastAPI · lifespan startup"]
        I2["model.py\nload_model() singleton\nFlan-T5 fine-tuned"]
        I3["logger.py\nW&B Weave tracing"]
        I1 --> I2
        I1 --> I3
    end

    %% ── Observability ────────────────────────────────────────
    WEAVE(["📊 W&B Weave\nObservability"])

    %% ── Connections ──────────────────────────────────────────
    DEV -->|"runs notebooks"| TRAIN
    DATASET -->|"datasets.load_dataset()"| T1
    T4 -->|"push_to_hub()"| HUB
    HUB -->|"from_pretrained()"| I2

    DEV -->|"git push main"| CICD
    CICD -->|"deploy new image"| INFER

    STUDENT -->|"POST /summarize"| I1
    I3 -->|"log traces"| WEAVE

    %% ── Styles ───────────────────────────────────────────────
    classDef training   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef inference  fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef cicd       fill:#ffedd5,stroke:#f97316,color:#7c2d12
    classDef external   fill:#f3f4f6,stroke:#9ca3af,color:#374151
    classDef hub        fill:#fef9c3,stroke:#eab308,color:#713f12

    class T1,T2,T3,T4 training
    class I1,I2,I3 inference
    class C1,C2,C3,C4 cicd
    class DEV,STUDENT,DATASET,WEAVE external
    class HUB hub
```

**Key takeaways for students:**
- The two halves are **decoupled** — training never calls the API and the API never calls the trainer.
- HuggingFace Hub acts as the **model registry** / handoff point.
- W&B Weave is **non-blocking**: the API works even if the tracing call fails.
- GitHub Actions only fires when code in `inference/**` changes — training notebooks are deployed manually.
