# CI/CD & Deployment Pipeline

This diagram shows the automated path from a developer pushing code to a live updated service on AWS App Runner. The pipeline has two jobs: a fast **test** gate that must pass before anything is deployed, and a **build-and-deploy** job that packages the service into a Docker image and tells App Runner to use it.

```mermaid
graph LR
    %% ── Trigger ──────────────────────────────────────────────
    subgraph TRIGGER["🔔  Trigger"]
        GH["git push\nbranch: main\npaths:\n• inference/**\n• pyproject.toml\n• uv.lock"]
    end

    %% ── Job 1: Test ──────────────────────────────────────────
    subgraph TEST["🧪  Job: test"]
        direction TB
        T1["actions/checkout@v4"]
        T2["Setup Python 3.12\nastral-sh/setup-uv@v3"]
        T3["uv sync --frozen\n(install deps from lockfile)"]
        T4["pytest -m 'not slow'\n(unit tests only)"]
        T1 --> T2 --> T3 --> T4
    end

    %% ── Job 2: Build & Deploy ────────────────────────────────
    subgraph BUILD["🐳  Job: build-and-deploy\n(needs: test)"]
        direction TB
        B1["aws-actions/configure-aws-credentials\n🔑 AWS_ACCESS_KEY_ID\n🔑 AWS_SECRET_ACCESS_KEY\n🔑 AWS_REGION"]
        B2["aws-actions/amazon-ecr-login\n🔑 ECR_REGISTRY"]
        B3["docker buildx build\n--platform linux/amd64\n-t ECR_REGISTRY/IMAGE:SHA\n-t ECR_REGISTRY/IMAGE:latest\n./inference/Dockerfile"]
        B4["docker push\nECR_REGISTRY/IMAGE:SHA\nECR_REGISTRY/IMAGE:latest"]
        B5["aws apprunner\nstart-deployment\n🔑 APP_RUNNER_SERVICE_ARN"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    %% ── App Runner Runtime ───────────────────────────────────
    subgraph RUNTIME["🚀  AWS App Runner Runtime"]
        direction TB
        R1["Pull new image\nfrom ECR"]
        R2["Start container\nuvicorn app:app --host 0.0.0.0"]
        R3["Health check\nGET /health → 200"]
        R4[["✅ Live\nhttps://your-service.awsapprunner.com"]]
        R1 --> R2 --> R3 --> R4
    end

    %% ── Flow ─────────────────────────────────────────────────
    GH -->|"workflow_dispatch\nor push event"| TEST
    TEST -->|"all tests pass ✅"| BUILD
    TEST -->|"tests fail ❌\n→ pipeline stops"| FAIL["🛑 Deploy blocked"]
    BUILD --> RUNTIME

    %% ── Styles ───────────────────────────────────────────────
    classDef cicd     fill:#ffedd5,stroke:#f97316,color:#7c2d12
    classDef infra    fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef external fill:#f3f4f6,stroke:#9ca3af,color:#374151
    classDef fail     fill:#fee2e2,stroke:#ef4444,color:#7f1d1d

    class T1,T2,T3,T4 cicd
    class B1,B2,B3,B4,B5 cicd
    class R1,R2,R3,R4 infra
    class GH external
    class FAIL fail
```

**Required GitHub Secrets:**

| Secret name | Used by | Purpose |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `configure-aws-credentials` | AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | `configure-aws-credentials` | AWS IAM secret |
| `AWS_REGION` | `configure-aws-credentials` | e.g. `us-east-1` |
| `ECR_REGISTRY` | `docker buildx`, `docker push` | AWS account ECR URL |
| `APP_RUNNER_SERVICE_ARN` | `aws apprunner start-deployment` | ARN of the App Runner service |

**Key files:**

| File | Role |
|---|---|
| `.github/workflows/deploy.yml` | Defines the full CI/CD pipeline |
| `inference/Dockerfile` | Multi-stage image built with `linux/amd64` for App Runner |

**Design decisions to discuss with students:**

- **Path-filtered trigger** — The workflow only runs when inference code or dependencies change. Pushing a notebook change won't trigger a redeploy.
- **`uv sync --frozen`** — Pins exact dependency versions from the lockfile, making CI reproducible.
- **Two image tags (`$SHA` + `latest`)** — `$SHA` gives traceability (you can roll back to any commit); `latest` makes it easy to pull the current image manually.
- **`linux/amd64` platform flag** — App Runner runs on x86 hardware. Without this flag, building on Apple Silicon (arm64) would produce an image that fails to start.
