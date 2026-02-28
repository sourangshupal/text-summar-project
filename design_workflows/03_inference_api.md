# Inference API — Request Lifecycle

This sequence diagram shows what happens from the moment the container starts up to a completed summarisation response. It highlights two important design decisions: the **lifespan singleton** (the model is loaded exactly once, not per request) and the **W&B Weave tracing** (every prediction is logged automatically via a decorator).

```mermaid
sequenceDiagram
    autonumber

    participant Docker as 🐳 Docker / App Runner
    participant App    as app.py<br/>(FastAPI)
    participant Model  as model.py<br/>(load_model · predict)
    participant HF     as 🤗 HuggingFace Hub
    participant Weave  as 📊 W&B Weave
    participant Client as 🧑‍💻 Client

    %% ── Container startup ────────────────────────────────────
    rect rgb(219, 234, 254)
        Note over Docker,HF: Container startup  (runs once)
        Docker->>App: uvicorn starts, lifespan() called
        App->>Model: load_model()
        Model->>HF: AutoTokenizer.from_pretrained()
        Model->>HF: T5ForConditionalGeneration.from_pretrained()
        HF-->>Model: tokenizer + model weights
        Model-->>App: _tokenizer, _model cached as module globals
        Note over App,Model: Singleton pattern — globals set once,<br/>reused for every subsequent request
    end

    %% ── Health check ─────────────────────────────────────────
    rect rgb(220, 252, 231)
        Note over Client,App: Health check  (GET /health)
        Client->>App: GET /health
        App-->>Client: 200 {"status": "ok"}
        Note over App: No model call — just confirms<br/>the server is alive
    end

    %% ── Summarise request ────────────────────────────────────
    rect rgb(255, 237, 213)
        Note over Client,Weave: Summarise request  (POST /summarize)
        Client->>App: POST /summarize<br/>{"text": "Alice: Hi...<br/> Bob: Hello..."}
        App->>App: Pydantic validates SummarizeRequest<br/>(text: str, non-empty)

        alt Validation fails
            App-->>Client: 422 Unprocessable Entity<br/>{"detail": [...]}
        else Validation passes
            App->>Model: predict(text)
            Note over Model: @weave.op() decorator fires<br/>— span starts automatically
            Model->>Model: "summarize: " + text
            Model->>Model: tokenizer(input, return_tensors="pt")
            Model->>Model: model.generate(max_new_tokens=128)
            Model->>Model: tokenizer.decode(skip_special_tokens=True)
            Model->>Weave: log inputs + output + latency
            Weave-->>Model: ack (non-blocking)
            Model-->>App: summary: str
            App-->>Client: 200 {"summary": "Alice greeted Bob."}
        end
    end

    %% ── Error path ───────────────────────────────────────────
    rect rgb(254, 226, 226)
        Note over Client,App: Unhandled error path
        Client->>App: POST /summarize (server error)
        App->>Model: predict(text)
        Model-->>App: raises Exception
        App-->>Client: 500 Internal Server Error
        Note over App: FastAPI default error handler.<br/>Weave span marked as failed.
    end
```

**Key files and their roles:**

| File | Responsibility |
|---|---|
| `app.py` | FastAPI app, `lifespan` context manager, route handlers, Pydantic models |
| `model.py` | `load_model()` singleton loader, `predict()` decorated with `@weave.op()` |
| `logger.py` | `weave.init()` call; imported at startup so Weave is ready before any request |

**Design decisions to discuss with students:**

- **Why a singleton?** Loading a transformer model takes 2–5 seconds and ~500 MB of RAM. Doing it per request would make the API unusably slow.
- **Why `lifespan` instead of a global `import`?** `lifespan` is the FastAPI-recommended pattern and guarantees the model is ready *before* the first request is accepted. It also handles graceful shutdown.
- **Why `@weave.op()`?** The decorator is non-intrusive — `predict()` doesn't know it's being traced. This keeps business logic clean.
