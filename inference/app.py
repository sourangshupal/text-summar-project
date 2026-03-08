"""FastAPI inference API for Flan-T5 summarization."""

import os
from contextlib import asynccontextmanager

import weave
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from inference.model import load_model, predict
from inference.logger import log_inference


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(strict=False)

    text: str
    max_length: int = 128


class SummarizeResponse(BaseModel):
    summary: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    project = os.environ.get("WANDB_PROJECT", "flan-t5-summarizer")
    weave.init(project_name=project)
    app.state.model = load_model()
    yield


app = FastAPI(title="Flan-T5 Summarizer API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    summary = predict(request.text, max_new_tokens=request.max_length)
    log_inference(query=request.text, response=summary)
    return SummarizeResponse(summary=summary)


print("Flan-T5 Summarizer API is ready!")
print("Flan-T5 Summarizer API is ready!")