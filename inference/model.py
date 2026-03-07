"""Model wrapper for inference (transformers 5.x compatible)."""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_model = None
_tokenizer = None

# Device selection:
#   - macOS arm64 : use MPS (Apple Silicon GPU) — CPU BLAS crashes with
#     SIGBUS on macOS 26 / PyTorch 2.10 due to an Accelerate framework bug.
#   - Linux / App Runner : CPU (no GPU available).
#   - Override anytime with TORCH_DEVICE env var.
def _select_device() -> str:
    override = os.environ.get("TORCH_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_DEVICE = _select_device()


def load_model():
    """Load and return the model + tokenizer (cached singleton).

    Returns the tokenizer so callers have a handle, but the model is also
    cached globally for use by predict().
    """
    global _model, _tokenizer
    if _model is not None:
        return _tokenizer  # already loaded

    model_id = os.environ.get("HF_MODEL_ID", "google/flan-t5-base")
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(_DEVICE)
    _model.eval()
    return _tokenizer


def predict(text: str, max_new_tokens: int = 128) -> str:
    """Run summarization on the given text.

    Args:
        text: Input dialogue / document to summarize.
        max_new_tokens: Maximum tokens in the generated summary.

    Returns:
        Generated summary string.
    """
    load_model()  # ensure loaded
    prefixed = "summarize: " + text
    inputs = {
        k: v.to(_DEVICE)
        for k, v in _tokenizer(
            prefixed,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).items()
    }
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
