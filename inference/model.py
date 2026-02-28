"""Model wrapper for inference (transformers 5.x compatible)."""

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_model = None
_tokenizer = None


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
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
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
    inputs = _tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
