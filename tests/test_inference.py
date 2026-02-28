"""Smoke tests for the inference API.

Fast tests (default): use mocks — no model download required.
Slow tests (opt-in):  require HF model downloaded; run with:  pytest -m slow
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Unit tests — model layer (mocked)
# ---------------------------------------------------------------------------

def test_model_loads():
    """load_model() returns a tokenizer without error (model patched)."""
    mock_tok = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    with patch("inference.model.AutoTokenizer") as mock_tok_cls, \
         patch("inference.model.AutoModelForSeq2SeqLM") as mock_model_cls:
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = mock_model

        # Reset cached singletons between tests
        import inference.model as m
        m._model = None
        m._tokenizer = None

        result = m.load_model()

    assert result is mock_tok


def test_predict_returns_string():
    """predict() returns the decoded string from model.generate()."""
    import torch
    import inference.model as m

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer.decode.return_value = "Amanda will bring cookies tomorrow."

    mock_model = MagicMock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

    m._model = mock_model
    m._tokenizer = mock_tokenizer

    dialogue = (
        "Amanda: I baked cookies. Do you want some?\r\n"
        "Jerry: Sure!\r\n"
        "Amanda: I'll bring you some tomorrow :-)"
    )
    result = m.predict(dialogue)

    assert isinstance(result, str)
    assert len(result) > 0

    # Cleanup
    m._model = None
    m._tokenizer = None


# ---------------------------------------------------------------------------
# HTTP tests — FastAPI endpoints (mocked model)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with model loading patched out."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import inference.model as m

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model

    m._model = None
    m._tokenizer = None

    with patch("inference.model.AutoTokenizer") as mock_tok_cls, \
         patch("inference.model.AutoModelForSeq2SeqLM") as mock_model_cls, \
         patch("inference.model.torch") as mock_torch, \
         patch("inference.logger.weave"):

        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tokenizer.decode.return_value = "B will bring cookies tomorrow."
        mock_torch.no_grad.return_value.__enter__ = lambda s: s
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_model.generate.return_value = MagicMock()

        from inference.app import app
        with TestClient(app) as c:
            yield c

    # Cleanup singletons after test module
    m._model = None
    m._tokenizer = None


def test_health_endpoint(client):
    """GET /health must return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarize_endpoint(client):
    """POST /summarize must return a non-empty summary."""
    payload = {
        "text": (
            "A: Hello! How are you?\r\n"
            "B: I'm fine, thanks. Just got back from the gym.\r\n"
            "A: Nice! What did you work on?\r\n"
            "B: Legs day. Pretty intense."
        )
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


# ---------------------------------------------------------------------------
# Slow integration tests (require actual HF model download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_loads():
    """[SLOW] Actually download and load flan-t5-base."""
    import inference.model as m
    m._model = None
    m._tokenizer = None
    tok = m.load_model()
    assert tok is not None
    m._model = None
    m._tokenizer = None


@pytest.mark.slow
def test_real_predict():
    """[SLOW] Actually run inference with flan-t5-base."""
    import inference.model as m
    m._model = None
    m._tokenizer = None
    result = m.predict("Amanda baked cookies and offered some to Jerry.")
    assert isinstance(result, str) and len(result) > 0
    m._model = None
    m._tokenizer = None
