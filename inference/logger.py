"""W&B Weave observability for inference logging."""

import os
import weave

_initialized = False


def _ensure_init():
    global _initialized
    if not _initialized:
        project = os.environ.get("WANDB_PROJECT", "flan-t5-summarizer")
        weave.init(project_name=project)
        _initialized = True


@weave.op()
def log_inference(query: str, response: str) -> dict:
    """Log a single inference call to W&B Weave.

    Args:
        query: The original input text sent to the model.
        response: The generated summary returned by the model.

    Returns:
        Dict with query and response fields (passed through for chaining).
    """
    _ensure_init()
    return {"query": query, "response": response}
