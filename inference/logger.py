"""W&B Weave observability for inference logging."""

import weave


@weave.op()
def log_inference(query: str, response: str) -> dict:
    """Log a single inference call to W&B Weave."""
    return {"query": query, "response": response}
