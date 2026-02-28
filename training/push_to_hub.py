"""Push fine-tuned model and tokenizer to Hugging Face Hub."""

import os
import sys
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def push(local_model_path: str, repo_id: str):
    """Upload model and tokenizer to the Hub.

    Args:
        local_model_path: Path to the saved checkpoint directory.
        repo_id: Hub repo in the form 'username/model-name'.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")

    login(token=token)

    print(f"Loading model from {local_model_path} …")
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    print(f"Pushing to Hub as {repo_id} …")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    print("Done — model available at https://huggingface.co/" + repo_id)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m training.push_to_hub <local_path> <repo_id>")
        print("Example: python -m training.push_to_hub results/flan-t5-samsum myuser/flan-t5-samsum-summarizer")
        sys.exit(1)
    push(sys.argv[1], sys.argv[2])
