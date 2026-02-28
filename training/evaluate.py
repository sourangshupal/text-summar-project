"""ROUGE evaluation: baseline vs fine-tuned Flan-T5 (transformers 5.x compatible)."""

import sys
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "google/flan-t5-base"


def _load(model_path_or_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_id)
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, text: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _compute_rouge(tokenizer, model, samples, max_new_tokens: int = 128) -> dict:
    rouge = evaluate.load("rouge")
    predictions = [_generate(tokenizer, model, s["dialogue"], max_new_tokens) for s in samples]
    references = [s["summary"] for s in samples]
    scores = rouge.compute(predictions=predictions, references=references)
    return {k: round(v * 100, 2) for k, v in scores.items()}


def evaluate_baseline(num_samples: int = 100) -> dict:
    """Evaluate the unmodified Flan-T5-base on SAMSum test set."""
    dataset = load_dataset("samsum", split=f"test[:{num_samples}]")
    tokenizer, model = _load(MODEL_ID)
    scores = _compute_rouge(tokenizer, model, dataset)
    print("=== Baseline ROUGE scores ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    return scores


def evaluate_finetuned(model_path: str, num_samples: int = 100) -> dict:
    """Evaluate a fine-tuned checkpoint on SAMSum test set."""
    dataset = load_dataset("samsum", split=f"test[:{num_samples}]")
    tokenizer, model = _load(model_path)
    scores = _compute_rouge(tokenizer, model, dataset)
    print(f"=== Fine-tuned ROUGE scores ({model_path}) ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    return scores


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "results/flan-t5-samsum"
    baseline = evaluate_baseline()
    finetuned = evaluate_finetuned(model_path)
    print("\n=== Improvement ===")
    for k in baseline:
        delta = finetuned[k] - baseline[k]
        print(f"  {k}: {delta:+.2f}")
