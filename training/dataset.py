"""SAMSum dataset loading and tokenization for Flan-T5 fine-tuning."""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "google/flan-t5-base"
INPUT_MAX_LENGTH = 512
LABEL_MAX_LENGTH = 128


def load_samsum():
    """Load the SAMSum dialogue summarization dataset."""
    return load_dataset("samsum")


def get_tokenizer():
    """Load the Flan-T5 tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize_function(examples, tokenizer):
    """Tokenize inputs and labels for seq2seq training."""
    inputs = ["summarize: " + dialogue for dialogue in examples["dialogue"]]
    model_inputs = tokenizer(
        inputs,
        max_length=INPUT_MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=LABEL_MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Replace padding token id in labels with -100 so it's ignored in loss
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs


def get_tokenized_dataset(tokenizer=None):
    """Return the fully tokenized SAMSum dataset."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    dataset = load_samsum()
    tokenized = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized
