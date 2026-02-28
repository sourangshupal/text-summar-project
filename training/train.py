"""Fine-tune Flan-T5-base on SAMSum using Seq2SeqTrainer."""

import os
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from training.dataset import get_tokenized_dataset, get_tokenizer, MODEL_ID

OUTPUT_DIR = "results/flan-t5-samsum"


def train():
    tokenizer = get_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    tokenized_dataset = get_tokenized_dataset(tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",          # replaces deprecated evaluation_strategy
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=True,                       # use T4 / A100 mixed precision
        logging_dir="results/logs",
        logging_steps=100,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,      # transformers 5.x: replaces tokenizer=
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    return trainer


if __name__ == "__main__":
    train()
