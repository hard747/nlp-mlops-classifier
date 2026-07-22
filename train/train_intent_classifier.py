"""Phase 2 - product model: customer support intent classification.

Fine-tunes DistilBERT on bitext/Bitext-customer-support-llm-chatbot-training-dataset
(26,872 rows, single train split, 27 balanced intent classes) using the `intent`
column as the label. This is the model actually served by the API; Phase 1
(train_phase1_benchmark.py) was a GPU pipeline benchmark on an unrelated dataset.
"""
from __future__ import annotations

import os

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from train.common import detect_device

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
CHECKPOINT = "distilbert-base-uncased"
OUTPUT_DIR = "./models/intent_classifier_customer_support"
MAX_LENGTH = 96
SEED = 42

# Experiment tracking: transformers' Trainer has a built-in MLflowCallback that
# activates automatically when "mlflow" is importable and these env vars are
# set. `docker compose up -d mlflow` (see infra/docker-compose.yml) exposes
# the tracking server at this URL; override MLFLOW_TRACKING_URI if it runs
# elsewhere. os.environ.setdefault so an externally-set value always wins.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "intent-classifier-phase2")
os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "TRUE")


def build_label_mappings(labels: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Builds id2label/label2id from a sorted, deduplicated label space.

    Sorting makes the mapping deterministic across runs, independent of the
    order labels first appear in the dataset.
    """
    unique_sorted = sorted(set(labels))
    id2label = {i: label for i, label in enumerate(unique_sorted)}
    label2id = {label: i for i, label in enumerate(unique_sorted)}
    return id2label, label2id


def main() -> None:
    device = detect_device()
    print(f"Training device: {device}")

    print(f"Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.class_encode_column("intent")

    id2label, label2id = build_label_mappings(dataset.features["intent"].names)
    num_labels = len(id2label)
    print(f"Detected {num_labels} intent classes.")

    split = dataset.train_test_split(test_size=0.1, stratify_by_column="intent", seed=SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize(batch):
        return tokenizer(batch["instruction"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)
    train_dataset = train_dataset.rename_column("intent", "labels")
    eval_dataset = eval_dataset.rename_column("intent", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {"accuracy": accuracy["accuracy"], "f1_macro": f1["f1"]}

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=device == "cuda",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to=["mlflow"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Long GPU runs can get interrupted (driver reset, power loss, etc.) - resume
    # from the last epoch checkpoint under output_dir instead of restarting cold.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            print(f"Found existing checkpoint, resuming from: {last_checkpoint}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    print(f"Saving final model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done. Promote the artifact manually with:")
    print(f"  cp -r {OUTPUT_DIR}/* src/infrastructure/ml_model/weights/")


if __name__ == "__main__":
    main()
