import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.domain.exceptions import ClassificationError
from src.domain.models import IntentPrediction


class TransformerClassifierAdapter:
    """Implements IntentClassifierPort using a local Hugging Face checkpoint.

    id2label is read from the model's own config.json, produced by
    train/train_intent_classifier.py, so no label file has to be kept in sync.
    """

    def __init__(self, model_path: str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        # low_cpu_mem_usage avoids materializing a second full copy of the
        # state dict while loading - the difference between fitting and OOMing
        # on a 512MB host (e.g. Render's free tier) for a ~270MB checkpoint.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path, low_cpu_mem_usage=True
        )
        self._model.eval()

    def predict(self, text: str) -> IntentPrediction:
        start = time.perf_counter()
        try:
            inputs = self._tokenizer(
                text, truncation=True, padding="max_length", max_length=96, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                label_id = int(torch.argmax(probs).item())
                confidence = float(probs[label_id].item())
        except Exception as exc:
            raise ClassificationError(str(exc)) from exc

        latency_ms = (time.perf_counter() - start) * 1000
        intent = self._model.config.id2label[label_id]
        return IntentPrediction(intent=intent, confidence=confidence, latency_ms=latency_ms)
