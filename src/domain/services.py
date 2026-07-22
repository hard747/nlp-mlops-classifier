from src.domain.models import IntentPrediction
from src.domain.ports import IntentClassifierPort


class ClassificationService:
    """Orchestrates intent classification through a classifier port only."""

    def __init__(self, classifier: IntentClassifierPort) -> None:
        self._classifier = classifier

    def classify(self, text: str) -> IntentPrediction:
        return self._classifier.predict(text)
