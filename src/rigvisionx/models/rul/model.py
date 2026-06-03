"""RUL model architecture"""


class RULModel:
    """Predicts remaining useful life of equipment."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build(self):
        """Build the model."""
        pass

    def predict(self, features):
        """Predict remaining useful life."""
        pass
