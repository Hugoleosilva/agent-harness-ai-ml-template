"""Model layer: start with a trivial baseline that any real model must beat."""
import numpy as np


class MajorityBaseline:
    """Classification baseline: always predicts the most frequent training label."""

    def fit(self, y_train):
        values, counts = np.unique(y_train, return_counts=True)
        self.label_ = values[counts.argmax()]
        return self

    def predict(self, n):
        return np.full(n, self.label_)
