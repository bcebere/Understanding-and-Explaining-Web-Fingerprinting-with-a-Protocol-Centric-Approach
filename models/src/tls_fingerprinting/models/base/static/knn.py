# stdlib
from typing import Any

# third party
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    """Classification plugin based on the KNeighborsClassifier classifier."""

    weights = ["uniform", "distance"]
    algorithm = ["auto", "ball_tree", "kd_tree", "brute"]

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: int = 0,
        algorithm: int = 0,
        leaf_size: int = 30,
        p: int = 2,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=KNNClassifier.algorithm[algorithm],
            weights=KNNClassifier.weights[weights],
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "KNNClassifier":
        self.model.fit(X, *args, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "knn"
