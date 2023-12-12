# third party
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM


class GMMClassifier:
    """Classification plugin based on the GaussianMixture classifier."""

    def __init__(
        self,
        covariance_type: str = "full",
        random_state: int = 0,
        n_init: int = 1,
    ) -> None:
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GMMClassifier":
        self.model = GMM(
            n_components=len(np.unique(y)),
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_proba(X)

    @staticmethod
    def name() -> str:
        return "gmm"
