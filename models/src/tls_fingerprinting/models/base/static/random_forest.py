# stdlib
import multiprocessing
from typing import Any, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:
    """Classification plugin based on Random forests.

    Method:
        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

    Args:
        n_estimators: int
            The number of trees in the forest.
        criterion: str
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        max_features: str
            The number of features to consider when looking for the best split.
        min_samples_split: int
            The minimum number of samples required to split an internal node.
        boostrap: bool
            Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        min_samples_leaf: int
            The minimum number of samples required to be at a leaf node.
    """

    criterions = ["gini", "entropy"]
    features = ["sqrt", "log2", None]

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: int = 0,
        max_features: int = 0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = 3,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=RFClassifier.criterions[criterion],
            max_features=RFClassifier.features[max_features],
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=max(1, int(multiprocessing.cpu_count() / 2)),
        )

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RFClassifier":
        if len(args) < 1:
            raise RuntimeError("please provide y for the fit method")

        X = np.asarray(X)
        y = np.asarray(args[0]).ravel()

        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        return self.model.predict_proba(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "random_forest"
