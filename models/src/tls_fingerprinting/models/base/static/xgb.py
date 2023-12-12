# stdlib
from typing import Any, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoostClassifier:
    """Classification plugin based on the XGBoost classifier.

    Method:
        Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm has a robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune.

    Args:
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        max_depth: int
            Maximum depth of a tree.
        reg_lambda: float
            L2 regularization term on weights (xgb’s lambda).
        reg_alpha: float
            L1 regularization term on weights (xgb’s alpha).
        colsample_bytree: float
            Subsample ratio of columns when constructing each tree.
        colsample_bynode: float
             Subsample ratio of columns for each split.
        colsample_bylevel: float
             Subsample ratio of columns for each level.
        subsample: float
            Subsample ratio of the training instance.
        lr: float
            Boosting learning rate
        booster: str
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
        max_bin: int
            Number of bins for histogram construction.
        random_state: float
            Random number seed.
    """

    booster = ["gbtree", "gblinear", "dart"]
    grow_policy = ["depthwise", "lossguide"]

    def __init__(
        self,
        n_estimators: int = 50,
        reg_lambda: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        max_depth: Optional[int] = 3,
        subsample: Optional[float] = None,
        lr: Optional[float] = None,
        min_child_weight: Optional[int] = None,
        max_bin: int = 256,
        booster: int = 0,
        grow_policy: int = 0,
        nthread: int = 4,
        random_state: int = 0,
        eta: float = 0.3,
        **kwargs: Any
    ) -> None:

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            max_depth=max_depth,
            subsample=subsample,
            lr=lr,
            min_child_weight=min_child_weight,
            max_bin=max_bin,
            eta=eta,
            verbosity=0,
            grow_policy=XGBoostClassifier.grow_policy[grow_policy],
            random_state=random_state,
            nthread=nthread,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "XGBoostClassifier":
        y = np.asarray(args[0])
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.encoder.inverse_transform(self.model.predict(X, *args, **kwargs))

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "xgboost"
