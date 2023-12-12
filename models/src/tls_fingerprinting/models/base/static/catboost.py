# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from catboost import CatBoostClassifier

# tls_fingerprinting absolute
from tls_fingerprinting.utils.constants import DEVICE


class CBClassifier:
    """Classification plugin based on the CatBoost framework.

    Method:
        CatBoost provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. It uses Ordered Boosting to overcome over fitting and Symmetric Trees for faster execution.

    Args:
        learning_rate: float
            The learning rate used for training.
        depth: int

        iterations: int
        grow_policy: int

    """

    grow_policies: List[Optional[str]] = [
        None,
        "Depthwise",
        "SymmetricTree",
        "Lossguide",
    ]

    def __init__(
        self,
        n_estimators: Optional[int] = 10,
        depth: Optional[int] = None,
        grow_policy: int = 0,
        random_state: int = 0,
        l2_leaf_reg: float = 3,
        learning_rate: float = 1e-3,
        min_data_in_leaf: int = 1,
        random_strength: float = 1,
        **kwargs: Any
    ) -> None:
        gpu_args = {}

        if DEVICE == "cuda":
            gpu_args = {
                "task_type": "GPU",
            }

        self.model = CatBoostClassifier(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="5gb",
            n_estimators=n_estimators,
            l2_leaf_reg=l2_leaf_reg,
            learning_rate=learning_rate,
            grow_policy=CBClassifier.grow_policies[grow_policy],
            random_state=random_state,
            min_data_in_leaf=min_data_in_leaf,
            random_strength=random_strength,
            od_type="Iter",
            od_wait=1000,
            **gpu_args,
        )

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CBClassifier":
        self.model.fit(X, *args, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "catboost"
