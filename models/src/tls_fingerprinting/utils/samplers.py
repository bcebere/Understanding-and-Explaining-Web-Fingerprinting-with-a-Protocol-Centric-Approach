# stdlib
from typing import Any, Generator, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pydantic import validate_arguments


class BaseSampler(torch.utils.data.sampler.Sampler):
    """DataSampler samples the conditional vector and corresponding data."""

    def get_dataset_conditionals(self) -> np.ndarray:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional(self, batch: int, **kwargs: Any) -> Optional[Tuple]:
        return None

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def sample_conditional_for_class(self, batch: int, c: int) -> Optional[np.ndarray]:
        return None

    def conditional_dimension(self) -> int:
        """Return the total number of categories."""
        return 0

    def conditional_probs(self) -> Optional[np.ndarray]:
        """Return the total number of categories."""
        return None

    def train_test(self) -> Tuple:
        raise NotImplementedError()


class ImbalancedDatasetSampler(BaseSampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset"""

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, labels: List, train_size: float = 0.8) -> None:
        # if indices is not provided, all elements in the dataset will be considered
        indices = list(range(len(labels)))

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_train_samples = len(indices)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = indices

        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        print(label_to_count)

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())
        self.indices = indices
        print(indices)

    def __iter__(self) -> Generator:
        return (
            self.indices[i]
            for i in torch.multinomial(
                self.weights, self.num_train_samples, replacement=True
            )
        )

    def __len__(self) -> int:
        return len(self.indices)
