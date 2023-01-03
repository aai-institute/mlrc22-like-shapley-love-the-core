from typing import Iterable

import numpy as np
from pydvl.utils.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

__all__ = ["FeatureValuationDataset"]


class FeatureValuationDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indices = np.arange(self.x_train.shape[1])
        self._data_names = self._indices

    def get_training_data(self, train_indices: Iterable[int] | None = None):
        if train_indices is None:
            x = self.x_train
        else:
            x = self.x_train[:, train_indices]
        y = self.y_train
        return x, y

    def get_test_data(self, test_indices: Iterable[int] | None = None):
        if test_indices is None:
            x = self.x_train
        else:
            x = self.x_train[:, test_indices]
        y = self.y_train
        return x, y

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
    ) -> "FeatureValuationDataset":
        """TODO: Remove this method once version 0.4.0 of pyDVL is released."""
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            train_size=train_size,
            random_state=random_state,
            stratify=data.target if stratify_by_target else None,
        )
        return cls(
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names=data.get("feature_names"),
            target_names=data.get("target_names"),
            description=data.get("DESCR"),
        )
