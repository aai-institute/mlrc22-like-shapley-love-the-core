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

    def __len__(self):
        return self.x_train.shape[1]
