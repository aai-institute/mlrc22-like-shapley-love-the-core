import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml, load_wine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .constants import BREAST_CANCER_OPENML_ID, HOUSE_VOTING_OPENML_ID, RANDOM_SEED
from .dataset import FeatureValuationDataset

__all__ = [
    "set_random_seed",
    "create_breast_cancer_dataset",
    "create_wine_dataset",
    "create_house_voting_dataset",
]


def set_random_seed(seed: int) -> None:
    """Taken verbatim from:
    https://koustuvsinha.com//practices_for_reproducibility/
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_breast_cancer_dataset() -> FeatureValuationDataset:
    X, _ = fetch_openml(data_id=BREAST_CANCER_OPENML_ID, return_X_y=True, parser="auto")
    X = X.drop(columns="id")
    X = X.dropna()
    y = X.pop("class")

    dataset = FeatureValuationDataset.from_arrays(
        X,
        y,
        train_size=0.8,
        random_state=RANDOM_SEED,
    )

    return dataset


def create_wine_dataset() -> FeatureValuationDataset:
    dataset = FeatureValuationDataset.from_sklearn(
        load_wine(),
        train_size=0.8,
        random_state=RANDOM_SEED,
    )
    return dataset


def create_house_voting_dataset() -> FeatureValuationDataset:
    X, y = fetch_openml(data_id=HOUSE_VOTING_OPENML_ID, return_X_y=True, parser="auto")
    # Fill NaN values with most frequent ones
    imputer = SimpleImputer(strategy="most_frequent")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    # Encode categorical features
    X = OrdinalEncoder().fit_transform(X)
    # Encode target labels
    y = LabelEncoder().fit_transform(y)
    dataset = FeatureValuationDataset.from_arrays(
        X,
        y,
        train_size=0.8,
        random_state=RANDOM_SEED,
    )
    return dataset
