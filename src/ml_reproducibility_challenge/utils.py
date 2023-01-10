import os
import random
import tarfile

import numpy as np
import pandas as pd
import requests
import torch
from pydvl.utils import Dataset
from sklearn.datasets import fetch_openml, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .constants import (
    BREAST_CANCER_OPENML_ID,
    DATA_DIR,
    HOUSE_VOTING_OPENML_ID,
    RANDOM_SEED,
)
from .dataset import FeatureValuationDataset

__all__ = [
    "set_random_seed",
    "create_breast_cancer_dataset",
    "create_wine_dataset",
    "create_house_voting_dataset",
    "create_enron_spam_datasets",
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


def create_enron_spam_datasets() -> tuple[Dataset, Dataset]:
    enron_spam_url = "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz"
    dataset_file = DATA_DIR / "enron1.tar.gz"
    dataset_dir = DATA_DIR / "enron1"
    # Download dataset file, if it does not exist already
    if not dataset_file.is_file():
        with requests.get(enron_spam_url, stream=True) as r:
            r.raise_for_status()
            with dataset_file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    if not dataset_dir.is_dir():
        with tarfile.open(dataset_file) as tar:
            tar.extractall(DATA_DIR)

    # Get list of all ham and spam files
    ham_files = list(dataset_dir.joinpath("ham").glob("*.txt"))
    spam_files = list(dataset_dir.joinpath("spam").glob("*.txt"))

    # Instantiate count vectorizer that will be used to convert
    # the text files into feature vectors
    vectorizer = CountVectorizer(
        input="filename",
        encoding="ISO-8859-1",
        analyzer="char_wb",
        strip_accents="ascii",
        lowercase=True,
    )

    # Generate the input array
    X = vectorizer.fit_transform(ham_files + spam_files).toarray()
    # Generate the target array
    # 0: ham
    # 1: spam
    y = np.concatenate(
        [
            np.zeros(len(ham_files), dtype=np.bool_),
            np.ones(len(spam_files), dtype=np.bool_),
        ]
    )

    # We split the data here in order to have a separate training and testing
    # dataset objects.
    # The first one will be used for computing the valuation
    # The second one will be used for the final performance evaluation

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=0.8, stratify=y_train
    )

    # We flip 20% of target labels
    indices = np.random.choice(np.arange(len(y_train)), size=int(0.2 * len(y_train)))
    y_train_flipped = y_train.copy()
    y_train_flipped[indices] = np.logical_not(y_train_flipped[indices])

    training_dataset = Dataset(
        x_train=x_train,
        y_train=y_train_flipped,
        x_test=x_val,
        y_test=y_val,
    )

    testing_dataset = Dataset(
        x_train=x_train,
        y_train=y_train_flipped,
        x_test=x_test,
        y_test=y_test,
    )
    return training_dataset, testing_dataset
