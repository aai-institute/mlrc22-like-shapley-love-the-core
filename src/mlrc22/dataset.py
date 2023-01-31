import os
import tarfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from datasets import load_from_disk
from numpy._typing import NDArray
from pydvl.utils.dataset import Dataset
from sklearn.datasets import fetch_openml, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from mlrc22.constants import (
    BREAST_CANCER_OPENML_ID,
    DATA_DIR,
    ENRON1_SPAM_DATASET_URL,
    HOUSE_VOTING_OPENML_ID,
    RANDOM_SEED,
)
from mlrc22.utils import (
    download_and_filter_imagenet,
    flip_labels,
    generate_inception_v3_embeddings,
)

__all__ = [
    "FeatureValuationDataset",
    "create_breast_cancer_dataset",
    "create_wine_dataset",
    "create_house_voting_dataset",
    "create_enron_spam_datasets",
    "create_synthetic_dataset",
    "create_dog_vs_fish_dataset",
]


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


def create_dog_vs_fish_dataset(seed: int = RANDOM_SEED) -> Dataset:
    dog_vs_fish_dataset_dir = DATA_DIR / "dog_vs_fish"
    dog_vs_fish_dataset_dir.mkdir(exist_ok=True)

    filtered_imagenet_dataset_dir = dog_vs_fish_dataset_dir / "filtered_imagenet"
    arrays_file = dog_vs_fish_dataset_dir / "arrays.npz"

    if filtered_imagenet_dataset_dir.is_dir():
        dog_fish_ds = load_from_disk(os.fspath(filtered_imagenet_dataset_dir))
    else:
        dog_fish_ds = download_and_filter_imagenet(seed=seed)
        # Save dataset to disk to avoid redoing the work
        dog_fish_ds.save_to_disk(filtered_imagenet_dataset_dir)

    if arrays_file.is_file():
        array_dict = np.load(arrays_file)
        X, y = array_dict["X"], array_dict["y"]
    else:
        X, y = generate_inception_v3_embeddings(dog_fish_ds)
        # Save arrays to disk to avoid redoing the work
        np.savez(arrays_file, X=X, y=y)

    dataset = Dataset.from_arrays(
        X=X,
        y=y,
        train_size=0.5,
        stratify_by_target=True,
    )

    dataset.target_names = ["dog", "fish"]

    return dataset


def create_synthetic_dataset(
    n_features: int,
    n_train_samples: int,
    n_test_samples: int,
    *,
    noise_level: float = 0.0,
    noise_fraction: float = 0.0,
    random_state: np.random.RandomState,
) -> tuple[Dataset, NDArray[np.int_]]:
    n_total_samples = n_train_samples + n_test_samples
    X = random_state.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_total_samples,
    )
    feature_mask = random_state.random(n_features) > 0.5
    Xb = X @ feature_mask
    Xb -= np.mean(X)
    pr = 1 / (1 + np.exp(-Xb))
    y = (pr > random_state.random(n_total_samples)).astype(int)

    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    if noise_fraction > 0.0:
        n_noisy_samples = int(noise_fraction * n_train_samples)
        noisy_indices = random_state.choice(
            np.arange(n_train_samples),
            size=n_noisy_samples,
            replace=False,
        )
        scale = noise_level * np.std(x_train)
        x_train[noisy_indices, :] += random_state.normal(
            scale=scale, size=x_train[noisy_indices].shape
        )
    else:
        noisy_indices = np.array([], dtype=int)

    dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    return dataset, noisy_indices


def create_breast_cancer_dataset(
    train_size: float = 0.8, *, random_state: np.random.RandomState = None
) -> FeatureValuationDataset:
    X, _ = fetch_openml(data_id=BREAST_CANCER_OPENML_ID, return_X_y=True, parser="auto")
    X = X.drop(columns="id")
    X = X.dropna()
    y = X.pop("class")

    dataset = FeatureValuationDataset.from_arrays(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
    )

    return dataset


def create_wine_dataset(
    train_size: float = 0.8, *, random_state: np.random.RandomState = None
) -> FeatureValuationDataset:
    dataset = FeatureValuationDataset.from_sklearn(
        load_wine(),
        train_size=train_size,
        random_state=random_state,
    )
    return dataset


def create_house_voting_dataset(
    train_size: float = 0.8, *, random_state: np.random.RandomState = None
) -> FeatureValuationDataset:
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
        train_size=train_size,
        random_state=random_state,
    )
    return dataset


def create_enron_spam_datasets(
    flip_percentage: float = 0.2, *, random_state: np.random.RandomState
) -> tuple[Dataset, Dataset, NDArray[np.int_]]:
    dataset_file = DATA_DIR / "enron1.tar.gz"
    dataset_dir = DATA_DIR / "enron1"
    # Download dataset file, if it does not exist already
    if not dataset_file.is_file():
        with requests.get(ENRON1_SPAM_DATASET_URL, stream=True) as r:
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

    # We only use a 1000 data points in total just like in the paper
    X, _, y, _ = train_test_split(
        X, y, stratify=y, train_size=1000, random_state=random_state
    )

    # We split the data here in order to have a separate training and testing
    # dataset objects.
    # The first one will be used for computing the valuation
    # The second one will be used for the final performance evaluation

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        stratify=y,
        random_state=random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        train_size=0.7,
        stratify=y_train,
        random_state=random_state,
    )

    # We flip a certain percentage of target labels in the training data
    y_train_flipped, flipped_indices = flip_labels(
        y_train, flip_percentage, random_state=random_state
    )

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
    return training_dataset, testing_dataset, flipped_indices
