import os
import random
import tarfile

import numpy as np
import pandas as pd
import requests
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from numpy.typing import NDArray
from pydvl.utils import Dataset
from sklearn.datasets import fetch_openml, load_wine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm.auto import tqdm

from mlrc22.constants import (
    BREAST_CANCER_OPENML_ID,
    DATA_DIR,
    ENRON1_SPAM_DATASET_URL,
    HOUSE_VOTING_OPENML_ID,
    IMAGENET_DOG_LABELS,
    IMAGENET_FISH_LABELS,
    RANDOM_SEED,
)

from .dataset import FeatureValuationDataset

__all__ = [
    "set_random_seed",
    "create_breast_cancer_dataset",
    "create_wine_dataset",
    "create_house_voting_dataset",
    "create_enron_spam_datasets",
    "create_synthetic_dataset",
    "create_dog_vs_fish_dataset",
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


def flip_labels(
    y: NDArray[np.int_], percentage: float, *, random_state: np.random.RandomState
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = random_state.choice(
        np.arange(len(y)), size=int(percentage * len(y)), replace=False
    )
    y = y.copy()
    y[indices] = np.logical_not(y[indices])
    return y, indices


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
    scale = 1 + 10 * random_state.uniform(low=-1.0, high=1.0, size=n_features)
    X *= scale
    feature_mask = random_state.random(n_features) > 0.5
    Xb = X @ feature_mask
    Xb -= np.mean(X)
    pr = 1 / (1 + np.exp(-Xb))
    y = (pr > random_state.random(n_total_samples)).astype(int)

    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    if noise_fraction > 0.0:
        n_noisy_samples = int(noise_fraction * n_train_samples)
        indices = random_state.choice(
            np.arange(n_train_samples),
            size=n_noisy_samples,
            replace=False,
        )
        x_noisy = x_train[indices, :]
        y_noisy = np.logical_not(y_train[indices])
        x_noisy += noise_level * random_state.standard_normal(
            size=x_train[indices].shape
        )
        x_train = np.concatenate([x_train, x_noisy], axis=0)
        y_train = np.concatenate([y_train, y_noisy], axis=0)
        noisy_indices = np.arange(n_train_samples, n_train_samples + n_noisy_samples)
    else:
        noisy_indices = np.array([], dtype=int)

    dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    return dataset, noisy_indices


def download_and_filter_imagenet(seed: int) -> DatasetDict:
    ds = load_dataset("imagenet-1k")

    # Keep only dog and fish labels
    def is_label_dog_fish(x):
        return x["label"] in (IMAGENET_DOG_LABELS + IMAGENET_FISH_LABELS)

    dog_fish_ds = ds.filter(is_label_dog_fish)

    # Assign new labels
    def new_label(x):
        if x["label"] in IMAGENET_DOG_LABELS:
            x["label"] = 0
        else:
            x["label"] = 1
        return x

    dog_fish_ds = dog_fish_ds.map(new_label)

    # Remove 'test' split because it is empty
    assert dog_fish_ds["test"].shape[0] == 0
    dog_fish_ds.pop("test")

    # Subsample the dataset to keep 1200 samples only
    dog_fish_ds["train"] = dog_fish_ds["train"].train_test_split(
        train_size=600, stratify_by_column="label", seed=seed
    )["train"]
    dog_fish_ds["validation"] = dog_fish_ds["validation"].train_test_split(
        train_size=600, stratify_by_column="label", seed=seed
    )["train"]
    return dog_fish_ds


def generate_inception_v3_embeddings(
    ds: DatasetDict,
) -> tuple[
    tuple[NDArray[np.float_], NDArray[np.float_]],
    tuple[NDArray[np.float_], NDArray[np.float_]],
]:
    # Instantiate Inception V3 model with pretrained weights
    pretrained_weights = Inception_V3_Weights.IMAGENET1K_V1
    inception_model = inception_v3(weights=pretrained_weights)
    inception_model.eval()

    # remove last layer
    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    inception_model.fc = Identity()
    preprocess = pretrained_weights.transforms()

    def transform(x):
        try:
            x["image"] = [preprocess(x["image"][0])]
        except RuntimeError:
            x["image"] = [preprocess(x["image"][0].convert("RGB"))]
        return x

    ds.set_transform(transform)

    # Run inference on images
    train_dataloader = DataLoader(ds["train"], batch_size=32)
    val_dataloader = DataLoader(ds["validation"], batch_size=32)

    train_inputs = []
    train_labels = []
    val_inputs = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(train_dataloader, desc="Training Set"):
            image, label = batch["image"], batch["label"]
            output = inception_model(image)
            train_inputs.append(output.numpy())
            train_labels.append(label.numpy())

        for batch in tqdm(val_dataloader, desc="Validation Set"):
            image, label = batch["image"], batch["label"]
            output = inception_model(image)
            val_inputs.append(output.numpy())
            val_labels.append(label.numpy())

    x_train = np.concatenate(train_inputs, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    x_test = np.concatenate(val_inputs, axis=0)
    y_test = np.concatenate(train_labels, axis=0)

    return (x_train, y_train), (x_test, y_test)


def create_dog_vs_fish_dataset(seed: int = RANDOM_SEED) -> Dataset:
    dog_vs_fish_dataset_dir = DATA_DIR / "dog_vs_fish"
    dog_vs_fish_dataset_dir.mkdir(exist_ok=True)

    filtered_imagenet_dataset_dir = dog_vs_fish_dataset_dir / "filtered_imagenet"
    train_array_file = dog_vs_fish_dataset_dir / "train_arrays.npz"
    val_array_file = dog_vs_fish_dataset_dir / "val_arrays.npz"

    if filtered_imagenet_dataset_dir.is_dir():
        dog_fish_ds = load_from_disk(os.fspath(filtered_imagenet_dataset_dir))
    else:
        dog_fish_ds = download_and_filter_imagenet(seed=seed)
        # Save dataset to disk to avoid redoing the work
        dog_fish_ds.save_to_disk(filtered_imagenet_dataset_dir)

    if train_array_file.is_file() and val_array_file.is_file():
        array_dict = np.load(train_array_file)
        x_train, y_train = array_dict["x_train"], array_dict["y_train"]
        array_dict = np.load(val_array_file)
        x_test, y_test = array_dict["x_test"], array_dict["y_test"]
    else:
        (x_train, y_train), (x_test, y_test) = generate_inception_v3_embeddings(
            dog_fish_ds
        )
        # Save arrays to disk to avoid redoing the work
        np.savez(train_array_file, x_train=x_train, y_train=y_train)
        np.savez(val_array_file, x_test=x_test, y_test=y_test)

    dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        target_names=["dog", "fish"],
    )

    return dataset
