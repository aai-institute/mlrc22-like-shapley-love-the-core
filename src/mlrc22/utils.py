import logging
import os
import random
from typing import Tuple

import numpy as np
import seaborn as sns
import torch
from datasets import DatasetDict, load_dataset
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm.auto import tqdm

from mlrc22.constants import IMAGENET_DOG_LABELS, IMAGENET_FISH_LABELS

__all__ = [
    "set_random_seed",
    "setup_logger",
    "flip_labels",
    "download_and_filter_imagenet",
    "generate_inception_v3_embeddings",
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


def setup_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logger


def setup_plotting():
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.set_context("paper", font_scale=1.5)


def flip_labels(
    y: NDArray[np.int_], percentage: float, *, random_state: np.random.RandomState
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    indices = random_state.choice(
        np.arange(len(y)), size=int(percentage * len(y)), replace=False
    )
    y = y.copy()
    y[indices] = np.logical_not(y[indices])
    return y, indices


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
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
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

    X = np.concatenate(train_inputs + val_inputs, axis=0)
    y = np.concatenate(train_labels + val_labels, axis=0)

    return X, y
