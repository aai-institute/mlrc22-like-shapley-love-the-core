from pathlib import Path

__all__ = [
    "RANDOM_SEED",
    "DATA_DIR",
    "OUTPUT_DIR",
    "BREAST_CANCER_OPENML_ID",
    "HOUSE_VOTING_OPENML_ID",
    "ENRON1_SPAM_DATASET_URL",
]

RANDOM_SEED = 16

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

BREAST_CANCER_OPENML_ID = 43611
HOUSE_VOTING_OPENML_ID = 56

ENRON1_SPAM_DATASET_URL = (
    "http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz"
)
