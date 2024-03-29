from pathlib import Path

__all__ = [
    "RANDOM_SEED",
    "DATA_DIR",
    "OUTPUT_DIR",
    "BREAST_CANCER_OPENML_ID",
    "HOUSE_VOTING_OPENML_ID",
    "ENRON1_SPAM_DATASET_URL",
    "IMAGENET_DOG_LABELS",
    "IMAGENET_FISH_LABELS",
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

IMAGENET_DOG_LABELS = [
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
]

IMAGENET_FISH_LABELS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
]
