from pathlib import Path

__all__ = ["RANDOM_SEED", "DATA_DIR"]

RANDOM_SEED = 16

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
