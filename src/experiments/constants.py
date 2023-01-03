from pathlib import Path

__all__ = ["DATA_DIR"]

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
