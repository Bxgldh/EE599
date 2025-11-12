"""
Data package for sentiment_llama.

This package provides access to:
- all-data.csv (raw dataset)
- utilities for loading and preprocessing datasets (in data_utils/)
"""

from pathlib import Path

# root data directory
DATA_DIR = Path(__file__).parent

# default data file path
DEFAULT_DATA_PATH = DATA_DIR / "all-data.csv"

__all__ = ["DATA_DIR", "DEFAULT_DATA_PATH"]
