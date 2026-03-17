"""
Reproducibility: set random seeds for Python, NumPy, and PyTorch.
"""

import random
from typing import Optional

import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer seed value. Use the same seed across runs for same results.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
