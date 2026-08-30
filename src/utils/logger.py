"""
Logging and reproducibility utilities for ev-flex-ml.
"""

import logging
import random
import sys
from typing import Optional

import numpy as np


def setup_logger(name: str = "ev_flex_ml", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger with detailed formatting.

    Args:
        name: Name of the logger instance.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Sets random seed across standard library, numpy, and torch for full reproducibility.

    Args:
        seed: Integer random seed.
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
        if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
