"""
Unit tests for src/utils module (logger, config, set_seed).
"""

import logging
import random
import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.utils.config import load_config
from src.utils.logger import set_seed, setup_logger


def test_setup_logger_propagate():
    """Test that setup_logger disables log propagation and configures standard handlers."""
    logger_name = "test_logger_propagate"
    logger = setup_logger(logger_name, level=logging.DEBUG)
    
    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 1


def test_set_seed():
    """Test set_seed for reproducibility across standard random, numpy, and torch."""
    seed = 1234
    set_seed(seed)
    r1 = random.randint(0, 100000)
    np1 = np.random.rand()

    set_seed(seed)
    r2 = random.randint(0, 100000)
    np2 = np.random.rand()

    assert r1 == r2
    assert np1 == np2

    try:
        import torch
        set_seed(seed)
        t1 = torch.rand(5)
        set_seed(seed)
        t2 = torch.rand(5)
        assert torch.equal(t1, t2)
    except ImportError:
        pass


def test_load_config_valid(tmp_path):
    """Test load_config with a valid YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("simulation:\n  time_steps: 24\n  dt: 1.0\n", encoding="utf-8")

    config = load_config(config_file)
    assert isinstance(config, dict)
    assert config["simulation"]["time_steps"] == 24


def test_load_config_file_not_found():
    """Test load_config raises FileNotFoundError when given non-existent file path."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("non_existent_config_12345.yaml"))
