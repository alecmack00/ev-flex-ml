"""
YAML Configuration loading and validation utility.
"""

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .logger import setup_logger

logger = setup_logger("config")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If configuration file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.absolute()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {path}")
        return config if config is not None else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration at {path}: {e}")
        raise
