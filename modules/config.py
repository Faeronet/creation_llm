"""
Configuration loading: YAML configs with optional merge from recommended training config.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from modules.constants import FILE_RECOMMENDED_CONFIG
from modules.exceptions import ConfigError
from modules.io_utils import read_yaml
from modules.logger import get_logger
from modules.paths import DATA_DIR, CONFIGS_DIR

logger = get_logger(__name__)


def load_config(
    config_name: str,
    configs_dir: Optional[Path] = None,
    merge_recommended: bool = False,
) -> Dict[str, Any]:
    """
    Load a YAML config from configs/ directory.

    Args:
        config_name: Base name of the config file (e.g. 'train' for train.yaml).
        configs_dir: Override configs directory. Default is CONFIGS_DIR.
        merge_recommended: If True and data/10_recommended_training_config.yaml
            exists, merge its keys into the loaded config (loaded config wins on conflict).

    Returns:
        Merged config dictionary.

    Raises:
        ConfigError: If config file is missing or invalid.
    """
    configs_dir = configs_dir or CONFIGS_DIR
    config_path = configs_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    cfg = read_yaml(config_path)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")

    if merge_recommended:
        recommended_path = DATA_DIR / FILE_RECOMMENDED_CONFIG
        if recommended_path.exists():
            rec = read_yaml(recommended_path)
            if isinstance(rec, dict):
                for k, v in rec.items():
                    if k not in cfg:
                        cfg[k] = v
                logger.info("Merged keys from %s", recommended_path)

    return cfg


def get_nested(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested key using dot notation (e.g. 'model.hidden_size').

    Args:
        cfg: Config dict.
        key_path: Dot-separated path.
        default: Value if path missing.

    Returns:
        Value at path or default.
    """
    keys = key_path.split(".")
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
