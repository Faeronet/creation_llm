#!/usr/bin/env python3
"""
CLI: Train answerability classifier (head on top of LM).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config
from modules.constants import FILE_ANSWERABILITY_TRAIN, FILE_ANSWERABILITY_VAL
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.paths import CONFIGS_DIR, DATA_DIR, MODEL_ANSWERABILITY, MODEL_LM, MODEL_TOKENIZER
from modules.answerability_trainer import train_answerability

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train answerability classifier")
    parser.add_argument("--config", type=str, default="answerability")
    parser.add_argument("--config-dir", type=Path, default=CONFIGS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=MODEL_TOKENIZER)
    parser.add_argument("--lm-dir", type=Path, default=MODEL_LM)
    parser.add_argument("--output-dir", type=Path, default=MODEL_ANSWERABILITY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from modules.seed import set_seed
    set_seed(args.seed)

    try:
        cfg = load_config(args.config, configs_dir=args.config_dir)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    # Load LM config from saved model
    config_path = args.lm_dir / "config.json"
    if config_path.exists():
        lm_config = LMConfig.load(config_path)
    else:
        lm_config = LMConfig()

    train_path = args.data_dir / FILE_ANSWERABILITY_TRAIN
    val_path = args.data_dir / FILE_ANSWERABILITY_VAL

    try:
        train_answerability(
            train_path=train_path,
            val_path=val_path if val_path.exists() else None,
            tokenizer_dir=args.tokenizer_dir,
            lm_config=lm_config,
            output_dir=args.output_dir,
            batch_size=cfg.get("batch_size", 16),
            epochs=cfg.get("epochs", 3),
            lr=cfg.get("learning_rate", 2e-5),
            max_length=cfg.get("max_length", 256),
            freeze_lm=cfg.get("freeze_lm", True),
        )
    except Exception as e:
        logger.exception("Answerability training failed: %s", e)
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
