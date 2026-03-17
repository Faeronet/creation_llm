#!/usr/bin/env python3
"""
CLI: Train decoder-only LM with DDP. Run with torchrun for multi-GPU.
  torchrun --nproc-per-node=2 scripts/train_lm.py --config train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.paths import CONFIGS_DIR, DATA_DIR, MODEL_LM, MODEL_TOKENIZER
from modules.seed import set_seed
from modules.trainer_lm import train_lm
from modules.constants import FILE_SFT_CHAT_TRAIN, FILE_SFT_CHAT_VAL

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LM for Angels Book (DDP)")
    parser.add_argument("--config", type=str, default="train", help="Config name from configs/")
    parser.add_argument("--config-dir", type=Path, default=CONFIGS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=MODEL_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=MODEL_LM, help="Final model output (model/lm)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoints (default: checkpoints/lm)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        cfg = load_config(args.config, configs_dir=args.config_dir, merge_recommended=True)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    model_cfg_dict = cfg.get("model", {})
    config = LMConfig(
        vocab_size=model_cfg_dict.get("vocab_size", 3072),
        hidden_size=model_cfg_dict.get("hidden_size", 768),
        num_layers=model_cfg_dict.get("num_layers", 12),
        num_heads=model_cfg_dict.get("num_heads", 12),
        intermediate_size=model_cfg_dict.get("intermediate_size", 2048),
        max_position_embeddings=model_cfg_dict.get("max_position_embeddings", 768),
        dropout=model_cfg_dict.get("dropout", 0.1),
        tie_embeddings=model_cfg_dict.get("tie_embeddings", True),
    )

    train_path = args.data_dir / FILE_SFT_CHAT_TRAIN
    val_path = args.data_dir / FILE_SFT_CHAT_VAL
    if not val_path.exists():
        val_path = None

    try:
        train_lm(
            train_path=train_path,
            val_path=val_path,
            tokenizer_dir=args.tokenizer_dir,
            config=config,
            output_checkpoint_dir=args.checkpoint_dir,
            output_model_dir=args.output_dir,
            batch_size=cfg.get("batch_size", 4),
            epochs=cfg.get("epochs", 3),
            lr=cfg.get("learning_rate", 1e-4),
            max_length=cfg.get("max_length", 512),
            checkpoint_every_steps=cfg.get("checkpoint_every_steps"),
            checkpoint_every_epoch=cfg.get("checkpoint_every_epoch", True),
            resume=not args.no_resume,
        )
    except Exception as e:
        logger.exception("LM training failed: %s", e)
        return 1

    logger.info("LM training finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
