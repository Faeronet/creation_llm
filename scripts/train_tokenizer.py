#!/usr/bin/env python3
"""
CLI: Train SentencePiece Unigram tokenizer on book + tokenizer corpus.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import load_config
from modules.logger import get_logger
from modules.paths import CONFIGS_DIR, MODEL_TOKENIZER
from modules.seed import set_seed
from modules.tokenizer_trainer import train_tokenizer

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train tokenizer for Angels Book LLM")
    parser.add_argument(
        "--config",
        type=str,
        default="tokenizer",
        help="Config name (without .yaml) from configs/",
    )
    parser.add_argument("--config-dir", type=Path, default=CONFIGS_DIR, help="Configs directory")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: model/tokenizer)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory (default: data/)")
    args = parser.parse_args()

    set_seed(args.seed)

    try:
        cfg = load_config(args.config, configs_dir=args.config_dir)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    vocab_size = args.vocab_size if args.vocab_size is not None else cfg.get("vocab_size", 3072)
    output_dir = args.output_dir or MODEL_TOKENIZER
    data_dir = args.data_dir

    try:
        train_tokenizer(
            data_dir=data_dir,
            output_dir=output_dir,
            vocab_size=vocab_size,
            model_prefix=cfg.get("model_prefix", "sp_model"),
            model_type=cfg.get("model_type", "unigram"),
            max_sentences=cfg.get("max_sentences"),
            character_coverage=cfg.get("character_coverage", 0.9995),
        )
    except Exception as e:
        logger.exception("Tokenizer training failed: %s", e)
        return 1

    logger.info("Done. Tokenizer saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
