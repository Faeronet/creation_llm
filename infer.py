#!/usr/bin/env python3
"""
CLI: Run inference pipeline. Single question from args or interactive.
  python infer.py "Ваш вопрос"
  python infer.py --interactive
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import load_config
from modules.inference_pipeline import InferencePipeline
from modules.logger import get_logger
from modules.paths import CONFIGS_DIR

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference: answer questions from the book or refuse.")
    parser.add_argument("question", nargs="?", default=None, help="Question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--config", type=str, default="inference", help="Config name from configs/")
    parser.add_argument("--config-dir", type=Path, default=CONFIGS_DIR)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, configs_dir=args.config_dir)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return 1

    try:
        pipeline = InferencePipeline(
            top_k=cfg.get("top_k", 5),
            max_new_tokens=cfg.get("max_new_tokens", 256),
            temperature=cfg.get("temperature", 0.8),
            use_fp16=cfg.get("use_fp16", True),
            answerability_threshold=cfg.get("answerability_threshold", 0.5),
        )
    except Exception as e:
        logger.exception("Failed to load pipeline: %s", e)
        return 1

    if args.interactive:
        print("Введите вопрос (пустая строка для выхода):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            response, refused = pipeline.run(line)
            if refused:
                print("[Отказ]", response)
            else:
                print(response)
            print()
        return 0

    if not args.question:
        print("Укажите вопрос или --interactive", file=sys.stderr)
        return 1
    response, refused = pipeline.run(args.question)
    if refused:
        print("[Отказ]", response)
    else:
        print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
