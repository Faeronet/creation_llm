#!/usr/bin/env python3
"""
CLI: Run evaluation (LM perplexity, answerability accuracy/F1).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.constants import FILE_SFT_CHAT_VAL, FILE_ANSWERABILITY_VAL
from modules.evaluator import run_evaluation
from modules.logger import get_logger
from modules.paths import DATA_DIR

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LM and answerability model")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON here")
    args = parser.parse_args()

    val_sft = args.data_dir / FILE_SFT_CHAT_VAL
    val_ans = args.data_dir / FILE_ANSWERABILITY_VAL

    results = run_evaluation(
        val_sft_path=val_sft,
        val_answerability_path=val_ans if val_ans.exists() else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Wrote metrics to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
