#!/usr/bin/env python3
"""
Unified training entrypoint. Subcommands: tokenizer, lm, answerability.
  python train.py tokenizer
  torchrun --nproc-per-node=2 train.py lm
  python train.py answerability
"""

import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: train.py <tokenizer|lm|answerability> [args...]", file=sys.stderr)
        print("  tokenizer  -> scripts/train_tokenizer.py", file=sys.stderr)
        print("  lm        -> scripts/train_lm.py (use torchrun for multi-GPU)", file=sys.stderr)
        print("  answerability -> scripts/train_answerability.py", file=sys.stderr)
        return 1

    subcommand = sys.argv[1].lower()
    # So that argparse in scripts sees only their args (e.g. --config)
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if subcommand == "tokenizer":
        from scripts.train_tokenizer import main as tokenizer_main
        return tokenizer_main()
    elif subcommand == "lm":
        from scripts.train_lm import main as lm_main
        return lm_main()
    elif subcommand == "answerability":
        from scripts.train_answerability import main as ans_main
        return ans_main()
    else:
        print(f"Unknown subcommand: {subcommand}. Use tokenizer, lm, or answerability.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
