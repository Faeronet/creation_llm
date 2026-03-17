#!/usr/bin/env python3
"""
CLI entrypoint for inference (same as root infer.py). Run from project root:
  python scripts/infer.py "Ваш вопрос"
  python scripts/infer.py --interactive
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from infer import main
    sys.exit(main())
