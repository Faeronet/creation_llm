#!/usr/bin/env python3
"""
CLI: Build BM25 index from 04_retrieval_chunks.jsonl and save to model/retrieval.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.paths import DATA_DIR, MODEL_RETRIEVAL
from modules.retrieval_index import build_index
from modules.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 retrieval index")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_RETRIEVAL)
    parser.add_argument("--chunks-file", type=str, default="04_retrieval_chunks.jsonl")
    args = parser.parse_args()

    chunks_path = args.data_dir / args.chunks_file
    try:
        build_index(chunks_path=chunks_path, output_dir=args.output_dir)
    except Exception as e:
        logger.exception("Build retrieval index failed: %s", e)
        return 1
    logger.info("Done. Index saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
