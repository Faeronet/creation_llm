"""
Build and save BM25 index from 04_retrieval_chunks.jsonl. Persist index and chunk_id -> text mapping.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.constants import RETRIEVAL_CHUNK_ID_KEYS, RETRIEVAL_CHUNK_TEXT_KEYS
from modules.io_utils import iter_jsonl, read_jsonl, write_json
from modules.logger import get_logger
from modules.paths import DATA_DIR, MODEL_RETRIEVAL

logger = get_logger(__name__)

INDEX_FILENAME = "bm25_index.pkl"
CHUNKS_META_FILENAME = "chunks_meta.json"


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple whitespace tokenization; lowercase for BM25."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.split() if text else []


def _chunk_id(record: Dict[str, Any]) -> str:
    for k in RETRIEVAL_CHUNK_ID_KEYS:
        if k in record:
            return str(record[k])
    return str(hash(str(record)))


def _chunk_text(record: Dict[str, Any]) -> str:
    for k in RETRIEVAL_CHUNK_TEXT_KEYS:
        if k in record and record[k]:
            return str(record[k]).strip()
    return ""


def build_index(
    chunks_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Build BM25 index from retrieval chunks JSONL.

    Args:
        chunks_path: Path to 04_retrieval_chunks.jsonl. Default: data/04_retrieval_chunks.jsonl.
        output_dir: Where to save index and metadata. Default: model/retrieval.

    Returns:
        (bm25_object, list of {chunk_id, text} for retrieval).
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("rank_bm25 is required for retrieval. Install with: pip install rank_bm25")

    chunks_path = Path(chunks_path or DATA_DIR / "04_retrieval_chunks.jsonl")
    output_dir = Path(output_dir or MODEL_RETRIEVAL)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    corpus_tokens: List[List[str]] = []
    chunks_meta: List[Dict[str, str]] = []
    for record in iter_jsonl(chunks_path):
        cid = _chunk_id(record)
        text = _chunk_text(record)
        if not text:
            continue
        corpus_tokens.append(_tokenize_for_bm25(text))
        chunks_meta.append({"chunk_id": cid, "text": text})

    if not corpus_tokens:
        raise ValueError(f"No valid chunks in {chunks_path}")

    bm25 = BM25Okapi(corpus_tokens)
    logger.info("Built BM25 index over %d chunks", len(corpus_tokens))

    # Save
    with open(output_dir / INDEX_FILENAME, "wb") as f:
        pickle.dump(bm25, f)
    write_json(output_dir / CHUNKS_META_FILENAME, chunks_meta)
    logger.info("Saved index to %s", output_dir)
    return bm25, chunks_meta
