"""
Retriever: load BM25 index and chunks metadata, expose retrieve(query, top_k) -> List[(chunk_id, text, score)].
"""

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

from modules.io_utils import read_json
from modules.logger import get_logger
from modules.paths import MODEL_RETRIEVAL
from modules.retrieval_index import CHUNKS_META_FILENAME, INDEX_FILENAME, _tokenize_for_bm25

logger = get_logger(__name__)


class Retriever:
    """Load BM25 index and retrieve top-k chunks by query."""

    def __init__(self, index_dir: Optional[Path] = None):
        """
        Load index and chunk metadata from index_dir (default: model/retrieval).
        """
        self.index_dir = Path(index_dir or MODEL_RETRIEVAL)
        index_path = self.index_dir / INDEX_FILENAME
        meta_path = self.index_dir / CHUNKS_META_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}. Run build_retrieval.py first.")
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        self.chunks_meta = read_json(meta_path)
        logger.info("Loaded retriever from %s (%d chunks)", self.index_dir, len(self.chunks_meta))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Return top-k (chunk_id, text, score) for the query.
        """
        if not query or not self.chunks_meta:
            return []
        tokenized_query = _tokenize_for_bm25(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in indices:
            meta = self.chunks_meta[i]
            out.append((meta["chunk_id"], meta["text"], float(scores[i])))
        return out
