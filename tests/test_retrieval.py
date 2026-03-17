"""Tests for BM25 retrieval index build and retrieve API."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_build_and_retrieve(small_retrieval_chunks, tmp_path):
    try:
        from modules.retrieval_index import build_index
        from modules.retriever import Retriever
    except ImportError as e:
        if "rank_bm25" in str(e):
            pytest.skip("rank_bm25 not installed")
        raise
    out = tmp_path / "retrieval"
    build_index(chunks_path=small_retrieval_chunks, output_dir=out)
    assert (out / "bm25_index.pkl").exists()
    assert (out / "chunks_meta.json").exists()
    retriever = Retriever(out)
    results = retriever.retrieve("ключи ангелы", top_k=2)
    assert len(results) <= 2
    for chunk_id, text, score in results:
        assert isinstance(chunk_id, str)
        assert isinstance(text, str)
        assert isinstance(score, (int, float))
