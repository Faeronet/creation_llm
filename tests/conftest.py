"""
Pytest fixtures: small mock data and paths for tests that do not require full dataset.
"""
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def small_jsonl_sft(tmp_path):
    """Minimal SFT JSONL (messages format)."""
    p = tmp_path / "sft.jsonl"
    records = [
        {"messages": [{"role": "user", "content": "Что такое ключ?"}, {"role": "assistant", "content": "Ключ — это символ."}]},
        {"messages": [{"role": "user", "content": "Кто ангел?"}, {"role": "assistant", "content": "Ангел — вестник."}]},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


@pytest.fixture
def small_jsonl_answerability(tmp_path):
    p = tmp_path / "answerability.jsonl"
    records = [
        {"question": "Что такое ключ?", "label": "answerable"},
        {"question": "Сколько весит Луна?", "label": "not_answerable"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


@pytest.fixture
def small_retrieval_chunks(tmp_path):
    p = tmp_path / "chunks.jsonl"
    records = [
        {"chunk_id": "c1", "text": "Первый чанк о ключах и ангелах."},
        {"chunk_id": "c2", "text": "Второй чанк о мастере и символах."},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p
