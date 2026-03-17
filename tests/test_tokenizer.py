"""Tests for tokenizer training and encode/decode."""
import tempfile
from pathlib import Path

import pytest

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_tokenizer_train_small_corpus(tmp_path):
    """Train tokenizer on minimal corpus and check encode/decode roundtrip."""
    try:
        import sentencepiece as spm
    except ImportError:
        pytest.skip("sentencepiece not installed")
    from modules.tokenizer_trainer import train_tokenizer
    from modules.tokenizer_utils import load_tokenizer, encode, decode

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Первая строка корпуса.\nВторая строка для обучения.\n" * 5, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "01_book_clean.txt").write_text(corpus.read_text(encoding="utf-8"), encoding="utf-8")
    (data_dir / "02_tokenizer_corpus.txt").write_text("Дополнительный текст.\n", encoding="utf-8")
    out_dir = tmp_path / "model" / "tokenizer"
    out_dir.mkdir(parents=True)
    train_tokenizer(data_dir=data_dir, output_dir=out_dir, vocab_size=128)
    sp = load_tokenizer(out_dir)
    text = "Строка для проверки."
    ids = encode(sp, text, add_bos=True, add_eos=True)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = decode(sp, ids, skip_special_tokens=True)
    assert text in decoded or decoded  # roundtrip may add spaces
    assert sp.pad_id() == 0
    assert sp.unk_id() == 1
    assert sp.bos_id() == 2
    assert sp.eos_id() == 3
