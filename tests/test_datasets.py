"""Tests for SFT and answerability dataset loading and batch format."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_sft_dataset_parses_messages(small_jsonl_sft, tmp_path):
    """SFT dataset parses messages format and returns input_ids/labels."""
    # Mock tokenizer to avoid needing trained tokenizer
    mock_sp = MagicMock()
    mock_sp.encode.return_value = [2, 10, 20, 30, 3]  # bos + tokens + eos
    mock_sp.pad_id.return_value = 0
    mock_sp.bos_id.return_value = 2
    mock_sp.eos_id.return_value = 3
    with patch("modules.dataset_sft.load_tokenizer", return_value=mock_sp):
        from modules.dataset_sft import SFTDataset
        ds = SFTDataset(small_jsonl_sft, max_length=128)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert isinstance(item["input_ids"], list)
    assert isinstance(item["labels"], list)


def test_answerability_dataset_parses(small_jsonl_answerability, tmp_path):
    """Answerability dataset parses question/label and returns input_ids, label."""
    mock_sp = MagicMock()
    mock_sp.encode.return_value = [2, 5, 6, 7, 3]
    mock_sp.pad_id.return_value = 0
    with patch("modules.dataset_answerability.load_tokenizer", return_value=mock_sp):
        from modules.dataset_answerability import AnswerabilityDataset
        ds = AnswerabilityDataset(small_jsonl_answerability, max_length=64)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "label" in item
    assert item["label"] in ("answerable", "not_answerable")
