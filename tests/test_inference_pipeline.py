"""Tests for inference pipeline: refusal and success paths with mocks."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.constants import DEFAULT_REFUSAL_MESSAGE
from modules.postcheck import answer_supported_by_context
from modules.prompt_builder import build_prompt, load_system_prompt


def test_postcheck_supported():
    chunks = [("id1", "В книге сказано про ключи и ангелов.", 1.0)]
    assert answer_supported_by_context("Ключи — это символы.", chunks) is True


def test_postcheck_unsupported():
    chunks = [("id1", "Текст о погоде.", 1.0)]
    assert answer_supported_by_context("Ключи открывают дверь в другой мир.", chunks) is False


def test_prompt_builder():
    chunks = [("c1", "Контекст из книги.", 0.9)]
    prompt = build_prompt("Вопрос?", chunks, system_prompt="Система.")
    assert "Контекст из книги" in prompt
    assert "Вопрос?" in prompt
    assert "Система" in prompt


def test_refusal_message_constant():
    assert "нет данных" in DEFAULT_REFUSAL_MESSAGE
