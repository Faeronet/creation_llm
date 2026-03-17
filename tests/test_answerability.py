"""Tests for answerability model forward and output shape."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.model_config import LMConfig
from modules.answerability_model import AnswerabilityModel


def test_answerability_logits_shape():
    config = LMConfig(vocab_size=64, hidden_size=32, num_layers=1, num_heads=2, intermediate_size=64, max_position_embeddings=32)
    model = AnswerabilityModel(config, freeze_lm=True)
    batch, seq_len = 4, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    logits, _ = model(input_ids)
    assert logits.shape == (batch, 2)
