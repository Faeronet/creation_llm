"""Tests for LM and answerability model shapes and tie_embeddings."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM
from modules.answerability_model import AnswerabilityModel


def test_lm_forward_shape():
    config = LMConfig(vocab_size=100, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128, max_position_embeddings=64)
    model = DecoderLM(config)
    batch, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    logits, loss = model(input_ids, labels=input_ids.clone())
    assert logits.shape == (batch, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0


def test_lm_tie_embeddings():
    config = LMConfig(vocab_size=50, hidden_size=32, num_layers=1, num_heads=2, tie_embeddings=True)
    model = DecoderLM(config)
    assert model.embed.weight is model.lm_head.weight


def test_answerability_forward_shape():
    config = LMConfig(vocab_size=100, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=128, max_position_embeddings=64)
    model = AnswerabilityModel(config, freeze_lm=True)
    batch, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    logits, loss = model(input_ids, labels=torch.tensor([0, 1]))
    assert logits.shape == (batch, 2)
    assert loss is not None
