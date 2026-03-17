"""
Decoder-only causal language model: transformer with causal self-attention and LM head.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model_config import LMConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional attention_mask."""

    def __init__(self, config: LMConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            # attention_mask: (B, T), 1 = attend, 0 = mask
            att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """FFN: linear -> gelu -> linear with dropout."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class DecoderBlock(nn.Module):
    """Single decoder block: LayerNorm -> CausalSelfAttention -> residual -> LayerNorm -> MLP -> residual."""

    def __init__(self, config: LMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderLM(nn.Module):
    """
    Decoder-only causal LM: embeddings + N decoder blocks + final ln + LM head.
    Optionally tie input embedding and LM head weights.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) token ids
            attention_mask: (batch, seq_len), 1 = valid, 0 = pad
            labels: (batch, seq_len) for loss; -100 ignored

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if labels provided, else None
        """
        B, T = input_ids.shape
        if T > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {T} > max_position_embeddings {self.config.max_position_embeddings}")

        pos = self.position_ids[:, :T].expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="mean",
            )
        return logits, loss

    def get_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states of the last layer (batch, seq_len, hidden_size) for answerability head."""
        B, T = input_ids.shape
        pos = self.position_ids[:, :T].expand(B, -1)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_f(x)
