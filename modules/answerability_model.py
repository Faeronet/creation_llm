"""
Answerability classifier: head on top of LM hidden states. Load LM from model/lm, freeze LM, add classification head.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM
from modules.paths import MODEL_LM, MODEL_ANSWERABILITY

logger = get_logger(__name__)

NUM_CLASSES = 2  # answerable, not_answerable


class AnswerabilityHead(nn.Module):
    """Classification head: pool last hidden (or [CLS]) -> linear -> logits."""

    def __init__(self, hidden_size: int, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden: (B, T, H). Pool: use last non-pad position per sequence
        if attention_mask is not None:
            # last token index per batch
            last_idx = attention_mask.sum(1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, last_idx, :]
        else:
            pooled = hidden[:, -1, :]
        pooled = torch.tanh(self.pool(pooled))
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class AnswerabilityModel(nn.Module):
    """
    LM + classification head. LM can be frozen. Forward returns logits for answerable/not_answerable.
    """

    def __init__(
        self,
        config: LMConfig,
        num_classes: int = NUM_CLASSES,
        freeze_lm: bool = True,
    ):
        super().__init__()
        self.config = config
        self.lm = DecoderLM(config)
        self.head = AnswerabilityHead(config.hidden_size, num_classes=num_classes, dropout=config.dropout)
        if freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = self.lm.get_last_hidden_state(input_ids, attention_mask)
        logits = self.head(hidden, attention_mask)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
        return logits, loss

    def load_lm_weights(self, path: Optional[Path] = None) -> None:
        """Load LM state_dict from model/lm (pytorch_model.pt)."""
        path = path or (MODEL_LM / "pytorch_model.pt")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LM weights not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.lm.load_state_dict(state, strict=False)
        logger.info("Loaded LM weights from %s", path)

    def save_pretrained(self, save_dir: Path) -> None:
        """Save head weights and config to save_dir (model/answerability/)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), save_dir / "head.pt")
        self.config.save(save_dir / "config.json")

    def load_head(self, save_dir: Path) -> None:
        """Load head weights from save_dir."""
        save_dir = Path(save_dir)
        path = save_dir / "head.pt"
        if not path.exists():
            raise FileNotFoundError(f"Head weights not found: {path}")
        self.head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        logger.info("Loaded answerability head from %s", path)
