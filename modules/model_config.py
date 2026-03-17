"""
Dataclass config for decoder-only causal LM. Serializable to JSON/YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from modules.io_utils import read_json, read_yaml, write_json, write_yaml


@dataclass
class LMConfig:
    """Configuration for decoder-only transformer LM."""

    vocab_size: int = 3072
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 768
    dropout: float = 0.1
    tie_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Export to dict for serialization."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LMConfig":
        """Create config from dict (e.g. loaded from JSON/YAML)."""
        return cls(
            vocab_size=d.get("vocab_size", 3072),
            hidden_size=d.get("hidden_size", 768),
            num_layers=d.get("num_layers", 12),
            num_heads=d.get("num_heads", 12),
            intermediate_size=d.get("intermediate_size", 2048),
            max_position_embeddings=d.get("max_position_embeddings", 768),
            dropout=d.get("dropout", 0.1),
            tie_embeddings=d.get("tie_embeddings", True),
            pad_token_id=d.get("pad_token_id", 0),
            bos_token_id=d.get("bos_token_id", 2),
            eos_token_id=d.get("eos_token_id", 3),
        )

    def save(self, path: Path, as_yaml: bool = False) -> None:
        """Save config to file."""
        path = Path(path)
        if as_yaml:
            write_yaml(path, self.to_dict())
        else:
            write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "LMConfig":
        """Load config from JSON or YAML file."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            d = read_yaml(path)
        else:
            d = read_json(path)
        return cls.from_dict(d)
