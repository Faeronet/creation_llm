"""
Central path definitions for the project. All paths derived from project root.
"""

from pathlib import Path

# Project root: parent of 'modules' package
_MODULES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULES_DIR.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_DIR = PROJECT_ROOT / "model"

# Checkpoint subdirs
CHECKPOINTS_TOKENIZER = CHECKPOINTS_DIR / "tokenizer"
CHECKPOINTS_LM = CHECKPOINTS_DIR / "lm"
CHECKPOINTS_ANSWERABILITY = CHECKPOINTS_DIR / "answerability"

# Model artifact subdirs
MODEL_TOKENIZER = MODEL_DIR / "tokenizer"
MODEL_LM = MODEL_DIR / "lm"
MODEL_ANSWERABILITY = MODEL_DIR / "answerability"
MODEL_RETRIEVAL = MODEL_DIR / "retrieval"
MODEL_INFERENCE = MODEL_DIR / "inference"

# Config directory
CONFIGS_DIR = PROJECT_ROOT / "configs"


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in (
        DATA_DIR,
        CHECKPOINTS_DIR,
        CHECKPOINTS_TOKENIZER,
        CHECKPOINTS_LM,
        CHECKPOINTS_ANSWERABILITY,
        MODEL_DIR,
        MODEL_TOKENIZER,
        MODEL_LM,
        MODEL_ANSWERABILITY,
        MODEL_RETRIEVAL,
        MODEL_INFERENCE,
        CONFIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
