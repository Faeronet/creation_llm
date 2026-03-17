"""
Load and use a trained SentencePiece tokenizer. Encode/decode with special tokens.
"""

from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

from modules.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from modules.exceptions import TokenizerError
from modules.logger import get_logger
from modules.paths import MODEL_TOKENIZER

logger = get_logger(__name__)

DEFAULT_MODEL_PREFIX = "sp_model"


def load_tokenizer(model_dir: Optional[Union[str, Path]] = None) -> spm.SentencePieceProcessor:
    """
    Load a trained SentencePiece model from model/tokenizer/ (or custom path).

    Expects either:
      - {model_dir}/sp_model.model (and .vocab if needed), or
      - {model_dir}/model.model

    Args:
        model_dir: Directory containing the .model file. Default: MODEL_TOKENIZER.

    Returns:
        Loaded SentencePiece processor.

    Raises:
        TokenizerError: If model file not found or load fails.
    """
    model_dir = Path(model_dir or MODEL_TOKENIZER)
    candidates = [
        model_dir / f"{DEFAULT_MODEL_PREFIX}.model",
        model_dir / "model.model",
        model_dir / "tokenizer.model",
    ]
    model_path = None
    for p in candidates:
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise TokenizerError(
            f"No SentencePiece model found in {model_dir}. "
            f"Expected one of: {[str(c) for c in candidates]}"
        )
    sp = spm.SentencePieceProcessor()
    try:
        sp.load(str(model_path))
    except Exception as e:
        raise TokenizerError(f"Failed to load tokenizer from {model_path}: {e}") from e
    logger.info("Loaded tokenizer from %s", model_path)
    return sp


def get_special_token_ids(sp: spm.SentencePieceProcessor) -> dict:
    """Return dict with pad_id, unk_id, bos_id, eos_id from the processor."""
    return {
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
    }


def encode(
    sp: spm.SentencePieceProcessor,
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
    out_type: type = list,
) -> Union[List[int], str]:
    """
    Encode text to token ids.

    Args:
        sp: Loaded SentencePiece processor.
        text: Input text.
        add_bos: Prepend BOS token.
        add_eos: Append EOS token.
        out_type: list (return List[int]) or str (return space-joined token strings).

    Returns:
        Token ids or token string sequence.
    """
    ids = sp.encode(text, add_bos=add_bos, add_eos=add_eos, out_type=int)
    if out_type is str:
        return " ".join(sp.id_to_piece(ids))
    return ids


def decode(
    sp: spm.SentencePieceProcessor,
    ids: List[int],
    skip_special_tokens: bool = True,
) -> str:
    """Decode token ids to text. Optionally skip special tokens."""
    if skip_special_tokens:
        ids = [i for i in ids if i not in (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())]
    return sp.decode(ids)
