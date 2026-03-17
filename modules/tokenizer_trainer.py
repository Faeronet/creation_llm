"""
Train a SentencePiece Unigram tokenizer on book + tokenizer corpus.
"""

from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

from modules.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    FILE_BOOK_CLEAN,
    FILE_TOKENIZER_CORPUS,
    PAD_TOKEN,
    UNK_TOKEN,
)
from modules.exceptions import DataNotFoundError
from modules.io_utils import ensure_data_file
from modules.logger import get_logger
from modules.paths import DATA_DIR, MODEL_TOKENIZER

logger = get_logger(__name__)

DEFAULT_VOCAB_SIZE = 3072
DEFAULT_MODEL_PREFIX = "sp_model"


def train_tokenizer(
    data_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
    model_type: str = "unigram",
    max_sentences: Optional[int] = None,
    character_coverage: float = 0.9995,
) -> Path:
    """
    Train SentencePiece Unigram tokenizer on 01_book_clean.txt and 02_tokenizer_corpus.txt.

    Args:
        data_dir: Directory containing corpus files. Default: DATA_DIR.
        output_dir: Where to save .model and .vocab. Default: MODEL_TOKENIZER.
        vocab_size: Target vocabulary size.
        model_prefix: Prefix for output files (e.g. sp_model -> sp_model.model).
        model_type: 'unigram' as required.
        max_sentences: If set, limit training to this many lines (for debugging).
        character_coverage: SentencePiece character_coverage.

    Returns:
        Path to the directory where the model was saved.

    Raises:
        DataNotFoundError: If at least one corpus file is missing.
    """
    data_dir = Path(data_dir or DATA_DIR)
    output_dir = Path(output_dir or MODEL_TOKENIZER)
    output_dir.mkdir(parents=True, exist_ok=True)

    book_path = data_dir / FILE_BOOK_CLEAN
    corpus_path = data_dir / FILE_TOKENIZER_CORPUS
    ensure_data_file(book_path, "book clean text")
    ensure_data_file(corpus_path, "tokenizer corpus")

    # SentencePiece accepts comma-separated input files
    input_files = ",".join(str(p) for p in [book_path, corpus_path] if p.exists())
    if not input_files:
        raise DataNotFoundError("At least one of 01_book_clean.txt or 02_tokenizer_corpus.txt must exist.")

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=str(output_dir / model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        character_coverage=character_coverage,
        user_defined_symbols=[],
    )

    logger.info("Tokenizer saved to %s (vocab_size=%s)", output_dir, vocab_size)
    return output_dir
