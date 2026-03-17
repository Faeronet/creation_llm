"""
Build prompt for generator: system prompt + retrieved chunks + user question in a single format.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from modules.constants import ASSISTANT_TOKEN, FILE_SYSTEM_PROMPT, USER_TOKEN
from modules.io_utils import read_text
from modules.paths import DATA_DIR
from modules.text_utils import truncate_by_chars

CONTEXT_PREFIX = "Контекст из книги:\n"
QUESTION_PREFIX = "Вопрос: "
ANSWER_PREFIX = "Ответ: "


def load_system_prompt(data_dir: Optional[Path] = None) -> str:
    """Load system prompt from 09_system_prompt.txt."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / FILE_SYSTEM_PROMPT
    if path.exists():
        return read_text(path).strip()
    return "Ответь на вопрос строго по приведённому контексту из книги. Если в контексте нет ответа — скажи об этом."


def build_prompt(
    question: str,
    chunks: List[Tuple[str, str, float]],
    system_prompt: Optional[str] = None,
    max_context_chars: int = 4000,
) -> str:
    """
    Build full prompt: system + context (from chunks) + question. No assistant prefix (model generates answer).

    Args:
        question: User question.
        chunks: List of (chunk_id, text, score) from retriever.
        system_prompt: Override system prompt. If None, load from 09_system_prompt.txt.
        max_context_chars: Truncate total context to this many characters.

    Returns:
        Single string to tokenize and feed to the model (without trailing assistant start).
    """
    if system_prompt is None:
        system_prompt = load_system_prompt()
    context_parts = [text for _, text, _ in chunks]
    context = "\n\n".join(context_parts)
    if len(context) > max_context_chars:
        context = truncate_by_chars(context, max_context_chars)
    block = f"{system_prompt}\n\n{CONTEXT_PREFIX}{context}\n\n{QUESTION_PREFIX}{question.strip()}\n\n{ANSWER_PREFIX}"
    return block
