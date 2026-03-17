"""
Post-check: verify that the generated answer is supported by the retrieved context.
Simple heuristic: key phrases from the answer should appear in context (or high overlap).
"""

import re
from typing import List, Tuple


def _normalize(s: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _token_set(s: str) -> set:
    """Set of words (for overlap)."""
    return set(_normalize(s).split())


def answer_supported_by_context(
    answer: str,
    context_chunks: List[Tuple[str, str, float]],
    min_overlap_ratio: float = 0.3,
    min_word_overlap: int = 2,
) -> bool:
    """
    Check if the generated answer is supported by the retrieved context.

    Heuristic: if the answer has very low word overlap with context, or is empty,
    consider it not supported (hallucination or irrelevant).

    Args:
        answer: Model-generated answer text.
        context_chunks: List of (chunk_id, text, score) that was used as context.
        min_overlap_ratio: Minimum ratio of answer words that should appear in context (0..1).
        min_word_overlap: Minimum number of non-stop answer words that must appear in context.

    Returns:
        True if answer appears supported by context, False otherwise (trigger refusal).
    """
    answer = answer.strip()
    if not answer or len(answer) < 3:
        return False
    context_text = " ".join(text for _, text, _ in context_chunks)
    if not context_text.strip():
        return False
    ans_tokens = _token_set(answer)
    ctx_tokens = _token_set(context_text)
    # Remove very short tokens
    ans_tokens = {t for t in ans_tokens if len(t) > 1}
    if not ans_tokens:
        return True
    # Basic token overlap check
    overlap = len(ans_tokens & ctx_tokens)
    if overlap >= min_word_overlap and overlap / len(ans_tokens) >= min_overlap_ratio:
        return True

    # Fallback: if the full answer text appears as a substring of any context chunk,
    # consider it supported even при низком токенном оверлэпе (морфология, пунктуация).
    for _, text, _ in context_chunks:
        if answer.lower() in text.lower():
            return True

    return False
