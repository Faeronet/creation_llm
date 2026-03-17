"""
Text normalization and truncation for retrieval and prompts.
"""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to single space and strip."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_by_chars(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate text to at most max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def truncate_by_tokens_approx(text: str, max_tokens: int, chars_per_token: int = 4) -> str:
    """
    Approximate truncation by token count using average chars per token.
    Use for display/context limits when exact tokenizer is not applied.
    """
    approx_chars = max_tokens * chars_per_token
    return truncate_by_chars(text, approx_chars)
