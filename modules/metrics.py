"""
Metrics: perplexity, accuracy, precision/recall/F1 for answerability and refusal.
"""

from typing import List, Optional

import torch


def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(loss)."""
    return float(torch.exp(torch.tensor(loss)).item())


def accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """Classification accuracy; ignore_index positions are skipped in labels."""
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def precision_recall_f1(
    preds: List[int],
    labels: List[int],
    positive_class: int = 1,
) -> tuple:
    """
    Precision, recall, F1 for the positive class (e.g. refusal or answerable).
    Returns (precision, recall, f1).
    """
    preds = list(preds)
    labels = list(labels)
    tp = sum(1 for p, l in zip(preds, labels) if p == positive_class and l == positive_class)
    fp = sum(1 for p, l in zip(preds, labels) if p == positive_class and l != positive_class)
    fn = sum(1 for p, l in zip(preds, labels) if p != positive_class and l == positive_class)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def refusal_rate(preds: List[int], positive_class: int = 1) -> float:
    """Fraction of predictions that are refusal (positive class)."""
    if not preds:
        return 0.0
    return sum(1 for p in preds if p == positive_class) / len(preds)


def correct_refusal_rate(
    preds: List[int],
    labels: List[int],
    positive_class: int = 1,
) -> float:
    """Among true refusals, fraction correctly predicted as refusal."""
    ref_idx = [i for i, l in enumerate(labels) if l == positive_class]
    if not ref_idx:
        return 0.0
    return sum(1 for i in ref_idx if preds[i] == positive_class) / len(ref_idx)
