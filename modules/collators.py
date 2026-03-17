"""
Data collators: padding and batching for LM and answerability.
"""

from typing import Any, Dict, List, Optional

import torch

from modules.constants import ANSWERABILITY_LABEL_NEGATIVE, ANSWERABILITY_LABEL_POSITIVE


class DataCollatorForCausalLM:
    """Pad sequences to max length in batch, create labels (copy of input_ids with -100 for non-target positions)."""

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        padding_side: str = "right",
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        features: list of dicts with 'input_ids' (and optionally 'labels').
        If 'labels' not in features, labels = input_ids with padding positions set to -100.
        """
        batch_size = len(features)
        input_ids = [f["input_ids"] for f in features]
        has_labels = "labels" in features[0]

        if self.max_length is not None:
            input_ids = [ids[: self.max_length] for ids in input_ids]

        max_len = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        padded = []
        labels_list = []
        for f in features:
            ids = f["input_ids"][:max_len]
            if has_labels:
                lab = f["labels"][:max_len]
            else:
                lab = list(ids)
            pad_len = max_len - len(ids)
            if self.padding_side == "right":
                padded.append(ids + [self.pad_token_id] * pad_len)
                labels_list.append(lab + [-100] * pad_len)
            else:
                padded.append([self.pad_token_id] * pad_len + ids)
                labels_list.append([-100] * pad_len + lab)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        return batch


class DataCollatorForAnswerability:
    """Pad input_ids and attention_mask; stack labels."""

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
        label_positive: str = ANSWERABILITY_LABEL_POSITIVE,
        label_negative: str = ANSWERABILITY_LABEL_NEGATIVE,
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.label_positive = label_positive
        self.label_negative = label_negative

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        input_ids = [f["input_ids"] for f in features]
        labels = [f["label"] for f in features]

        if self.max_length is not None:
            input_ids = [ids[: self.max_length] for ids in input_ids]

        max_len = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        padded = []
        for ids in input_ids:
            pad_len = max_len - len(ids)
            padded.append(ids + [self.pad_token_id] * pad_len)

        # Convert labels: answerable -> 1, not_answerable -> 0
        label_ids = []
        for lab in labels:
            if lab in (self.label_positive, 1, True, "1"):
                label_ids.append(1)
            else:
                label_ids.append(0)

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": (torch.tensor(padded, dtype=torch.long) != self.pad_token_id).long(),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
