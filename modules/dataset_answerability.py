"""
Answerability dataset: load 07/08 JSONL, tokenize question (and optional context), return input_ids, attention_mask, label.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset

from modules.constants import (
    ANSWERABILITY_LABEL_KEYS,
    ANSWERABILITY_LABEL_NEGATIVE,
    ANSWERABILITY_LABEL_POSITIVE,
    ANSWERABILITY_QUESTION_KEYS,
)
from modules.io_utils import read_jsonl
from modules.logger import get_logger
from modules.tokenizer_utils import load_tokenizer

logger = get_logger(__name__)


def _get_question(record: Dict[str, Any]) -> str:
    for k in ANSWERABILITY_QUESTION_KEYS:
        if k in record and record[k]:
            return str(record[k]).strip()
    return ""


def _get_label(record: Dict[str, Any]) -> str:
    for k in ANSWERABILITY_LABEL_KEYS:
        if k in record:
            v = record[k]
            if isinstance(v, bool):
                return ANSWERABILITY_LABEL_POSITIVE if v else ANSWERABILITY_LABEL_NEGATIVE
            if isinstance(v, (int, float)):
                return ANSWERABILITY_LABEL_POSITIVE if v else ANSWERABILITY_LABEL_NEGATIVE
            return str(v).strip().lower()
    return ANSWERABILITY_LABEL_NEGATIVE


class AnswerabilityDataset(Dataset):
    """
    Dataset for answerability classification from 07/08 JSONL.
    Each item: input_ids, attention_mask, label (0 = not_answerable, 1 = answerable).
    Optionally concatenate context (retrieved chunks) with question for training.
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer_dir: Optional[Union[str, Path]] = None,
        max_length: int = 256,
        context_key: Optional[str] = None,
    ):
        """
        Args:
            path: Path to JSONL (07_answerability_train.jsonl or 08_answerability_val.jsonl).
            tokenizer_dir: Path to trained tokenizer.
            max_length: Max sequence length (question or question + context).
            context_key: If set, record[context_key] is prepended to question (e.g. "context").
        """
        self.path = Path(path)
        self.max_length = max_length
        self.context_key = context_key
        self.sp = load_tokenizer(tokenizer_dir)
        self.pad_id = self.sp.pad_id()

        self.records: List[Dict[str, Any]] = []
        if self.path.exists():
            self.records = read_jsonl(self.path)
        else:
            logger.warning("Answerability file not found: %s", self.path)
        logger.info("Loaded %d answerability records from %s", len(self.records), self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        question = _get_question(record)
        label = _get_label(record)

        if self.context_key and self.context_key in record and record[self.context_key]:
            text = str(record[self.context_key]).strip() + " " + question
        else:
            text = question

        ids = self.sp.encode(text, add_bos=True, add_eos=True, out_type=int)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
        return {
            "input_ids": ids,
            "label": label,
        }
