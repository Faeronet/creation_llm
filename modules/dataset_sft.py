"""
SFT dataset: load chat JSONL (messages or prompt/completion), format with special tokens, return input_ids/labels.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset

from modules.constants import (
    ASSISTANT_TOKEN,
    FILE_SFT_CHAT_TRAIN,
    FILE_SFT_CHAT_VAL,
    MESSAGE_CONTENT_KEY,
    MESSAGE_ROLE_KEY,
    SFT_COMPLETION_KEY,
    SFT_MESSAGES_KEY,
    SFT_PROMPT_KEY,
    SYSTEM_TOKEN,
    USER_TOKEN,
)
from modules.io_utils import read_jsonl
from modules.logger import get_logger
from modules.paths import DATA_DIR
from modules.tokenizer_utils import load_tokenizer

logger = get_logger(__name__)


def _format_messages(messages: List[Dict[str, str]]) -> str:
    """Turn messages into a single string with role tokens."""
    parts = []
    for m in messages:
        role = m.get(MESSAGE_ROLE_KEY, "user").lower()
        content = m.get(MESSAGE_CONTENT_KEY, "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"{SYSTEM_TOKEN}\n{content}")
        elif role == "user":
            parts.append(f"{USER_TOKEN}\n{content}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}\n{content}")
        else:
            parts.append(content)
    return "\n".join(parts)


def _format_prompt_completion(prompt: str, completion: str) -> str:
    """Format prompt + completion with special tokens."""
    return f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n{completion}"


class SFTDataset(Dataset):
    """
    Dataset for SFT from 05_sft_chat_train.jsonl / 06_sft_chat_val.jsonl.
    Each item: {"input_ids": [...], "labels": [...]} (labels = input_ids with non-assistant positions masked by -100 if needed).
    We use full sequence as target (labels = input_ids) for causal LM; optionally mask non-completion in labels.
    """

    def __init__(
        self,
        path: Union[str, Path],
        tokenizer_dir: Optional[Union[str, Path]] = None,
        max_length: Optional[int] = 512,
        mask_prompt_in_labels: bool = True,
    ):
        """
        Args:
            path: Path to JSONL file.
            tokenizer_dir: Directory with trained tokenizer. If None, use default MODEL_TOKENIZER.
            max_length: Truncate sequences to this length.
            mask_prompt_in_labels: If True, set labels to -100 for prompt positions (only assistant part is trained).
        """
        self.path = Path(path)
        self.max_length = max_length
        self.mask_prompt_in_labels = mask_prompt_in_labels
        self.sp = load_tokenizer(tokenizer_dir)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        self.records: List[Dict[str, Any]] = []
        if self.path.exists():
            self.records = read_jsonl(self.path)
        else:
            logger.warning("SFT file not found: %s", self.path)
        logger.info("Loaded %d SFT records from %s", len(self.records), self.path)

    def _record_to_text(self, record: Dict[str, Any]) -> str:
        if SFT_MESSAGES_KEY in record:
            return _format_messages(record[SFT_MESSAGES_KEY])
        prompt = record.get(SFT_PROMPT_KEY, "")
        completion = record.get(SFT_COMPLETION_KEY, "")
        return _format_prompt_completion(prompt, completion)

    def _text_to_ids_and_labels(self, text: str) -> tuple:
        ids = self.sp.encode(text, add_bos=True, add_eos=True, out_type=int)
        if self.max_length:
            ids = ids[: self.max_length]
        if self.mask_prompt_in_labels:
            # For prompt/completion format we could mask prompt; for full chat we train on all (simpler)
            labels = list(ids)
        else:
            labels = list(ids)
        return ids, labels

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        text = self._record_to_text(record)
        input_ids, labels = self._text_to_ids_and_labels(text)
        return {"input_ids": input_ids, "labels": labels}
