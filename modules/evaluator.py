"""
Evaluation: LM loss/perplexity, answerability accuracy/F1, optional inference run for refusal metrics.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from modules.collators import DataCollatorForCausalLM, DataCollatorForAnswerability
from modules.dataset_answerability import AnswerabilityDataset
from modules.dataset_sft import SFTDataset
from modules.logger import get_logger
from modules.metrics import compute_perplexity, correct_refusal_rate, precision_recall_f1, refusal_rate
from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM
from modules.answerability_model import AnswerabilityModel
from modules.paths import MODEL_ANSWERABILITY, MODEL_LM, MODEL_TOKENIZER

logger = get_logger(__name__)


def evaluate_lm(
    model: DecoderLM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute average loss and perplexity on the given dataloader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            _, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if loss is not None:
                total_loss += loss.item()
                n_batches += 1
    avg_loss = total_loss / n_batches if n_batches else 0.0
    return {"loss": avg_loss, "perplexity": compute_perplexity(avg_loss)}


def evaluate_answerability(
    model: AnswerabilityModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute accuracy and F1 for answerability classifier."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0.0
    prec, rec, f1 = precision_recall_f1(all_preds, all_labels, positive_class=1)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "refusal_rate": refusal_rate(all_preds, positive_class=1),
        "correct_refusal_rate": correct_refusal_rate(all_preds, all_labels, positive_class=1),
    }


def run_evaluation(
    val_sft_path: Optional[Path] = None,
    val_answerability_path: Optional[Path] = None,
    tokenizer_dir: Optional[Path] = None,
    lm_dir: Optional[Path] = None,
    answerability_dir: Optional[Path] = None,
    batch_size: int = 8,
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    Run full evaluation: LM val loss/perplexity and answerability val accuracy/F1.
    Returns dict of metric name -> value.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_dir = tokenizer_dir or MODEL_TOKENIZER
    lm_dir = lm_dir or MODEL_LM
    answerability_dir = answerability_dir or MODEL_ANSWERABILITY

    results: Dict[str, Any] = {}

    if val_sft_path and val_sft_path.exists():
        config_path = lm_dir / "config.json"
        if config_path.exists():
            config = LMConfig.load(config_path)
        else:
            config = LMConfig()
        model = DecoderLM(config)
        state = torch.load(lm_dir / "pytorch_model.pt", map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        ds = SFTDataset(val_sft_path, tokenizer_dir=tokenizer_dir, max_length=max_length)
        collator = DataCollatorForCausalLM(pad_token_id=config.pad_token_id, max_length=max_length)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
        results["lm"] = evaluate_lm(model, loader, device)
        logger.info("LM eval: %s", results["lm"])

    if val_answerability_path and val_answerability_path.exists():
        config_path = lm_dir / "config.json"
        if config_path.exists():
            config = LMConfig.load(config_path)
        else:
            config = LMConfig()
        model = AnswerabilityModel(config, freeze_lm=True)
        model.load_lm_weights(lm_dir / "pytorch_model.pt")
        head_path = answerability_dir / "head.pt"
        if head_path.exists():
            model.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        model = model.to(device)
        ds = AnswerabilityDataset(val_answerability_path, tokenizer_dir=tokenizer_dir, max_length=256)
        collator = DataCollatorForAnswerability(pad_token_id=config.pad_token_id, max_length=256)
        loader = DataLoader(ds, batch_size=batch_size, collate_fn=collator)
        results["answerability"] = evaluate_answerability(model, loader, device)
        logger.info("Answerability eval: %s", results["answerability"])

    return results
