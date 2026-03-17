"""
Train answerability classifier: loop over batches, CE loss, validation accuracy, save best to model/answerability.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from modules.answerability_model import AnswerabilityModel
from modules.collators import DataCollatorForAnswerability
from modules.dataset_answerability import AnswerabilityDataset
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.paths import MODEL_ANSWERABILITY, MODEL_LM, CHECKPOINTS_ANSWERABILITY

logger = get_logger(__name__)


def train_answerability(
    train_path: Path,
    val_path: Optional[Path],
    tokenizer_dir: Path,
    lm_config: LMConfig,
    output_dir: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    max_length: int = 256,
    device: Optional[torch.device] = None,
    freeze_lm: bool = True,
) -> None:
    """
    Train answerability head on top of frozen LM.

    Args:
        train_path: Path to 07_answerability_train.jsonl.
        val_path: Path to 08_answerability_val.jsonl (optional).
        tokenizer_dir: Path to trained tokenizer.
        lm_config: LMConfig (for loading LM).
        output_dir: Save best model here (default: model/answerability).
        checkpoint_dir: Save checkpoints here (default: checkpoints/answerability).
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate for head only.
        max_length: Max sequence length.
        device: Device (default: cuda if available).
        freeze_lm: Whether to freeze LM parameters.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    output_dir = Path(output_dir or MODEL_ANSWERABILITY)
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS_ANSWERABILITY)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AnswerabilityDataset(train_path, tokenizer_dir=tokenizer_dir, max_length=max_length)
    collator = DataCollatorForAnswerability(pad_token_id=lm_config.pad_token_id, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)

    val_loader = None
    if val_path and val_path.exists():
        val_ds = AnswerabilityDataset(val_path, tokenizer_dir=tokenizer_dir, max_length=max_length)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    model = AnswerabilityModel(lm_config, freeze_lm=freeze_lm)
    model.load_lm_weights(MODEL_LM / "pytorch_model.pt")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, loss = model(input_ids, attention_mask, labels)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total else 0
        avg_loss = train_loss / len(train_loader) if train_loader else 0
        logger.info("epoch=%s train_loss=%.4f train_acc=%.4f", epoch, avg_loss, train_acc)

        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    logits, _ = model(input_ids, attention_mask)
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total if val_total else 0
            logger.info("epoch=%s val_acc=%.4f", epoch, val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(output_dir)
                logger.info("Saved best model to %s (val_acc=%.4f)", output_dir, val_acc)

        # Save checkpoint each epoch
        torch.save(
            {"epoch": epoch, "head_state_dict": model.head.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
        )

    if not val_loader:
        model.save_pretrained(output_dir)
        logger.info("Saved model to %s", output_dir)
