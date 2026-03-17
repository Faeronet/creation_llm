"""
LM training loop: DataLoader with DistributedSampler, forward, loss, backward, checkpointing, logging.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modules.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from modules.collators import DataCollatorForCausalLM
from modules.distributed import get_device, get_world_size, init_process_group, is_main_process, wrap_model_ddp
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM
from modules.paths import CHECKPOINTS_LM, MODEL_LM
from modules.dataset_sft import SFTDataset

logger = get_logger(__name__)


def train_lm(
    train_path: Path,
    val_path: Optional[Path],
    tokenizer_dir: Path,
    config: LMConfig,
    output_checkpoint_dir: Optional[Path] = None,
    output_model_dir: Optional[Path] = None,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 1e-4,
    max_length: int = 512,
    checkpoint_every_steps: Optional[int] = None,
    checkpoint_every_epoch: bool = True,
    resume: bool = True,
) -> None:
    """
    Train decoder-only LM. Uses DDP if world_size > 1.

    Args:
        train_path: Path to 05_sft_chat_train.jsonl.
        val_path: Path to 06_sft_chat_val.jsonl (optional).
        tokenizer_dir: Path to trained tokenizer.
        config: LMConfig.
        output_checkpoint_dir: Where to save checkpoints. Default: checkpoints/lm.
        output_model_dir: Where to save final model. Default: model/lm.
        batch_size: Per-device batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        max_length: Max sequence length.
        checkpoint_every_steps: Save checkpoint every N steps (optional).
        checkpoint_every_epoch: Save checkpoint at end of each epoch.
        resume: If True, resume from latest checkpoint in output_checkpoint_dir.
    """
    if get_world_size() > 1:
        init_process_group()
    device = get_device()
    output_checkpoint_dir = Path(output_checkpoint_dir or CHECKPOINTS_LM)
    output_model_dir = Path(output_model_dir or MODEL_LM)
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SFTDataset(train_path, tokenizer_dir=tokenizer_dir, max_length=max_length)
    collator = DataCollatorForCausalLM(pad_token_id=config.pad_token_id, max_length=max_length)

    sampler = None
    shuffle = True
    if get_world_size() > 1:
        sampler = DistributedSampler(train_ds, shuffle=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    model = DecoderLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start_step = 0
    start_epoch = 0

    if resume:
        latest = find_latest_checkpoint(output_checkpoint_dir)
        if latest is not None:
            ckpt_info = load_checkpoint(latest, model, optimizer, device)
            start_step = ckpt_info.get("step", 0)
            start_epoch = ckpt_info.get("epoch", 0)
            if is_main_process():
                logger.info("Resumed from step %s epoch %s", start_step, start_epoch)

    model = wrap_model_ddp(model, device)
    model.train()

    global_step = start_step
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            _, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning("Invalid loss, skipping batch")
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()
            num_batches += 1

            if checkpoint_every_steps and global_step % checkpoint_every_steps == 0 and is_main_process():
                raw = model.module if hasattr(model, "module") else model
                save_checkpoint(raw, optimizer, global_step, epoch, config, checkpoint_dir=output_checkpoint_dir)

            if is_main_process() and num_batches % 10 == 0:
                ppl = torch.exp(loss.detach()).item()
                logger.info("step=%s loss=%.4f perplexity=%.2f", global_step, loss.item(), ppl)

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            if is_main_process():
                logger.info("epoch=%s avg_loss=%.4f perplexity=%.2f", epoch, avg_loss, torch.exp(torch.tensor(avg_loss)).item())

        if checkpoint_every_epoch and is_main_process():
            raw = model.module if hasattr(model, "module") else model
            save_checkpoint(raw, optimizer, global_step, epoch, config, checkpoint_dir=output_checkpoint_dir)

    if is_main_process():
        raw = model.module if hasattr(model, "module") else model
        torch.save({"model_state_dict": raw.state_dict(), "config": config.to_dict()}, output_model_dir / "pytorch_model.pt")
        config.save(output_model_dir / "config.json")
        logger.info("Saved final model to %s", output_model_dir)
