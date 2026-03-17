"""
Distributed training: DDP init, rank/world_size, wrap model, barriers.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.logger import get_logger

logger = get_logger(__name__)


def get_local_rank() -> int:
    """Get local rank from env (set by torchrun)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    """Get global rank. 0 if not distributed."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size. 1 if not distributed."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """True if rank 0 (only this process should save checkpoints, log)."""
    return get_rank() == 0


def init_process_group(backend: str = "nccl") -> None:
    """
    Initialize torch.distributed process group.
    Expects RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (set by torchrun).
    """
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)
    if is_main_process():
        logger.info("Distributed init: world_size=%s", get_world_size())


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """Return the device for this process (cuda:local_rank or cuda:0)."""
    if local_rank is None:
        local_rank = get_local_rank()
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def wrap_model_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Wrap model in DDP for multi-GPU. If world_size <= 1, return model as-is."""
    if get_world_size() <= 1:
        return model.to(device)
    model = model.to(device)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    return model


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
