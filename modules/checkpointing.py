"""
Checkpoint save/load for LM: state_dict, optimizer, step, config. Only rank 0 saves.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from modules.distributed import barrier, is_main_process
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.paths import CHECKPOINTS_LM

logger = get_logger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    epoch: int,
    config: LMConfig,
    checkpoint_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save LM checkpoint. Only rank 0 writes to disk; all processes barrier after.

    Args:
        model: Model (may be DDP-wrapped; we save module.state_dict()).
        optimizer: Optimizer state (optional).
        step: Global training step.
        epoch: Current epoch.
        config: LMConfig to save.
        checkpoint_dir: Directory for checkpoints. Default: CHECKPOINTS_LM.
        extra: Optional extra dict to save (e.g. scheduler).

    Returns:
        Path to the saved checkpoint directory.
    """
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINTS_LM)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "config": config.to_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra

    if is_main_process():
        path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(state, path)
        config_path = checkpoint_dir / "config.json"
        config.save(config_path, as_yaml=False)
        logger.info("Saved checkpoint to %s (step=%s)", path, step)

    barrier()
    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load LM checkpoint. Returns dict with step, epoch, config (and optional optimizer).

    Args:
        checkpoint_path: Path to .pt file or directory containing checkpoint_step_*.pt.
        model: Model to load state into (may be DDP-wrapped).
        optimizer: If provided, load optimizer state.
        device: Device for tensors.

    Returns:
        Dict with at least 'step', 'epoch', 'config'.
    """
    path = Path(checkpoint_path)
    if path.is_dir():
        # Load latest by step number
        pts = list(path.glob("checkpoint_step_*.pt"))
        if not pts:
            raise FileNotFoundError(f"No checkpoint_step_*.pt in {path}")
        path = max(pts, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (step=%s)", path, state.get("step"))
    return {
        "step": state.get("step", 0),
        "epoch": state.get("epoch", 0),
        "config": state.get("config"),
        "extra": state.get("extra"),
    }


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return path to latest checkpoint .pt in directory, or None."""
    pts = list(Path(checkpoint_dir).glob("checkpoint_step_*.pt"))
    if not pts:
        return None
    return max(pts, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
