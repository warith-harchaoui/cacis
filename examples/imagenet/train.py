"""
examples.imagenet.train
=======================

Train a ResNet from scratch on ImageNet with a cost-aware loss whose cost
matrix encodes semantic similarity between class names (built once offline
from FastText, see :mod:`examples.imagenet.cost_matrix`).

Cloud-friendly defaults
-----------------------
- ``torchrun``-driven DDP — works on a single GPU, multi-GPU, or multi-node.
- Mixed precision (``torch.cuda.amp``) is on by default.
- Inputs come from a mounted ``ImageFolder`` directory; checkpoints and logs
  go to a mounted output directory. No cloud SDK calls are made from this
  script — bring-your-own bucket sync if needed.
- Precomputed ε. ``offdiag_mean`` on a (1000, 1000) cost matrix would
  re-materialize a (B, 1000²) mask each batch. We compute the off-diagonal
  statistic **once** here and pass ``epsilon_mode='constant'`` to the loss.

Single-node, single-GPU
-----------------------
::

    python -m examples.imagenet.train \\
        --data-root /data/imagenet \\
        --cost-matrix /models/cost_matrix.pt \\
        --loss sinkhorn_envelope \\
        --batch-size 64 --epochs 90

Vast.ai (single RTX 4090 24 GB, the supported cloud target)
-----------------------------------------------------------
::

    torchrun --standalone --nproc-per-node=1 \\
        -m examples.imagenet.train \\
        --data-root /data/imagefolder \\
        --cost-matrix /workspace/cost_matrix.pt \\
        --loss sinkhorn_envelope \\
        --batch-size 32 --epochs 45

The bootstrap script (``examples/imagenet/cloud/vast_bootstrap.sh``) wraps
this with credential setup, ImageNet download from Kaggle, rclone upload to
Backblaze B2, and self-destruction.

DDP scaffolding remains in this script for future multi-GPU runs; it is
inactive on single-GPU setups.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from cost_aware_losses import (
    SinkhornEnvelopeLoss,
    SinkhornFullAutodiffLoss,
    SinkhornPOTLoss,
)

from examples.imagenet.data import build_loaders
from examples.imagenet.model import build_resnet
from examples.imagenet.plots import plot_curves

logger = logging.getLogger(__name__)


# =============================================================================
# Distributed setup
# =============================================================================

@dataclass
class DistInfo:
    """Snapshot of distributed-training state for the current rank."""
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    is_main: bool


def init_distributed() -> DistInfo:
    """
    Initialize ``torch.distributed`` from ``torchrun`` environment variables.

    Falls back to single-process mode when ``RANK`` / ``WORLD_SIZE`` are unset.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return DistInfo(rank, world_size, local_rank, True, rank == 0)
    return DistInfo(rank=0, world_size=1, local_rank=0, is_distributed=False, is_main=True)


def cleanup_distributed(info: DistInfo) -> None:
    if info.is_distributed:
        dist.destroy_process_group()


def _select_device(args: argparse.Namespace, dist_info: DistInfo) -> torch.device:
    """
    Pick a device, allowing CPU/MPS fallback when ``--allow-cpu`` is set.

    Distributed training still requires CUDA — Gloo on CPU is supported by PyTorch
    but the rest of the pipeline (AMP, NCCL) assumes a GPU. Local smoke-tests run
    single-process, so this restriction does not block them.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{dist_info.local_rank}")
    if dist_info.is_distributed:
        raise RuntimeError("Distributed training requires CUDA.")
    if not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Pass --allow-cpu for local smoke-tests "
            "(see examples.imagenet.smoke_test)."
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Loss factory
# =============================================================================

_SINKHORN_LOSSES = {
    "sinkhorn_envelope": SinkhornEnvelopeLoss,
    "sinkhorn_autodiff": SinkhornFullAutodiffLoss,
    "sinkhorn_pot": SinkhornPOTLoss,
}


def precompute_epsilon(C: torch.Tensor, *, mode: str, scale: float) -> float:
    """
    Compute ε once from a shared cost matrix, avoiding per-batch recomputation.

    For ``mode='constant'`` this is a no-op (returns ``scale``).
    """
    if mode == "constant":
        return float(scale)
    K = C.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
    od = C[mask]
    if mode == "offdiag_mean":
        base = float(od.mean())
    elif mode == "offdiag_median":
        base = float(od.median())
    elif mode == "offdiag_max":
        base = float(od.max())
    else:
        raise ValueError(f"Unknown epsilon mode: {mode}")
    return base * float(scale)


def build_loss(
    loss_name: str,
    *,
    cost_matrix: Optional[torch.Tensor],
    epsilon: Optional[float],
    sinkhorn_max_iter: int,
    label_smoothing: float,
) -> nn.Module:
    """
    Construct the training loss.

    ``cross_entropy`` returns a stock CE; the cost-aware variants are configured
    in ``constant`` ε mode with the precomputed ε passed in via ``epsilon``.
    """
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if cost_matrix is None or epsilon is None:
        raise ValueError(f"Loss '{loss_name}' requires --cost-matrix and a precomputed epsilon.")

    cls = _SINKHORN_LOSSES.get(loss_name)
    if cls is None:
        raise ValueError(
            f"Unknown loss: {loss_name}. Choose from "
            f"{['cross_entropy'] + list(_SINKHORN_LOSSES)}."
        )

    kwargs: Dict[str, Any] = dict(
        epsilon_mode="constant",
        epsilon=float(epsilon),
        epsilon_scale=1.0,
        max_iter=sinkhorn_max_iter,
        label_smoothing=label_smoothing,
    )
    if cls is SinkhornPOTLoss:
        kwargs["allow_numpy_fallback"] = False
    return cls(**kwargs)


def compute_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    cost_matrix: Optional[torch.Tensor],
) -> torch.Tensor:
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return loss_fn(logits, targets)
    return loss_fn(logits, targets, C=cost_matrix)


# =============================================================================
# Metrics
# =============================================================================

@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 5)) -> Dict[int, int]:
    """Top-k correct counts (sums, not averages) for a single batch."""
    max_k = max(ks)
    _, pred = logits.topk(max_k, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    return {k: int(correct[:, :k].any(dim=1).sum().item()) for k in ks}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    cost_matrix: Optional[torch.Tensor],
    dist_info: DistInfo,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute Top-1 / Top-5 accuracy and, when ``cost_matrix`` is provided, the
    realized semantic regret under the cost-optimal decision rule.

    Decision rule: pick the action that minimizes expected cost under the
    softmax distribution::

        a*(p) = argmin_a  ⟨p, C[:, a]⟩

    All sums are all-reduced across ranks before averaging.
    """
    model.eval()
    n_total = n_top1 = n_top5 = 0
    sum_regret = 0.0

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)

        correct = topk_correct(logits, targets, ks=(1, 5))
        n_top1 += correct[1]
        n_top5 += correct[5]
        n_total += targets.size(0)

        if cost_matrix is not None:
            probs = F.softmax(logits, dim=1)            # (B, K)
            expected_cost = probs @ cost_matrix          # (B, K) — E[cost | action]
            actions = expected_cost.argmin(dim=1)        # (B,)
            sum_regret += float(cost_matrix[targets, actions].sum().item())

    if dist_info.is_distributed:
        t = torch.tensor(
            [n_total, n_top1, n_top5, sum_regret],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        n_total, n_top1, n_top5, sum_regret = t.tolist()
        n_total, n_top1, n_top5 = int(n_total), int(n_top1), int(n_top5)

    out = {
        "top1": n_top1 / max(n_total, 1),
        "top5": n_top5 / max(n_total, 1),
    }
    if cost_matrix is not None:
        out["realized_regret"] = sum_regret / max(n_total, 1)
    return out


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    *,
    epoch: int,
    dist_info: DistInfo,
    log_every: int,
    use_amp: bool,
    cost_matrix: Optional[torch.Tensor],
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    running_loss = running_top1 = running_top5 = 0.0
    n_seen = 0
    t0 = time.time()

    for step, (images, targets) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = compute_loss(loss_fn, logits, targets, cost_matrix)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = targets.size(0)
        correct = topk_correct(logits.detach(), targets, ks=(1, 5))
        running_loss += float(loss.item()) * bs
        running_top1 += correct[1]
        running_top5 += correct[5]
        n_seen += bs

        if dist_info.is_main and (step + 1) % log_every == 0:
            elapsed = max(time.time() - t0, 1e-9)
            ips = n_seen / elapsed * dist_info.world_size
            logger.info(
                "ep %d step %d/%d | loss=%.4f top1=%.3f top5=%.3f lr=%.2e | %.0f img/s (global)",
                epoch, step + 1, len(loader),
                running_loss / n_seen,
                running_top1 / n_seen,
                running_top5 / n_seen,
                scheduler.get_last_lr()[0],
                ips,
            )

    return {
        "loss": running_loss / max(n_seen, 1),
        "top1": running_top1 / max(n_seen, 1),
        "top5": running_top5 / max(n_seen, 1),
    }


# =============================================================================
# Checkpointing
# =============================================================================

def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_acc1: float,
    args: argparse.Namespace,
) -> None:
    state = {
        "epoch": epoch,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc1": float(best_acc1),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Restore state in-place and return ``(next_epoch, best_acc1)``.

    ``weights_only=False`` because we store our own ``argparse.Namespace``
    (with ``pathlib.PosixPath``) in the bundle; PyTorch ≥ 2.6 rejects that by
    default. The file is produced by this script and trusted.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    _unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1, float(ckpt.get("best_acc1", 0.0))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImageNet training with cost-aware losses.")

    # Data
    p.add_argument("--data-root", type=Path, required=True,
                   help="ImageNet root (with train/, val/, LOC_synset_mapping.txt).")
    p.add_argument("--cost-matrix", type=Path, default=None,
                   help="Path to a cost_matrix.pt bundle produced by examples.imagenet.cost_matrix")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)

    # Model
    p.add_argument("--arch", type=str, default="resnet50",
                   choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    p.add_argument("--num-classes", type=int, default=1000,
                   help="Output classes. 1000 for real ImageNet; smaller for smoke tests.")

    # Loss
    p.add_argument("--loss", type=str, default="sinkhorn_envelope",
                   choices=["cross_entropy", "sinkhorn_envelope",
                            "sinkhorn_autodiff", "sinkhorn_pot"])
    p.add_argument("--epsilon-mode", type=str, default="offdiag_mean",
                   choices=["offdiag_mean", "offdiag_median", "offdiag_max", "constant"])
    p.add_argument("--epsilon-scale", type=float, default=1.0)
    p.add_argument("--sinkhorn-max-iter", type=int, default=20)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # Optimizer / schedule
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--lr", type=float, default=0.1,
                   help="Reference LR for global batch=256. Scaled linearly to actual global batch.")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed precision (debug only — much slower / more memory).")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Allow CPU/MPS fallback when CUDA is unavailable (local smoke-tests).")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Cap training batches per epoch (smoke-tests).")
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="Cap validation batches per evaluation (smoke-tests).")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test preset: --max-train-batches 5 --max-val-batches 2.")

    # Output
    p.add_argument("--output-dir", type=Path, default=Path("imagenet_output"))
    p.add_argument("--run-id", type=str, default="default")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # --quick is sugar for the two batch caps; explicit caps still win.
    if args.quick:
        if args.max_train_batches is None:
            args.max_train_batches = 5
        if args.max_val_batches is None:
            args.max_val_batches = 2

    dist_info = init_distributed()

    logging.basicConfig(
        level=logging.INFO if dist_info.is_main else logging.WARNING,
        format=f"[rank {dist_info.rank}] %(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = _select_device(args, dist_info)

    torch.manual_seed(args.seed + dist_info.rank)
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    run_dir = args.output_dir / args.run_id
    if dist_info.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "args.json", "w") as f:
            json.dump(
                {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                f, indent=2,
            )

    # -------------------------
    # Cost matrix → precomputed ε
    # -------------------------
    cost_matrix: Optional[torch.Tensor] = None
    epsilon: Optional[float] = None
    if args.cost_matrix is not None:
        bundle = torch.load(args.cost_matrix, map_location="cpu")
        cost_matrix = bundle["C"].float().to(device)
        K = cost_matrix.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=device)
        od = cost_matrix[mask]
        epsilon = precompute_epsilon(cost_matrix, mode=args.epsilon_mode, scale=args.epsilon_scale)
        if dist_info.is_main:
            logger.info(
                "Cost matrix %s | offdiag mean=%.4f median=%.4f max=%.4f | ε=%.6f (%s × %.2f)",
                tuple(cost_matrix.shape),
                float(od.mean()), float(od.median()), float(od.max()),
                epsilon, args.epsilon_mode, args.epsilon_scale,
            )

    # -------------------------
    # Data
    # -------------------------
    train_loader, val_loader, train_sampler = build_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        distributed=dist_info.is_distributed,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------
    # Model
    # -------------------------
    model: nn.Module = build_resnet(args.arch, num_classes=args.num_classes).to(device)
    if dist_info.is_distributed:
        model = DDP(model, device_ids=[dist_info.local_rank])

    # -------------------------
    # Loss
    # -------------------------
    loss_fn = build_loss(
        args.loss,
        cost_matrix=cost_matrix,
        epsilon=epsilon,
        sinkhorn_max_iter=args.sinkhorn_max_iter,
        label_smoothing=args.label_smoothing,
    )

    # -------------------------
    # Optimizer + LR schedule
    # -------------------------
    global_batch = args.batch_size * dist_info.world_size
    scaled_lr = args.lr * global_batch / 256.0
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=scaled_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, steps_per_epoch * args.warmup_epochs)

    warmup = LambdaLR(optimizer, lr_lambda=lambda step: float(step) / float(warmup_steps))
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )

    # AMP is CUDA-only; silently disable on CPU/MPS.
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # -------------------------
    # Resume
    # -------------------------
    ckpt_last = run_dir / "checkpoint_last.pt"
    ckpt_best = run_dir / "checkpoint_best.pt"
    epoch_start = 0
    # ``-inf`` so the first epoch's val_top1 (however small) always wins and
    # writes a checkpoint_best.pt — even on degenerate smoke runs where every
    # val_top1 is 0.0. Real runs comfortably beat -inf on epoch 0 too.
    best_acc1 = float("-inf")
    if args.resume and ckpt_last.exists():
        epoch_start, best_acc1 = load_checkpoint(
            ckpt_last, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
        )
        if dist_info.is_main:
            logger.info(
                "Resumed from %s at epoch %d (best_top1=%.4f).",
                ckpt_last, epoch_start, best_acc1,
            )

    # -------------------------
    # Train
    # -------------------------
    for epoch in range(epoch_start, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device,
            epoch=epoch, dist_info=dist_info, log_every=args.log_every,
            use_amp=use_amp, cost_matrix=cost_matrix,
            max_batches=args.max_train_batches,
        )
        val_stats = evaluate(
            model, val_loader, device,
            cost_matrix=cost_matrix, dist_info=dist_info,
            max_batches=args.max_val_batches,
        )

        if dist_info.is_main:
            tail = (
                f" val_regret={val_stats['realized_regret']:.4f}"
                if "realized_regret" in val_stats else ""
            )
            logger.info(
                "[epoch %d] train_loss=%.4f train_top1=%.3f | val_top1=%.3f val_top5=%.3f%s",
                epoch, train_stats["loss"], train_stats["top1"],
                val_stats["top1"], val_stats["top5"], tail,
            )

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_stats.items()})
            row.update({f"val_{k}": v for k, v in val_stats.items()})
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

            # Re-render curves from metrics.jsonl after every epoch.
            # Each figure is fully re-creatable from the on-disk metrics file via
            #   python -m examples.imagenet.plots --run-dir <run_dir>
            plot_curves(run_dir / "metrics.jsonl", run_dir)

            save_checkpoint(
                ckpt_last, model=model, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_acc1=best_acc1, args=args,
            )
            if val_stats["top1"] > best_acc1:
                best_acc1 = val_stats["top1"]
                save_checkpoint(
                    ckpt_best, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler,
                    epoch=epoch, best_acc1=best_acc1, args=args,
                )

    cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
