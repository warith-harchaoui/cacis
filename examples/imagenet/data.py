"""
examples.imagenet.data
======================

ImageNet ``DataLoader`` factory.

Expects the standard ``ImageFolder`` layout::

    data_root/
        train/
            n01440764/  *.JPEG
            ...
        val/
            n01440764/  *.JPEG
            ...
        LOC_synset_mapping.txt

The loaders are DDP-aware: pass ``distributed=True`` and the train sampler
will be a :class:`torch.utils.data.distributed.DistributedSampler` that you
must ``set_epoch(epoch)`` once per epoch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Standard ImageNet train / val pipelines."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])
    return train_tf, val_tf


def build_loaders(
    data_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    distributed: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Build train and val ``DataLoader``s.

    Parameters
    ----------
    data_root:
        ImageNet root with ``train/`` and ``val/`` subdirectories.
    batch_size:
        Per-process batch size. Global batch is ``batch_size * world_size``.
    num_workers:
        DataLoader worker processes per rank.
    image_size:
        Crop size (default 224).
    distributed:
        Wrap with ``DistributedSampler`` for both splits if True.
    pin_memory:
        Pinned-memory transfers. Disable in CPU-only debug runs.

    Returns
    -------
    train_loader, val_loader, train_sampler
        ``train_sampler`` is ``None`` when ``distributed`` is False; otherwise
        the caller must call ``train_sampler.set_epoch(epoch)`` each epoch.
    """
    train_dir = Path(data_root) / "train"
    val_dir = Path(data_root) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected ImageFolder layout under {data_root} with train/ and val/."
        )

    train_tf, val_tf = _build_transforms(image_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    logger.info(
        "ImageNet: %d train | %d val | %d classes",
        len(train_ds), len(val_ds), len(train_ds.classes),
    )

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_sampler
