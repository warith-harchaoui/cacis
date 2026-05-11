"""
examples.imagenet.inference
===========================

Run a trained cost-aware ImageNet classifier on a single image, a directory,
or an ``ImageFolder``-style validation set.

Two prediction modes
--------------------
The whole point of cost-aware training is that ``argmax(p)`` is **not** the
right decision rule when costs are asymmetric. This script reports both:

- **argmax**: the standard top-1 / top-k prediction from the softmax.
- **cost-optimal**: the action ``a*`` that minimizes the expected cost under
  the model's predicted probabilities::

      a*(p) = argmin_a  Σ_i p_i · C[i, a]

  When a cost matrix is provided, the action label is selected via this rule.
  Without a cost matrix, the action equals the argmax.

Outputs
-------
- Single image:  prints top-k probabilities + the cost-optimal action.
- Directory:     writes ``predictions.csv`` (one row per image).
- ``ImageFolder`` validation set: reports Top-1, Top-5, and realized semantic
  regret under both decision rules.

CLI
---
::

    # Single image
    python -m examples.imagenet.inference \\
        --checkpoint imagenet_output/resnet50_fasttext/checkpoint_best.pt \\
        --cost-matrix cost_matrix.pt \\
        --image /path/to/photo.jpg \\
        --topk 5

    # Directory of images
    python -m examples.imagenet.inference \\
        --checkpoint .../checkpoint_best.pt \\
        --cost-matrix cost_matrix.pt \\
        --input-dir /path/to/photos/ \\
        --output predictions.csv

    # ImageFolder-style val set (full evaluation)
    python -m examples.imagenet.inference \\
        --checkpoint .../checkpoint_best.pt \\
        --cost-matrix cost_matrix.pt \\
        --val-dir /data/imagenet/val

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from examples.imagenet.model import build_resnet

logger = logging.getLogger(__name__)


_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# =============================================================================
# Loading
# =============================================================================

def load_checkpoint_bundle(path: Path) -> Dict:
    """Load a checkpoint produced by ``examples.imagenet.train.save_checkpoint``."""
    return torch.load(path, map_location="cpu")


def build_model_from_checkpoint(
    bundle: Dict, *, arch: Optional[str] = None, device: torch.device
) -> torch.nn.Module:
    """
    Instantiate a ResNet matching the architecture used at training time.

    The training script stores its ``argparse.Namespace`` in ``bundle['args']``,
    so we read ``arch`` from there by default and allow an explicit override.
    """
    saved_args = bundle.get("args", {})
    if arch is None:
        arch = saved_args.get("arch", "resnet50")
    model = build_resnet(arch, num_classes=1000)
    state_dict = bundle["model"]
    # Strip a possible ``module.`` prefix from DDP-saved checkpoints.
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_cost_matrix(path: Optional[Path], device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    """Load ``cost_matrix.pt`` bundle (``{C, class_names}``)."""
    if path is None:
        return None, None
    bundle = torch.load(path, map_location="cpu")
    C = bundle["C"].float().to(device)
    class_names = bundle.get("class_names")
    return C, class_names


# =============================================================================
# Transforms
# =============================================================================

def build_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


# =============================================================================
# Predictions
# =============================================================================

@torch.no_grad()
def predict_probs(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Return the softmax distribution for an ``(B, 3, H, W)`` batch."""
    return F.softmax(model(batch), dim=1)


def cost_optimal_action(probs: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    Action minimizing expected cost under the predicted distribution.

    Parameters
    ----------
    probs : (B, K) softmax distribution.
    C     : (K, K) cost matrix where ``C[i, j]`` is the cost of action ``j``
            given true class ``i``.

    Returns
    -------
    (B,) int64 tensor of chosen action indices.
    """
    expected_cost = probs @ C            # (B, K)
    return expected_cost.argmin(dim=1)


def topk_with_labels(
    probs: torch.Tensor,
    class_names: Optional[List[str]],
    k: int,
) -> List[List[Tuple[int, float, str]]]:
    """
    Top-k indices, probabilities, and labels per row of ``probs``.

    Returns a list (length B) of lists (length k) of ``(idx, prob, label)``.
    """
    top_p, top_i = probs.topk(k, dim=1)
    out: List[List[Tuple[int, float, str]]] = []
    for row_i, row_p in zip(top_i.tolist(), top_p.tolist()):
        out.append([
            (int(i), float(p), (class_names[i] if class_names else str(i)))
            for i, p in zip(row_i, row_p)
        ])
    return out


# =============================================================================
# Single-image / directory entrypoints
# =============================================================================

def predict_single(
    model: torch.nn.Module,
    image_path: Path,
    *,
    transform: transforms.Compose,
    device: torch.device,
    C: Optional[torch.Tensor],
    class_names: Optional[List[str]],
    topk: int,
) -> Dict:
    """Predict on one image; return a JSON-serializable dict."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    probs = predict_probs(model, tensor)
    top = topk_with_labels(probs, class_names, topk)[0]

    out: Dict = {
        "path": str(image_path),
        "argmax_label": top[0][2],
        "argmax_prob": top[0][1],
        f"top{topk}": [
            {"class_idx": idx, "prob": p, "label": name} for idx, p, name in top
        ],
    }

    if C is not None:
        action = int(cost_optimal_action(probs, C).item())
        out["cost_optimal_action_idx"] = action
        out["cost_optimal_action_label"] = (
            class_names[action] if class_names else str(action)
        )
    return out


def _iter_images(directory: Path) -> List[Path]:
    """All files under ``directory`` (recursively) with a known image extension."""
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def predict_directory(
    model: torch.nn.Module,
    directory: Path,
    *,
    transform: transforms.Compose,
    device: torch.device,
    C: Optional[torch.Tensor],
    class_names: Optional[List[str]],
    batch_size: int,
    output_csv: Path,
) -> None:
    """
    Write one row per image to ``output_csv``.

    Columns: path, argmax_idx, argmax_label, argmax_prob,
             [cost_optimal_idx, cost_optimal_label].
    """
    paths = _iter_images(directory)
    if not paths:
        raise FileNotFoundError(f"No image files found under {directory}.")

    logger.info("Predicting on %d images...", len(paths))

    fieldnames = ["path", "argmax_idx", "argmax_label", "argmax_prob"]
    if C is not None:
        fieldnames += ["cost_optimal_idx", "cost_optimal_label"]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            batch = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in chunk
            ]).to(device)
            probs = predict_probs(model, batch)
            argmax = probs.argmax(dim=1).tolist()
            argmax_p = probs.gather(1, probs.argmax(dim=1, keepdim=True)).squeeze(1).tolist()

            actions = (
                cost_optimal_action(probs, C).tolist() if C is not None else argmax
            )

            for path, a_idx, a_p, c_idx in zip(chunk, argmax, argmax_p, actions):
                row = {
                    "path": str(path),
                    "argmax_idx": a_idx,
                    "argmax_label": class_names[a_idx] if class_names else str(a_idx),
                    "argmax_prob": f"{a_p:.6f}",
                }
                if C is not None:
                    row["cost_optimal_idx"] = c_idx
                    row["cost_optimal_label"] = class_names[c_idx] if class_names else str(c_idx)
                writer.writerow(row)
    logger.info("Wrote %s", output_csv)


# =============================================================================
# Full validation
# =============================================================================

@torch.no_grad()
def evaluate_imagefolder(
    model: torch.nn.Module,
    val_dir: Path,
    *,
    device: torch.device,
    C: Optional[torch.Tensor],
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    """
    Top-1 / Top-5 and (if ``C`` is provided) realized semantic regret under
    both decision rules.
    """
    transform = build_val_transform()
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    n_total = n_top1 = n_top5 = 0
    sum_regret_argmax = 0.0
    sum_regret_cost_opt = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        probs = predict_probs(model, images)

        _, top5 = probs.topk(5, dim=1)
        correct = top5.eq(targets.view(-1, 1))
        n_top1 += int(correct[:, :1].sum().item())
        n_top5 += int(correct.any(dim=1).sum().item())
        n_total += targets.size(0)

        if C is not None:
            argmax = probs.argmax(dim=1)
            action = cost_optimal_action(probs, C)
            sum_regret_argmax += float(C[targets, argmax].sum().item())
            sum_regret_cost_opt += float(C[targets, action].sum().item())

    out: Dict[str, float] = {
        "n_total": n_total,
        "top1": n_top1 / max(n_total, 1),
        "top5": n_top5 / max(n_total, 1),
    }
    if C is not None:
        out["realized_regret_argmax"] = sum_regret_argmax / max(n_total, 1)
        out["realized_regret_cost_optimal"] = sum_regret_cost_opt / max(n_total, 1)
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-aware ImageNet inference.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a checkpoint_best.pt / checkpoint_last.pt from train.py.")
    p.add_argument("--cost-matrix", type=Path, default=None,
                   help="Optional cost_matrix.pt bundle for cost-optimal predictions + regret.")
    p.add_argument("--arch", type=str, default=None,
                   help="ResNet variant; defaults to the value stored in the checkpoint.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image file.")
    src.add_argument("--input-dir", type=Path, help="Directory of images (recursive).")
    src.add_argument("--val-dir", type=Path, help="ImageFolder val set for full evaluation.")

    p.add_argument("--output", type=Path, default=Path("predictions.csv"),
                   help="CSV path when --input-dir is used.")
    return p.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    bundle = load_checkpoint_bundle(args.checkpoint)
    model = build_model_from_checkpoint(bundle, arch=args.arch, device=device)

    C, class_names = load_cost_matrix(args.cost_matrix, device=device)
    if C is None:
        logger.info("No cost matrix provided — predictions use argmax only.")
    else:
        logger.info("Cost matrix loaded: shape=%s, classes labeled=%s",
                    tuple(C.shape), bool(class_names))

    if args.image is not None:
        result = predict_single(
            model, args.image,
            transform=build_val_transform(),
            device=device, C=C, class_names=class_names, topk=args.topk,
        )
        print(json.dumps(result, indent=2))
        return

    if args.input_dir is not None:
        predict_directory(
            model, args.input_dir,
            transform=build_val_transform(),
            device=device, C=C, class_names=class_names,
            batch_size=args.batch_size, output_csv=args.output,
        )
        return

    if args.val_dir is not None:
        metrics = evaluate_imagefolder(
            model, args.val_dir,
            device=device, C=C,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
