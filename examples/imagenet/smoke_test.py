"""
examples.imagenet.smoke_test
============================

End-to-end local smoke test for the ImageNet cost-aware pipeline.

What it does
------------
1. Downloads **Imagenette** (the canonical 10-class ImageNet subset from
   fast.ai, ~95 MB, ``imagenette2-160``) to ``~/.cache/cacis/`` if it isn't
   already cached. Subsequent runs are offline.
2. Samples real JPEGs into a small ``train/`` / ``val/`` / ``test/`` layout
   under the smoke workdir, with the real WordNet synset directory names
   (``n01440764``, …) and human-readable labels in ``LOC_synset_mapping.txt``.
3. Builds a (10, 10) random cost matrix stand-in for the FastText one (the
   point of the smoke test is the *pipeline*, not the embedding quality).
4. Runs :mod:`examples.imagenet.train` on CPU/MPS for 2 epochs of 4 batches
   each, producing checkpoints, ``metrics.jsonl``, ``metrics.csv``, and the
   three curve PNGs.
5. Re-renders the curves from disk via :mod:`examples.imagenet.plots`, proving
   each figure is reproducible from the data files alone.
6. Runs :mod:`examples.imagenet.inference` in all three modes (single image,
   directory, ``ImageFolder`` val/test set) against the test split.
7. Lists every artifact produced and verifies the critical ones are present.

First run takes ~1–2 min (most of it the imagenette download); subsequent
runs are ~30 s.

CLI
---
::

    python -m examples.imagenet.smoke_test
    python -m examples.imagenet.smoke_test --loss cross_entropy
    python -m examples.imagenet.smoke_test --loss all   # try every loss in sequence
    python -m examples.imagenet.smoke_test --keep       # don't clean up workdir

Exit code is 0 on success, non-zero on any failure (subprocess error or missing
artifact). Safe to wire into CI.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# Critical artifacts the run directory must contain at the end of training.
_REQUIRED_RUN_ARTIFACTS = (
    "args.json",
    "metrics.jsonl",
    "metrics.csv",
    "checkpoint_last.pt",
    "checkpoint_best.pt",
    "curve_loss.png",
    "curve_accuracy.png",
    "curve_regret.png",   # cost-aware runs only; checked conditionally below
)

_ALL_LOSSES = (
    "cross_entropy",
    "sinkhorn_envelope",
    "sinkhorn_autodiff",
    "sinkhorn_pot",
)


# =============================================================================
# Imagenette dataset (real ImageNet images, 10 classes, ~95 MB cached once)
# =============================================================================

# Real ImageNet labels for the 10 Imagenette synsets — used to write a
# LOC_synset_mapping.txt that matches the real distribution.
_IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

_IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
_IMAGENETTE_CACHE = Path.home() / ".cache" / "cacis"


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Minimal urlretrieve progress hook — one line, updated in place."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, 100.0 * downloaded / total_size)
    print(
        f"\r  → imagenette download: {pct:5.1f}%  ({downloaded // (1024*1024)} / "
        f"{total_size // (1024*1024)} MB)",
        end="", flush=True,
    )
    if downloaded >= total_size:
        print()  # newline at the end


def ensure_imagenette(cache_root: Path = _IMAGENETTE_CACHE) -> Path:
    """
    Return the path to a cached ``imagenette2-160`` tree, downloading it once.

    Layout after extract::

        cache_root/imagenette2-160/
            train/<synset>/*.JPEG
            val/<synset>/*.JPEG
    """
    extracted = cache_root / "imagenette2-160"
    if (extracted / "train").exists() and (extracted / "val").exists():
        return extracted

    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / "imagenette2-160.tgz"

    if not archive.exists():
        print(f"  → Fetching Imagenette (~95 MB) from {_IMAGENETTE_URL}")
        urllib.request.urlretrieve(_IMAGENETTE_URL, archive, _reporthook)

    print(f"  → Extracting to {cache_root}")
    with tarfile.open(archive) as tf:
        tf.extractall(cache_root)
    return extracted


def _build_imagenette_smoke_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
) -> List[str]:
    """
    Populate ``dst_root/{train,val,test}/<synset>/`` from real Imagenette JPEGs.

    Sampling is **disjoint** across splits within each class so we don't leak
    the same image into multiple roles. Returns the sorted synset list.
    """
    synsets = sorted(d.name for d in (src_root / "train").iterdir() if d.is_dir())
    rng = random.Random(42)

    n_needed = train_per_class + val_per_class + test_per_class

    for synset in synsets:
        candidates = sorted((src_root / "train" / synset).glob("*.JPEG"))
        rng.shuffle(candidates)
        if len(candidates) < n_needed:
            raise RuntimeError(
                f"Class {synset} has {len(candidates)} images, "
                f"need {n_needed} for the smoke test."
            )
        a = train_per_class
        b = a + val_per_class
        c = b + test_per_class
        splits = {
            "train": candidates[:a],
            "val":   candidates[a:b],
            "test":  candidates[b:c],
        }
        for split, imgs in splits.items():
            out_dir = dst_root / split / synset
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, out_dir / img.name)

    # LOC_synset_mapping.txt with the real ImageNet labels for these synsets.
    with open(dst_root / "LOC_synset_mapping.txt", "w") as f:
        for s in synsets:
            f.write(f"{s} {_IMAGENETTE_LABELS.get(s, 'unknown')}\n")

    return synsets


def _build_synthetic_cost_matrix(n_classes: int, path: Path) -> None:
    """Random symmetric cost matrix with zero diagonal."""
    rng = np.random.default_rng(seed=42)
    C = rng.uniform(0.0, 1.0, size=(n_classes, n_classes)).astype(np.float32)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 0.0)
    torch.save(
        {
            "C": torch.from_numpy(C),
            "class_names": [f"class_{i}" for i in range(n_classes)],
        },
        path,
    )


# =============================================================================
# Subprocess runners
# =============================================================================

def _run(cmd: List[str], *, env: dict = None) -> int:
    """Run a subprocess, streaming output, returning the exit code."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env)


def _run_train(
    *,
    data_root: Path,
    cost_matrix_path: Path,
    output_dir: Path,
    run_id: str,
    loss: str,
    n_classes: int,
) -> int:
    cmd = [
        sys.executable, "-m", "examples.imagenet.train",
        "--data-root", str(data_root),
        "--cost-matrix", str(cost_matrix_path),
        "--loss", loss,
        "--arch", "resnet18",
        "--num-classes", str(n_classes),
        "--batch-size", "4",
        "--epochs", "2",
        "--warmup-epochs", "1",
        "--quick",                 # 5 train / 2 val batches per epoch
        "--num-workers", "0",
        "--sinkhorn-max-iter", "5",
        "--output-dir", str(output_dir),
        "--run-id", run_id,
        "--allow-cpu",
        "--no-amp",
        "--log-every", "1",
    ]
    return _run(cmd)


def _run_inference_modes(
    *,
    checkpoint: Path,
    cost_matrix_path: Path,
    test_dir: Path,
    workdir: Path,
) -> List[Tuple[str, int]]:
    """Run inference in all three modes; return [(mode, exit_code), ...]."""
    single_image = next(test_dir.rglob("*.JPEG"))
    results: List[Tuple[str, int]] = []
    for mode, extra in [
        ("single-image", ["--image", str(single_image)]),
        ("directory", ["--input-dir", str(test_dir), "--output", str(workdir / "predictions.csv")]),
        ("imagefolder", ["--val-dir", str(test_dir)]),
    ]:
        print(f"\n--- inference: {mode} ---")
        rc = _run([
            sys.executable, "-m", "examples.imagenet.inference",
            "--checkpoint", str(checkpoint),
            "--cost-matrix", str(cost_matrix_path),
            "--device", "cpu",
        ] + extra)
        results.append((mode, rc))
    return results


# =============================================================================
# Verification
# =============================================================================

def _verify_artifacts(run_dir: Path, loss: str) -> List[str]:
    """
    Return the list of missing required artifacts (empty list = success).
    """
    missing: List[str] = []
    for art in _REQUIRED_RUN_ARTIFACTS:
        if art == "curve_regret.png" and loss == "cross_entropy":
            # cross-entropy doesn't produce regret curves
            continue
        if not (run_dir / art).exists():
            missing.append(art)
    return missing


def _list_artifacts(run_dir: Path, workdir: Path) -> None:
    print("\n=== Artifacts ===")
    for root in (run_dir, workdir):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and "synthetic_imagenet" not in p.parts:
                size_kb = p.stat().st_size / 1024
                print(f"  {p.relative_to(workdir.parent)}  ({size_kb:,.1f} KB)")


# =============================================================================
# Driver
# =============================================================================

def run_one_loss(
    *,
    workdir: Path,
    data_root: Path,
    cost_matrix_path: Path,
    output_dir: Path,
    loss: str,
    n_classes: int,
) -> int:
    """Run a full smoke cycle for a single loss. Returns 0 on success."""
    run_id = f"smoke-{loss}"
    run_dir = output_dir / run_id

    print(f"\n{'=' * 60}")
    print(f"=== Smoke test: loss = {loss}")
    print(f"{'=' * 60}")

    print(f"\n[train] running 2 epochs × 5 train batches × 2 val batches ...")
    rc = _run_train(
        data_root=data_root,
        cost_matrix_path=cost_matrix_path,
        output_dir=output_dir,
        run_id=run_id,
        loss=loss,
        n_classes=n_classes,
    )
    if rc != 0:
        print(f"\n  ✗ training failed (exit {rc})", file=sys.stderr)
        return rc

    print("\n[replot] regenerating curves from metrics.jsonl ...")
    rc = _run([
        sys.executable, "-m", "examples.imagenet.plots",
        "--run-dir", str(run_dir),
    ])
    if rc != 0:
        print(f"\n  ✗ replot failed (exit {rc})", file=sys.stderr)
        return rc

    print("\n[inference] running 3 modes ...")
    results = _run_inference_modes(
        checkpoint=run_dir / "checkpoint_best.pt",
        cost_matrix_path=cost_matrix_path,
        test_dir=data_root / "test",
        workdir=workdir,
    )
    for mode, rc in results:
        if rc != 0:
            print(f"\n  ✗ inference ({mode}) failed (exit {rc})", file=sys.stderr)
            return rc

    missing = _verify_artifacts(run_dir, loss)
    if missing:
        print(f"\n  ✗ missing artifacts: {missing}", file=sys.stderr)
        return 1

    print(f"\n  ✓ {loss}: all artifacts present")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end local smoke test for the ImageNet cost-aware pipeline.",
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path(".smoke-test"),
        help="Where synthetic data and outputs go (default: ./.smoke-test).",
    )
    parser.add_argument(
        "--loss", type=str, default="sinkhorn_envelope",
        choices=("all",) + _ALL_LOSSES,
        help="Loss to test; pass 'all' to cycle through every loss.",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Don't delete workdir at the end (useful for inspecting artifacts).",
    )
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    data_root = workdir / "imagenette_smoke"
    cost_matrix_path = workdir / "cost_matrix.pt"
    output_dir = workdir / "output"

    print(f"workdir         : {workdir}")
    print(f"loss(es)        : {args.loss}")
    print(f"python          : {sys.executable}")

    print("\n[setup] ensuring Imagenette is cached ...")
    imagenette_root = ensure_imagenette()
    print(f"  ✓ Imagenette available at {imagenette_root}")

    print("[setup] sampling real JPEGs into train/val/test ...")
    synsets = _build_imagenette_smoke_dataset(
        imagenette_root, data_root,
        train_per_class=8, val_per_class=4, test_per_class=4,
    )
    n_classes = len(synsets)
    print(f"  ✓ {n_classes} classes, 8 train / 4 val / 4 test each")

    print("[setup] generating cost matrix ...")
    _build_synthetic_cost_matrix(n_classes, cost_matrix_path)

    losses = _ALL_LOSSES if args.loss == "all" else (args.loss,)
    failures: List[str] = []
    for loss in losses:
        rc = run_one_loss(
            workdir=workdir,
            data_root=data_root,
            cost_matrix_path=cost_matrix_path,
            output_dir=output_dir,
            loss=loss,
            n_classes=n_classes,
        )
        if rc != 0:
            failures.append(loss)

    if losses == _ALL_LOSSES:
        _list_artifacts(
            run_dir=output_dir / f"smoke-{losses[-1]}",
            workdir=workdir,
        )
    else:
        _list_artifacts(
            run_dir=output_dir / f"smoke-{losses[0]}",
            workdir=workdir,
        )

    print()
    if failures:
        print(f"✗ FAILED for: {failures}")
        return 1

    print(f"✓ Smoke test passed for: {list(losses)}")
    if args.keep:
        print(f"(workdir preserved at {workdir})")
    else:
        shutil.rmtree(workdir)
        print(f"(cleaned up {workdir} — pass --keep to inspect artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
