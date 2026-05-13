"""
scripts/resize_imagenet_for_cloud.py
====================================

Pre-resize a full ImageNet ILSVRC2012 tree so the bytes that travel to the
cloud are 4-5× smaller, without sacrificing measurable Top-1 accuracy.

Why
---
Standard ResNet ImageNet pipelines do ``RandomResizedCrop(224)`` on training
and ``Resize(256) + CenterCrop(224)`` on validation. The network never sees
more than 224×224 pixels regardless of source resolution. Storing originals
(avg ~500×400) is a 4-5× I/O / network tax for ~0.3 % top-1 difference.

This script:
  - Resizes every JPEG so its **shortest side** is ``--short-side`` (256 by default).
  - Preserves aspect ratio.
  - Re-encodes as JPEG at ``--quality`` (90 by default).
  - Mirrors the source directory tree verbatim into ``--dst``.
  - Uses multiprocessing — saturates CPU cores; ~1-2 h on a 10-core Mac.

Layout
------
Source (after extracting the Kaggle zip)::

    /Volumes/orange/imagenet-raw/
        ILSVRC/Data/CLS-LOC/train/<synset>/*.JPEG
        ILSVRC/Data/CLS-LOC/val/*.JPEG          ← still flat at this stage
        LOC_synset_mapping.txt
        LOC_val_solution.csv

Destination (mirrors source, resized; val/ stays flat — the vast bootstrap
already knows how to reorganize it into synset subfolders)::

    /Volumes/orange/imagenet-resized-256/
        ILSVRC/Data/CLS-LOC/train/<synset>/*.JPEG
        ILSVRC/Data/CLS-LOC/val/*.JPEG
        LOC_synset_mapping.txt    (copied verbatim)
        LOC_val_solution.csv      (copied verbatim)

Then::

    cd /Volumes/orange
    zip -r -1 imagenet-resized-256.zip imagenet-resized-256/
    # ~ 30 GB output — small enough for deraison.ai

CLI
---
    python scripts/resize_imagenet_for_cloud.py \\
        --src /Volumes/orange/imagenet-raw \\
        --dst /Volumes/orange/imagenet-resized-256 \\
        --short-side 256 \\
        --quality 90 \\
        --workers 10

Resume-safe: existing destination files are skipped, so killing the script
and restarting just picks up where it left off.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageFile

# Some Kaggle images have truncated EXIF — let PIL load anyway.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# Per-image worker
# =============================================================================

def _resize_one(args: Tuple[Path, Path, int, int]) -> Tuple[Path, str]:
    """
    Resize one JPEG. Returns ``(path, status)`` where status is one of
    ``ok`` / ``skipped`` / ``error:<reason>``.

    Errors are tolerated — corrupt files happen, and one bad apple
    shouldn't kill a multi-hour run.
    """
    src, dst, short_side, quality = args
    if dst.exists() and dst.stat().st_size > 0:
        return src, "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src) as im:
            # JPEGs from ImageNet are often loaded as RGB by default but some
            # are 'L' (grayscale) or 'P' (palette) or 'CMYK'. Force RGB.
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if min(w, h) > short_side:
                if w < h:
                    new_w = short_side
                    new_h = int(round(h * short_side / w))
                else:
                    new_h = short_side
                    new_w = int(round(w * short_side / h))
                im = im.resize((new_w, new_h), Image.BICUBIC)
            im.save(dst, format="JPEG", quality=quality, optimize=True, progressive=False)
        return src, "ok"
    except Exception as exc:  # pragma: no cover — defensive
        return src, f"error:{type(exc).__name__}:{exc}"


# =============================================================================
# Tree walk
# =============================================================================

def _iter_jobs(src_root: Path, dst_root: Path, short_side: int, quality: int) -> Iterable[Tuple[Path, Path, int, int]]:
    """Yield (src_jpeg, dst_jpeg, short_side, quality) tuples."""
    for jpeg in src_root.rglob("*.JPEG"):
        rel = jpeg.relative_to(src_root)
        yield (jpeg, dst_root / rel, short_side, quality)
    # Also handle .jpg/.jpeg lowercase (rare in ImageNet but cheap to support)
    for ext in ("*.jpg", "*.jpeg"):
        for jpeg in src_root.rglob(ext):
            rel = jpeg.relative_to(src_root)
            yield (jpeg, dst_root / rel, short_side, quality)


def _copy_metadata(src_root: Path, dst_root: Path) -> None:
    """Copy the non-image companion files verbatim (synset map, val solutions)."""
    for name in ("LOC_synset_mapping.txt", "LOC_val_solution.csv",
                 "LOC_train_solution.csv", "LOC_sample_submission.csv"):
        src = src_root / name
        if src.exists():
            dst = dst_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ✓ copied metadata: {name}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resize an ImageNet ILSVRC2012 tree for cloud upload.",
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Directory containing the extracted Kaggle dataset (with ILSVRC/Data/CLS-LOC/...).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output directory (will mirror src). Resume-safe.")
    p.add_argument("--short-side", type=int, default=256,
                   help="Resize so the shorter edge equals this many pixels (default: 256).")
    p.add_argument("--quality", type=int, default=90,
                   help="JPEG re-encode quality 1-95 (default: 90).")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2),
                   help="Parallel worker processes (default: cpu_count − 2).")
    p.add_argument("--progress-every", type=int, default=5000,
                   help="Print a progress line every N images (default: 5000).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.src.exists():
        print(f"✗ source does not exist: {args.src}", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"src         : {args.src}")
    print(f"dst         : {args.dst}")
    print(f"short side  : {args.short_side}")
    print(f"quality     : {args.quality}")
    print(f"workers     : {args.workers}")
    print()

    print("→ Copying metadata files ...")
    _copy_metadata(args.src, args.dst)
    print()

    print("→ Counting source JPEGs ...")
    jobs = list(_iter_jobs(args.src, args.dst, args.short_side, args.quality))
    total = len(jobs)
    print(f"  found {total:,} images")
    print()

    if total == 0:
        print("Nothing to do.")
        return 0

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errs_shown = 0

    print(f"→ Resizing with {args.workers} workers ...")
    with mp.Pool(processes=args.workers) as pool:
        for i, (src, status) in enumerate(
            pool.imap_unordered(_resize_one, jobs, chunksize=64), start=1
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                if errs_shown < 10:
                    print(f"  ✗ {src}: {status}")
                    errs_shown += 1
                elif errs_shown == 10:
                    print("  (further error details suppressed)")
                    errs_shown += 1

            if i % args.progress_every == 0:
                rate = i / max(time.time() - t0, 1e-9)
                eta_min = (total - i) / max(rate, 1e-9) / 60.0
                print(f"  {i:>9,}/{total:,}  ({100.0*i/total:5.1f}%)  "
                      f"{rate:7.0f} img/s  eta {eta_min:5.1f} min  "
                      f"[ok={n_ok:,} skipped={n_skip:,} err={n_err:,}]")

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  ok     : {n_ok:,}")
    print(f"  skipped: {n_skip:,}")
    print(f"  errors : {n_err:,}")
    if n_err > 0:
        print(f"  (corrupt source JPEGs are tolerated; {n_err} files left unwritten in dst.)")

    # Helpful next-step hint
    print()
    print("Next:")
    print(f"  cd {args.dst.parent}")
    print(f"  zip -r -1 {args.dst.name}.zip {args.dst.name}/")
    print(f"  rsync -avP --progress {args.dst.name}.zip <user>@deraison.ai:~/web/deraison/ai/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
