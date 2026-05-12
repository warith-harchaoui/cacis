"""
examples.imagenet.plots
=======================

Render per-epoch training curves from a run directory.

Design contract: **every figure must be re-creatable, on any machine, from data
that lives in the run directory.** The single source of truth is
``metrics.jsonl`` — one JSON object per line, one line per epoch — written by
``examples.imagenet.train``. This module reads it back and produces the PNGs.

The training loop calls :func:`plot_curves` after each epoch; you can also
regenerate the figures at any time via the CLI::

    python -m examples.imagenet.plots --run-dir imagenet_output/<run_id>

Three PNGs are produced, plus a small ``metrics.csv`` companion file (same
data, spreadsheet-friendly):

- ``curve_loss.png``      — training loss vs epoch
- ``curve_accuracy.png``  — train top-1, val top-1, val top-5 vs epoch
- ``curve_regret.png``    — val realized semantic regret vs epoch (cost-aware runs only)
- ``metrics.csv``         — same rows as ``metrics.jsonl`` in CSV form

Matplotlib is the only runtime dependency; it ships with the base requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importing examples.utils both registers Montserrat with matplotlib (via
# setup_plot_style()) and exposes the shared color palette. Every figure in
# the repo — fraud + ImageNet — picks up the same look from this single source.
from examples.utils import COLORS, setup_plot_style

logger = logging.getLogger(__name__)


# Semantic role → hex (from the shared palette).
_ROLE_COLORS: Dict[str, str] = {
    "train_loss": COLORS["blue"],
    "train_top1": COLORS["blue"],
    "val_top1":   COLORS["green"],
    "val_top5":   COLORS["purple"],
    "val_regret": COLORS["red"],
}


def _read_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Read ``metrics.jsonl`` into a list of dicts (one per epoch)."""
    if not metrics_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_metrics_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Mirror ``metrics.jsonl`` as ``metrics.csv`` for spreadsheet users."""
    if not rows:
        return
    # Stable column order: epoch first, then sorted remaining keys.
    keys: List[str] = ["epoch"] + sorted(
        {k for r in rows for k in r.keys() if k != "epoch"}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def plot_curves(metrics_path: Path, out_dir: Path) -> None:
    """
    Render the three standard curves into ``out_dir``.

    Also writes ``metrics.csv`` next to ``metrics.jsonl`` so the figures' data
    is available in two formats.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover — matplotlib is in requirements.txt
        logger.warning("matplotlib not installed — skipping curve generation.")
        return

    # Idempotent: registers Montserrat + applies rcParams the first time only.
    setup_plot_style()

    rows = _read_metrics(metrics_path)
    if not rows:
        logger.warning("No rows in %s — nothing to plot.", metrics_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(rows, out_dir / "metrics.csv")

    epochs = [r["epoch"] for r in rows]

    # --- Loss ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_loss" in rows[0]:
        ax.plot(
            epochs, [r["train_loss"] for r in rows],
            marker="o", color=_ROLE_COLORS["train_loss"], label="train loss",
        )
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title(r"Training Loss  ($\downarrow$ lower is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "curve_loss.png")
    plt.close(fig)

    # --- Accuracy -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    series_drawn = False
    if "train_top1" in rows[0]:
        ax.plot(
            epochs, [r["train_top1"] for r in rows],
            marker="o", color=_ROLE_COLORS["train_top1"], label="train top-1",
        )
        series_drawn = True
    if "val_top1" in rows[0]:
        ax.plot(
            epochs, [r["val_top1"] for r in rows],
            marker="s", color=_ROLE_COLORS["val_top1"], label="val top-1",
        )
        series_drawn = True
    if "val_top5" in rows[0]:
        ax.plot(
            epochs, [r["val_top5"] for r in rows],
            marker="^", color=_ROLE_COLORS["val_top5"], label="val top-5",
        )
        series_drawn = True
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    if series_drawn:
        ax.legend()
    ax.set_title(r"Top-k Accuracy  ($\uparrow$ higher is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "curve_accuracy.png")
    plt.close(fig)

    # --- Regret (cost-aware only) -------------------------------------------
    if any("val_realized_regret" in r for r in rows):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            epochs,
            [r.get("val_realized_regret", float("nan")) for r in rows],
            marker="o", color=_ROLE_COLORS["val_regret"], label="val realized regret",
        )
        ax.set_xlabel("Epoch"); ax.set_ylabel("Regret")
        ax.legend()
        ax.set_title(r"Validation Realized Regret  ($\downarrow$ lower is better)")
        fig.tight_layout()
        fig.savefig(out_dir / "curve_regret.png")
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-generate per-epoch curves from a run directory's metrics.jsonl.",
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory containing metrics.jsonl (e.g. imagenet_output/<run_id>/).",
    )
    parser.add_argument(
        "--metrics-file", type=str, default="metrics.jsonl",
        help="Override the metrics filename (default: metrics.jsonl).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    metrics_path: Path = args.run_dir / args.metrics_file
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file at {metrics_path}.")

    plot_curves(metrics_path, args.run_dir)
    print(f"Wrote curves + metrics.csv to {args.run_dir}/")


if __name__ == "__main__":
    main()
