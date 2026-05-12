"""
examples.utils
========================

Utilities for the IEEE-CIS fraud detection benchmark.

This module focuses on:
- logging configuration
- device selection (CUDA / MPS / CPU)
- a TrainingState container for iteration-based metric tracking
- plotting metric trajectories over iterations (cost/regret, AP, etc.)
- plotting Precision–Recall curves
- checkpoint serialization helpers (save/load TrainingState)

Design notes
------------
Fraud detection is highly imbalanced, so **Precision–Recall** metrics (PR-AUC)
are often more informative than ROC-AUC. Additionally, in cost-aware
settings, it is crucial to track **business-aligned regret/cost** metrics rather
than relying on raw training losses, which may not be comparable across different
surrogate objectives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


# =============================================================================
# Graphics / Colors
# =============================================================================

# Standardized color palette — mirrors https://harchaoui.org/warith/colors
# so every figure in the repo shares one visual vocabulary.
COLORS = {
    # Base palette
    "red":       "#FF3B30",
    "orange":    "#FF9500",
    "yellow":    "#FFCC00",
    "green":     "#28CD41",
    "turquoise": "#79DBDC",
    "blue":      "#007AFF",
    "purple":    "#AF52DE",
    "pink":      "#FF2D55",
    # Light variants (good for fills, light backgrounds, shaded bands)
    "light_red":       "#FFD8D6",
    "light_orange":    "#FFEACC",
    "light_yellow":    "#FFF5CC",
    "light_green":     "#D4F5D9",
    "light_turquoise": "#00FFEF",
    "light_blue":      "#CCE4FF",
    "light_purple":    "#EFDCF8",
    "light_pink":      "#FFD5DD",
    # Neutrals
    "black": "#000000",
    "gray":  "#808080",
    "brown": "#A52A2A",
    "white": "#F8F8F8",
}


# =============================================================================
# Plot style (Montserrat font + rcParams)
# =============================================================================
#
# Every figure in this repo (fraud + ImageNet) uses Montserrat-Regular for
# typographic consistency.  The TTF file lives in ``assets/fonts/`` and is
# fetched by ``scripts/download_fonts.sh``.  If the file is missing, we fall
# back to matplotlib's default sans-serif with a single warning.

_FONT_NAME = "Montserrat"
_FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-Regular.ttf"

_STYLE_APPLIED = False  # idempotency flag


def setup_plot_style() -> None:
    """
    Register Montserrat with matplotlib's font manager and apply repo-wide
    rcParams. Safe to call many times; only the first call does work.

    Run ``bash scripts/download_fonts.sh`` once if the font file is missing.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return

    import matplotlib as mpl

    # Headless backend — safe for SSH, CI, cloud workers.
    mpl.use("Agg")

    family = "sans-serif"
    if _FONT_PATH.exists():
        try:
            from matplotlib import font_manager
            font_manager.fontManager.addfont(str(_FONT_PATH))
            family = _FONT_NAME
        except Exception as exc:  # pragma: no cover — defensive
            logging.getLogger(__name__).warning(
                "Could not register Montserrat from %s: %s. Falling back to default sans-serif.",
                _FONT_PATH, exc,
            )
    else:
        logging.getLogger(__name__).warning(
            "Montserrat-Regular.ttf not found at %s — using default sans-serif. "
            "Run: bash scripts/download_fonts.sh",
            _FONT_PATH,
        )

    mpl.rcParams.update({
        # Typography
        "font.family": family,
        "font.sans-serif": [_FONT_NAME, "DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Spines / grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Output
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        # Lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })
    _STYLE_APPLIED = True


# Apply on import so any module that imports utils (fraud_detection, plots, …)
# gets the right font + rcParams without an explicit call.
setup_plot_style()


# =============================================================================
# Logging / device
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure basic logging.

    Parameters
    ----------
    level:
        Logging level (e.g., ``logging.INFO``).
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Returns
    -------
    torch.device
        ``cuda`` if available, else ``mps`` if available, else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Training state
# =============================================================================

@dataclass
class TrainingState:
    """
    Container for iteration-based metric tracking.

    Attributes
    ----------
    batch_size:
        Training batch size (used in plot labels).
    current_iter:
        Global optimizer step count.
    train_smoothed:
        Smoothed averages of training-batch metrics.
        Keys are metric names; values are sequences aligned with ``train_smoothed_iters``.
    train_smoothed_iters:
        Iteration indices for smoothing points.
    val_points:
        Periodic evaluation points on a validation subset.
        Keys are metric names; values are sequences aligned with ``val_iters``.
    val_iters:
        Iteration indices for validation evaluation points.
    epoch_iters:
        Iteration indices at epoch boundaries (for vertical lines in plots).
    """
    batch_size: int = 0
    current_iter: int = 0

    train_smoothed: Dict[str, List[float]] = field(default_factory=dict)
    train_smoothed_iters: List[int] = field(default_factory=list)

    val_points: Dict[str, List[float]] = field(default_factory=dict)
    val_iters: List[int] = field(default_factory=list)

    epoch_iters: List[int] = field(default_factory=lambda: [0])


def training_state_to_dict(state: TrainingState) -> Dict[str, Any]:
    """
    Convert TrainingState to a JSON-serializable dictionary.

    Parameters
    ----------
    state:
        The TrainingState instance.

    Returns
    -------
    dict
        Serializable dictionary.
    """
    return {
        "batch_size": int(state.batch_size),
        "current_iter": int(state.current_iter),
        "train_smoothed": {k: list(map(float, v)) for k, v in state.train_smoothed.items()},
        "train_smoothed_iters": list(map(int, state.train_smoothed_iters)),
        "val_points": {k: list(map(float, v)) for k, v in state.val_points.items()},
        "val_iters": list(map(int, state.val_iters)),
        "epoch_iters": list(map(int, state.epoch_iters)),
    }


def training_state_from_dict(d: Dict[str, Any]) -> TrainingState:
    """
    Restore a TrainingState from a dictionary.

    Parameters
    ----------
    d:
        Dictionary produced by :func:`training_state_to_dict`.

    Returns
    -------
    TrainingState
        Reconstructed state.
    """
    state = TrainingState(
        batch_size=int(d.get("batch_size", 0)),
        current_iter=int(d.get("current_iter", 0)),
    )
    state.train_smoothed = {str(k): list(map(float, v)) for k, v in (d.get("train_smoothed") or d.get("train_ema") or {}).items()}
    state.train_smoothed_iters = list(map(int, d.get("train_smoothed_iters") or d.get("train_ema_iters") or []))
    state.val_points = {str(k): list(map(float, v)) for k, v in (d.get("val_points") or d.get("probe_points") or {}).items()}
    state.val_iters = list(map(int, d.get("val_iters") or d.get("probe_iters") or []))
    state.epoch_iters = list(map(int, d.get("epoch_iters") or [0]))
    return state


# =============================================================================
# Smoothing helpers
# =============================================================================

def smooth_update(prev: Optional[float], x: float, alpha: float) -> float:
    """
    Update a smoothed average.

    Parameters
    ----------
    prev:
        Previous smoothed value, or None if uninitialized.
    x:
        New observation.
    alpha:
        Smoothing factor in (0, 1]. Larger alpha = less smoothing.

    Returns
    -------
    float
        Updated smoothed value.
    """
    if prev is None:
        return float(x)
    return float(alpha) * float(x) + (1.0 - float(alpha)) * float(prev)


# =============================================================================
# Plotting
# =============================================================================

def _direction_suffix(better: Optional[Literal["higher", "lower"]]) -> str:
    """
    Return a mathtext-rendered direction indicator for the title.

    >>> _direction_suffix("higher")
    '  ($\\\\uparrow$ higher is better)'
    >>> _direction_suffix("lower")
    '  ($\\\\downarrow$ lower is better)'
    >>> _direction_suffix(None)
    ''
    """
    if better == "higher":
        return r"  ($\uparrow$ higher is better)"
    if better == "lower":
        return r"  ($\downarrow$ lower is better)"
    return ""


def plot_metric_trajectory(
    *,
    iters: Sequence[int],
    values: Sequence[float],
    out_path: PathLike,
    title: str,
    ylabel: str,
    epoch_iters: Optional[Sequence[int]] = None,
    y_quantile_max: Optional[float] = 0.98,

    baselines: Optional[Dict[str, Sequence[float]]] = None,
    better: Optional[Literal["higher", "lower"]] = None,
) -> None:
    """
    Plot a single metric vs training iterations.

    Parameters
    ----------
    baselines:
        Optional dictionary of {label: values} for baseline curves.
    better:
        ``"higher"`` or ``"lower"`` appends a ``($\\uparrow$ higher is better)`` /
        ``($\\downarrow$ lower is better)`` indicator to the title. ``None``
        leaves the title untouched.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = title + _direction_suffix(better)

    it = np.asarray(list(iters), dtype=int)
    val = np.asarray(list(values), dtype=float)

    plt.figure(figsize=(12, 4.5))
    plt.plot(it, val, linewidth=2, color=COLORS["blue"], label="Model")

    if baselines:
        for label, b_vals in baselines.items():
            b_arr = np.asarray(list(b_vals), dtype=float)
            min_len = min(len(it), len(b_arr))
            
            # Map labels to specified colors
            color = COLORS["gray"]
            style = "--"
            if "Approve" in label:
                color = COLORS["green"]
                style = "--"
            elif "Decline" in label:
                color = COLORS["red"]
                style = "--"
            
            plt.plot(it[:min_len], b_arr[:min_len], 
                     linestyle=style, color=color, alpha=0.7, label=label)

    if epoch_iters:
        first = True
        for e in epoch_iters:
            plt.axvline(e, linestyle=":", alpha=0.25, label="Epoch" if first else None)
            first = False

    plt.legend(loc="best")
    plt.xlabel("Optimizer iterations")
    plt.ylabel(ylabel)
    plt.title(title)

    if y_quantile_max is not None and val.size > 0:
        top = float(np.quantile(val, float(y_quantile_max)))
        bottom = float(np.min(val))

        if baselines:
            for b_vals in baselines.values():
                b_arr = np.asarray(list(b_vals), dtype=float)
                if len(b_arr) > 0:
                    top = max(top, float(np.quantile(b_arr, float(y_quantile_max))))
                    bottom = min(bottom, float(np.min(b_arr)))

        pad = 0.1 * (top - bottom + 1e-12)
        plt.ylim(bottom - pad, top + pad)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    *,
    out_path: PathLike,
    title: str,
    average_precision: Optional[float] = None,
    prevalence: Optional[float] = None,
) -> None:
    """
    Plot a precision–recall curve.

    Parameters
    ----------
    precision:
        Precision values (sklearn).
    recall:
        Recall values (sklearn).
    out_path:
        Destination PNG path.
    title:
        Plot title.
    average_precision:
        Optional Average Precision (AP) for annotation.
    prevalence:
        Prevalence (positive rate) for Naive baselines.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    
    label = f"AP={average_precision:.4f}" if average_precision is not None else None
    plt.plot(recall, precision, linewidth=2, color=COLORS["blue"], label=label, marker=".")

    if prevalence is not None:
        # Standardized colors: Decline All (Red), Approve All (Green)
        plt.axhline(prevalence, color=COLORS["red"], linestyle="--", label=f"Decline All [{prevalence:.4f}]")
        plt.axhline(0.0, color=COLORS["green"], linestyle="--", label="Approve All [0.0000]")

    # PR-AUC is always higher-is-better.
    plt.title(title + _direction_suffix("higher"))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(-0.05, 1.05)
    plt.xlim(0.0, 1.0)
    plt.legend(loc="best")
    plt.grid(True)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
def plot_temporal_split(
    train_dt: np.ndarray,
    val_dt: np.ndarray,
    out_path: PathLike,
    *,
    bins: int = 60,
    seconds_per_day: float = 86_400.0,
) -> None:
    """
    Plot the distribution of `TransactionDT` for train/validation splits with a human-friendly x-axis.

    The IEEE-CIS `TransactionDT` feature is typically a *relative* timestamp expressed in seconds since
    an arbitrary reference point (the true calendar origin is not provided). Plotting raw seconds
    yields an unreadable x-axis (e.g., values around 1e7). This helper converts seconds to
    **days since the first observed transaction**, which makes temporal splits easier to inspect.

    Parameters
    ----------
    train_dt:
        1D array of `TransactionDT` values for the training split (units: seconds).
    val_dt:
        1D array of `TransactionDT` values for the validation split (units: seconds).
    out_path:
        Destination path for the saved PNG figure.
    bins:
        Number of histogram bins to use (shared between train and validation).
    seconds_per_day:
        Conversion factor from seconds to days. Defaults to 86400.

    Returns
    -------
    None
        The function saves the figure to `out_path` and closes the Matplotlib figure.

    Notes
    -----
    - This plot is meant to validate that the validation set occurs *after* the training set in time
      (i.e., no temporal leakage).
    - The histograms use the same bin edges to ensure a fair visual comparison.
    """
    # --- Defensive conversions & basic validation -----------------------------------------------
    # Accept numpy arrays (or array-like) and ensure we work with flat float arrays.
    train_dt_arr: np.ndarray = np.asarray(train_dt, dtype=np.float64).ravel()
    val_dt_arr: np.ndarray = np.asarray(val_dt, dtype=np.float64).ravel()

    if train_dt_arr.size == 0:
        raise ValueError("`train_dt` is empty; cannot plot a temporal split.")
    if val_dt_arr.size == 0:
        raise ValueError("`val_dt` is empty; cannot plot a temporal split.")
    if not np.isfinite(train_dt_arr).all():
        raise ValueError("`train_dt` contains NaN or infinite values; please clean/filter first.")
    if not np.isfinite(val_dt_arr).all():
        raise ValueError("`val_dt` contains NaN or infinite values; please clean/filter first.")
    if seconds_per_day <= 0:
        raise ValueError("`seconds_per_day` must be strictly positive.")

    # --- Convert seconds to a more interpretable unit (days) -----------------------------------
    # We also shift time to start at day 0 for the earliest transaction across both splits.
    # This makes the axis "days since first transaction", which is stable and readable.
    all_dt_seconds: np.ndarray = np.concatenate([train_dt_arr, val_dt_arr])
    t0_seconds: float = float(all_dt_seconds.min())

    train_days: np.ndarray = (train_dt_arr - t0_seconds) / seconds_per_day
    val_days: np.ndarray = (val_dt_arr - t0_seconds) / seconds_per_day

    # --- Shared binning for fair histogram comparison ------------------------------------------
    # Using shared bin edges ensures "Train" and "Validation" bars align.
    all_days: np.ndarray = np.concatenate([train_days, val_days])
    day_min: float = float(all_days.min())
    day_max: float = float(all_days.max())

    # In degenerate cases (all timestamps identical), slightly widen the range so hist() works.
    if day_min == day_max:
        day_min -= 0.5
        day_max += 0.5

    bin_edges: np.ndarray = np.linspace(day_min, day_max, int(bins) + 1)

    # --- Create output directory ---------------------------------------------------------------
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    # --- Plot ------------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))

    # Histogram for training days
    plt.hist(
        train_days,
        bins=bin_edges,
        alpha=0.7,
        label="Train",
        color=COLORS["blue"],
    )

    # Histogram for validation days
    plt.hist(
        val_days,
        bins=bin_edges,
        alpha=0.7,
        label="Validation",
        color=COLORS["orange"],
    )

    # A vertical line at the earliest validation timestamp is often a helpful split indicator.
    # This is optional but usually improves interpretability.
    val_start: float = float(val_days.min())
    plt.axvline(val_start, linestyle="--", linewidth=1.5, alpha=0.8, color=COLORS["red"], label="Train / Val Split")

    # Labels and cosmetics
    plt.title("Temporal Split (TransactionDT)")
    plt.xlabel("Days since first transaction")
    plt.ylabel("Occurences")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save and close to avoid memory leaks in long-running scripts / notebooks.
    plt.savefig(out_path_p, dpi=150)
    plt.close()
