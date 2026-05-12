"""
scripts/make_paper_figures.py
=============================

Generate paper-ready comparison figures from a fraud_output/<run_id>/ directory.

The training script writes one ``val_metrics.csv`` per loss; this script overlays
all losses on the same axes for three metrics and writes the PNGs into
``docs/latex/figures/`` for ``\\includegraphics`` in the KDD paper.

Outputs
-------
- ``fig_fraud_pr_auc.png``           — Validation PR-AUC vs optimizer iters (↑)
- ``fig_fraud_realized_regret.png``  — Validation realized regret vs iters (↓)
- ``fig_fraud_expected_regret.png``  — Validation expected optimal regret (↓)
- ``fig_fraud_summary_table.tex``    — LaTeX table from summary.csv (final numbers)

Run
---
    python scripts/make_paper_figures.py \\
        --run-dir fraud_output/comprehensive_benchmark \\
        --out docs/latex/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project-wide palette + Montserrat font registration
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.utils import COLORS, setup_plot_style

setup_plot_style()


# Stable mapping of loss → palette colour + display label.
_LOSS_STYLE = {
    "cross_entropy":          {"color": COLORS["gray"],   "label": "Cross-Entropy",         "ls": "-"},
    "cross_entropy_weighted": {"color": COLORS["orange"], "label": "Weighted CE",           "ls": "-"},
    "sinkhorn_fenchel_young": {"color": COLORS["purple"], "label": "Sinkhorn-FY (CACIS)",   "ls": "-"},
    "sinkhorn_envelope":      {"color": COLORS["blue"],   "label": "Sinkhorn Envelope",     "ls": "-"},
    "sinkhorn_autodiff":      {"color": COLORS["pink"],   "label": "Sinkhorn Autodiff",     "ls": "--"},
    "sinkhorn_pot":           {"color": COLORS["green"],  "label": "Sinkhorn (POT backend)", "ls": "-"},
}


def _load_val_metrics(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Return ``{loss_name: DataFrame}`` for every loss subdir with val_metrics.csv."""
    out: dict[str, pd.DataFrame] = {}
    for loss in sorted(_LOSS_STYLE.keys()):
        path = run_dir / loss / "val_metrics.csv"
        if path.exists():
            out[loss] = pd.read_csv(path)
    return out


def _plot_overlay(
    metrics: dict[str, pd.DataFrame],
    column: str,
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    better: str,
    show_baselines: bool = False,
    baseline_column: str | None = None,
    baseline_label: str | None = None,
    smooth_window: int | None = None,
) -> None:
    """
    Overlay ``column`` from every loss's val_metrics on one figure.

    ``better`` is ``'higher'`` or ``'lower'`` and stamps the title with the
    project-standard arrow indicator.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    arrow = r"$\uparrow$ higher is better" if better == "higher" else r"$\downarrow$ lower is better"

    for loss, df in metrics.items():
        style = _LOSS_STYLE[loss]
        y = df[column]
        if smooth_window and smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        ax.plot(df["iter"], y, color=style["color"], linestyle=style["ls"],
                label=style["label"], linewidth=2)

    if show_baselines and baseline_column and metrics:
        # All losses see the same naive baselines — take the first.
        first = next(iter(metrics.values()))
        ax.plot(first["iter"], first[baseline_column],
                color=COLORS["red"], linestyle=":", linewidth=1.5, alpha=0.7,
                label=baseline_label or "Naive baseline")

    ax.set_xlabel("Optimizer iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ({arrow})")
    ax.legend(loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _write_summary_table(summary_csv: Path, out_path: Path) -> None:
    """Render the loss-vs-loss summary as a paper-ready LaTeX table fragment."""
    df = pd.read_csv(summary_csv, index_col=0)

    keep = ["val_pr_auc", "val_realized_regret", "val_expected_opt_regret"]
    pretty = {
        "val_pr_auc":              r"PR-AUC $\uparrow$",
        "val_realized_regret":     r"Realized regret \$ $\downarrow$",
        "val_expected_opt_regret": r"Expected opt.\ regret \$ $\downarrow$",
    }
    name_map = {
        "cross_entropy":          "Cross-Entropy",
        "cross_entropy_weighted": "Weighted CE",
        "sinkhorn_fenchel_young": r"\CACIS{} (FY)",
        "sinkhorn_envelope":      r"\CACIS{} (envelope)",
        "sinkhorn_autodiff":      r"\CACIS{} (autodiff)",
        "sinkhorn_pot":           r"\CACIS{} (POT)",
    }

    rows = []
    for idx in df.index:
        if idx not in name_map:
            continue
        row = [name_map[idx]]
        for col in keep:
            row.append(f"{float(df.loc[idx, col]):.3f}")
        rows.append(row)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(f"\\textbf{{{pretty[c]}}}" for c in keep) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(r) + r" \\" for r in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper figures from a fraud benchmark run.")
    p.add_argument("--run-dir", type=Path,
                   default=Path("fraud_output/comprehensive_benchmark"),
                   help="Directory containing per-loss subdirs with val_metrics.csv")
    p.add_argument("--out", type=Path, default=Path("docs/latex/figures"),
                   help="Where to write the figures and the LaTeX table fragment")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling-window smoothing for the overlay plots (1 disables)")
    args = p.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"run dir {args.run_dir} does not exist")
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = _load_val_metrics(args.run_dir)
    print(f"Loaded val_metrics for {len(metrics)} losses: {list(metrics.keys())}")

    _plot_overlay(
        metrics, "pr_auc",
        out_path=args.out / "fig_fraud_pr_auc.png",
        title="IEEE-CIS / Vesta — Validation PR-AUC",
        ylabel="PR-AUC",
        better="higher",
        smooth_window=args.smooth,
    )
    _plot_overlay(
        metrics, "realized_regret",
        out_path=args.out / "fig_fraud_realized_regret.png",
        title="IEEE-CIS / Vesta — Validation realized regret (\\$ per transaction)",
        ylabel=r"Realized regret (\$)",
        better="lower",
        show_baselines=True,
        baseline_column="approve_all_realized_cost",
        baseline_label="Naive baseline (Approve all)",
        smooth_window=args.smooth,
    )
    _plot_overlay(
        metrics, "expected_opt_regret",
        out_path=args.out / "fig_fraud_expected_regret.png",
        title="IEEE-CIS / Vesta — Validation expected optimal regret",
        ylabel=r"Expected optimal regret (\$)",
        better="lower",
        smooth_window=args.smooth,
    )

    summary_csv = args.run_dir / "summary.csv"
    if summary_csv.exists():
        _write_summary_table(summary_csv, args.out / "fig_fraud_summary_table.tex")

    # Also copy the temporal-split visualization so the LaTeX figure path is local.
    src_temporal = args.run_dir / "temporal_split.png"
    if src_temporal.exists():
        import shutil
        dst_temporal = args.out / "fig_fraud_temporal_split.png"
        shutil.copy2(src_temporal, dst_temporal)
        print(f"  copied {src_temporal.name} -> {dst_temporal}")


if __name__ == "__main__":
    main()
