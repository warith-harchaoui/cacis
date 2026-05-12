"""
scripts/eps_sweep_table.py
==========================

Aggregate ``fraud_output/<RUN_PREFIX>/<tag>/summary.csv`` rows from an ε-sweep
into a single side-by-side comparison, sorted by validation realized regret.

Used by ``scripts/fraud_epsilon_sweep.sh``.

Usage
-----
    python scripts/eps_sweep_table.py --root fraud_output/eps_sweep
    python scripts/eps_sweep_table.py --root fraud_output/eps_sweep --out sweep.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _load_args(run_dir: Path) -> dict:
    """
    Recover the CLI args saved in any per-loss subdir as args.json.

    Returns an empty dict if no args.json is present (e.g. CE-only runs may not
    save one in this style).
    """
    for sub in run_dir.iterdir():
        if sub.is_dir():
            j = sub / "args.json"
            if j.exists():
                try:
                    return json.loads(j.read_text())
                except Exception:
                    pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a comparison table across ε-sweep runs.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Root directory containing one subdir per (tag) run.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional CSV path. Stdout is always populated.")
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"{args.root} does not exist")

    rows = []
    for tag_dir in sorted(args.root.iterdir()):
        if not tag_dir.is_dir():
            continue
        summary = tag_dir / "summary.csv"
        if not summary.exists():
            continue

        df = pd.read_csv(summary, index_col=0)
        cli = _load_args(tag_dir)
        eps_mode = cli.get("epsilon_mode", "")
        eps_scale = cli.get("epsilon_scale", "")

        for loss, row in df.iterrows():
            approve = float(row.get("val_approve_all_realized_cost", float("nan")))
            decline = float(row.get("val_decline_all_realized_cost", float("nan")))
            realized = float(row.get("val_realized_regret", float("nan")))
            beats_naive = bool(realized + 1e-3 < min(approve, decline))
            rows.append({
                "tag": tag_dir.name,
                "loss": loss,
                "eps_mode": eps_mode if loss != "cross_entropy" else "—",
                "eps_scale": eps_scale if loss != "cross_entropy" else "—",
                "val_pr_auc": round(float(row["val_pr_auc"]), 4),
                "val_realized_regret": round(realized, 4),
                "val_expected_opt_regret": round(float(row["val_expected_opt_regret"]), 4),
                "approve_all": round(approve, 4),
                "decline_all": round(decline, 4),
                "beats_naive?": "✓" if beats_naive else "✗",
            })

    if not rows:
        print(f"No completed runs found under {args.root}.")
        print("(Each tag dir must contain a summary.csv.)")
        return

    df = pd.DataFrame(rows).sort_values("val_realized_regret").reset_index(drop=True)

    with pd.option_context("display.max_columns", None,
                           "display.width", 160,
                           "display.max_colwidth", 28):
        print(df.to_string(index=False))

    print()
    n_beats = int((df["beats_naive?"] == "✓").sum())
    print(f"{n_beats}/{len(df)} runs beat the naive baseline.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
