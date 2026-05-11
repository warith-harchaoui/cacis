"""
examples.fraud_detection
==================================

Benchmark runner for IEEE-CIS (Kaggle) fraud detection comparing:

- cross_entropy
- cross_entropy_weighted (sample-weighted CE with w_i = C_i[y_i, 1-y_i])
- sinkhorn_fenchel_young (Fenchel–Young / Sinkhorn-FY-style)
- sinkhorn_envelope (custom Sinkhorn OT-loss, envelope gradient)
- sinkhorn_autodiff (custom Sinkhorn OT-loss, full autodiff)
- sinkhorn_pot (POT library Sinkhorn OT-loss, envelope gradient)

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

Key design choices
------------------
- Focus on fraud-appropriate evaluation: Precision–Recall (Average Precision),
  plus business-aligned regret/cost metrics.
- Plot **metrics vs optimizer iterations** (not raw losses).
- Support checkpoint saving/loading and a resume ("continue") mechanism.

Resume semantics
----------------
If you pass ``--resume``, the script will load the latest checkpoint for each
run directory and continue training for ``--epochs`` *additional* epochs.

Example
-------
Train 5 epochs from scratch:

    python fraud_detection.py --loss all --epochs 5 --run-id demo1

Continue training for 3 more epochs:

    python fraud_detection.py --loss all --epochs 3 --run-id demo1 --resume

Dependencies
------------
    pip install torch pandas numpy matplotlib scikit-learn tqdm

Optional (for Sinkhorn variants)
--------------------------------
    pip install POT[backend-torch]
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from examples.tabular_models import (
    BackboneName,
    TabularModelConfig,
    TabularRiskModel,
    compute_smart_architecture_defaults,
)
from examples.utils import (
    TrainingState,
    smooth_update,
    get_device,
    plot_metric_trajectory,
    plot_precision_recall_curve,
    plot_temporal_split,
    setup_logging,
    training_state_from_dict,
    training_state_to_dict,
)


# =============================================================================
# Business model → per-example regret/cost matrix
# =============================================================================

@dataclass(frozen=True)
class BusinessParams:
    """
    Business parameters used to derive per-example cost matrices.

    We use a regret construction consistent with an approve/decline value model:

        C_legit,decline = rho_fd * M
        C_fraud,approve = L_fraud(M) = lambda_cb * M + F_cb

    where M is TransactionAmt.
    """
    rho_fd: float = 0.10
    lambda_cb: float = 1.50
    F_cb: float = 15.0


def build_cost_matrix(amount: np.ndarray, params: BusinessParams) -> np.ndarray:
    """
    Build per-example cost matrices C_i (2x2).

    Mapping:
    - label 0 = legit, label 1 = fraud
    - action 0 = approve (predict legit), action 1 = decline (predict fraud)

    C_i =
        [[0, rho_fd * M_i],
         [lambda_cb * M_i + F_cb, 0]]
    """
    M = amount.astype(np.float32)
    c_fd = params.rho_fd * M
    c_cb = params.lambda_cb * M + params.F_cb

    C = np.zeros((M.shape[0], 2, 2), dtype=np.float32)
    C[:, 0, 1] = c_fd
    C[:, 1, 0] = c_cb
    return C


def sample_weight_from_C(y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Compute sample weights w_i = C_i[y_i, 1-y_i] for binary costs.
    """
    y = y.astype(np.int64)
    wrong = 1 - y
    return C[np.arange(y.shape[0]), y, wrong].astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================

class FraudDataset(Dataset):
    """
    IEEE-CIS fraud dataset with features, labels, costs, and sample weights.

    Returns per item:
    - x : float32 features, shape (D,)
    - y : int64 label in {0,1}
    - C : float32 cost matrix, shape (2,2)
    - w : float32 weight (for weighted CE), scalar (median-normalized globally)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        params: BusinessParams,
        *,
        weight_norm_median: float,
    ) -> None:
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df["isFraud"].values.astype(np.int64)
        if "TransactionAmt_raw" in df.columns:
            self.amount = df["TransactionAmt_raw"].values.astype(np.float32)
        else:
            self.amount = df["TransactionAmt"].values.astype(np.float32)

        self.C = build_cost_matrix(self.amount, params)

        w_raw = sample_weight_from_C(self.y, self.C)
        denom = float(weight_norm_median) if float(weight_norm_median) > 0 else 1.0
        self.w = (w_raw / denom).astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.from_numpy(self.C[idx]),
            torch.tensor(self.w[idx], dtype=torch.float32),
        )


# =============================================================================
# Metrics: PR + business regret
# =============================================================================

@torch.no_grad()
def batch_regret_metrics(scores: Tensor, y: Tensor, C: Tensor) -> Dict[str, float]:
    """
    Compute regret metrics on a batch (for iteration-level tracking).

    Returns
    -------
    dict
        - train_expected_opt_regret
        - train_realized_regret
    """
    # P(fraud) from model predictions
    prob_fraud = torch.softmax(scores, dim=1)[:, 1]
    prob_legit = 1.0 - prob_fraud
    
    # Expected cost for each action, properly accounting for both label outcomes:
    # E[cost | action=approve] = P(fraud) * C[fraud, approve] + P(legit) * C[legit, approve]
    # E[cost | action=decline] = P(fraud) * C[fraud, decline] + P(legit) * C[legit, decline]
    exp_approve = prob_fraud * C[:, 1, 0] + prob_legit * C[:, 0, 0]
    exp_decline = prob_fraud * C[:, 1, 1] + prob_legit * C[:, 0, 1]

    # Optimal action minimizes expected cost
    exp_opt = torch.minimum(exp_approve, exp_decline)
    
    # Realized cost: C[true_label, chosen_action]
    action = torch.where(exp_decline < exp_approve, torch.ones_like(y), torch.zeros_like(y)).long()
    realized = C[torch.arange(C.shape[0], device=C.device), y, action]

    # Naive baselines (Always Approve vs Always Decline)
    # 1. Naive Expected
    mean_exp_approve = exp_approve.mean().item()
    mean_exp_decline = exp_decline.mean().item()
    naive_exp_cost = min(mean_exp_approve, mean_exp_decline)
    
    # 2. Naive Realized
    # Cost if we always approved: C[i, y[i], 0]
    # Cost if we always declined: C[i, y[i], 1]
    cost_approve_all = C[torch.arange(C.shape[0], device=C.device), y, 0]
    cost_decline_all = C[torch.arange(C.shape[0], device=C.device), y, 1]
    naive_realized_cost = min(cost_approve_all.mean().item(), cost_decline_all.mean().item())
    
    # Optimal Expected Cost
    mean_exp_opt = exp_opt.mean().item()
    
    # Optimal Realized Cost = min(C[i, y, 0], C[i, y, 1])
    # Note: 'train_realized_regret' assumes we want regret vs perfect foresight?
    # Usually realized regret = realized_cost - optimal_realized_cost
    # But optimal_realized_cost is usually 0 if C has 0s on diagonal?
    # Let's check cost matrix structure.
    # C_i = [[0, ...], [..., 0]]. So optimal cost is 0.
    # If standard costs are [[0, FP], [FN, 0]], then optimal decision always yields 0 cost.
    # So regret = cost.
    # But let's be safe and subtract min entry.
    opt_realized = torch.minimum(cost_approve_all, cost_decline_all).mean().item()

    # PR-AUC for the batch (if both classes present and predictions are finite)
    if y.float().std() == 0 or not torch.isfinite(prob_fraud).all():
        batch_pr_auc = float("nan")
    else:
        # sklearn avg precision needs cpu numpy
        batch_pr_auc = average_precision_score(
            y.detach().cpu().numpy(), 
            prob_fraud.detach().cpu().numpy()
        )

    return {
        "train_expected_opt_regret": mean_exp_opt,
        "train_realized_regret": float(realized.mean().item()),
        "train_pr_auc": float(batch_pr_auc),
        # Baseline strategies (fixed)
        "train_approve_all_expected_regret": float(cost_approve_all.mean().item()),
        "train_decline_all_expected_regret": float(cost_decline_all.mean().item()),
        "train_approve_all_realized_regret": float(cost_approve_all.mean().item()),
        "train_decline_all_realized_regret": float(cost_decline_all.mean().item()),
    }



@torch.no_grad()
def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Evaluate PR and regret metrics on a labeled loader.
    """
    model.eval()

    y_true: List[int] = []
    p_fraud: List[float] = []
    realized_regrets: List[float] = []
    expected_opt_regrets: List[float] = []
    
    # Accumulators for baseline strategies
    total_approve_all_realized = 0.0
    total_decline_all_realized = 0.0
    total_approve_all_expected = 0.0
    total_decline_all_expected = 0.0
    total_samples = 0

    for x, y, C, _w in loader:
        x = x.to(device)
        y = y.to(device)
        C = C.to(device)
        B = x.size(0)

        scores = model(x)
        prob_fraud = torch.softmax(scores, dim=1)[:, 1]
        prob_legit = 1.0 - prob_fraud

        y_true.extend(y.detach().cpu().numpy().tolist())
        p_fraud.extend(prob_fraud.detach().cpu().numpy().tolist())

        # Expected cost for each action
        exp_approve = prob_fraud * C[:, 1, 0] + prob_legit * C[:, 0, 0]
        exp_decline = prob_fraud * C[:, 1, 1] + prob_legit * C[:, 0, 1]
        
        exp_opt = torch.minimum(exp_approve, exp_decline)
        action = torch.where(exp_decline < exp_approve, torch.ones_like(y), torch.zeros_like(y)).long()

        realized = C[torch.arange(B, device=device), y, action]

        realized_regrets.extend(realized.detach().cpu().numpy().tolist())
        expected_opt_regrets.extend(exp_opt.detach().cpu().numpy().tolist())
        
        # Realized (baseline strategies)
        c_app = C[torch.arange(B, device=device), y, 0]
        c_dec = C[torch.arange(B, device=device), y, 1]
        
        total_approve_all_realized += float(c_app.sum().item())
        total_decline_all_realized += float(c_dec.sum().item())
        
        # For "Expected" cost, we also use the Realized sum as the unbiased estimator.
        # This prevents the baseline from drifting with the model's probabilities.
        total_approve_all_expected += float(c_app.sum().item())
        total_decline_all_expected += float(c_dec.sum().item())
        
        total_samples += B

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p_fraud, dtype=float)

    pr_auc = float(average_precision_score(y_arr, p_arr))
    
    # Averages
    if total_samples > 0:
        app_all_real = total_approve_all_realized / total_samples
        dec_all_real = total_decline_all_realized / total_samples
        app_all_exp = total_approve_all_expected / total_samples
        dec_all_exp = total_decline_all_expected / total_samples
    else:
        app_all_real = dec_all_real = app_all_exp = dec_all_exp = float("nan")

    return {
        "pr_auc": pr_auc,
        "realized_regret": float(np.mean(realized_regrets)) if realized_regrets else float("nan"),
        "expected_opt_regret": float(np.mean(expected_opt_regrets)) if expected_opt_regrets else float("nan"),
        
        "approve_all_realized_cost": app_all_real,
        "decline_all_realized_cost": dec_all_real,
        "approve_all_expected_cost": app_all_exp,
        "decline_all_expected_cost": dec_all_exp,
    }


@torch.no_grad()
def pr_curve(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision–Recall curve and Average Precision on a loader.
    """
    model.eval()
    y_true: List[int] = []
    p_fraud: List[float] = []

    for x, y, _C, _w in loader:
        x = x.to(device)
        scores = model(x)
        prob = torch.softmax(scores, dim=1)[:, 1]
        y_true.extend(y.numpy().tolist())
        p_fraud.extend(prob.detach().cpu().numpy().tolist())

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(p_fraud, dtype=float)

    precision, recall, _thr = precision_recall_curve(y_arr, p_arr)
    pr_auc = float(average_precision_score(y_arr, p_arr))
    return precision, recall, pr_auc


# =============================================================================
# Feature engineering
# =============================================================================

def make_features(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build aligned one-hot encoded features for train and validation.
    """
    for df in (train_df, val_df):
        df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype(float))

    drop_cols = {"isFraud", "TransactionID"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    categorical_cols = train_df[feature_cols].select_dtypes(include=["object"]).columns
    train_df2 = pd.get_dummies(train_df, columns=categorical_cols, dummy_na=True)
    val_df2 = pd.get_dummies(val_df, columns=categorical_cols, dummy_na=True)

    train_df2, val_df2 = train_df2.align(val_df2, join="left", axis=1, fill_value=0)

    feature_cols2 = [c for c in train_df2.columns if c not in drop_cols]
    train_df2[feature_cols2] = train_df2[feature_cols2].fillna(0.0)
    val_df2[feature_cols2] = val_df2[feature_cols2].fillna(0.0)

    return train_df2, val_df2, feature_cols2


# =============================================================================
# Loss construction
# =============================================================================

LossName = Literal[
    "cross_entropy",
    "cross_entropy_weighted",
    "sinkhorn_fenchel_young",
    "sinkhorn_envelope",
    "sinkhorn_autodiff",
    "sinkhorn_pot",
]

def make_cost_aware_loss(
    loss_name: LossName,
    *,
    epsilon_mode: str,
    epsilon_scale: float,
    epsilon: Optional[float],
    sinkhorn_max_iter: int,
    cacis_solver_iter: int,
    epsilon_schedule: Optional[str],
    schedule_start_mult: float,
    schedule_end_mult: float,
    total_epochs: int,
):
    """
    Construct a cost-aware loss module.

    Notes
    -----
    The cost-aware losses are expected to live in the ``cost_aware_losses`` package
    you generated earlier (SinkhornFenchelYoungLoss, SinkhornEnvelopeLoss, SinkhornFullAutodiffLoss).
    """
    from cost_aware_losses import (  # type: ignore
        SinkhornFenchelYoungLoss,
        SinkhornEnvelopeLoss,
        SinkhornFullAutodiffLoss,
        SinkhornPOTLoss,
    )

    # Use provided epsilon_mode and epsilon_scale
    # If epsilon is provided (constant), override mode to "constant"
    if epsilon is not None:
        eps_mode = "constant"
    else:
        eps_mode = epsilon_mode

    if loss_name == "sinkhorn_fenchel_young":
        return SinkhornFenchelYoungLoss(
            epsilon_mode=eps_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            solver_iter=cacis_solver_iter,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=total_epochs,
        )
    if loss_name == "sinkhorn_envelope":
        return SinkhornEnvelopeLoss(
            epsilon_mode=eps_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            max_iter=sinkhorn_max_iter,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=total_epochs,
        )
    if loss_name == "sinkhorn_autodiff":
        return SinkhornFullAutodiffLoss(
            epsilon_mode=eps_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            max_iter=sinkhorn_max_iter,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=total_epochs,
        )
    if loss_name == "sinkhorn_pot":
        return SinkhornPOTLoss(
            epsilon_mode=eps_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            max_iter=sinkhorn_max_iter,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=total_epochs,
        )

    raise ValueError(f"Not a cost-aware loss: {loss_name}")


def weighted_cross_entropy(scores: Tensor, y: Tensor, w: Tensor) -> Tensor:
    """
    Sample-weighted cross entropy with normalized weights.

    We use reduction='none' then aggregate with normalized weights:

        loss = sum_i w_i * CE_i / sum_i w_i
    """
    ce_i = nn.functional.cross_entropy(scores, y, reduction="none")
    w = torch.clamp(w, min=0.0)
    denom = torch.clamp(w.sum(), min=1e-12)
    return (w * ce_i).sum() / denom


# =============================================================================
# Checkpointing
# =============================================================================

def _best_mode_for_metric(metric: str) -> Literal["min", "max"]:
    """
    Decide whether to minimize or maximize a given metric.
    """
    if metric in ("pr_auc",):
        return "max"
    # regrets are minimized
    return "min"


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    smooth_buf: Dict[str, Optional[float]],
    epoch_next: int,
    model_config: Dict[str, Any],
    run_config: Dict[str, Any],
    best_score: Optional[float],
) -> None:
    """
    Save a training checkpoint.

    Parameters
    ----------
    path:
        Destination file.
    model:
        Model to save.
    optimizer:
        Optimizer to save.
    state:
        TrainingState (metrics history).
    smooth_buf:
        Smoothing buffers (last smoothed values).
    epoch_next:
        Next epoch index to run (resume will start here).
    model_config:
        Model hyperparameters/config.
    run_config:
        Run arguments/config snapshot for reproducibility.
    best_score:
        Best score so far (according to save-best-by metric).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "epoch_next": int(epoch_next),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": training_state_to_dict(state),
        "smooth_buf": {k: (None if v is None else float(v)) for k, v in smooth_buf.items()},
        "model_config": dict(model_config),
        "run_config": dict(run_config),
        "best_score": None if best_score is None else float(best_score),
        "rng": {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    path:
        Checkpoint file path.
    device:
        Device to map tensors to.

    Returns
    -------
    dict
        The loaded checkpoint dictionary.
    """
    return torch.load(path, map_location=device)


# =============================================================================
# Training
# =============================================================================

def save_state_csvs(state: TrainingState, out_dir: Path) -> None:
    """
    Save state metric series to CSV for downstream analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if state.train_smoothed_iters:
        train_df = pd.DataFrame({"iter": state.train_smoothed_iters})
        for k, v in state.train_smoothed.items():
            train_df[k] = v
        train_df.to_csv(out_dir / "train_smoothed_metrics.csv", index=False)

    if state.val_iters:
        val_df = pd.DataFrame({"iter": state.val_iters})
        for k, v in state.val_points.items():
            val_df[k] = v
        val_df.to_csv(out_dir / "val_metrics.csv", index=False)


def train_one(
    *,
    run_dir: Path,
    loss_name: LossName,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_prevalence: float,
    train_prevalence: float,
    val_loader_subset: DataLoader,
    device: torch.device,
    epochs_additional: int,
    quick: bool,
    eval_every: int,
    smoothing_alpha: float,
    epsilon_mode: str,
    epsilon_scale: float,
    epsilon: Optional[float],
    sinkhorn_max_iter: int,
    cacis_solver_iter: int,
    epsilon_schedule: Optional[str],
    schedule_start_mult: float,
    schedule_end_mult: float,
    resume: bool,
    save_best_by: str,
    checkpoint_every_iters: int,
    checkpoint_every_epochs: int,
    run_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Train one method, with optional resume from checkpoint.

    Returns
    -------
    dict
        Final validation metrics for summary.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = run_dir / "checkpoint_last.pt"
    ckpt_best = run_dir / "checkpoint_best.pt"

    # -----------------------
    # Restore from checkpoint
    # -----------------------
    state = TrainingState(batch_size=int(train_loader.batch_size or 0))
    smooth_buf: Dict[str, Optional[float]] = {
        "train_expected_opt_regret": None,
        "train_realized_regret": None,
    }
    best_score: Optional[float] = None
    epoch_start = 0

    if resume and ckpt_last.exists():
        ckpt = load_checkpoint(ckpt_last, device=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        state = training_state_from_dict(ckpt["training_state"])
        smooth_buf = {k: (None if v is None else float(v)) for k, v in (ckpt.get("smooth_buf") or ckpt.get("ema_buf") or {}).items()}
        epoch_start = int(ckpt.get("epoch_next", 0))
        best_score = ckpt.get("best_score", None)

        # Restore RNG states if present (best-effort).
        rng = ckpt.get("rng", {})
        try:
            if rng.get("torch") is not None:
                torch.random.set_rng_state(rng["torch"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            logging.warning("[%s] RNG state restore failed (continuing anyway).", loss_name)

        logging.info("[%s] Resumed from %s at epoch=%d iter=%d", loss_name, ckpt_last, epoch_start, state.current_iter)
    elif resume:
        logging.warning("[%s] --resume set but no checkpoint found at %s; starting fresh.", loss_name, ckpt_last)

    # -----------------------
    # Loss construction
    # -----------------------
    ce_loss = nn.CrossEntropyLoss()
    cost_aware_loss = None
    if loss_name in ("sinkhorn_fenchel_young", "sinkhorn_envelope", "sinkhorn_autodiff", "sinkhorn_pot"):
        cost_aware_loss = make_cost_aware_loss(
            loss_name,
            epsilon_mode=epsilon_mode,
            epsilon_scale=epsilon_scale,
            epsilon=epsilon,
            sinkhorn_max_iter=sinkhorn_max_iter,
            cacis_solver_iter=cacis_solver_iter,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=epochs_additional,
        )

    best_mode = _best_mode_for_metric(save_best_by)

    def is_better(new: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return (new > best) if best_mode == "max" else (new < best)

    # -----------------------
    # Training loop
    # -----------------------
    target_epochs = epoch_start + int(epochs_additional)
    logging.info("[%s] Training epochs: %d -> %d (additional=%d)", loss_name, epoch_start, target_epochs, epochs_additional)

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=target_epochs, eta_min=1e-6
    )
    # Fast-forward scheduler if resuming
    if epoch_start > 0:
         for _ in range(epoch_start):
             scheduler.step()

    for epoch in range(epoch_start, target_epochs):
        # Update epsilon schedule if using cost-aware loss
        if cost_aware_loss is not None:
            cost_aware_loss.set_epoch(epoch - epoch_start)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"[{loss_name}] Epoch {epoch+1}/{target_epochs}", total=len(train_loader), mininterval=2.0, maxinterval=2.0)

        for step, (x, y, C, w) in enumerate(pbar):
            if quick and step >= 5:
                break

            x = x.to(device)
            y = y.to(device)
            C = C.to(device)
            w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            scores = model(x)

            # Training objective
            if loss_name == "cross_entropy":
                loss = ce_loss(scores, y)
            elif loss_name == "cross_entropy_weighted":
                loss = weighted_cross_entropy(scores, y, w)
            else:
                assert cost_aware_loss is not None
                loss = cost_aware_loss(scores, y, C=C)

            loss.backward()
            optimizer.step()

            state.current_iter += 1

            # Iteration-level metrics (Smoothing)
            train_metrics = batch_regret_metrics(scores, y, C)
            state.train_smoothed_iters.append(state.current_iter)
            for k, v in train_metrics.items():
                smooth_buf[k] = smooth_update(smooth_buf.get(k), float(v), alpha=smoothing_alpha)
                state.train_smoothed.setdefault(k, []).append(float(smooth_buf[k]))

            pbar.set_postfix({
                "loss": f"{float(loss.item()):.4f}",
                "regret*": f"{train_metrics['train_expected_opt_regret']:.2f}",
            })

            # Periodic validation evaluation (subset)
            if eval_every > 0 and (state.current_iter % eval_every == 0):
                val_subset_metrics = eval_on_loader(model, val_loader_subset, device)
                state.val_iters.append(state.current_iter)
                for k, v in val_subset_metrics.items():
                    state.val_points.setdefault(k, []).append(float(v))
                logging.info(
                    "[%s] iter=%d | val(sub) PR-AUC=%.4f | regret(exp*)=%.4f | regret(real)=%.4f",
                    loss_name,
                    state.current_iter,
                    val_subset_metrics["pr_auc"],
                    val_subset_metrics["expected_opt_regret"],
                    val_subset_metrics["realized_regret"],
                )

            # Iteration-based checkpointing
            if checkpoint_every_iters > 0 and (state.current_iter % checkpoint_every_iters == 0):
                save_checkpoint(
                    ckpt_last,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    smooth_buf=smooth_buf,
                    epoch_next=epoch,  # we are still inside epoch
                    model_config=model_config,
                    run_config=run_config,
                    best_score=best_score,
                )

        # Epoch boundary marker
        state.epoch_iters.append(state.current_iter)

        # Full validation at epoch end
        val_metrics = eval_on_loader(model, val_loader, device)
        logging.info(
            "[%s] Epoch %02d | VAL PR-AUC=%.4f | regret(exp*)=%.4f | regret(real)=%.4f",
            loss_name,
            epoch + 1,
            val_metrics["pr_auc"],
            val_metrics["expected_opt_regret"],
            val_metrics["realized_regret"],
        )

        # Save CSVs + plots
        save_state_csvs(state, run_dir)
        
        # Scheduler Step
        scheduler.step()
        logging.info("[%s] LR: %.2e", loss_name, scheduler.get_last_lr()[0])

        # Train Smoothing plots
        if "train_expected_opt_regret" in state.train_smoothed:
            baselines = {}
            if "train_approve_all_expected_regret" in state.train_smoothed:
                baselines["Naive (Approve)"] = state.train_smoothed["train_approve_all_expected_regret"]
            if "train_decline_all_expected_regret" in state.train_smoothed:
                baselines["Naive (Decline)"] = state.train_smoothed["train_decline_all_expected_regret"]

            plot_metric_trajectory(
                iters=state.train_smoothed_iters,
                values=state.train_smoothed["train_expected_opt_regret"],
                out_path=run_dir / "train_expected_opt_regret.png",
                title=f"{loss_name} — Train expected optimal regret (Smoothed)",
                ylabel="Expected Regret",
                epoch_iters=state.epoch_iters,
                baselines=baselines,
            )

        if "train_realized_regret" in state.train_smoothed:
            baselines_real = {}
            if "train_approve_all_realized_regret" in state.train_smoothed:
                baselines_real["Naive (Approve)"] = state.train_smoothed["train_approve_all_realized_regret"]
            if "train_decline_all_realized_regret" in state.train_smoothed:
                baselines_real["Naive (Decline)"] = state.train_smoothed["train_decline_all_realized_regret"]

            plot_metric_trajectory(
                iters=state.train_smoothed_iters,
                values=state.train_smoothed["train_realized_regret"],
                out_path=run_dir / "train_realized_regret.png",
                title=f"{loss_name} — Train realized regret (Smoothed)",
                ylabel="Realized Regret",
                epoch_iters=state.epoch_iters,
                baselines=baselines_real,
            )

        if "train_pr_auc" in state.train_smoothed:
            # Baseline is luck prevalence
            baselines_pr = {"Luck": [train_prevalence] * len(state.train_smoothed_iters)}
            
            n_iters = len(state.train_smoothed_iters)
            # Naive (Decline) = perfect recall (1.0) but only for the positive class.
            # Precision for Naive (Decline) = prevalence.
            baselines_pr["Naive (Decline)"] = [train_prevalence] * n_iters
            # Naive (Approve) = zero recall for positive class. Precision = 0.
            baselines_pr["Naive (Approve)"] = [0.0] * n_iters

            plot_metric_trajectory(
                iters=state.train_smoothed_iters,
                values=state.train_smoothed["train_pr_auc"],
                out_path=run_dir / "train_pr_auc.png",
                title=f"{loss_name} — Train PR-AUC (Smoothed)",
                ylabel="PR-AUC",
                epoch_iters=state.epoch_iters,
                baselines=baselines_pr,
                y_quantile_max=None
            )
        # Validation (Subset) plots
        if state.val_iters:
            if "pr_auc" in state.val_points:
                n_v_iters = len(state.val_iters)
                baselines_val_pr = {
                    "Naive (Decline)": [val_prevalence] * n_v_iters,
                    "Naive (Approve)": [0.0] * n_v_iters,
                }
                plot_metric_trajectory(
                    iters=state.val_iters,
                    values=state.val_points["pr_auc"],
                    out_path=run_dir / "val_pr_auc.png",
                    title=f"{loss_name} — Validation PR-AUC vs iteration",
                    ylabel="PR-AUC",
                    epoch_iters=state.epoch_iters,
                    baselines=baselines_val_pr,
                    y_quantile_max=None
                )
            if "expected_opt_regret" in state.val_points:
                baselines_val = {}
                if "approve_all_expected_cost" in state.val_points:
                    baselines_val["Naive (Approve)"] = state.val_points["approve_all_expected_cost"]
                if "decline_all_expected_cost" in state.val_points:
                    baselines_val["Naive (Decline)"] = state.val_points["decline_all_expected_cost"]

                plot_metric_trajectory(
                    iters=state.val_iters,
                    values=state.val_points["expected_opt_regret"],
                    out_path=run_dir / "val_expected_opt_regret.png",
                    title=f"{loss_name} — Validation expected optimal regret vs iteration",
                    ylabel="€ regret (expected, optimal action)",
                    epoch_iters=state.epoch_iters,
                    baselines=baselines_val,
                    y_quantile_max=None
                )
            if "realized_regret" in state.val_points:
                baselines_val_real = {}
                if "approve_all_realized_cost" in state.val_points:
                    baselines_val_real["Naive (Approve)"] = state.val_points["approve_all_realized_cost"]
                if "decline_all_realized_cost" in state.val_points:
                    baselines_val_real["Naive (Decline)"] = state.val_points["decline_all_realized_cost"]

                plot_metric_trajectory(
                    iters=state.val_iters,
                    values=state.val_points["realized_regret"],
                    out_path=run_dir / "val_realized_regret.png",
                    title=f"{loss_name} — Validation realized regret vs iteration",
                    ylabel="€ regret (realized)",
                    epoch_iters=state.epoch_iters,
                    baselines=baselines_val_real,   
                    y_quantile_max=None
                )

        # PR curve on full validation
        prec, rec, pr_auc = pr_curve(model, val_loader, device)
        
        plot_precision_recall_curve(
            precision=prec,
            recall=rec,
            out_path=run_dir / "val_precision_recall_curve.png",
            title=f"{loss_name} — Validation Precision–Recall (Epoch {epoch+1})",
            average_precision=pr_auc,
            prevalence=val_prevalence,
        )

        # PR curve on training set (for monitoring overfitting)
        # We reuse the train_loader (shuffled, but safe for global PR curve) or a subset if extremely large.
        # Given MLP speed, full pass is acceptable.
        if not quick or epoch == target_epochs - 1: # Always for full run, or last epoch of quick
            prec_train, rec_train, pr_auc_train = pr_curve(model, train_loader, device)
            plot_precision_recall_curve(
                precision=prec_train,
                recall=rec_train,
                out_path=run_dir / "train_precision_recall_curve.png",
                title=f"{loss_name} — Train Precision–Recall (Epoch {epoch+1})",
                average_precision=pr_auc_train,
                prevalence=train_prevalence,
            )

        # Best checkpoint update
        score = float(val_metrics.get(save_best_by, float("nan")))
        if not np.isnan(score) and is_better(score, best_score):
            best_score = score
            save_checkpoint(
                ckpt_best,
                model=model,
                optimizer=optimizer,
                state=state,
                smooth_buf=smooth_buf,
                epoch_next=epoch + 1,
                model_config=model_config,
                run_config=run_config,
                best_score=best_score,
            )

        # Epoch-based checkpointing
        if checkpoint_every_epochs > 0 and ((epoch + 1) % checkpoint_every_epochs == 0):
            save_checkpoint(
                ckpt_last,
                model=model,
                optimizer=optimizer,
                state=state,
                smooth_buf=smooth_buf,
                epoch_next=epoch + 1,
                model_config=model_config,
                run_config=run_config,
                best_score=best_score,
            )

    # Always save a final "last" checkpoint
    save_checkpoint(
        ckpt_last,
        model=model,
        optimizer=optimizer,
        state=state,
        smooth_buf=smooth_buf,
        epoch_next=target_epochs,
        model_config=model_config,
        run_config=run_config,
        best_score=best_score,
    )

    final_val = eval_on_loader(model, val_loader, device)
    final_train = {k: float(v) if v is not None else 0.0 for k, v in smooth_buf.items()}
    
    return {"train": final_train, "val": final_val}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="IEEE-CIS fraud benchmark (resume + iteration plots)")
    parser.add_argument("--run-id", type=str, default="default_run", help="Run identifier (subfolder under --out)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint (continues for --epochs additional epochs)")

    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (additional epochs if --resume is used)")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training and validation batch size (default: 256). Lower values (128-256) recommended for cost-aware losses for better speed.")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate for AdamW optimizer (default: 1e-5)")
    parser.add_argument("--split", type=float, default=0.3,
                       help="Validation set fraction (default: 0.3 = 30%% validation, 70%% training)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode: use only a few batches per epoch for rapid debugging")
    parser.add_argument("--out", type=str, default="fraud_output",
                       help="Output root directory for all runs (default: fraud_output)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training: auto (detect), cpu, cuda (NVIDIA GPU), or mps (Apple Silicon). "
                            "Note: For POT-based losses (sinkhorn_pot) on Apple Silicon, CPU is often faster than MPS.")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    parser.add_argument(
        "--loss",
        type=str,
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "cross_entropy",
            "cross_entropy_weighted",
            "sinkhorn_fenchel_young",
            "sinkhorn_envelope",
            "sinkhorn_autodiff",
            "sinkhorn_pot",
        ],
    )

    # Model architecture
    parser.add_argument("--backbone", type=str, default="mlp", choices=["linear", "mlp"],
                       help="Model architecture: 'linear' (logistic regression) or 'mlp' (multi-layer perceptron)")
    parser.add_argument("--hidden-dims", type=str, default="", 
                       help="Comma-separated hidden layer dimensions for MLP (e.g., '512,256,128'). Empty string uses smart defaults based on input dimension and dataset size.")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate between hidden layers (default: 0.1, auto-tuned if using smart defaults)")
    parser.add_argument("--no-batchnorm", action="store_true", 
                       help="Disable batch normalization in MLP (default: enabled)")

    # Cost-aware epsilon settings
    parser.add_argument("--epsilon-mode", type=str, default="offdiag_mean",
                       choices=["offdiag_mean", "offdiag_median", "offdiag_max", "constant"],
                       help="Epsilon computation mode (default: offdiag_mean)")
    parser.add_argument("--epsilon-scale", type=float, default=1.0,
                       help="Multiplicative scale factor for adaptive epsilon (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=None, 
                       help="Fixed epsilon value for cost-aware losses (overrides --epsilon-mode). Not recommended; use adaptive modes instead.")
    parser.add_argument("--sinkhorn-max-iter", type=int, default=50,
                       help="Maximum Sinkhorn iterations for OT-based losses (default: 50). Lower values (10-20) for faster training.")
    parser.add_argument("--cacis-solver-iter", type=int, default=30,
                       help="Frank-Wolfe solver iterations for Fenchel-Young loss (default: 30)")
    
    # Epsilon scheduling
    parser.add_argument("--epsilon-schedule", type=str, default=None, 
                       choices=[None, "exponential_decay"],
                       help="Epsilon scheduling strategy (default: None = static)")
    parser.add_argument("--epsilon-schedule-start-mult", type=float, default=10.0,
                       help="Starting multiplier for epsilon schedule (default: 10.0)")
    parser.add_argument("--epsilon-schedule-end-mult", type=float, default=0.1,
                       help="Ending multiplier for epsilon schedule (default: 0.1)")

    # Iteration-based evaluation / smoothing
    parser.add_argument("--eval-every", type=int, default=500, help="Validation eval period in iterations (0 disables)")
    parser.add_argument("--smoothing-alpha", type=float, default=0.05, help="Smoothing alpha for train-batch metrics")
    parser.add_argument("--val-subset-size", type=int, default=20000, help="Validation subset size for iteration plots")

    # Checkpointing
    parser.add_argument("--checkpoint-every-iters", type=int, default=0, help="Save checkpoint every N iters (0 disables)")
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1, help="Save checkpoint every N epochs (0 disables)")
    parser.add_argument("--save-best-by", type=str, default="expected_opt_regret", 
                       choices=["expected_opt_regret", "realized_regret", "pr_auc"],
                       help="Metric for selecting best checkpoint: 'expected_opt_regret' (cost under optimal decisions), 'realized_regret' (actual cost), or 'pr_auc' (precision-recall AUC for imbalanced data)")

    # Business cost parameters (see docs/fraud_business_and_cost_matrix.md)
    parser.add_argument("--rho-fd", type=float, default=0.10,
                       help="False decline friction parameter: relative cost of declining a legitimate transaction (default: 0.10 = 10%% additional loss beyond the sale). Total cost = (1+rho_fd)*amount.")
    parser.add_argument("--lambda-cb", type=float, default=1.50,
                       help="Chargeback multiplier: fraud loss factor (default: 1.50 = 1.5× transaction amount). Accounts for principal + fees + operational overhead.")
    parser.add_argument("--F-cb", type=float, default=15.0,
                       help="Fixed chargeback fee per fraud (default: 15.0 currency units). Makes decision threshold amount-dependent. Total fraud cost = lambda_cb*amount + F_cb.")

    args = parser.parse_args()

    setup_logging()
    
    # Smart device selection
    if args.device == "auto":
        device = get_device()
        # Override for POT-based losses on Apple Silicon: CPU is faster than MPS
        # Override for POT-based losses on Apple Silicon: CPU is faster than MPS
        if "sinkhorn_pot" in args.loss and device.type == "mps":
            logging.info("Overriding device: MPS → CPU (faster for sinkhorn_pot on Apple Silicon)")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logging.info("Using device: %s", device)

    # Seeds when not resuming (best-effort reproducibility).
    if not args.resume:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))

    out_root = Path(args.out) / str(args.run_id)
    out_root.mkdir(parents=True, exist_ok=True)

    # Dataset path
    data_folder = "ieee-fraud-detection"
    train_csv = os.path.join(data_folder, "train_transaction.csv")

    dwnld_msg = (
        "rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true\n"
        "mkdir ieee-fraud-detection\n"
        "wget -c http://deraison.ai/ai/ieee-fraud-detection.zip\n"
        "unzip ieee-fraud-detection.zip -d ieee-fraud-detection"
    )
    if not Path(train_csv).exists():
        logging.error("Missing file: %s\nRun:\n%s", train_csv, dwnld_msg)
        return

    logging.info("Loading train transactions from %s ...", train_csv)
    full_df = pd.read_csv(train_csv, engine="python")
    if "TransactionAmt" not in full_df.columns or "isFraud" not in full_df.columns or "TransactionDT" not in full_df.columns:
        raise ValueError("Expected IEEE-CIS columns: TransactionAmt, isFraud, TransactionDT.")

    # Sort by TransactionDT for temporal split
    logging.info("Sorting by TransactionDT for temporal split...")
    full_df = full_df.sort_values("TransactionDT").reset_index(drop=True)

    # Split
    split_idx = int(len(full_df) * (1 - float(args.split)))
    train_df = full_df.iloc[:split_idx].copy()
    val_df = full_df.iloc[split_idx:].copy()
    
    # train_test_split removed in favor of temporal split
    logging.info("Temporal Split: train=%d, val=%d (split_idx=%d)", len(train_df), len(val_df), split_idx)

    # Visualize temporal split
    plot_temporal_split(
        train_df["TransactionDT"].values,
        val_df["TransactionDT"].values,
        out_root / "temporal_split.png"
    )

    train_df2, val_df2, feature_cols = make_features(train_df, val_df)

    # De-fragment (consolidate blocks) once to avoid PerformanceWarning on inserts
    train_df2 = train_df2.copy()
    val_df2 = val_df2.copy()

    # Save raw amount for cost calculation before scaling
    train_df2["TransactionAmt_raw"] = train_df2["TransactionAmt"].to_numpy(copy=True)
    val_df2["TransactionAmt_raw"] = val_df2["TransactionAmt"].to_numpy(copy=True)

    # Apply RobustScaler to features
    logging.info("Applying RobustScaler to %d features...", len(feature_cols))
    scaler = RobustScaler()
    train_df2[feature_cols] = scaler.fit_transform(train_df2[feature_cols])
    val_df2[feature_cols] = scaler.transform(val_df2[feature_cols])

    input_dim = len(feature_cols)
    logging.info("Feature dimension: %d", input_dim)

    params = BusinessParams(rho_fd=float(args.rho_fd), lambda_cb=float(args.lambda_cb), F_cb=float(args.F_cb))
    logging.info("Business params: rho_fd=%.3f, lambda_cb=%.3f, F_cb=%.2f", params.rho_fd, params.lambda_cb, params.F_cb)

    # Median-normalize CE weights once (robust scaling)
    train_amount = train_df2["TransactionAmt_raw"].values.astype(np.float32)
    train_y = train_df2["isFraud"].values.astype(np.int64)
    train_C = build_cost_matrix(train_amount, params)
    w_raw = sample_weight_from_C(train_y, train_C)
    w_median = float(np.median(w_raw[w_raw > 0])) if np.any(w_raw > 0) else 1.0
    logging.info("Weighted-CE normalization: median(w_raw)=%.6f", w_median)

    train_ds = FraudDataset(train_df2, feature_cols, params, weight_norm_median=w_median)
    val_ds = FraudDataset(val_df2, feature_cols, params, weight_norm_median=w_median)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    
    val_prevalence = float(val_df2["isFraud"].mean())
    train_prevalence = float(train_df2["isFraud"].mean())
    logging.info("Prevalence: train=%.4f, val=%.4f", train_prevalence, val_prevalence)

    # Fixed validation subset for iteration plots (deterministic)
    val_subset_size = int(args.val_subset_size)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(val_ds), size=min(val_subset_size, len(val_ds)), replace=False)
    val_subset = torch.utils.data.Subset(val_ds, idx.tolist())
    val_loader_subset = DataLoader(val_subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    logging.info("Validation subset size: %d", len(val_subset))

    # Parse hidden dims
    hidden_dims: Tuple[int, ...]
    if str(args.hidden_dims).strip() == "":
        hidden_dims = ()
    else:
        hidden_dims = tuple(int(s) for s in str(args.hidden_dims).split(",") if s.strip() != "")

    # Apply smart defaults (function is imported from tabular_models)
    if not hidden_dims:
        smart_hidden_dims, smart_dropout = compute_smart_architecture_defaults(
            input_dim=int(input_dim),
            n_train=len(train_ds),
            n_classes=2,
        )
    else:
        smart_hidden_dims = hidden_dims
        smart_dropout = float(args.dropout)
    
    # Use user-provided dropout if specified, otherwise use smart default
    if args.dropout != 0.1:  # 0.1 is argparse default
        final_dropout = float(args.dropout)
    else:
        final_dropout = smart_dropout
    
    logging.info("Architecture: input_dim=%d, hidden_dims=%s, dropout=%.2f", 
                 input_dim, smart_hidden_dims, final_dropout)

    backbone: BackboneName = "mlp" if args.backbone == "mlp" else "linear"
    model_cfg = TabularModelConfig(
        input_dim=int(input_dim),
        backbone=backbone,
        hidden_dims=smart_hidden_dims,
        dropout=final_dropout,
        use_batchnorm=not bool(args.no_batchnorm),
    )
    model_config_dict = asdict(model_cfg)

    # Methods
    methods: List[LossName]
    if "all" in args.loss:
        methods = [
            "cross_entropy",
            "cross_entropy_weighted",
            "sinkhorn_fenchel_young",
            "sinkhorn_envelope",
            "sinkhorn_autodiff",
            "sinkhorn_pot",
        ]
    else:
        methods = args.loss  # type: ignore[assignment]

    run_config: Dict[str, Any] = vars(args).copy()
    run_config["device_resolved"] = str(device)
    run_config["feature_dim"] = int(input_dim)
    run_config["weight_median"] = float(w_median)

    results: Dict[str, Dict[str, float]] = {}
    for loss_name in methods:
        run_dir = out_root / str(loss_name)
        model = TabularRiskModel(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )
        # Initialize bias to reflect prevalence (helps calibration and initial regret)
        model.initialize_output_bias(train_prevalence)
        logging.info("[%s] Initialized output bias to reflect train_prevalence=%.4f", loss_name, train_prevalence)

        try:
            metrics = train_one(
                run_dir=run_dir,
                loss_name=loss_name,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                val_prevalence=val_prevalence,
                train_prevalence=train_prevalence,
                val_loader_subset=val_loader_subset,
                device=device,
                epochs_additional=int(args.epochs),
                quick=bool(args.quick),
                eval_every=int(args.eval_every),
                smoothing_alpha=float(args.smoothing_alpha),
                epsilon_mode=str(args.epsilon_mode),
                epsilon_scale=float(args.epsilon_scale),
                epsilon=args.epsilon,
                sinkhorn_max_iter=int(args.sinkhorn_max_iter),
                cacis_solver_iter=int(args.cacis_solver_iter),
                epsilon_schedule=args.epsilon_schedule,
                schedule_start_mult=float(args.epsilon_schedule_start_mult),
                schedule_end_mult=float(args.epsilon_schedule_end_mult),
                resume=bool(args.resume),
                save_best_by=str(args.save_best_by),
                checkpoint_every_iters=int(args.checkpoint_every_iters),
                checkpoint_every_epochs=int(args.checkpoint_every_epochs),
                run_config=run_config,
                model_config=model_config_dict,
            )
            # Merge train and val into a single row
            row = metrics["train"].copy()
            # Prefix validation metrics
            for k, v in metrics["val"].items():
                row[f"val_{k}"] = v
            
            results[loss_name] = row
        except ImportError as e:
            msg = f"[{loss_name}] Skipped (missing dependency): {str(e)}"
            logging.error(msg)
            raise ValueError(msg)

        if results:
            summary = pd.DataFrame(results).T
            summary_path = out_root / "summary.csv"
            summary.to_csv(summary_path, index=True)
            logging.info("Saved central summary to: %s", summary_path)
            
            with pd.option_context("display.max_columns", 50, "display.width", 140):
                logging.info("Updated Summary (Train + Val):\n%s", summary)

    logging.info("Done.")


if __name__ == "__main__":
    main()
