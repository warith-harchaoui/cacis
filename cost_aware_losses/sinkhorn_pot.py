"""
cost_aware_losses.sinkhorn_pot
==============================

Entropic OT-loss for classification using the Python Optimal Transport (POT) library.

This module provides :class:`~cost_aware_losses.sinkhorn_pot.SinkhornPOTLoss`, a
cost-aware classification loss based on *balanced* entropic optimal transport (OT).

Core idea
---------
Given a predicted distribution ``p`` (from ``softmax(scores)``) and a target
distribution ``q`` (one-hot with optional label smoothing), we compute an entropic
optimal transport plan ``P*`` that solves the regularized OT problem for each
example. We then evaluate the *primal* objective:

    <P*, C> + ε * KL(P* || p ⊗ q)

Differentiability + performance
-------------------------------
To keep gradients stable and avoid backpropagating through Sinkhorn iterations,
we compute the transport plan ``P*`` under ``torch.no_grad()`` (envelope gradient),
then compute a differentiable objective that depends on ``p`` (and optionally on
``ε(C)`` and ``C``). This yields meaningful gradients w.r.t. ``scores`` while
keeping the Sinkhorn solve out of the autograd graph.

This implementation tries to run POT on the **PyTorch backend** (including CUDA)
to avoid expensive NumPy/CPU round-trips. If POT backend dispatch fails, you may
optionally allow a NumPy fallback (CPU), controlled by ``allow_numpy_fallback``.

Usage example
-------------
>>> import torch
>>> from cost_aware_losses import SinkhornPOTLoss
>>>
>>> scores = torch.randn(8, 4, requires_grad=True)
>>> targets = torch.randint(0, 4, (8,))
>>>
>>> # Shared cost matrix (K, K) is allowed:
>>> C = torch.rand(4, 4)
>>> C.fill_diagonal_(0.0)
>>>
>>> loss_fn = SinkhornPOTLoss(max_iter=20, allow_numpy_fallback=True)
>>> loss = loss_fn(scores, targets, C=C)
>>> loss.backward()

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>

Notes
-----
- POT uses ``reg`` for entropic regularization (our ``epsilon``).
- Envelope gradient: ``P*`` is treated as constant in backward.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Union

import numpy as np
import torch
from torch import Tensor

import ot

from .base import CostAwareLoss

logger = logging.getLogger(__name__)


def _pot_sinkhorn_plan(
    p: Tensor,
    q: Tensor,
    C: Tensor,
    eps: Tensor,
    *,
    max_iter: int,
    stopThr: float = 1e-9,
    method: Literal["sinkhorn", "sinkhorn_log"] = "sinkhorn",
    allow_numpy_fallback: bool = False,
) -> Tensor:
    """
    Compute the entropic optimal transport plan using POT's Sinkhorn solver.

    Supports either a batched cost matrix ``C`` of shape ``(B, K, K)`` or a shared
    cost matrix of shape ``(K, K)`` (which will be broadcast to the batch).

    Parameters
    ----------
    p, q : torch.Tensor
        Distributions of shape ``(B, K)``. Each row should sum to 1.
    C : torch.Tensor
        Cost matrix, either:
        - ``(B, K, K)`` batched, or
        - ``(K, K)`` shared across batch.
    eps : torch.Tensor
        Regularization ε, either scalar tensor or shape ``(B,)``.
    max_iter : int
        Maximum number of Sinkhorn iterations.
    stopThr : float, default=1e-9
        Convergence threshold used by POT.
    method : {"sinkhorn", "sinkhorn_log"}, default="sinkhorn"
        POT Sinkhorn variant. ``"sinkhorn_log"`` is often more stable for small ε.
    allow_numpy_fallback : bool, default=False
        If True, fallback to CPU/NumPy when POT torch backend fails.
        If False, raise after logging.

    Returns
    -------
    torch.Tensor
        Transport plans of shape ``(B, K, K)``.
    torch.Tensor
        Dual potentials f = eps*log(u) of shape ``(B, K)``.
    torch.Tensor
        Dual potentials g = eps*log(v) of shape ``(B, K)``.
    """
    # -------------------------
    # Validate p, q shapes
    # -------------------------
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(f"p and q must have shape (B, K). Got p={p.shape}, q={q.shape}.")
    if p.shape != q.shape:
        raise ValueError(f"p and q must share shape (B, K). Got p={p.shape}, q={q.shape}.")

    B, K = p.shape
    device, dtype = p.device, p.dtype

    # -------------------------
    # Accept C as (K, K) or (B, K, K)
    # -------------------------
    if C.ndim == 2:
        if C.shape != (K, K):
            raise ValueError(f"Shared C must have shape (K, K)={(K, K)}. Got C={C.shape}.")
        # Broadcast to (B, K, K) WITHOUT allocating if possible.
        # NOTE: We still index C[i] later; expand provides a view.
        Cb = C.unsqueeze(0).expand(B, K, K)
    elif C.ndim == 3:
        if C.shape != (B, K, K):
            raise ValueError(f"Batched C must have shape (B, K, K)={(B, K, K)}. Got C={C.shape}.")
        Cb = C
    else:
        raise ValueError(f"C must have shape (K,K) or (B,K,K). Got C={C.shape}.")

    # -------------------------
    # Normalize eps to (B,)
    # -------------------------
    if eps.ndim == 0:
        eps_batch = eps.expand(B)
    elif eps.ndim == 1 and eps.shape[0] == B:
        eps_batch = eps
    else:
        raise ValueError(f"eps must be scalar or (B,), got shape {eps.shape}.")

    # -------------------------
    # Allocate output P (B, K, K), f (B, K), g (B, K)
    # f, g are dual potentials: f = eps * log(u), g = eps * log(v)
    # -------------------------
    P = torch.empty((B, K, K), device=device, dtype=dtype)
    f = torch.empty((B, K), device=device, dtype=dtype)
    g = torch.empty((B, K), device=device, dtype=dtype)

    # -------------------------
    # Solve per example (POT sinkhorn is 2D per call)
    # -------------------------
    for i in range(B):
        reg_i = eps_batch[i]

        # Fast path: torch backend (keeps GPU if tensors are CUDA)
        try:
            # We need the log dictionary to get u and v
            out = ot.sinkhorn(
                a=p[i],
                b=q[i],
                M=Cb[i],
                reg=reg_i,
                numItermax=int(max_iter),
                stopThr=float(stopThr),
                method=method,
                log=True,
            )
            
            # Unpack based on return type
            if isinstance(out, tuple) and len(out) == 2:
                P_i, log_dict = out
            else:
                # Should not happen with log=True, but safety fallback
                P_i, log_dict = out, {}

            # Function to extract potentials
            def get_potentials(ld, current_reg):
                if "log_u" in ld and "log_v" in ld:
                    lu = ld["log_u"]
                    lv = ld["log_v"]
                    lu = lu.to(device=device, dtype=dtype) if isinstance(lu, torch.Tensor) else torch.as_tensor(lu, device=device, dtype=dtype)
                    lv = lv.to(device=device, dtype=dtype) if isinstance(lv, torch.Tensor) else torch.as_tensor(lv, device=device, dtype=dtype)
                    return current_reg * lu, current_reg * lv
                elif "alpha" in ld and "beta" in ld:
                    alpha = ld["alpha"]
                    beta = ld["beta"]
                    alpha = alpha.to(device=device, dtype=dtype) if isinstance(alpha, torch.Tensor) else torch.as_tensor(alpha, device=device, dtype=dtype)
                    beta = beta.to(device=device, dtype=dtype) if isinstance(beta, torch.Tensor) else torch.as_tensor(beta, device=device, dtype=dtype)
                    return alpha, beta
                elif "u" in ld and "v" in ld:
                    u_raw = ld["u"]
                    v_raw = ld["v"]
                    u_i = u_raw.to(device=device, dtype=dtype) if isinstance(u_raw, torch.Tensor) else torch.as_tensor(u_raw, device=device, dtype=dtype)
                    v_i = v_raw.to(device=device, dtype=dtype) if isinstance(v_raw, torch.Tensor) else torch.as_tensor(v_raw, device=device, dtype=dtype)
                    return current_reg * torch.log(u_i + 1e-16), current_reg * torch.log(v_i + 1e-16)
                return torch.zeros(K, device=device, dtype=dtype), torch.zeros(K, device=device, dtype=dtype)

            f_i, g_i = get_potentials(log_dict, reg_i)

            # Retry logic if non-finite
            if not torch.isfinite(f_i).all() or not torch.isfinite(g_i).all():
                logger.warning(
                    f"POT sinkhorn returned non-finite potentials for batch index {i}. "
                    "Retrying with double iterations."
                )
                out = ot.sinkhorn(
                    a=p[i],
                    b=q[i],
                    M=Cb[i],
                    reg=reg_i,
                    numItermax=int(2 * max_iter),  # Double iterations
                    stopThr=float(stopThr),
                    method=method,
                    log=True,
                )
                if isinstance(out, tuple) and len(out) == 2:
                    P_i, log_dict = out
                else:
                    P_i, log_dict = out, {}
                f_i, g_i = get_potentials(log_dict, reg_i)

            # Final stabilization
            if not torch.isfinite(f_i).all() or not torch.isfinite(g_i).all():
                logger.warning(f"POT sinkhorn still non-finite after retry for batch index {i}. Stabilizing.")
                f_i = torch.nan_to_num(f_i, nan=0.0, posinf=1e6, neginf=-1e6)
                g_i = torch.nan_to_num(g_i, nan=0.0, posinf=1e6, neginf=-1e6)

            P[i] = P_i.to(device=device, dtype=dtype) if isinstance(P_i, torch.Tensor) else torch.as_tensor(P_i, device=device, dtype=dtype)
            f[i], g[i] = f_i, g_i

        except Exception as e:
            msg = (
                "POT sinkhorn torch-backend call failed for batch index "
                f"{i}/{B - 1}. allow_numpy_fallback={allow_numpy_fallback}. "
                f"Original error: {repr(e)}"
            )
            logger.error(msg)

            if not allow_numpy_fallback:
                raise RuntimeError(msg) from e

            # Fallback: CPU NumPy
            p_i_np = p[i].detach().cpu().numpy()
            q_i_np = q[i].detach().cpu().numpy()
            C_i_np = Cb[i].detach().cpu().numpy()
            reg_np = float(reg_i.detach().cpu().item())

            out_np = ot.sinkhorn(
                a=p_i_np,
                b=q_i_np,
                M=C_i_np,
                reg=reg_np,
                numItermax=int(max_iter),
                stopThr=float(stopThr),
                method=method,
                log=True,
            )
            
            if isinstance(out_np, tuple) and len(out_np) == 2:
                P_i_np, log_dict_np = out_np
            else:
                P_i_np, log_dict_np = out_np, {}

            P[i] = torch.from_numpy(P_i_np).to(device=device, dtype=dtype)
            
            # Extract potentials from numpy log
            if "log_u" in log_dict_np and "log_v" in log_dict_np:
                 f[i] = reg_i * torch.from_numpy(log_dict_np["log_u"]).to(device=device, dtype=dtype)
                 g[i] = reg_i * torch.from_numpy(log_dict_np["log_v"]).to(device=device, dtype=dtype)
            elif "alpha" in log_dict_np and "beta" in log_dict_np:
                 f[i] = torch.from_numpy(log_dict_np["alpha"]).to(device=device, dtype=dtype)
                 g[i] = torch.from_numpy(log_dict_np["beta"]).to(device=device, dtype=dtype)
            elif "u" in log_dict_np and "v" in log_dict_np:
                 u_np = log_dict_np["u"]
                 v_np = log_dict_np["v"]
                 f_np = reg_np * np.log(u_np + 1e-16)
                 g_np = reg_np * np.log(v_np + 1e-16)
                 f[i] = torch.from_numpy(f_np).to(device=device, dtype=dtype)
                 g[i] = torch.from_numpy(g_np).to(device=device, dtype=dtype)
            else:
                 f[i].fill_(0.0)
                 g[i].fill_(0.0)

    return P, f, g


def _entropy_kl_objective(P: Tensor, p: Tensor, q: Tensor, eps: Tensor, C: Tensor) -> Tensor:
    """
    Entropic OT primal objective with KL regularization against p ⊗ q.

    Objective per example:
        <P, C> + ε * KL(P || p ⊗ q)

    Parameters
    ----------
    P : torch.Tensor
        Transport plan of shape ``(B, K, K)``.
    p, q : torch.Tensor
        Marginals of shape ``(B, K)``.
    eps : torch.Tensor
        Regularization ε (scalar or ``(B,)``).
    C : torch.Tensor
        Cost matrices of shape ``(B, K, K)``.

    Returns
    -------
    torch.Tensor
        Objective values of shape ``(B,)``.
    """
    tiny = torch.as_tensor(1e-12, device=P.device, dtype=P.dtype)

    # Clamp to keep logs finite and handle NaNs
    P_safe = torch.nan_to_num(P, nan=1e-12).clamp(min=tiny)
    p_safe = torch.nan_to_num(p, nan=1e-12).clamp(min=tiny)
    q_safe = torch.nan_to_num(q, nan=1e-12).clamp(min=tiny)

    # <P, C>
    cost_term = (P_safe * C).sum(dim=(1, 2))

    # KL(P || p ⊗ q)
    logP = torch.log(P_safe)
    logp = torch.log(p_safe).unsqueeze(2)  # (B, K, 1)
    logq = torch.log(q_safe).unsqueeze(1)  # (B, 1, K)
    kl = (P_safe * (logP - logp - logq)).sum(dim=(1, 2))

    return cost_term + eps * kl


class SinkhornPOTLoss(CostAwareLoss):
    """
    Sinkhorn OT-loss using the Python Optimal Transport (POT) library.

    Parameters
    ----------
    epsilon_mode, epsilon, epsilon_scale, epsilon_min :
        See :class:`~cost_aware_losses.base.CostAwareLoss`.
    max_iter : int, default=20
        Maximum number of Sinkhorn iterations in POT.
    stopThr : float, default=1e-9
        Convergence threshold for POT's Sinkhorn.
    label_smoothing : float, default=1e-3
        Label smoothing applied to one-hot ``q``.
    method : {"sinkhorn", "sinkhorn_log"}, default="sinkhorn"
        POT Sinkhorn method.
    allow_numpy_fallback : bool, default=False
        Enable CPU/NumPy fallback if POT torch backend fails.

    Notes
    -----
    - ``C`` can be provided either as ``(K, K)`` (shared) or ``(B, K, K)`` (batched).
    - Envelope gradient: compute ``P*`` under ``no_grad`` and detach, while keeping
      differentiability through ``p = softmax(scores)``.
    """

    def __init__(
        self,
        *,
        epsilon_mode: str = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        max_iter: int = 50,
        stopThr: float = 1e-5,
        label_smoothing: float = 1e-3,
        method: Literal["sinkhorn", "sinkhorn_log"] = "sinkhorn_log",
        allow_numpy_fallback: bool = False,
        epsilon_schedule: Optional[str] = None,
        schedule_start_mult: float = 10.0,
        schedule_end_mult: float = 0.1,
        total_epochs: Optional[int] = None,
    ) -> None:
        super().__init__(
            epsilon_mode=epsilon_mode,  # type: ignore[arg-type]
            epsilon=epsilon,
            epsilon_scale=epsilon_scale,
            epsilon_min=epsilon_min,
            epsilon_schedule=epsilon_schedule,
            schedule_start_mult=schedule_start_mult,
            schedule_end_mult=schedule_end_mult,
            total_epochs=total_epochs,
        )
        self.max_iter = int(max_iter)
        self.stopThr = float(stopThr)
        self.label_smoothing = float(label_smoothing)
        self.method = method
        self.allow_numpy_fallback = bool(allow_numpy_fallback)

    def _loss_per_example(self, scores: Tensor, targets: Tensor, Cb: Tensor) -> Tensor:
        """
        Per-example loss values using POT Sinkhorn.

        Parameters
        ----------
        scores : torch.Tensor
            Logits of shape ``(B, K)``.
        targets : torch.Tensor
            Integer labels of shape ``(B,)``.
        Cb : torch.Tensor
            Cost matrix, either ``(K, K)`` shared or ``(B, K, K)`` batched.

        Returns
        -------
        torch.Tensor
            Loss values of shape ``(B,)``.
        """
        # -------------------------
        # Basic sizes
        # -------------------------
        B, K = scores.shape

        # -------------------------
        # Predicted distribution p (grad flows here)
        # -------------------------
        # Clamp logits for numerical stability before softmax
        scores_stab = torch.clamp(scores, min=-100.0, max=100.0)
        p = torch.softmax(scores_stab, dim=1)

        # -------------------------
        # Target distribution q (one-hot + optional smoothing)
        # -------------------------
        q = torch.zeros((B, K), device=scores.device, dtype=scores.dtype)
        q.scatter_(1, targets.view(-1, 1), 1.0)

        if self.label_smoothing > 0.0:
            delta = self.label_smoothing
            q = (1.0 - delta) * q + delta / float(K)

        # -------------------------
        # Broadcast shared cost (K,K) to (B,K,K) for epsilon computation + objective
        # -------------------------
        if Cb.ndim == 2:
            if Cb.shape != (K, K):
                raise ValueError(f"Shared C must have shape (K,K)={(K,K)}. Got {Cb.shape}.")
            Cb_full = Cb.unsqueeze(0).expand(B, K, K)
        elif Cb.ndim == 3:
            if Cb.shape != (B, K, K):
                raise ValueError(f"Batched C must have shape (B,K,K)={(B,K,K)}. Got {Cb.shape}.")
            Cb_full = Cb
        else:
            raise ValueError(f"C must have shape (K,K) or (B,K,K). Got {Cb.shape}.")

        # -------------------------
        # Epsilon may depend on C (depending on epsilon_mode)
        # -------------------------
        eps = self.compute_epsilon(Cb_full)  # scalar or (B,)

        # -------------------------
        # Solve for P* without tracking gradients
        # -------------------------
        # -------------------------
        # Solve for P* and potentials f, g
        # -------------------------
        with torch.no_grad():
            P, f, g = _pot_sinkhorn_plan(
                p=p.detach(),
                q=q,  # no grad anyway
                C=Cb_full.detach(),
                eps=eps.detach(),
                max_iter=self.max_iter,
                stopThr=self.stopThr,
                method=self.method,
                allow_numpy_fallback=self.allow_numpy_fallback,
            )

        # -------------------------
        # Helper: Primal value
        # -------------------------
        P = P.detach()
        primal_val = _entropy_kl_objective(P, p, q, eps, Cb_full)

        # -------------------------
        # Gradient Grafting
        # -------------------------
        # Correct gradient w.r.t p is: f - eps * (1 + log(p))
        # Or simply: f - eps * log(p)   (constants vanish for softmax)
        #
        # We implement this by adding term: (f * p).sum() - eps * (p * log(p)).sum()
        # The sum(p * log(p)) term ensures we pick up the -eps*(1+log p) gradient.
        
        tiny = 1e-16
        grad_term_f = (f.detach() * p).sum(dim=1)
        
        # Entropy term for correction
        # H_part = sum(p log p)
        # We subtract eps * H_part.
        # Gradient is -eps * (1 + log p). Correct.
        eps_b = eps.view(-1) if eps.ndim == 1 else eps
        entropy_part = (p * torch.log(p + tiny)).sum(dim=1)
        correction_term = - eps_b * entropy_part

        # Combined graft
        graft = grad_term_f + correction_term
        
        # Identity subtraction to detach value, with NaN safeguard
        res = primal_val.detach() + graft - graft.detach()
        return torch.nan_to_num(res, nan=0.0)
