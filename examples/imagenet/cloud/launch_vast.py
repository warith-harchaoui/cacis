"""
examples.imagenet.cloud.launch_vast
===================================

Launch an ImageNet cost-aware training run on a single Vast.ai instance
(default: RTX 4090 24 GB). Sequentially runs every loss listed in the config
on one box and syncs results to a Backblaze B2 bucket via rclone.

What it does
------------
1. Loads ``config/cloud-vast.yaml`` via :class:`VastConfig`.
2. Reads the secrets named in ``credentials`` from your shell environment.
3. Searches Vast.ai for the cheapest offer matching your criteria.
4. Creates an instance running the ``cacis-imagenet`` image with the bootstrap
   script (baked into the image at
   ``/workspace/examples/imagenet/cloud/vast_bootstrap.sh``) as ``onstart``.
5. Prints the instance id and a one-liner to tail logs:
   ``vastai logs <id> --tail 200 --follow``.

The instance self-destroys after training (config: ``auto_destroy: true``).

Usage
-----
::

    # Render and inspect the env/onstart that *would* be sent (no API calls):
    python -m examples.imagenet.cloud.launch_vast \\
        --config config/cloud-vast.yaml --dry-run

    # Actually launch:
    python -m examples.imagenet.cloud.launch_vast \\
        --config config/cloud-vast.yaml

Prereqs
-------
- ``pip install vastai`` and ``vastai set api-key <YOUR-KEY>`` once.
- Environment variables set in your shell (names configurable in the YAML):
  ``KAGGLE_USERNAME, KAGGLE_KEY, B2_KEY_ID, B2_APP_KEY, COST_MATRIX_URL,
  VASTAI_API_KEY``.
- A public-readable ``cost_matrix.pt`` somewhere (``COST_MATRIX_URL``).
- A Backblaze B2 bucket reachable through the rclone remote in ``output.rclone_remote``.
- An image pushed to Docker Hub (or another public registry); see
  ``scripts/build_and_push.sh``.

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from examples.imagenet.cloud.config import VastConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Secret resolution
# =============================================================================

def _resolve_secrets(cfg: VastConfig, *, allow_missing: bool = False) -> Dict[str, str]:
    """
    Read each ``*_env`` from the shell and return ``{ENV_NAME: VALUE}``.

    When ``allow_missing`` is True (used by ``--dry-run``), missing variables
    are filled with the placeholder ``<UNSET:VARNAME>`` instead of raising. The
    placeholder is obvious in dry-run output and makes the rest of the
    rendering path work without real credentials.
    """
    creds = cfg.credentials
    needed = {
        creds.vastai_api_key_env:    "vast.ai API key",
        creds.kaggle_username_env:   "Kaggle username",
        creds.kaggle_key_env:        "Kaggle API key",
        creds.b2_key_id_env:         "Backblaze B2 key id",
        creds.b2_app_key_env:        "Backblaze B2 app key",
        creds.cost_matrix_url_env:   "public URL of cost_matrix.pt",
    }
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for env_name, desc in needed.items():
        val = os.environ.get(env_name)
        if not val:
            missing.append(f"  - ${env_name}  ({desc})")
            resolved[env_name] = f"<UNSET:{env_name}>"
        else:
            resolved[env_name] = val
    if missing and not allow_missing:
        raise RuntimeError(
            "The following environment variables are not set:\n"
            + "\n".join(missing)
            + "\nExport them in your shell before launching, "
            + "or use --dry-run to inspect what would be sent."
        )
    if missing and allow_missing:
        logger.warning(
            "%d env var(s) missing — using placeholders for --dry-run:\n%s",
            len(missing), "\n".join(missing),
        )
    return resolved


# =============================================================================
# Render bootstrap environment
# =============================================================================

def build_instance_env(cfg: VastConfig, secrets: Dict[str, str]) -> Dict[str, str]:
    """
    Build the env dict shipped into the Vast.ai container.

    The bootstrap script reads from these keys (not the ``*_env`` indirection
    keys used in the YAML).
    """
    creds = cfg.credentials
    t = cfg.training
    env: Dict[str, str] = {
        # Credentials (resolved from the user's shell)
        "VAST_API_KEY":     secrets[creds.vastai_api_key_env],
        "KAGGLE_USERNAME":  secrets[creds.kaggle_username_env],
        "KAGGLE_KEY":       secrets[creds.kaggle_key_env],
        "B2_KEY_ID":        secrets[creds.b2_key_id_env],
        "B2_APP_KEY":       secrets[creds.b2_app_key_env],
        "COST_MATRIX_URL":  secrets[creds.cost_matrix_url_env],
        # Output destination
        "RCLONE_DEST":      f"{cfg.output.rclone_remote.rstrip('/')}/{cfg.run_id}",
        # Run definition
        "RUN_PREFIX":       cfg.run_id,
        "LOSSES":           " ".join(cfg.losses),
        "AUTO_DESTROY":     "true" if cfg.auto_destroy else "false",
        # Training hyperparameters
        "ARCH":             t.arch,
        "NUM_CLASSES":      str(t.num_classes),
        "BATCH_SIZE":       str(t.batch_size),
        "EPOCHS":           str(t.epochs),
        "WARMUP_EPOCHS":    str(t.warmup_epochs),
        "LR":               repr(t.lr),
        "WEIGHT_DECAY":     repr(t.weight_decay),
        "LABEL_SMOOTHING":  repr(t.label_smoothing),
        "SINKHORN_MAX_ITER": str(t.sinkhorn_max_iter),
        "EPSILON_MODE":     t.epsilon_mode,
        "EPSILON_SCALE":    repr(t.epsilon_scale),
        "NUM_WORKERS":      str(t.num_workers),
    }
    return env


def env_to_vastai_flag(env: Dict[str, str]) -> str:
    """
    Render the env dict as the single ``-e KEY=VAL`` blob Vast.ai expects.

    ``vastai create instance`` takes ``--env "-e A=1 -e B=2 …"``.
    Values are shell-quoted to survive Vast.ai's reshelling.
    """
    parts = [f"-e {k}={shlex.quote(v)}" for k, v in env.items()]
    return " ".join(parts)


_ONSTART_CMD = "bash /workspace/examples/imagenet/cloud/vast_bootstrap.sh"


# =============================================================================
# Vast.ai CLI shell-outs
# =============================================================================

def _run_vastai(args: List[str], *, api_key: str, capture: bool = True) -> str:
    """Invoke the ``vastai`` CLI with a temporary API key in the env."""
    env = {**os.environ, "VAST_API_KEY": api_key}
    cmd = ["vastai", *args]
    logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, env=env,
        capture_output=capture, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"vastai {' '.join(args)} failed (exit {proc.returncode})\n"
            f"stderr: {proc.stderr.strip()}"
        )
    return proc.stdout


def find_cheapest_offer(cfg: VastConfig, api_key: str) -> Dict[str, Any]:
    """
    Run ``vastai search offers`` and return the cheapest matching offer.

    The CLI's search query is a single quoted string of filters; we sort
    ascending by dollars-per-hour and take the first.
    """
    i = cfg.instance
    query = (
        f"reliability >= {i.min_reliability} "
        f"num_gpus = {i.num_gpus} "
        f"gpu_name = {i.gpu_name} "
        f"disk_space >= {i.min_disk_gb} "
        f"dph_total <= {i.max_price_per_hour} "
        f"cuda_max_good >= {i.cuda_max_good}"
    )
    raw = _run_vastai(
        ["search", "offers", query, "-o", "dph_total+", "--raw"],
        api_key=api_key,
    )
    offers = json.loads(raw)
    if not offers:
        raise RuntimeError(
            f"No Vast.ai offers match your criteria:\n  {query}\n"
            "Try raising max_price_per_hour, lowering min_disk_gb, or "
            "relaxing min_reliability."
        )
    return offers[0]


def create_instance(
    offer_id: int,
    *,
    image: str,
    disk_gb: int,
    env_flag: str,
    onstart_cmd: str,
    api_key: str,
) -> Dict[str, Any]:
    """Create the instance and return the JSON response from Vast.ai."""
    raw = _run_vastai(
        [
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--env", env_flag,
            "--onstart-cmd", onstart_cmd,
            "--raw",
        ],
        api_key=api_key,
    )
    return json.loads(raw)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a Vast.ai ImageNet training run.",
    )
    p.add_argument(
        "--config", type=Path, required=True,
        help="Path to a cloud-vast.yaml (start from config/cloud-vast.yaml.example).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Render the env + onstart and print them; do not call Vast.ai.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = VastConfig.from_yaml(args.config)
    secrets = _resolve_secrets(cfg, allow_missing=args.dry_run)
    env = build_instance_env(cfg, secrets)
    env_flag = env_to_vastai_flag(env)

    if args.dry_run:
        logger.info("Dry run — no Vast.ai calls.")
        logger.info("Bootstrap onstart cmd: %s", _ONSTART_CMD)
        logger.info("Environment that would be shipped:")
        for k, v in env.items():
            shown = v if k not in (
                "VAST_API_KEY", "KAGGLE_KEY", "B2_APP_KEY",
            ) else f"<{len(v)} chars hidden>"
            logger.info("  %s = %s", k, shown)
        return

    logger.info("Searching for cheapest matching Vast.ai offer ...")
    offer = find_cheapest_offer(cfg, secrets[cfg.credentials.vastai_api_key_env])
    logger.info(
        "Picked offer %s — %s × %d  @  $%.4f/hr  (reliability %.2f)",
        offer.get("id"), offer.get("gpu_name"), offer.get("num_gpus", 1),
        offer.get("dph_total", 0.0), offer.get("reliability2", 0.0),
    )

    logger.info("Creating instance ...")
    response = create_instance(
        offer["id"],
        image=cfg.container.image,
        disk_gb=cfg.instance.min_disk_gb,
        env_flag=env_flag,
        onstart_cmd=_ONSTART_CMD,
        api_key=secrets[cfg.credentials.vastai_api_key_env],
    )
    instance_id = response.get("new_contract") or response.get("id")
    logger.info("✓ Launched instance id = %s", instance_id)
    logger.info("")
    logger.info("Monitor with:")
    logger.info("    vastai logs %s --tail 200 --follow", instance_id)
    logger.info("    vastai ssh-url %s", instance_id)
    logger.info("")
    logger.info("Outputs will appear at:")
    logger.info("    %s/%s/", cfg.output.rclone_remote.rstrip("/"), cfg.run_id)
    logger.info("")
    logger.info(
        "auto_destroy=%s — instance %s self-terminate when training finishes.",
        cfg.auto_destroy, "will" if cfg.auto_destroy else "will NOT",
    )


if __name__ == "__main__":
    main()
