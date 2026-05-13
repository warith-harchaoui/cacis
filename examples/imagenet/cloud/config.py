"""
examples.imagenet.cloud.config
==============================

Typed loader for ``config/cloud-vast.yaml`` — the Vast.ai launcher's config.

We pin to Vast.ai with RTX 4090 24 GB. Secrets are passed as environment
variable **names** in the YAML, never as values, so the file can live on disk
without leaking credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _require(d: Dict[str, Any], key: str, section: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"cloud-vast.yaml: missing required key '{section}.{key}'.")
    return d[key]


@dataclass
class CredentialsConfig:
    """
    Environment-variable **names** holding the actual secret values, OR literal
    values (the launcher accepts both; ALL_CAPS_WITH_UNDERSCORES is treated as
    an env-var lookup, anything else as a literal).

    Use ``--dry-run`` to render the bootstrap with no Vast.ai calls if you
    don't yet have the env vars set.
    """
    vastai_api_key_env: str = "VASTAI_API_KEY"
    kaggle_username_env: str = "KAGGLE_USERNAME"
    kaggle_key_env: str = "KAGGLE_KEY"
    b2_key_id_env: str = "B2_KEY_ID"
    b2_app_key_env: str = "B2_APP_KEY"
    cost_matrix_url_env: str = "COST_MATRIX_URL"
    # Optional fast mirror for the ImageNet zip. If set, the bootstrap will
    # curl from this URL instead of going through Kaggle's throttled CDN.
    # Empty value → fall back to Kaggle.
    imagenet_zip_url_env: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CredentialsConfig":
        return cls(
            vastai_api_key_env=d.get("vastai_api_key_env", cls.vastai_api_key_env),
            kaggle_username_env=d.get("kaggle_username_env", cls.kaggle_username_env),
            kaggle_key_env=d.get("kaggle_key_env", cls.kaggle_key_env),
            b2_key_id_env=d.get("b2_key_id_env", cls.b2_key_id_env),
            b2_app_key_env=d.get("b2_app_key_env", cls.b2_app_key_env),
            cost_matrix_url_env=d.get("cost_matrix_url_env", cls.cost_matrix_url_env),
            imagenet_zip_url_env=d.get("imagenet_zip_url_env", cls.imagenet_zip_url_env),
        )


@dataclass
class ContainerConfig:
    image: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContainerConfig":
        return cls(image=_require(d, "image", "container"))


@dataclass
class InstanceConfig:
    """Vast.ai offer criteria."""
    gpu_name: str = "RTX_4090"
    num_gpus: int = 1
    min_disk_gb: int = 250
    max_price_per_hour: float = 0.50
    min_reliability: float = 0.99   # raised from 0.95 after seeing CDI / driver failures
    cuda_max_good: float = 12.4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InstanceConfig":
        return cls(
            gpu_name=d.get("gpu_name", cls.gpu_name),
            num_gpus=int(d.get("num_gpus", cls.num_gpus)),
            min_disk_gb=int(d.get("min_disk_gb", cls.min_disk_gb)),
            max_price_per_hour=float(d.get("max_price_per_hour", cls.max_price_per_hour)),
            min_reliability=float(d.get("min_reliability", cls.min_reliability)),
            cuda_max_good=float(d.get("cuda_max_good", cls.cuda_max_good)),
        )


@dataclass
class OutputConfig:
    """rclone destination, e.g. ``b2:my-bucket/cacis-imagenet``."""
    rclone_remote: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        remote = _require(d, "rclone_remote", "output")
        if ":" not in remote:
            raise ValueError(
                f"output.rclone_remote must look like '<remote>:<bucket>/path' (got '{remote}')."
            )
        return cls(rclone_remote=remote)


@dataclass
class TrainingConfig:
    """Hyperparameters shared by every loss run."""
    arch: str = "resnet50"
    num_classes: int = 1000
    batch_size: int = 32
    epochs: int = 45
    warmup_epochs: int = 3
    lr: float = 0.1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    sinkhorn_max_iter: int = 20
    epsilon_mode: str = "offdiag_mean"
    epsilon_scale: float = 1.0
    num_workers: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        defaults = cls()
        kwargs = {}
        for f in defaults.__dataclass_fields__:
            if f in d:
                kwargs[f] = d[f]
        return cls(**kwargs)


_VALID_LOSSES = {
    "cross_entropy",
    "sinkhorn_envelope",
    "sinkhorn_autodiff",
    "sinkhorn_pot",
}


@dataclass
class VastConfig:
    run_id: str
    credentials: CredentialsConfig
    container: ContainerConfig
    instance: InstanceConfig
    output: OutputConfig
    losses: List[str]
    training: TrainingConfig
    auto_destroy: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VastConfig":
        losses = _require(d, "losses", "(root)")
        if not isinstance(losses, list) or not losses:
            raise ValueError("'losses' must be a non-empty list.")
        unknown = set(losses) - _VALID_LOSSES
        if unknown:
            raise ValueError(
                f"Unknown losses: {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_LOSSES)}."
            )

        return cls(
            run_id=_require(d, "run_id", "(root)"),
            credentials=CredentialsConfig.from_dict(d.get("credentials", {})),
            container=ContainerConfig.from_dict(_require(d, "container", "(root)")),
            instance=InstanceConfig.from_dict(d.get("instance", {})),
            output=OutputConfig.from_dict(_require(d, "output", "(root)")),
            losses=list(losses),
            training=TrainingConfig.from_dict(d.get("training", {})),
            auto_destroy=bool(d.get("auto_destroy", True)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "VastConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Top-level YAML in {path} must be a mapping.")
        return cls.from_dict(raw)
