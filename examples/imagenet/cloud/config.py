"""
examples.imagenet.cloud.config
==============================

Typed loader for ``config/cloud.yaml``.

Parses the YAML into nested dataclasses, validates required fields, and surfaces
descriptive errors. Has only one runtime dependency (PyYAML) — the actual AWS
launcher imports boto3 lazily.

Usage
-----
::

    from examples.imagenet.cloud.config import CloudConfig
    cfg = CloudConfig.from_yaml(Path("config/cloud.yaml"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _require(d: Dict[str, Any], key: str, section: str) -> Any:
    """Fetch ``d[key]``; raise a precise error if missing or empty."""
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"cloud.yaml: missing required key '{section}.{key}'.")
    return d[key]


@dataclass
class AwsConfig:
    region: str
    profile: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AwsConfig":
        return cls(
            region=_require(d, "region", "aws"),
            profile=d.get("profile"),
        )


@dataclass
class ContainerConfig:
    image_uri: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContainerConfig":
        image_uri = _require(d, "image_uri", "container")
        if ".dkr.ecr." not in image_uri:
            # Not strictly required, but ECR is what the launcher's auth step assumes.
            raise ValueError(
                f"container.image_uri ('{image_uri}') does not look like an ECR URI."
            )
        return cls(image_uri=image_uri)


@dataclass
class Ec2Config:
    instance_type: str
    ami_id: str
    key_name: str
    subnet_id: str
    security_group_ids: List[str]
    iam_instance_profile: str
    root_volume_size_gb: int = 200
    terminate_on_complete: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Ec2Config":
        sgs = _require(d, "security_group_ids", "ec2")
        if not isinstance(sgs, list) or not sgs:
            raise ValueError("ec2.security_group_ids must be a non-empty list.")
        return cls(
            instance_type=_require(d, "instance_type", "ec2"),
            ami_id=_require(d, "ami_id", "ec2"),
            key_name=_require(d, "key_name", "ec2"),
            subnet_id=_require(d, "subnet_id", "ec2"),
            security_group_ids=list(sgs),
            iam_instance_profile=_require(d, "iam_instance_profile", "ec2"),
            root_volume_size_gb=int(d.get("root_volume_size_gb", 200)),
            terminate_on_complete=bool(d.get("terminate_on_complete", True)),
        )


@dataclass
class DataConfig:
    imagenet_s3: str
    cost_matrix_s3: str
    output_s3_prefix: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        for key in ("imagenet_s3", "cost_matrix_s3", "output_s3_prefix"):
            val = _require(d, key, "data")
            if not str(val).startswith("s3://"):
                raise ValueError(f"data.{key} must start with s3:// (got '{val}').")
        return cls(
            imagenet_s3=d["imagenet_s3"],
            cost_matrix_s3=d["cost_matrix_s3"],
            output_s3_prefix=d["output_s3_prefix"],
        )


@dataclass
class TrainingConfig:
    """
    Hyperparameters passed verbatim to ``examples.imagenet.train``.

    Anything not listed here can still be added to the YAML — unrecognized
    keys are forwarded as ``--<key-with-dashes>`` flags.
    """
    loss: str = "sinkhorn_envelope"
    arch: str = "resnet50"
    batch_size: int = 64
    epochs: int = 90
    lr: float = 0.1
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    sinkhorn_max_iter: int = 20
    epsilon_mode: str = "offdiag_mean"
    epsilon_scale: float = 1.0
    num_workers: int = 8
    resume: bool = False
    # Anything else — forwarded transparently.
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(extra=extra, **kwargs)


@dataclass
class CloudConfig:
    run_id: str
    aws: AwsConfig
    container: ContainerConfig
    ec2: Ec2Config
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CloudConfig":
        return cls(
            run_id=_require(d, "run_id", "(root)"),
            aws=AwsConfig.from_dict(_require(d, "aws", "(root)")),
            container=ContainerConfig.from_dict(_require(d, "container", "(root)")),
            ec2=Ec2Config.from_dict(_require(d, "ec2", "(root)")),
            data=DataConfig.from_dict(_require(d, "data", "(root)")),
            training=TrainingConfig.from_dict(d.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "CloudConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Top-level YAML in {path} must be a mapping.")
        return cls.from_dict(raw)
