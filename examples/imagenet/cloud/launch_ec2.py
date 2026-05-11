"""
examples.imagenet.cloud.launch_ec2
==================================

Launch an ImageNet cost-aware training run on a single multi-GPU EC2 instance.

What this script does
---------------------
1. Loads ``config/cloud.yaml`` via :class:`CloudConfig`.
2. Renders a self-contained bash *user-data* script that:

   - Logs in to ECR and pulls the training container image.
   - Stages ImageNet from S3 to the instance NVMe SSD.
   - Stages the precomputed cost matrix.
   - Runs ``torchrun -m examples.imagenet.train`` inside Docker with the right
     mounts.
   - Syncs ``imagenet_output/`` back to ``s3://.../output_s3_prefix/<run_id>/``.
   - Self-terminates (if configured).

3. Calls ``ec2:RunInstances`` with the user-data attached.

The script returns the instance id and a one-line ``aws ssm start-session``
command to attach to the running instance.

Usage
-----
::

    python -m examples.imagenet.cloud.launch_ec2 \\
        --config config/cloud.yaml

    # Render and print the user-data without launching anything:
    python -m examples.imagenet.cloud.launch_ec2 \\
        --config config/cloud.yaml --dry-run

Required IAM (for the launcher user)
-----------------------------------
- ``ec2:RunInstances``, ``ec2:CreateTags``, ``ec2:DescribeInstances``
- ``iam:PassRole`` on the instance profile referenced in ``ec2.iam_instance_profile``

Required IAM (for the instance profile)
---------------------------------------
- ECR: ``ecr:GetAuthorizationToken``, ``ecr:BatchGetImage``,
       ``ecr:GetDownloadUrlForLayer``, ``ecr:BatchCheckLayerAvailability``
- S3 (data + output buckets): ``s3:GetObject``, ``s3:PutObject``,
       ``s3:ListBucket``
- Self-terminate: ``ec2:TerminateInstances`` scoped to the instance's ARN
- SSM (optional, recommended for attach): ``AmazonSSMManagedInstanceCore``

Author
------
Warith Harchaoui <wharchaoui@nexton-group.com>
"""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path
from string import Template
from typing import Any, Dict, List

from examples.imagenet.cloud.config import CloudConfig, TrainingConfig

logger = logging.getLogger(__name__)


# =============================================================================
# user-data script
# =============================================================================
#
# Python substitutes ``@@VAR@@`` tokens (custom Template delimiter so that
# ``$VAR`` bash variables don't conflict). Bash is responsible for its own
# expansion at instance boot time.
#
# The script writes everything to /var/log/cacis-bootstrap.log; attach with
# ``aws ssm start-session --target <iid>`` and ``tail -f /var/log/cacis-bootstrap.log``.

_USER_DATA = r"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/cacis-bootstrap.log) 2>&1

echo "=== CACIS bootstrap starting at $(date -u) ==="

REGION="@@REGION@@"
IMAGE_URI="@@IMAGE_URI@@"
IMAGENET_S3="@@IMAGENET_S3@@"
COST_MATRIX_S3="@@COST_MATRIX_S3@@"
OUTPUT_S3_PREFIX="@@OUTPUT_S3_PREFIX@@"
RUN_ID="@@RUN_ID@@"
TERMINATE_ON_COMPLETE="@@TERMINATE_ON_COMPLETE@@"
TRAIN_ARGS=(@@TRAIN_ARGS@@)

# --- 1. Bring up docker --------------------------------------------------
systemctl is-active --quiet docker || systemctl start docker
docker --version

# --- 2. Mount instance NVMe (if any) -------------------------------------
# Deep Learning AMIs expose instance store as /dev/nvme1n1 (or similar) on
# p4d / g5 / p5. We use it as scratch because EBS is too slow for 1.28M images.
DATA_ROOT=/data
mkdir -p "$DATA_ROOT"
NVME_DEV=$(lsblk -dno NAME,SIZE,MOUNTPOINT | awk '$3=="" && $1 ~ /^nvme[1-9]/ {print "/dev/"$1; exit}')
if [[ -n "${NVME_DEV:-}" ]]; then
    echo "Mounting instance NVMe $NVME_DEV at $DATA_ROOT"
    mkfs.ext4 -F -q "$NVME_DEV"
    mount "$NVME_DEV" "$DATA_ROOT"
else
    echo "No instance NVMe detected; using EBS root volume for scratch (slower)."
fi
mkdir -p "$DATA_ROOT/imagenet" "$DATA_ROOT/models" "$DATA_ROOT/output"

# --- 3. ECR login + image pull ------------------------------------------
ECR_REGISTRY="${IMAGE_URI%%/*}"
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$IMAGE_URI"

# --- 4. Stage data from S3 ----------------------------------------------
echo "Syncing ImageNet from $IMAGENET_S3 ..."
aws s3 sync --only-show-errors "$IMAGENET_S3" "$DATA_ROOT/imagenet/"

echo "Fetching cost matrix from $COST_MATRIX_S3 ..."
aws s3 cp --only-show-errors "$COST_MATRIX_S3" "$DATA_ROOT/models/cost_matrix.pt"

# --- 5. Run training -----------------------------------------------------
NPROC=$(nvidia-smi -L | wc -l)
echo "Detected $NPROC GPU(s)."

set +e
docker run --gpus all --shm-size=8g --rm \
    -v "$DATA_ROOT/imagenet:/data/imagenet:ro" \
    -v "$DATA_ROOT/models:/models:ro" \
    -v "$DATA_ROOT/output:/workspace/imagenet_output" \
    "$IMAGE_URI" \
    torchrun --standalone --nproc-per-node="$NPROC" \
        -m examples.imagenet.train \
        --data-root /data/imagenet \
        --cost-matrix /models/cost_matrix.pt \
        --output-dir /workspace/imagenet_output \
        --run-id "$RUN_ID" \
        "${TRAIN_ARGS[@]}"
TRAIN_RC=$?
set -e

echo "=== Training exit code: $TRAIN_RC ==="

# --- 6. Upload outputs back to S3 ---------------------------------------
DEST="${OUTPUT_S3_PREFIX%/}/${RUN_ID}/"
echo "Uploading outputs to $DEST ..."
aws s3 sync --only-show-errors "$DATA_ROOT/output/" "$DEST"

# Always upload the bootstrap log too.
aws s3 cp --only-show-errors /var/log/cacis-bootstrap.log "${DEST}cacis-bootstrap.log" || true

# --- 7. Self-terminate (optional) ---------------------------------------
if [[ "$TERMINATE_ON_COMPLETE" == "true" ]]; then
    INSTANCE_ID=$(curl -fsS http://169.254.169.254/latest/meta-data/instance-id)
    echo "Self-terminating $INSTANCE_ID"
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" || true
fi

echo "=== CACIS bootstrap finished at $(date -u) ==="
exit $TRAIN_RC
"""


class _AtTemplate(Template):
    """``Template`` subclass using ``@@VAR@@`` delimiters to avoid clash with bash ``$``."""
    delimiter = "@@"
    pattern = r"""
    @@(?:
        (?P<escaped>@@)            |    # @@@@ → literal @@
        (?P<named>[A-Za-z][A-Za-z0-9_]*)@@ |
        (?P<braced>[A-Za-z][A-Za-z0-9_]*)@@ |
        (?P<invalid>)
    )
    """


# =============================================================================
# CLI flag construction
# =============================================================================

def _flag_value(value: Any) -> str:
    """Render a YAML scalar as a shell-quoted CLI value."""
    if isinstance(value, float):
        return shlex.quote(repr(value))
    return shlex.quote(str(value))


def build_train_args(t: TrainingConfig) -> List[str]:
    """
    Build the list of ``--flag value`` tokens for ``examples.imagenet.train``.

    Boolean fields become bare flags (only emitted when True). Unknown fields
    in ``training.extra`` are passed through transparently as ``--key value``.
    """
    parts: List[str] = []
    skip_known = {"resume", "extra"}
    for key in (
        "loss", "arch", "batch_size", "epochs", "lr",
        "warmup_epochs", "weight_decay", "label_smoothing",
        "sinkhorn_max_iter", "epsilon_mode", "epsilon_scale",
        "num_workers",
    ):
        parts += [f"--{key.replace('_', '-')}", _flag_value(getattr(t, key))]
    if t.resume:
        parts.append("--resume")

    for k, v in t.extra.items():
        if k in skip_known:
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts += [flag, _flag_value(v)]
    return parts


# =============================================================================
# User-data rendering
# =============================================================================

def render_user_data(cfg: CloudConfig) -> str:
    """Render the bootstrap bash script for ``cfg``."""
    train_args = " ".join(build_train_args(cfg.training))
    return _AtTemplate(_USER_DATA).substitute(
        REGION=cfg.aws.region,
        IMAGE_URI=cfg.container.image_uri,
        IMAGENET_S3=cfg.data.imagenet_s3,
        COST_MATRIX_S3=cfg.data.cost_matrix_s3,
        OUTPUT_S3_PREFIX=cfg.data.output_s3_prefix,
        RUN_ID=cfg.run_id,
        TERMINATE_ON_COMPLETE=str(cfg.ec2.terminate_on_complete).lower(),
        TRAIN_ARGS=train_args,
    )


# =============================================================================
# Launcher
# =============================================================================

def launch(cfg: CloudConfig) -> str:
    """
    Submit ``ec2:RunInstances`` for the given config and return the instance id.

    Imports ``boto3`` lazily so that ``--dry-run`` users don't need it installed.
    """
    import boto3  # type: ignore[import-not-found]

    session = boto3.Session(region_name=cfg.aws.region, profile_name=cfg.aws.profile)
    ec2 = session.client("ec2")

    user_data = render_user_data(cfg)
    if len(user_data.encode("utf-8")) > 15_000:
        # EC2 limit is 16 KB pre-base64. Warn loudly.
        logger.warning("user-data is %d bytes — close to the 16KB EC2 limit.", len(user_data))

    shutdown_behavior = "terminate" if cfg.ec2.terminate_on_complete else "stop"

    params: Dict[str, Any] = dict(
        ImageId=cfg.ec2.ami_id,
        InstanceType=cfg.ec2.instance_type,
        KeyName=cfg.ec2.key_name,
        SubnetId=cfg.ec2.subnet_id,
        SecurityGroupIds=cfg.ec2.security_group_ids,
        IamInstanceProfile={"Name": cfg.ec2.iam_instance_profile},
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": cfg.ec2.root_volume_size_gb,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                },
            },
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"cacis-{cfg.run_id}"},
                    {"Key": "Project", "Value": "cacis-imagenet"},
                    {"Key": "RunId", "Value": cfg.run_id},
                ],
            }
        ],
        MinCount=1,
        MaxCount=1,
        InstanceInitiatedShutdownBehavior=shutdown_behavior,
    )

    response = ec2.run_instances(**params)
    instance_id = response["Instances"][0]["InstanceId"]
    return instance_id


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ImageNet training on AWS EC2.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to cloud.yaml (start from config/cloud.yaml.example).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render the user-data script and print it; do not call AWS.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = CloudConfig.from_yaml(args.config)

    if args.dry_run:
        print(render_user_data(cfg))
        return

    instance_id = launch(cfg)

    logger.info("Launched instance: %s", instance_id)
    logger.info("Tag: Name=cacis-%s  RunId=%s", cfg.run_id, cfg.run_id)
    logger.info("Attach (requires SSM agent + permissions on the instance role):")
    logger.info("    aws ssm start-session --region %s --target %s", cfg.aws.region, instance_id)
    logger.info("Then: tail -f /var/log/cacis-bootstrap.log")
    logger.info("Outputs will be uploaded to: %s%s/",
                cfg.data.output_s3_prefix.rstrip("/") + "/", cfg.run_id)


if __name__ == "__main__":
    main()
