# syntax=docker/dockerfile:1.6
#
# ImageNet cost-aware training image.
# Targets multi-GPU CUDA boxes on AWS (p4d/p4de/p5) and GCP (a2-highgpu-*).
#
# ---------------------------------------------------------------------------
# Build (from the repository root):
#
#   docker build -t cacis-imagenet -f examples/imagenet/Dockerfile .
#
# ---------------------------------------------------------------------------
# AWS ECR push:
#
#   AWS_ACCT=<acct-id>
#   AWS_REGION=us-east-1
#   aws ecr get-login-password --region $AWS_REGION \
#     | docker login --username AWS --password-stdin \
#         $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com
#   docker tag cacis-imagenet:latest \
#     $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com/cacis-imagenet:latest
#   docker push \
#     $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com/cacis-imagenet:latest
#
# ---------------------------------------------------------------------------
# GCP Artifact Registry push:
#
#   REGION=us-central1
#   PROJECT=<project-id>
#   REPO=ml
#   gcloud auth configure-docker $REGION-docker.pkg.dev
#   docker tag cacis-imagenet:latest \
#     $REGION-docker.pkg.dev/$PROJECT/$REPO/cacis-imagenet:latest
#   docker push \
#     $REGION-docker.pkg.dev/$PROJECT/$REPO/cacis-imagenet:latest
#
# ---------------------------------------------------------------------------
# Run on a single multi-GPU node (8 × A100 example):
#
#   docker run --gpus all --shm-size=8g --rm \
#     -v /mnt/imagenet:/data/imagenet:ro \
#     -v /mnt/output:/workspace/imagenet_output \
#     -v /mnt/models:/models:ro \
#     cacis-imagenet \
#     torchrun --standalone --nproc-per-node=8 \
#       -m examples.imagenet.train \
#         --data-root /data/imagenet \
#         --cost-matrix /models/cost_matrix.pt \
#         --loss sinkhorn_envelope \
#         --batch-size 64 --epochs 90 \
#         --output-dir /workspace/imagenet_output \
#         --run-id resnet50_fasttext
#
# Multi-node usage is in the docstring of examples/imagenet/train.py.
# ---------------------------------------------------------------------------

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# Minimal system deps: libgomp for FastText, libgl1 for some torchvision codepaths.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies. We split base + imagenet-specific so layers can cache.
COPY requirements.txt /workspace/requirements.txt
COPY examples/imagenet/requirements.txt /workspace/requirements-imagenet.txt
RUN pip install --no-cache-dir \
        -r /workspace/requirements.txt \
        -r /workspace/requirements-imagenet.txt

# Source. Mounting at runtime would also work, but baking in keeps the image
# self-contained and the launch command simple.
COPY cost_aware_losses /workspace/cost_aware_losses
COPY examples /workspace/examples
COPY setup.py /workspace/setup.py
RUN pip install --no-cache-dir -e /workspace

ENV PYTHONUNBUFFERED=1
ENV NCCL_DEBUG=WARN

# No ENTRYPOINT — pass the full torchrun command at run-time (see header).
