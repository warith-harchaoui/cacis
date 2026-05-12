# syntax=docker/dockerfile:1.6
#
# cacis-imagenet — runtime image for Vast.ai (RTX 4090 24 GB) training of the
# ImageNet cost-aware pipeline. The bootstrap script at
# /workspace/examples/imagenet/cloud/vast_bootstrap.sh is the Vast.ai onstart
# command (see examples/imagenet/cloud/launch_vast.py).
#
# ---------------------------------------------------------------------------
# macOS prereq: you need a running Docker daemon. The `docker` CLI alone is
# not enough. Easiest option is Colima:
#
#   brew install colima docker-buildx
#   mkdir -p ~/.docker/cli-plugins && \
#     ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" \
#             ~/.docker/cli-plugins/docker-buildx
#   colima start --cpu 4 --memory 8 --disk 60
#
# (Docker Desktop or OrbStack work too. See config/README.md Step 5a.)
#
# ---------------------------------------------------------------------------
# Build (from the repository root):
#
#   docker build -t cacis-imagenet -f examples/imagenet/Dockerfile .
#
# ---------------------------------------------------------------------------
# Push to Docker Hub (public repo, free):
#
#   DOCKER_USER=<your-dockerhub-username>
#   docker login -u $DOCKER_USER
#   docker tag cacis-imagenet:latest $DOCKER_USER/cacis-imagenet:latest
#   docker push $DOCKER_USER/cacis-imagenet:latest
#
# Then put `docker.io/$DOCKER_USER/cacis-imagenet:latest` into
# `container.image` in your config/cloud-vast.yaml.
#
# ---------------------------------------------------------------------------
# Run locally (debug, GPU optional):
#
#   docker run --rm --gpus all --shm-size=8g \
#     -e KAGGLE_USERNAME=... -e KAGGLE_KEY=... \
#     -e B2_KEY_ID=... -e B2_APP_KEY=... \
#     -e RCLONE_DEST=b2:your-bucket/cacis -e COST_MATRIX_URL=https://... \
#     -e LOSSES="cross_entropy sinkhorn_envelope" -e RUN_PREFIX=local-test \
#     -e ARCH=resnet50 -e NUM_CLASSES=1000 -e BATCH_SIZE=32 -e EPOCHS=2 \
#     -e WARMUP_EPOCHS=1 -e LR=0.1 -e WEIGHT_DECAY=1e-4 \
#     -e LABEL_SMOOTHING=0.1 -e SINKHORN_MAX_ITER=20 \
#     -e EPSILON_MODE=offdiag_mean -e EPSILON_SCALE=1.0 -e NUM_WORKERS=4 \
#     -e AUTO_DESTROY=false \
#     cacis-imagenet \
#     bash /workspace/examples/imagenet/cloud/vast_bootstrap.sh
#
# ---------------------------------------------------------------------------

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# System deps:
#   - unzip, curl   : used by the Vast.ai bootstrap
#   - rclone        : sync outputs to Backblaze B2
#   - libgomp/libgl : torchvision codepaths + FastText if used at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip curl rclone libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps split so layers cache well
COPY requirements.txt /workspace/requirements.txt
COPY examples/imagenet/requirements.txt /workspace/requirements-imagenet.txt
RUN pip install --no-cache-dir \
        -r /workspace/requirements.txt \
        -r /workspace/requirements-imagenet.txt \
    && pip install --no-cache-dir kaggle

# Source
COPY cost_aware_losses /workspace/cost_aware_losses
COPY examples           /workspace/examples
COPY scripts            /workspace/scripts
COPY setup.py           /workspace/setup.py
RUN pip install --no-cache-dir -e /workspace

# Montserrat font for figures (downloaded at build time so the image is
# self-contained even when the developer didn't run scripts/download_fonts.sh
# locally). Falls back gracefully to default sans-serif if the download fails.
RUN bash /workspace/scripts/download_fonts.sh \
    || echo "WARN: font download failed — plots will use default sans-serif."

# Make the Vast.ai bootstrap directly executable
RUN chmod +x /workspace/examples/imagenet/cloud/vast_bootstrap.sh

ENV PYTHONUNBUFFERED=1
ENV NCCL_DEBUG=WARN

# No ENTRYPOINT — Vast.ai injects the onstart command at runtime.
