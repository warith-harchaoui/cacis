# Vast.ai cloud setup

This folder holds the YAML config consumed by `examples.imagenet.cloud.launch_vast`. **One Vast.ai instance, one RTX 4090 24 GB, four losses sequential.** Budget estimate at default settings (45 epochs, batch 32): ~$30–50 per loss × 4 ≈ **$120–200 total**.

## Files

| File | Tracked? | Purpose |
|------|----------|---------|
| `cloud-vast.yaml.example` | yes | Annotated template — copy to `cloud-vast.yaml`. |
| `README.md` | yes | This guide. |
| `cloud-vast.yaml` | **gitignored** | Your filled-in config (do not commit). |

---

## Step 0 — Local smoke test (do this first, every time)

Before paying for any GPU time, prove the pipeline works locally:

```bash
# 1. Fetch the figure font (one-time)
bash scripts/download_fonts.sh

# 2. Full pipeline end-to-end on real Imagenette images, on your CPU/MPS
python -m examples.imagenet.smoke_test
# all four losses
python -m examples.imagenet.smoke_test --loss all
```

Only proceed to Step 1 once this prints `✓ Smoke test passed`. If it fails on your laptop, it will fail on Vast.ai — fix locally first.

---

## Step 1 — Create a Vast.ai account

1. Go to <https://vast.ai/console/create/>, create an account, verify your email and SSH key.
2. Top up credit. **Add the budget you're willing to lose**, e.g. $200. Vast.ai does not auto-bill beyond what you deposit. Set a low ceiling first, you can always add more.
3. Install the CLI and store your API key locally:
   ```bash
   pip install vastai
   vastai set api-key <YOUR-API-KEY-FROM-https://cloud.vast.ai/account/>
   vastai show user           # sanity check — should print your account info
   ```

---

## Step 2 — Create a Kaggle account (for ImageNet)

The bootstrap downloads ImageNet from Kaggle's `imagenet-object-localization-challenge` competition.

1. Sign up at <https://www.kaggle.com/>.
2. Go to <https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules> and click **I Understand and Accept** (required once).
3. Generate an API token: <https://www.kaggle.com/settings> → **Create New Token** → downloads `kaggle.json`.
4. Open `kaggle.json` and note `username` and `key`. Export them:
   ```bash
   export KAGGLE_USERNAME=<username from kaggle.json>
   export KAGGLE_KEY=<key from kaggle.json>
   ```

---

## Step 3 — Create a Backblaze B2 bucket (for outputs)

10 GB free, no credit card required for the free tier.

1. Sign up at <https://www.backblaze.com/cloud-storage>.
2. Create a private bucket, e.g. `cacis-imagenet`.
3. Create an application key: **Account → App Keys → Add a New Application Key**. Pick the bucket from step 2, grant `readWrite`, save the `keyID` and `applicationKey`.
4. Export them:
   ```bash
   export B2_KEY_ID=<keyID>
   export B2_APP_KEY=<applicationKey>
   ```
5. (Optional) Test rclone locally:
   ```bash
   brew install rclone   # or apt install / scoop / …
   rclone config         # add a remote called `b2`, type Backblaze B2,
                         # paste the keyID + applicationKey
   rclone lsd b2:cacis-imagenet
   ```

---

## Step 4 — Create a Docker Hub account (to host the image)

1. Sign up at <https://hub.docker.com/signup>.
2. Create a public repository, name it `cacis-imagenet`. (Public is free; private has free quotas.)
3. Log in locally:
   ```bash
   docker login -u <your-dockerhub-username>
   ```

---

## Step 5 — Build and push the image

### 5a. Make sure a Docker daemon is running

On macOS the `docker` CLI alone is not enough — you also need a daemon. Three options:

| Option | Install | Trade-off |
|---|---|---|
| **Colima** (recommended) | `brew install colima docker-buildx` | Free, CLI-only, lightweight, native on Apple Silicon. |
| Docker Desktop | `brew install --cask docker` then open Docker.app | GUI app, heavier resource use; free for personal/small business. |
| OrbStack | `brew install --cask orbstack` | Very fast, but commercial license needed above $1M revenue. |

The rest of this guide assumes **Colima**.

```bash
# One-time install (skip lines that are already done)
brew install colima docker-buildx

# Register buildx as a CLI plugin (silences the "legacy builder" deprecation)
mkdir -p ~/.docker/cli-plugins
ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" \
        ~/.docker/cli-plugins/docker-buildx

# Start the VM with enough RAM/disk for a ~2 GB image build with a 4 GB CUDA base
colima start --cpu 4 --memory 8 --disk 60

# Verify
docker info | head -4
docker buildx version
```

When you're done with cloud work, `colima stop` frees the VM's RAM. Re-launch with `colima start` next time (your VM and pulled images persist).

### 5b. Build and push

From the repository root:

```bash
DOCKER_USER=warithharchaoui
docker build -t cacis-imagenet -f examples/imagenet/Dockerfile .
docker tag cacis-imagenet:latest $DOCKER_USER/cacis-imagenet:latest
docker push $DOCKER_USER/cacis-imagenet:latest
```

First build downloads the ~4 GB PyTorch+CUDA base image (5–15 min depending on bandwidth) and the final image is ~5 GB. Subsequent builds are much faster (layer cache). Put `docker.io/$DOCKER_USER/cacis-imagenet:latest` into `container.image` in `cloud-vast.yaml` (Step 7).

### Troubleshooting

- **`failed to connect to the docker API at unix:///var/run/docker.sock`** — daemon isn't running. `colima start` (or open Docker Desktop).
- **`DEPRECATED: The legacy builder…`** — fix by registering the buildx plugin (the `ln -sfn` line above), then re-run.
- **Stale daemon after macOS sleep** — `colima restart`.

---

## Step 6 — Precompute the cost matrix and upload it

Run **locally** (small, fast, free):

```bash
# Download a FastText English model once (~7 GB compressed; the .bin variant
# supports subword OOV fallback — recommended).
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

# Produce cost_matrix.pt (~4 MB)
python -m examples.imagenet.cost_matrix \
    --data-root <path-with-LOC_synset_mapping.txt> \
    --fasttext cc.en.300.bin \
    --out cost_matrix.pt
```

You only need `LOC_synset_mapping.txt` locally (it ships with the Kaggle ImageNet download). If you don't have it yet, grab the file from <https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data> (the file alone is tiny).

Upload `cost_matrix.pt` somewhere publicly readable and export the URL:

```bash
# Easiest: drop it in the same B2 bucket and make it public
rclone copy cost_matrix.pt b2:cacis-imagenet/
# Then in the B2 web UI, mark cost_matrix.pt as public, copy its URL.
export COST_MATRIX_URL=https://f000.backblazeb2.com/file/cacis-imagenet/cost_matrix.pt
```

---

## Step 7 — Fill in `cloud-vast.yaml`

```bash
cp config/cloud-vast.yaml.example config/cloud-vast.yaml
```

Edit `config/cloud-vast.yaml` and replace:

| Field | Value |
|-------|-------|
| `run_id` | A short tag, e.g. `cacis-001` (becomes the B2 path prefix). |
| `container.image` | `docker.io/<your-dockerhub-username>/cacis-imagenet:latest` |
| `output.rclone_remote` | `b2:cacis-imagenet/runs` (or wherever you want in B2) |
| `instance.max_price_per_hour` | Bid ceiling. Start at `0.50`, raise if no offers found. |
| `losses` | Keep all four for the full benchmark; trim for ablations. |
| `training.epochs` | `45` (default) fits the budget. `90` if you want the full recipe and have ~$400. |

Leave the `credentials.*_env` fields alone unless you want different env-var names — the defaults match what you exported in Steps 1–3 and 6.

---

## Step 8 — Sanity-check with `--dry-run`

```bash
# Make sure ALL required env vars are exported in the current shell:
echo "${KAGGLE_USERNAME:?}" "${KAGGLE_KEY:?}" "${B2_KEY_ID:?}" \
     "${B2_APP_KEY:?}" "${VASTAI_API_KEY:?}" "${COST_MATRIX_URL:?}" > /dev/null

# Render the env + onstart cmd that would be sent (no Vast.ai calls yet):
python -m examples.imagenet.cloud.launch_vast \
    --config config/cloud-vast.yaml --dry-run
```

Read the printed env carefully. Secrets are redacted; the rest should match your intent.

If `VASTAI_API_KEY` is missing, set it (the `vastai` CLI stores its own copy, but the launcher reads from the shell):

```bash
export VASTAI_API_KEY=$(grep '"api_key"' ~/.config/vastai/vast_api_key 2>/dev/null \
    || cat ~/.vastai_api_key 2>/dev/null || echo "")
# Or just paste it from https://cloud.vast.ai/account/:
export VASTAI_API_KEY=<your-key>
```

---

## Step 9 — Launch

```bash
python -m examples.imagenet.cloud.launch_vast --config config/cloud-vast.yaml
```

The launcher prints the instance id and a tail-log command:

```
✓ Launched instance id = 12345678
Monitor with:
    vastai logs 12345678 --tail 200 --follow
    vastai ssh-url 12345678
Outputs will appear at:
    b2:cacis-imagenet/runs/cacis-001/
auto_destroy=true — instance will self-terminate when training finishes.
```

Follow the logs. The bootstrap will:

1. Download ImageNet from Kaggle (~30–60 min).
2. Reorganize `val/` into synset subfolders.
3. Download `cost_matrix.pt` from `$COST_MATRIX_URL`.
4. Train each loss sequentially; after each finishes, rclone-sync that loss's outputs to B2.
5. Self-destroy the instance.

---

## Step 10 — Retrieve and inspect results

```bash
mkdir -p ~/cacis-runs && cd ~/cacis-runs
rclone copy b2:cacis-imagenet/runs/cacis-001 ./cacis-001 -v
ls cacis-001/cacis-001-sinkhorn_envelope/
#   args.json
#   checkpoint_best.pt
#   checkpoint_last.pt
#   curve_accuracy.png      ← ↑ higher is better
#   curve_loss.png          ← ↓ lower is better
#   curve_regret.png        ← ↓ lower is better
#   metrics.csv
#   metrics.jsonl
```

Regenerate the curves locally (e.g. after tweaking the plotter):

```bash
python -m examples.imagenet.plots --run-dir cacis-001/cacis-001-sinkhorn_envelope
```

Run inference against the trained model:

```bash
python -m examples.imagenet.inference \
    --checkpoint cacis-001/cacis-001-sinkhorn_envelope/checkpoint_best.pt \
    --cost-matrix /path/to/cost_matrix.pt \
    --val-dir /path/to/imagenet/val
```

---

## Troubleshooting

**`No Vast.ai offers match your criteria`** — `instance.max_price_per_hour` is too low or `min_disk_gb` is too high. Try `0.60` and `200`.

**`vastai: command not found`** — install with `pip install vastai`. The launcher shells out to it.

**`KAGGLE_USERNAME is not set`** — exported only in another shell. Run `env | grep KAGGLE_` in the *exact* shell from which you'll launch.

**Instance launches but bootstrap fails** — `vastai logs <id> --follow` shows the bootstrap log. The same log is also uploaded to B2 at `runs/<run_id>/cacis-bootstrap.log` on completion.

**Training is much slower than expected** — RTX 4090 is single-GPU. At 45 epochs the default fits the budget but takes ~3–4 days per loss × 4 = ~2 weeks sequential. To go faster, launch four parallel instances (run this whole flow four times, each with `losses: [<single>]`).

**Cost discipline** — Vast.ai shows the live `$/hr` per instance in the dashboard. If a host is misbehaving, destroy it manually:
```bash
vastai destroy instance <id>
```
