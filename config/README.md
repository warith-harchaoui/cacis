# Cloud configs

This folder holds the YAML configs consumed by `examples.imagenet.cloud.launch_ec2`. Each config provisions a single multi-GPU EC2 instance that pulls the training container, stages ImageNet from S3, trains, syncs results back, and self-terminates.

## Files

| File | Tracked? | Purpose |
|------|----------|---------|
| `cloud.yaml.example` | yes (tracked) | Annotated template — copy into one of the per-loss files below if you start from scratch. |
| `README.md` | yes (tracked) | This document. |
| `cloud-cross_entropy.yaml` | **gitignored** | Baseline run with `nn.CrossEntropyLoss`. |
| `cloud-sinkhorn_envelope.yaml` | **gitignored** | Recommended cost-aware run. Default for the paper's ImageNet experiments. |
| `cloud-sinkhorn_autodiff.yaml` | **gitignored** | Ablation: full-autodiff Sinkhorn. Memory-sensitive. |
| `cloud-sinkhorn_pot.yaml` | **gitignored** | Ablation: POT-backed Sinkhorn. Slow at K=1000. |
| `cloud.yaml` | **gitignored** | Optional generic config (if you don't want per-loss files). |

`.gitignore` rules track only `cloud.yaml.example` and `README.md`; everything else is your local secrets.

## Step 1 — Provision AWS resources (once)

You need each of the following ahead of time. The launcher does **not** create them for you (and intentionally so — they outlive any single run).

- **ECR repository** holding the training image. Build and push from the repo root:
  ```bash
  docker build -t cacis-imagenet -f examples/imagenet/Dockerfile .
  AWS_ACCT=<your-acct-id>; AWS_REGION=us-east-1
  aws ecr get-login-password --region $AWS_REGION \
    | docker login --username AWS --password-stdin \
        $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com
  docker tag cacis-imagenet:latest \
    $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com/cacis-imagenet:latest
  docker push \
    $AWS_ACCT.dkr.ecr.$AWS_REGION.amazonaws.com/cacis-imagenet:latest
  ```
  Put the resulting image URI into `container.image_uri`.

- **S3 bucket** with ImageNet under `train/`, `val/`, and `LOC_synset_mapping.txt`:
  ```bash
  aws s3 sync /local/imagenet/ s3://my-bucket/datasets/imagenet/
  ```

- **Precomputed cost matrix** uploaded to S3:
  ```bash
  python -m examples.imagenet.cost_matrix \
      --data-root /local/imagenet \
      --fasttext /local/cc.en.300.bin \
      --out cost_matrix.pt
  aws s3 cp cost_matrix.pt s3://my-bucket/cacis/cost_matrix.pt
  ```

- **EC2 networking** — a key pair (for SSH), a VPC subnet that supports your GPU instance family, and a security group that allows egress to S3 / ECR and (optionally) SSH from your IP.

- **IAM instance profile** attached to the instance. Minimum policy:
  - ECR pull: `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, `ecr:BatchCheckLayerAvailability`
  - S3 r/w on the data and output buckets used here
  - `ec2:TerminateInstances` scoped to itself (only when `terminate_on_complete: true`)
  - `AmazonSSMManagedInstanceCore` (optional, recommended — lets you attach with `aws ssm start-session`)

- **IAM permissions for the user launching the script:** `ec2:RunInstances`, `ec2:CreateTags`, `ec2:DescribeInstances`, and `iam:PassRole` on the instance profile.

## Step 2 — Fill in a per-loss config

Each `cloud-<loss>.yaml` was created from `cloud.yaml.example` with placeholder values. Open one and replace every `XXXX` or `123456789012` with your real values:

| Field | Where it comes from |
|-------|---------------------|
| `container.image_uri` | ECR URI of `cacis-imagenet:latest` |
| `ec2.ami_id` | A GPU Deep Learning AMI in your region. List them: |
| | `aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch 2.* (Ubuntu 22.04)*"` |
| `ec2.key_name` | An existing EC2 key pair |
| `ec2.subnet_id` | A VPC subnet that supports `instance_type` |
| `ec2.security_group_ids` | One or more SG ids permitting S3/ECR egress |
| `ec2.iam_instance_profile` | The **name** (not ARN) of the instance profile from Step 1 |
| `data.*` | S3 paths from Step 1 |

The `training` block already differs across the per-loss files in batch size and iteration count — those defaults reflect what fits on a single p4d.24xlarge with AMP. Tweak `epochs`, `lr`, etc. as needed.

## Step 3 — Dry-run, then launch

Always start with `--dry-run`. It renders the bash bootstrap that would be sent to the instance, without making any AWS calls:

```bash
python -m examples.imagenet.cloud.launch_ec2 \
    --config config/cloud-sinkhorn_envelope.yaml \
    --dry-run
```

Read the output — make sure the S3 paths, image URI, and training flags look right. Then launch:

```bash
python -m examples.imagenet.cloud.launch_ec2 \
    --config config/cloud-sinkhorn_envelope.yaml
```

The script prints the instance id and an `aws ssm start-session` command for attaching. Once attached, follow progress with:

```bash
tail -f /var/log/cacis-bootstrap.log
```

The same log is also uploaded to S3 alongside the model outputs when training completes (or fails).

## Step 4 — Find your outputs

Outputs land at `<output_s3_prefix>/<run_id>/`. Each run produces:

- `checkpoint_last.pt` and `checkpoint_best.pt`
- `metrics.jsonl` — one row per epoch with train + val metrics
- `args.json` — exact CLI args used
- `cacis-bootstrap.log` — the instance's bootstrap log

To compare losses, download the `metrics.jsonl` files from each run and plot or harvest them locally.

## Running all four in sequence

```bash
for cfg in config/cloud-cross_entropy.yaml \
           config/cloud-sinkhorn_envelope.yaml \
           config/cloud-sinkhorn_autodiff.yaml \
           config/cloud-sinkhorn_pot.yaml; do
    python -m examples.imagenet.cloud.launch_ec2 --config "$cfg"
done
```

Each launch returns immediately (`run_instances` is asynchronous); the four instances run independently and self-terminate when done.

## Cost note

A p4d.24xlarge in `us-east-1` runs around US\$32/hour on-demand. A full 90-epoch ResNet-50 ImageNet pass takes roughly 12–24h depending on the loss, so budget US\$400–800 per run. Spot pricing typically cuts this 60–70% if you can tolerate interruption (set `terminate_on_complete: true` regardless — spot reclamation also triggers shutdown). For development, switch to `g5.48xlarge` (8× A10G, ≈US\$16/h, ≈2× slower).
