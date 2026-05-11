# 🚀 Cost-Aware Classification + Fraud Benchmark (IEEE-CIS)

**Author:** Warith Harchaoui <wharchaoui@nexton-group.com>

Corporate research/engineering repository for **cost-aware classification** with **example-dependent misclassification costs**. This toolbox transforms traditional machine learning from "matching labels" to "**maximizing business profit**".

## 🎯 The Business Problem

Traditional classification (Cross-Entropy) treats all errors as equal. In the real world, **some mistakes are far more expensive than others**:
- **False Decline:** Turning away a legit customer costs the transaction margin + customer frustration + possible churn.
- **False Approval (Fraud):** Accepting a stolen card costs the full transaction amount + chargeback fees + operational overhead.

This repository implements **Optimal Transport (OT)** based loss functions that "understand" these costs during training, allowing models to make decisions that minimize financial regret rather than just counting errors.

---

## 📍 Table of Contents

- [The Business Problem](#-the-business-problem)
- [Quick Start for Decisions Makers](#-quick-start-for-decisions-makers)
- [Available Loss Functions](#-available-loss-functions)
  - [Baseline Losses](#1-baseline-losses)
  - [Cost-Aware Losses (Optimal Transport)](#2-cost-aware-losses-optimal-transport)
- [Epsilon (ε) Tuning Guide](#️-epsilon-ε-tuning-guide)
- [Performance Tips](#-performance-tips)
- [Complete Usage Guide](#-complete-usage-guide)
- [Metrics & Evaluation](#-metrics--evaluation)
- [Choosing a Loss Function](#-choosing-a-loss-function)
- [Documentation & Resources](#-documentation)
- [Tests](#-tests)
- [ImageNet with FastText Semantic Cost](#-imagenet-with-fasttext-semantic-cost)
- [Citation](#-citation)
- [License](#-license)

---

## 💡 Quick Start for Decisions Makers

If you want to see the business impact immediately, run the comprehensive benchmark:

```bash
# Benchmark all losses against financial baselines
python -m examples.fraud_detection --loss all --epochs 15 --run-id business_impact
```

**What to look for in results:**
- **Realized Regret:** The actual money lost in production.
- **Expected Optimal Regret:** The theoretical minimum loss possible.
- **Naive Baseline:** What happens if you simply "Approve All" or "Decline All".

---

## 📋 Available Loss Functions

### Baseline Losses

#### 1. **Cross-Entropy** (`cross_entropy`)
Standard cross-entropy loss without cost awareness.

**When to use:** Baseline comparison when all misclassifications have equal cost.

**Run command:**
```bash
python -m examples.fraud_detection --loss cross_entropy --epochs 5 --run-id baseline
```

#### 2. **Weighted Cross-Entropy** (`cross_entropy_weighted`)
Sample-weighted cross-entropy with weights $w_i = C_i[y_i, 1-y_i]$ derived from the cost matrix.

**When to use:** Simple cost-aware baseline that reweights examples by their misclassification cost.

**Run command:**
```bash
python -m examples.fraud_detection --loss cross_entropy_weighted --epochs 5 --run-id weighted_baseline
```

**Additional parameters:**
- Automatically uses cost matrix to compute sample weights
- Weights are normalized by median for stability

### Cost-Aware Losses (Optimal Transport)

All OT-based losses use a cost matrix $C$ where $C_{ij}$ represents the cost of predicting class $j$ when the true class is $i$.

#### Understanding Epsilon (ε) Regularization

The entropic regularization parameter ε controls the smoothness of the optimal transport. **By default, ε is computed adaptively from the cost matrix using rule-of-thumb heuristics:**

**Adaptive Epsilon Modes** (recommended):
- **`offdiag_mean`** (default): ε = mean of off-diagonal costs × scale factor
- **`offdiag_median`**: ε = median of off-diagonal costs × scale factor  
- **`offdiag_max`**: ε = maximum off-diagonal cost × scale factor

**Benefits of adaptive ε:**
- Automatically scales with your cost matrix magnitude
- No manual tuning required
- Robust across different problem domains
- Maintains numerical stability

**Override with constant ε only if:**
- You have domain expertise suggesting a specific value
- You're doing controlled experiments comparing different ε values
- You've validated that a fixed ε outperforms adaptive methods

#### 3. **Sinkhorn-Fenchel-Young Loss** (`sinkhorn_fenchel_young`)
Implicit Fenchel–Young loss with Frank–Wolfe inner solver. Does not differentiate through the inner optimization.

**Theory:**
- Uses envelope theorem: computes optimal solution without backpropagating through solver
- Frank-Wolfe algorithm solves inner QP on the simplex
- Stable gradients, efficient computation

**When to use:** When you want provably stable gradients from implicit differentiation.

**Run command:**
```bash
# Default: adaptive epsilon (offdiag_mean)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 --run-id fenchel_young
```

**Epsilon control options:**
```bash
# Use median-based adaptive epsilon (more robust to outliers)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon-mode offdiag_median --run-id fy_median

# Use max-based adaptive epsilon (more conservative)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon-mode offdiag_max --run-id fy_max

# Scale the adaptive epsilon by 0.5 (tighter regularization)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon-scale 0.5 --run-id fy_scale05

# Scale by 2.0 (looser regularization)
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon-scale 2.0 --run-id fy_scale20
```

**Advanced: constant epsilon (not recommended unless you know what you're doing):**
```bash
# Override with constant epsilon=0.1
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 \
  --epsilon 0.1 --run-id fy_constant_eps
```

**Hyperparameters:**
- `--epsilon-mode`: Adaptive method - `offdiag_mean` (default), `offdiag_median`, `offdiag_max`
- `--epsilon-scale`: Multiplicative scale factor for adaptive ε (default: 1.0)
- `--epsilon`: Fixed ε value (overrides adaptive mode; use sparingly)
- `--cacis-solver-iter`: Frank-Wolfe iterations (default: 50)

#### 4. **Sinkhorn Envelope Loss** (`sinkhorn_envelope`)
Entropic OT loss with custom Sinkhorn solver and envelope-style gradients.

**Theory:**
- Solves entropic OT with custom Sinkhorn iterations
- Envelope gradient: treats optimal transport plan as constant during backprop
- Keeps explicit dependence on predictions through KL term
- Memory efficient, stable gradients

**When to use:** When you want full control over the Sinkhorn implementation with stable gradients.

**Run command:**
```bash
# Default: adaptive epsilon
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 --run-id envelope
```

**Epsilon control options:**
```bash
# Use median-based adaptive epsilon
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --epsilon-mode offdiag_median --run-id env_median

# Tighter regularization via scaling
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --epsilon-scale 0.5 --run-id env_tight

# More Sinkhorn iterations for convergence
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 \
  --sinkhorn-max-iter 100 --run-id envelope_iter100
```

**Hyperparameters:**
- `--epsilon-mode`: Adaptive method (default: `offdiag_mean`)
- `--epsilon-scale`: Scale factor for adaptive ε (default: 1.0)
- `--sinkhorn-max-iter`: Sinkhorn iterations (default: 50)

#### 5. **Sinkhorn Full Autodiff Loss** (`sinkhorn_autodiff`)
Entropic OT loss with full differentiation through Sinkhorn iterations.

**Theory:**
- Backpropagates through all Sinkhorn iterations
- More "end-to-end" but higher memory usage
- May have less stable gradients for many iterations

**When to use:** When you want to compare end-to-end learning vs envelope gradients, or when iteration count is small.

**Run command:**
```bash
# Default: adaptive epsilon
python -m examples.fraud_detection --loss sinkhorn_autodiff --epochs 5 --run-id autodiff
```

**Epsilon control options:**
```bash
# Median-based epsilon with fewer iterations (saves memory)
python -m examples.fraud_detection --loss sinkhorn_autodiff --epochs 5 \
  --epsilon-mode offdiag_median --sinkhorn-max-iter 30 --run-id autodiff_efficient
```

**Hyperparameters:**
- `--epsilon-mode`: Adaptive method (default: `offdiag_mean`)
- `--epsilon-scale`: Scale factor (default: 1.0)
- `--sinkhorn-max-iter`: Sinkhorn iterations (default: 50, consider reducing for memory)

**Memory considerations:** This loss stores intermediate tensors for backprop. Use fewer iterations if memory is limited.

#### 6. **Sinkhorn POT Loss** (`sinkhorn_pot`)
Entropic OT loss using the [Python Optimal Transport (POT)](https://pythonot.github.io/) library with envelope gradients.

**Theory:**
- Uses POT's battle-tested, optimized Sinkhorn implementation
- Envelope gradients (same as `sinkhorn_envelope`) for stability
- GPU acceleration via POT's backend
- Best numerical stability from mature implementation

**When to use:** Production scenarios requiring reliability, or when comparing custom implementations against established library.

**Run command:**
```bash
# Default: adaptive epsilon
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 --run-id pot
```

**Epsilon control options:**
```bash
# Use median-based epsilon (recommended for production)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon-mode offdiag_median --run-id pot_production

# Conservative regularization with max-based epsilon
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon-mode offdiag_max --sinkhorn-max-iter 100 --run-id pot_conservative

# Fine-tune with epsilon scaling
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon-scale 0.8 --run-id pot_tuned
```

**Hyperparameters:**
- `--epsilon-mode`: Adaptive method (default: `offdiag_mean`)
- `--epsilon-scale`: Scale factor (default: 1.0)
- `--sinkhorn-max-iter`: Maximum Sinkhorn iterations (default: 50)

**Benefits:**
- Mature, optimized implementation
- GPU support through POT backends
- Numerical stability improvements
- Active maintenance by POT community

## 🎛️ Epsilon (ε) Tuning Guide

### Quick Reference

| Epsilon Mode | When to Use | Characteristics |
|--------------|-------------|-----------------|
| `offdiag_mean` ✅ | **Default choice** | Balanced, works well in most cases |
| `offdiag_median` | Outlier-heavy costs | Robust to extreme cost values |
| `offdiag_max` | Conservative needs | Ensures all costs are regularized |

### Scaling Strategy

The `--epsilon-scale` parameter multiplies the adaptive ε:

| Scale | Effect | Use When |
|-------|--------|----------|
| < 1.0 | Tighter regularization, sharper solutions | Costs have clear structure, want crisp decisions |
| = 1.0 | **Default**, balanced | Start here |
| > 1.0 | Looser regularization, smoother solutions | Costs are noisy, want robust solutions |

**Example workflow:**
```bash
# 1. Start with default
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 --run-id baseline

# 2. If overfitting, increase scale
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon-scale 2.0 --run-id smoother

# 3. If underfitting, decrease scale
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --epsilon-scale 0.5 --run-id sharper
```

### Epsilon Scheduling (Advanced)

For longer training runs, you can use **epsilon scheduling** to automatically adjust epsilon over epochs:

#### Exponential Decay Strategy

Start with high epsilon (smooth, stable) and gradually decrease to low epsilon (sharp, decisive):

```bash
# Exponential decay: 10× base epsilon at epoch 0 → 0.1× at final epoch
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 \
  --epsilon-schedule exponential_decay --run-id scheduled
```

**How it works:**
- **Epoch 0**: ε = 10 × base_epsilon (very smooth OT, stable gradients)
- **Mid-training**: ε gradually decreases exponentially
- **Final epoch**: ε = 0.1 × base_epsilon (sharp decisions)

**Customize the schedule:**
```bash
# Start at 20× and end at 0.05×
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 \
  --epsilon-schedule exponential_decay \
  --epsilon-schedule-start-mult 20.0 \
  --epsilon-schedule-end-mult 0.05 \
  --run-id custom_schedule
```

**When to use scheduling:**
- ✅ Long training runs (>10 epochs)
- ✅ Want stable early training with sharp final predictions
- ✅ Dealing with difficult optimization landscapes
- ❌ Not needed for short experiments (<5 epochs)

**Example output** (10 epochs, default schedule):
```
Epoch 0: ε = 15.00 (10.0× base)
Epoch 1: ε =  8.99 (6.0× base)
Epoch 2: ε =  5.39 (3.6× base)
...
Epoch 9: ε =  0.15 (0.1× base)
```

**Parameters:**
- `--epsilon-schedule`: `None` (default) or `exponential_decay`
- `--epsilon-schedule-start-mult`: Starting multiplier (default: 10.0)
- `--epsilon-schedule-end-mult`: Ending multiplier (default: 0.1)

## 📊 Metrics & Evaluation

To truly measure business success, we move beyond Accuracy and ROC-AUC:

- **PR-AUC (Precision-Recall Area Under Curve):** Primary classification metric for imbalanced fraud data (**higher is better**).
- **Luck Baseline:** A horizontal line on the PR curve representing a random classifier (equivalent to fraud prevalence).
- **Expected Optimal Regret:** The expected business cost incurred if we make the mathematically optimal decision based on our model's predictions (**lower is better**).
- **Realized Regret:** The actual money lost by following the model's decisions on a test set (**lower is better**). Includes:
  - Total $ lost to accepted fraud.
  - Total $ value lost due to false declines.
- **Naive Baseline:** The smoothed average of the better of two constant strategies: "Approve Everything" ($0 fraud detection, massive fraud losses) or "Decline Everything" ($0 fraud losses, massive lost sales). **Your model must beat this to be useful.**

---

## ⚡ Performance Tips

### Robust Data Loading
We recommend using the Python engine for reading large IEEE-CIS CSV files to avoid `ParserError` issues:
```python
df = pd.read_csv('train_transaction.csv', engine='python')
```

### Numerical Stability
Fraud datasets often have extreme values. We recommend:
- **`RobustScaler`**: To handle outliers in transaction amounts.
- **`CosineAnnealingLR`**: For smoother convergence during training.
- **Lower Learning Rates**: Starting at `1e-5` often provides more stable gradients for Sinkhorn-based losses.

### Device Selection

**Apple Silicon (M*):**
```bash
# CPU is often faster than MPS for POT-based losses
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 --device cpu

# MPS works well for other losses
python -m examples.fraud_detection --loss sinkhorn_envelope --epochs 5 --device mps
```

**NVIDIA GPU:**
```bash
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 --device cuda
```

### Speed Optimizations

**Faster training (lower accuracy):**
```bash
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --batch-size 128 --sinkhorn-max-iter 10
```

**Balanced (recommended):**
```bash
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --batch-size 256 --sinkhorn-max-iter 20
```

**Quick testing:**
```bash
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 2 --quick
```

## 🚀 Complete Usage Guide

### Installation

Install conda if you haven't already. See [conda](https://harchaoui.org/warith/4ml) for instructions.

```bash
PROJECT=cost_aware_fraud
GITHUB_OWNER=warith-harchaoui

REPOURL=https://github.com/$GITHUB_OWNER/$PROJECT.git
ENV=env4$PROJECT

git clone $REPOURL
cd $PROJECT
conda update -y -n base -c defaults conda
conda create -y -n $ENV python=3.10
conda activate $ENV
conda install -y pip
pip install -r requirements.txt
```

### Dataset Download

Download the IEEE-CIS fraud detection dataset:

```bash
rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true
mkdir ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip
unzip ieee-fraud-detection.zip -d ieee-fraud-detection
```

### Basic Usage Examples

#### Run a Single Loss
```bash
# Train with cross-entropy for 5 epochs
python -m examples.fraud_detection --loss cross_entropy --epochs 5 --run-id exp1

# Train with SinkhornPOTLoss (adaptive epsilon)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 --run-id pot_exp

# Train with Fenchel-Young loss
python -m examples.fraud_detection --loss sinkhorn_fenchel_young --epochs 5 --run-id fy_exp
```

#### Run All Losses (Benchmark)
```bash
# Compare all 6 loss functions
python -m examples.fraud_detection --loss all --epochs 5 --run-id benchmark1
```

This will train 6 separate models (one per loss) and save results in:
```
fraud_output/benchmark1/cross_entropy/
fraud_output/benchmark1/cross_entropy_weighted/
fraud_output/benchmark1/sinkhorn_fenchel_young/
fraud_output/benchmark1/sinkhorn_envelope/
fraud_output/benchmark1/sinkhorn_autodiff/
fraud_output/benchmark1/sinkhorn_pot/
```

### Advanced Options

#### Resume Training
Continue training from a checkpoint for additional epochs:

```bash
# Train 5 epochs initially
python -m examples.fraud_detection --loss all --epochs 5 --run-id longrun

# Continue for 3 MORE epochs (total: 8)
python -m examples.fraud_detection --loss all --epochs 3 --run-id longrun --resume
```

#### Model Architecture
```bash
# Use linear model instead of MLP
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --backbone linear --run-id linear_model

# Custom MLP architecture
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --backbone mlp --hidden-dims 512,256,128 --dropout 0.2 --run-id deep_mlp

# Disable batch normalization
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --no-batchnorm --run-id no_bn
```

#### Training Configuration
```bash
# Custom batch size and learning rate
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 10 \
  --batch-size 1024 --lr 5e-4 --run-id custom_training

# Quick test run (few batches per epoch)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 2 \
  --quick --run-id quick_test

# Custom train/val split
python -m examples.fraud_detection --loss all --epochs 5 \
  --split 0.2 --run-id split20
```

#### Device Selection
```bash
# Force CPU
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device cpu --run-id cpu_run

# Force CUDA (if available)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device cuda --run-id gpu_run

# Use MPS (Apple Silicon)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device mps --run-id mps_run

# Auto-detect (default)
python -m examples.fraud_detection --loss sinkhorn_pot --epochs 5 \
  --device auto --run-id auto_device
```

### Output Artifacts

Each run creates a directory `fraud_output/<run-id>/<loss_name>/` with:

```
checkpoint_last.pt          # Latest checkpoint (for resuming)
checkpoint_best.pt          # Best checkpoint (by PR-AUC)
train_smoothed_metrics.csv       # Training metrics (smoothed)
val_metrics.csv               # Validation metrics over time
val_precision_recall_curve.png  # PR curve visualization
val_pr_auc.png                  # PR-AUC vs iteration (the higher, the better)
val_expected_opt_regret.png     # Expected optimal regret (the lower, the better)
val_realized_regret.png         # Realized regret (the lower, the better)
train_expected_opt_regret.png   # Training regret (smoothed, the lower, the better)
train_realized_regret.png       # Training realized regret (the lower, the better)
train_pr_auc.png                # Training PR-AUC (smoothed, the higher, the better)
train_precision_recall_curve.png # Training PR curve
```


## 🎯 Choosing a Loss Function

| Loss | Pros | Cons | Best For |
|------|------|------|----------|
| `cross_entropy` | Simple, fast, well-understood | Ignores costs | Baseline comparison |
| `cross_entropy_weighted` | Simple cost integration | Doesn't optimize transport | Quick cost-aware baseline |
| `sinkhorn_fenchel_young` | Provably stable gradients | Custom implementation | Research, reproducibility |
| `sinkhorn_envelope` | Stable, memory efficient | Custom implementation | When memory is limited |
| `sinkhorn_autodiff` | End-to-end learning | High memory, less stable | Research comparison |
| `sinkhorn_pot` | Production-ready, optimized | External dependency | **Production deployments** ⭐ |

**Recommendation for production:** Start with `sinkhorn_pot` with default adaptive epsilon (`offdiag_mean`).

**Recommendation for research:** Compare `sinkhorn_envelope`, `sinkhorn_autodiff`, and `sinkhorn_fenchel_young` to understand gradient quality vs computational trade-offs.

## 📚 Documentation & Resources

- [**`docs/math.md`**](docs/math.md) — Mathematical foundations and the explicit mapping between $\varepsilon$ and POT's `reg` parameter.
- [**`docs/fraud_business_and_cost_matrix.md`**](docs/fraud_business_and_cost_matrix.md) — Business value model and per-example cost matrix construction for fraud detection.
- [**`examples/sinkhorn_pot_example.py`**](examples/sinkhorn_pot_example.py) — Standalone example demonstrating `SinkhornPOTLoss` usage.

## 🖼️ ImageNet with FastText Semantic Cost

Beyond fraud, the same cost-aware machinery applies to **semantic classification**: a confusion between *tiger* and *leopard* is much less costly than a confusion between *tiger* and *minivan*. The `examples/imagenet/` subpackage trains a torchvision ResNet from scratch on ImageNet-1k with a 1000×1000 cost matrix built from **FastText** class-name embeddings.

### Pipeline

1. **Precompute the cost matrix (once, on a laptop).** Embeds each of the 1000 ImageNet class names with FastText and writes `C[i, j] = 1 − cos(emb_i, emb_j)`.
   ```bash
   python -m examples.imagenet.cost_matrix \
       --data-root /data/imagenet \
       --fasttext /models/cc.en.300.bin \
       --out cost_matrix.pt
   ```
   The result is a small (~4 MB) `cost_matrix.pt` file mounted into training as a model artifact.
2. **Train.** DDP-aware, AMP-on, cosine LR with linear warmup.
   ```bash
   torchrun --standalone --nproc-per-node=8 \
       -m examples.imagenet.train \
       --data-root /data/imagenet \
       --cost-matrix cost_matrix.pt \
       --loss sinkhorn_envelope \
       --batch-size 64 --epochs 90
   ```

### Loss recommendation for K=1000

- **`sinkhorn_envelope`** (default) — GPU-native, memory-efficient envelope gradient. **Recommended for ImageNet.**
- **`sinkhorn_autodiff`** — only for ablation; memory cost scales with `sinkhorn-max-iter`.
- **`sinkhorn_pot`** — slow at K=1000 because POT's solver runs per example. Reserve for K ≲ 100.

`offdiag_mean` ε is computed **once** from the shared cost matrix at startup and converted to a constant — avoiding a per-batch reduction over a (B, 1000²) mask.

### Cloud setup (AWS / GCP)

A self-contained Dockerfile lives at `examples/imagenet/Dockerfile`. The header comments include the exact ECR push, Artifact Registry push, and `docker run` commands for both clouds.

**AWS — single multi-GPU EC2 (e.g. `p4d.24xlarge`):**
```bash
docker run --gpus all --shm-size=8g \
    -v /mnt/imagenet:/data/imagenet:ro \
    -v /mnt/output:/workspace/imagenet_output \
    -v /mnt/models:/models:ro \
    <acct>.dkr.ecr.<region>.amazonaws.com/cacis-imagenet:latest \
    torchrun --standalone --nproc-per-node=8 \
      -m examples.imagenet.train \
        --data-root /data/imagenet \
        --cost-matrix /models/cost_matrix.pt \
        --loss sinkhorn_envelope \
        --batch-size 64 --epochs 90 \
        --output-dir /workspace/imagenet_output \
        --run-id resnet50_fasttext
```

**GCP — single multi-GPU `a2-highgpu-8g`:** same command with the GCP-pushed image and a GCSFuse / persistent-disk mount in place of `/mnt/imagenet`.

**Multi-node** (e.g. two `p4d.24xlarge` for 16 × A100): set `MASTER_ADDR` to the first node's IP and launch on each node with:
```bash
torchrun --nnodes=2 --node-rank=$NODE_RANK \
    --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29500 \
    -m examples.imagenet.train ...
```

### Files

| File | Purpose |
|------|---------|
| `examples/imagenet/cost_matrix.py` | Build `C` from FastText (`.bin` or `.vec`). CLI. |
| `examples/imagenet/classnames.py` | Load ImageNet class labels in the canonical `ImageFolder` order. |
| `examples/imagenet/data.py` | DDP-aware ImageNet DataLoaders. |
| `examples/imagenet/model.py` | torchvision ResNet factory (no pretrained weights). |
| `examples/imagenet/train.py` | DDP + AMP training loop. Reports Top-1, Top-5, and realized semantic regret. |
| `examples/imagenet/Dockerfile` | Container for AWS/GCP GPU instances. |

## ✍️ Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{harchaoui2026cacis,
  title={Cost-Aware Classification with Optimal Transport for E-commerce Fraud Detection},
  author={Harchaoui, Warith and Pantanacce, Laurent},
  booktitle={The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
```

## 🔬 Example: Comprehensive Benchmark

```bash
# Benchmark all losses with optimized settings
python -m examples.fraud_detection \
  --loss all \
  --epochs 15 \
  --run-id comprehensive_benchmark

# Results will be in fraud_output/comprehensive_benchmark/<loss_name>/
```

Compare results by examining:
1. **PR-AUC**: Which loss achieves highest precision-recall?
2. **Regret metrics**: Which loss minimizes business costs?
3. **Training curves**: Which loss converges fastest/most stably?
4. **Computation time**: Which loss is most efficient?

## 🧪 Tests

To ensure the reliability and mathematical correctness of the custom Sinkhorn loss implementations, we provide a comprehensive test suite.

### 1. Installation for Testing
Tests require the package to be installed in **editable mode** so that `cost_aware_losses` is importable by `pytest`:

```bash
pip install -e .
```

### 2. Running Tests

Run the full suite with:

```bash
pytest tests
```

### 3. Test Categories

- **Consistency Tests** (`tests/test_sinkhorn_consistency.py`):
  - Verifies that different implementations (`SinkhornPOTLoss`, `SinkhornEnvelopeLoss`, `SinkhornFullAutodiffLoss`) output consistent loss values for the same inputs.
  - Checks that gradients match across implementations (e.g., confirming that the envelope gradient graft matches the autodiff gradient).

- **Advanced Verification** (`tests/test_sinkhorn_advanced.py`):
  - **`test_gradcheck`**: Uses `torch.autograd.gradcheck` (finite differences) to mathematically prove the correctness of our custom backward passes. This is critical for confirming the "Gradient Grafting" technique used in `SinkhornPOTLoss` and `SinkhornEnvelopeLoss` (fixing the zero-gradient issue of standard envelope theorems on discrete measures).
  - **`test_epsilon_limit_convergence`**: Verifies that as entropic regularization $\varepsilon \to 0$, the Sinkhorn loss converges to the exact Optimal Transport cost (Earth Mover's Distance).
  - **`test_extreme_costs`**: Checks numerical stability with very large cost values (e.g., $10^5$), ensuring no `NaN` or `Inf` outputs.
  - **`test_cost_shift_invariance`**: Verifies theoretical properties, such as the loss increasing by exactly $k$ when a constant $k$ is added to the cost matrix, while gradients remain invariant.

## 📜 License

**Unlicense** — This is free and unencumbered software released into the public domain.  
See [UNLICENSE](https://unlicense.org) for details.

## 🙏 Acknowledgments

- POT library: https://pythonot.github.io/
- IEEE-CIS Kaggle competition for the dataset
